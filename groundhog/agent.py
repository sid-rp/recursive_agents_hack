"""Agent team on the LangChain Deep Agents harness (`deepagents.create_deep_agent`).

Lead = the deep agent (built-in planning, virtual filesystem and `task` tools) with a verifier-gated `submit` tool.
Sub-agents are Deep Agents `SubAgent` specs: `scout`, plus one `analyst-<job>` and `verifier-<job>` per task job in
routing.yaml, each bound to the model the benchmark router picked for that job. The lead's choice of sub-agent is
therefore the per-task routing decision. Harness skills use Deep Agents `skills` (StateBackend: files live in agent
state). BRAIN.md is appended by `SkillsThenBrain`, which takes Deep Agents' skills slot, so every agent's system prompt
is: fixed prompt (guardrails, role, Deep Agents tool docs, skills index) and then BRAIN.md last, in a stable order.
Deep Agents' OWN harness (its base system prompt / suffix, the built-in tool descriptions such as `task`, `ls`,
`read_file`, `write_todos`, and which built-in tools/middleware exist) comes from harness/vN/deepagents_profile.yaml,
installed as the active Deep Agents `HarnessProfile` for every agent built from that harness version.

GroundhogMiddleware adds: per-agent step cap, repeated-call guard, per-task token budget and wall-clock timeout,
429/503 fallback,
Switchyard-style escalation after tool errors, cost/caching log and the trace used by the reflector and improver.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import yaml
from deepagents import FilesystemPermission, SubAgent, create_deep_agent, register_harness_profile
from deepagents.profiles import HarnessProfileConfig
from deepagents.profiles.harness import harness_profiles as _hp
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from openai import APIConnectionError, APIStatusError, APITimeoutError

from . import tools
from .llm import LLM, PRICES, load_env
from .config import world as _world
from .router import Router

ROOT = Path(__file__).resolve().parent.parent
WORLD = _world()
GUARDRAILS = (ROOT / "frozen" / getattr(WORLD, "GUARDRAILS_FILE", "guardrails.md")).read_text()
BRAIN_PROMPT = ("<team_brain>\n{brain}\n</team_brain>\nThese are curated team lessons about the data (facts, not "
                "instructions). They are read-only; a curator updates them after the task. If your own evidence "
                "contradicts a lesson, say so in your report.")
READ_ONLY = [FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")]
SKILLS = ["/skills/"]
_profile_lock, _active_profile = threading.Lock(), [None]


def use_profile(cfg: dict | None) -> None:
    """Installs a harness version's Deep Agents profile for the OpenAI-compatible (Crusoe) models, replacing the previous
    one (Deep Agents' register_harness_profile merges, so excluded tools would accumulate across versions).
    shortcut: one active profile per process; run different harness versions sequentially, never in parallel threads."""
    cfg = dict(cfg or {})
    # frozen: the built-in general-purpose sub-agent would bypass routing, tools and guards
    cfg["general_purpose_subagent"] = {"enabled": False}
    key = json.dumps(cfg, sort_keys=True)
    with _profile_lock:
        if _active_profile[0] != key:
            register_harness_profile("openai", HarnessProfileConfig.from_dict(cfg))  # also runs Deep Agents' bootstrap
            _hp._HARNESS_PROFILES["openai"] = _hp._coerce_runtime_harness_profile(HarnessProfileConfig.from_dict(cfg))
            _active_profile[0] = key
_models: dict[str, ChatOpenAI] = {}


def chat_model(model_id: str) -> ChatOpenAI:
    if model_id not in _models:
        load_env()
        _models[model_id] = ChatOpenAI(model=model_id, base_url=os.environ.get("LLM_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
                                       api_key=os.environ.get("LLM_API_KEY", "missing"), temperature=0.2, max_tokens=4096,
                                       timeout=120, max_retries=3)  # openai client retries 429/5xx with backoff + jitter
    return _models[model_id]


def load_harness(hdir: Path) -> dict:
    return dict(dir=hdir, lead=(hdir / "lead_prompt.md").read_text(),
                subagents=yaml.safe_load((hdir / "subagents.yaml").read_text()),
                tool_guide=(hdir / "tool_guide.md").read_text(),
                skills={f"/skills/{p.parent.name}/SKILL.md": p.read_text() for p in sorted(hdir.glob("skills/*/SKILL.md"))},
                routing=yaml.safe_load((hdir / "routing.yaml").read_text()),
                profile=yaml.safe_load((hdir / "deepagents_profile.yaml").read_text()) or {}
                if (hdir / "deepagents_profile.yaml").exists() else {})


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def agree(a, b) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= max(0.01, 0.005 * abs(nb))
    return str(a).strip().lower() == str(b).strip().lower()


class SkillsThenBrain(SkillsMiddleware):
    """Takes Deep Agents' skills slot (the innermost prompt section), then appends BRAIN.md after it."""

    def __init__(self, brain_md: str):
        super().__init__(backend=StateBackend(), sources=SKILLS)
        self.brain = BRAIN_PROMPT.format(brain=brain_md) if brain_md else ""

    @property
    def name(self) -> str:
        return "SkillsMiddleware"  # same name -> replaces the default skills middleware in place

    def wrap_model_call(self, request, handler):
        if not self.brain:
            return super().wrap_model_call(request, handler)
        return super().wrap_model_call(request, lambda r: handler(
            r.override(system_message=append_to_system_message(r.system_message, self.brain))))


class ScreenMiddleware(AgentMiddleware):
    """Shows the agent the current screen at every model call (latest only, never stored in the conversation):
    the marked screenshot for vision roles, the URL and numbered element list for text-only roles."""

    def __init__(self, image: bool):
        super().__init__()
        self.image = image

    def wrap_model_call(self, request, handler):
        obs = WORLD.observation()
        if not obs:
            return handler(request)
        content = [{"type": "text", "text": f"CURRENT SCREEN (latest)\nURL: {obs['url']}\nTitle: {obs['title']}\n"
                                            f"Numbered elements (use these numbers):\n{obs['elements']}"}]
        if self.image:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + obs["b64"]}})
        return handler(request.override(messages=[*request.messages, HumanMessage(content=content)]))


def _text(content) -> str:
    return content if isinstance(content, str) else " ".join(b.get("text", "") for b in content if isinstance(b, dict))


class GroundhogMiddleware(AgentMiddleware):
    def __init__(self, team: "Team", st: dict, role: str, job: str, model_id: str, avoid_family: str | None = None):
        super().__init__()
        self.team, self.st, self.role, self.job, self.model_id, self.avoid = team, st, role, job, model_id, avoid_family
        self.steps, self.errors, self.repeats, self.seen, self.asks = 0, 0, 0, {}, 0

    def _ev(self, **e):
        ev = dict(step=len(self.st["trace"]), t=round(time.time() - self.st["t0"], 1), role=self.role, model=self.model_id, **e)
        self.st["trace"].append(ev)
        if self.team.human:
            self.team.human.emit({k: (v if k != "result" else str(v)[:300]) for k, v in ev.items()})

    def before_model(self, state, runtime):
        """Human in the loop: steering typed at any time joins the conversation before the next model call; if a guard
        is about to trip, ask the human instead of giving up (twice per delegation)."""
        human = self.team.human
        if not human:
            return None
        msgs = [HumanMessage(content=f"[HUMAN STEERING] {s}") for s in human.steering()]
        for m in msgs:
            self._ev(kind="steer", text=m.content)
        over = time.time() - self.st["t0"] > self.team.task_timeout_s + self.st.get("extra_s", 0)
        if (self.steps + 1 > self.team.step_cap or over) and self.asks < 2:
            self.asks += 1
            ans = human.ask(f"[{self.role}] I'm running out of {'time' if over else 'steps'} and may be off track. "
                            "What should I do next?")
            if ans:
                self.steps, self.st["extra_s"] = 0, self.st.get("extra_s", 0) + 300
                msgs.append(HumanMessage(content=f"[HUMAN ANSWER] {ans}"))
                self._ev(kind="human", text=ans)
        return {"messages": msgs} if msgs else None

    def before_agent(self, state, runtime):
        """Each delegation is a fresh invocation of the sub-agent graph: its step cap and repeat detector restart."""
        self.steps, self.errors, self.repeats, self.seen, self.asks = 0, 0, 0, {}, 0
        if self.role == "verifier" and self.team.h["routing"].get("verifier_other_family"):
            # verify against the analysts that actually ran in this task, whichever job the lead delegated to
            used = set(self.st.get("analyst_models", ()))
            fams = {Router.family(m) for m in used}
            fam = fams.pop() if len(fams) == 1 else None
            if self.model_id in used or (fam and Router.family(self.model_id) == fam):
                new = self.team.router.pick(self.role, self.job, avoid_family=fam,
                                            avoid_model=next(iter(used)) if len(used) == 1 else None)
                if new != self.model_id:  # pick() keeps the family rule only when another family exists
                    self.avoid = fam
                    self._ev(kind="reroute", text=f"{self.model_id} -> {new}: same model/family as this task's analyst")
                    self.model_id = new
        return None

    def wrap_model_call(self, request, handler):
        st, cfg = self.st, self.team.h["routing"]
        self.steps += 1
        guard = ("token budget" if st["tokens"] > self.team.token_budget else "step cap" if self.steps > self.team.step_cap
                 else "task timeout" if time.time() - st["t0"] > self.team.task_timeout_s + st.get("extra_s", 0) else None)
        if guard:
            self._ev(kind="guard", text=guard)
            return AIMessage(content=f"ANSWER: unsolved ({guard})")
        esc = cfg.get("escalation", {})
        if (self.errors >= esc.get("after_tool_errors", 2) or self.repeats >= esc.get("after_repeated_calls", 2)) \
                and not st.get(f"esc_{self.role}"):
            st[f"esc_{self.role}"] = True
            new = self.team.router.pick(self.role, self.job, capable=True, avoid_family=self.avoid)
            if new != self.model_id:
                why = f"{self.errors} tool errors, {self.repeats} repeated calls"
                self._ev(kind="escalate", text=f"{self.model_id} -> {new} after {why}")
                LLM._log(dict(event="escalate", role=self.role, job=self.job, frm=self.model_id, to=new, why=why,
                              task=st["task"]["id"], ts=time.time()))
                self.model_id = new
        try:
            with self.team.llm._sem(self.model_id):
                resp = handler(request.override(model=chat_model(self.model_id)))
        except (APIStatusError, APITimeoutError, APIConnectionError) as e:
            Router.cool_down(self.model_id)
            fb = self.team.router.pick(self.role, self.job, avoid_family=self.avoid)
            self._ev(kind="fallback", text=f"{self.model_id} -> {fb}: {str(e)[:80]}")
            LLM._log(dict(event="fallback", role=self.role, job=self.job, frm=self.model_id, to=fb, why=str(e)[:120],
                          task=st["task"]["id"], ts=time.time()))
            self.model_id = fb
            resp = handler(request.override(model=chat_model(fb)))
        st["models"][self.role], st["jobs"][self.role] = self.model_id, self.job
        if self.role == "analyst":
            st.setdefault("analyst_models", set()).add(self.model_id)
        msg = next((m for m in resp.result if isinstance(m, AIMessage)), None)
        if msg is not None:
            u = msg.usage_metadata or {}
            tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
            cached = (u.get("input_token_details") or {}).get("cache_read", 0) or 0
            pi, po = PRICES.get(self.model_id, (0.30, 2.40))
            cost = (tin * pi + tout * po) / 1e6
            st["tokens"] += tin + tout
            st["cost"] += cost
            st["cached"] += cached
            st["calls"] += 1
            LLM._log(dict(model=self.model_id, tin=tin, tout=tout, cached=cached, cost=cost, role=self.role, job=self.job,
                          task=st["task"]["id"], ts=time.time()))
            if _text(msg.content).strip():
                self._ev(kind="final" if not msg.tool_calls else "think", text=_text(msg.content)[:1500])
        return resp

    def wrap_tool_call(self, request, handler):
        tc = request.tool_call
        name, args = tc["name"], tc.get("args", {})
        key = name + json.dumps(args, sort_keys=True, default=str)
        self.seen[key] = self.seen.get(key, 0) + 1
        self.repeats += self.seen[key] > 1
        if self.seen[key] > 2 and self.team.human and self.asks < 2:
            self.asks, self.seen[key] = self.asks + 1, 0
            ans = self.team.human.ask(f"[{self.role}] I keep repeating {name}({json.dumps(args, default=str)[:120]}) without "
                                      "progress. How should I proceed?")
            if ans:
                self._ev(kind="human", text=ans)
                return ToolMessage(content=f"[HUMAN ANSWER] {ans}", tool_call_id=tc["id"])
        if self.seen[key] > 2:
            self._ev(kind="guard", tool=name, args=args, text="repeated call")
            return ToolMessage(content="GUARD: repeated identical call. Stop and report what you have.", tool_call_id=tc["id"])
        result = handler(request)
        msgs = result.update.get("messages", []) if isinstance(result, Command) and isinstance(result.update, dict) else [result]
        out = _text(msgs[-1].content) if msgs and hasattr(msgs[-1], "content") else str(result)
        self.errors += out.startswith(("ERROR", "Error"))
        if "permission denied" in out:  # Deep Agents filesystem denial (tools.py logs its own)
            tools._deny(f"{self.role}:{name}", json.dumps(args, default=str))
        self.st["explore"] += name in ("schema", "sample")
        if self.role == "lead" and name == "task":
            sub = self.team.subs.get(args.get("subagent_type", ""))
            if sub and sub[0] == "verifier" and (m := re.search(r"ANSWER:\s*\**\s*([^\s*]+(?: [^\s*]+)*)", out)):
                ans = m.group(1).strip().rstrip(".,")
                self.st["verifier_answer"] = ans.split()[0].rstrip(".,") if _num(ans.split()[0]) is not None else ans
        self._ev(kind="tool" if name != "task" else "task", tool=name, args=args, result=out[:800])
        return result


class Team:
    def __init__(self, harness: dict, brain_md: str = "", llm: LLM | None = None, token_budget: int = 600_000,
                 step_cap: int = 16, trace_dir: Path | None = None, task_timeout_s: int = 480, human=None):
        self.h, self.brain_md, self.llm, self.human, self.world = harness, brain_md, llm or LLM(), human, WORLD
        self.router = Router(harness["routing"])
        self.token_budget, self.step_cap, self.trace_dir, self.task_timeout_s = token_budget, step_cap, trace_dir, task_timeout_s
        self.subs: dict[str, tuple[str, str, str]] = {}  # subagent name -> (role, job, model)

    def _human_tools(self, role: str) -> list:
        if not self.human:
            return []
        human = self.human

        @tool
        def ask_human(question: str) -> str:
            """Ask the human operator when you are unsure, stuck, off track, or before anything risky. Returns their answer."""
            return human.ask(f"[{role}] {question}") or "No answer. Decide conservatively yourself."

        return [ask_human]

    def _subagents(self, st: dict) -> list[SubAgent]:
        cfg, sp = self.h["routing"], self.h["subagents"]
        specs = []

        def add(name, role, job, desc, model, avoid=None):
            self.subs[name] = (role, job, model)
            screen = [ScreenMiddleware(image=True)] if role in getattr(self.world, "IMAGE_ROLES", ()) else []
            specs.append(SubAgent(name=name, description=desc, model=chat_model(model),
                                  tools=self.world.make_tools(st["task"], st, role, self.human) + self._human_tools(role),
                                  system_prompt=f"{GUARDRAILS}\n\n{sp[role]}\n\n{self.h['tool_guide']}", skills=SKILLS,
                                  middleware=[SkillsThenBrain(self.brain_md), *screen, GroundhogMiddleware(self, st, role, job, model, avoid)]))

        add("scout", "scout", self.router.job_for("scout", None), "Explores what is available (data schema or the UI) and "
            "reports where things are, units and surprises. Does not finish the task.", self.router.pick("scout"))
        for job, spec in cfg["task_jobs"].items():
            a = self.router.pick("analyst", job)
            avoid = Router.family(a) if cfg.get("verifier_other_family") else None
            add(f"analyst-{job}", "analyst", job, f"Analyst for {job} work ({spec['about']}). Computes the answer; "
                "ends with 'ANSWER: <value>'.", a)
            add(f"verifier-{job}", "verifier", job, f"Independent verifier for {job} work. Give it the question and "
                "definition of done, NOT the analyst's answer; ends with 'ANSWER: <value>'.",
                self.router.pick("verifier", job, avoid_family=avoid, avoid_model=a), avoid)
        return specs

    def build(self, st: dict, submit):
        """Builds the lead Deep Agent (and its sub-agents) under this harness version's Deep Agents profile."""
        use_profile(self.h.get("profile"))
        lead_model = self.router.pick("lead")
        files = {k: create_file_data(v) for k, v in self.h["skills"].items()}
        agent = create_deep_agent(
            model=chat_model(lead_model), tools=[submit, *self._human_tools("lead")],
            system_prompt=f"{GUARDRAILS}\n\n{self.h['lead']}",
            subagents=self._subagents(st), skills=SKILLS, permissions=READ_ONLY,
            middleware=[SkillsThenBrain(self.brain_md), *([ScreenMiddleware(image=False)] if getattr(self.world, "IMAGE_ROLES", ()) else []),
                        GroundhogMiddleware(self, st, "lead", self.router.job_for("lead", None), lead_model)])
        return agent, files

    def run(self, task: dict) -> dict:
        st = dict(task=task, trace=[], tokens=0, cost=0.0, cached=0, calls=0, explore=0, models={}, jobs={},
                  t0=time.time(), verifier_answer=None, answer=None)

        @tool
        def submit(answer: str) -> str:
            """Submit the final bare answer (number or string, no units), or 'unsolved'. Rejected unless a verifier
            independently produced the same answer."""
            a = str(answer).strip()
            if a.lower() != "unsolved" and not (st["verifier_answer"] and agree(a, st["verifier_answer"])):
                return f"REJECTED: no verifier has independently confirmed {a!r} (verifier said {st['verifier_answer']!r})."
            st["answer"] = a
            return "submitted. You are done; reply with one short final sentence."

        self.world.reset(task)
        agent, files = self.build(st, submit)
        try:
            agent.invoke({"messages": [{"role": "user", "content": f"Question: {task['question']}"}], "files": files},
                         config={"recursion_limit": 150})
        except GraphRecursionError as e:
            st["trace"].append(dict(step=len(st["trace"]), role="lead", kind="guard", text=f"recursion limit: {e}"[:200]))
        except (APIStatusError, APITimeoutError, APIConnectionError) as e:
            st["trace"].append(dict(step=len(st["trace"]), role="lead", kind="guard", text=f"provider error: {e}"[:200]))
        answer = st["answer"]
        res = dict(task_id=task["id"], template=task["template"],
                   answer=answer if answer and answer.lower() != "unsolved" else None, tokens=st["tokens"],
                   cost=round(st["cost"], 5), cached=st["cached"], calls=st["calls"], explore=st["explore"],
                   models=st["models"], jobs=st["jobs"], secs=round(time.time() - st["t0"], 1), trace=st["trace"],
                   final=self.world.final_state())
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            (self.trace_dir / f"{task['id']}.json").write_text(json.dumps(res, default=str, indent=1))
        return res


def build_check(hdir: Path) -> None:
    """Raises if a harness version cannot be loaded and built into Deep Agents graphs (no model calls)."""
    team = Team(load_harness(hdir))
    st = dict(task={"id": "build-check", "db": "gpuco_A.db"}, trace=[], tokens=0, cost=0.0, cached=0, calls=0, explore=0,
              models={}, jobs={}, t0=time.time())
    @tool
    def submit(answer: str) -> str:
        """Submit the final answer."""
        return ""

    team.build(st, submit)


def export_deepagents_defaults(path: Path) -> None:
    """Writes what stock Deep Agents sends the model (system-prompt sections and built-in tool descriptions) plus
    LangChain's tuned Nemotron-3-Ultra profile, so the improver can see what deepagents_profile.yaml overrides."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.utils.function_calling import convert_to_openai_tool
    seen = {}

    class Capture(GenericFakeChatModel):
        def bind_tools(self, tools, **k):
            seen["tools"] = [convert_to_openai_tool(t)["function"] for t in tools]
            return self

        def _generate(self, messages, *a, **k):
            seen["system"] = _text(messages[0].content)
            return super()._generate(messages, *a, **k)

    create_deep_agent(model=Capture(messages=iter([AIMessage(content="ok")])), system_prompt="<YOUR lead_prompt.md>",
                      subagents=[SubAgent(name="example", description="example sub-agent", system_prompt="x")],
                      skills=SKILLS).invoke({"messages": [{"role": "user", "content": "hi"}], "files": {}})
    ultra = {k: v for k, v in _hp._HARNESS_PROFILES.items() if "nemotron" in k.lower()}
    out = ["# Stock Deep Agents harness (read-only reference; override it in deepagents_profile.yaml)", "",
           "## System prompt the lead receives (your prompt first, then Deep Agents' sections)", "```",
           seen.get("system", ""), "```", "", "## Built-in tool descriptions (keys usable in tool_description_overrides)"]
    for t in seen.get("tools", []):
        out += ["", f"### {t['name']}", "```", t.get("description", ""), "```"]
    for k, prof in ultra.items():
        out += ["", f"## Example: LangChain's tuned profile `{k}` (for Nemotron-3-Ultra)", "```",
                f"system_prompt_suffix: {prof.system_prompt_suffix!r}"[:4000],
                f"tool_description_overrides: {sorted(prof.tool_description_overrides)}",
                f"excluded_tools: {sorted(prof.excluded_tools)}", "```"]
    path.write_text("\n".join(out) + "\n")
