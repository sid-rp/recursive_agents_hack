# Groundhog: an agent team that doesn't relive the same day

A Nemotron/DeepSeek agent team on Crusoe that shares one curated brain (what the team *knows*) and rewrites its own
harness between runs (how the team *works*). A harness change is kept only if it wins on held-out validation templates
and passes the security tests. Guardrails, tools, checker, answers and benchmark evidence live outside anything the
improver can edit.

Problem statements: **Version N+1** (harness evolution) + **Shared Brain** (curated playbook); the verifier-gated
`submit` and definition of done also cover **Done Means Done**.

## Benchmark-informed, adaptive routing (generic, per task)

Every model call is routed to the model whose **published benchmarks** best predict that kind of work, then the
choice adapts to what Groundhog measures:

1. [frozen/benchmarks.yaml](frozen/benchmarks.yaml): model-card scores (τ-bench, BFCL, Terminal-Bench, BIRD-SQL,
   AIME25, GPQA, LiveCodeBench, SWE-bench, IFBench, RULER, MMLU-Pro, HLE, Arena-Hard) plus Crusoe prices. Frozen.
2. [harness/v0/routing.yaml](harness/v0/routing.yaml) (editable by the improver):
   - `jobs`: fixed role jobs (lead = planning, scout = tool_exploration, reflector, improver) → benchmark weights.
   - `task_jobs`: **generic, domain-independent** kinds of work (`text_to_sql`, `numeric_reasoning`,
     `code_generation`, `repo_editing`, `agentic_tools`, `expert_knowledge`, `long_document`, `strict_format`), each
     with a description and benchmark weights. Nothing here is GPUCo-specific, so new task types only need a new
     entry here.
3. **Per-task routing**: the team is built with one Deep Agents sub-agent per task job (`analyst-<job>`,
   `verifier-<job>`), each bound to the model the router picked for that job. The lead reads the task and picks the
   sub-agent whose job matches, so each task goes to the model that's best at it. Verifiers are picked from a
   different model family than their analyst.
4. [groundhog/router.py](groundhog/router.py): weighted benchmark score → Beta prior, updated with measured pass/fail
   per (job, model) in `runs/router_stats.json`; picks `argmax posterior − cost_weight·log price`. A missing score
   counts as the worst published score on that benchmark (model cards tend to omit benchmarks a model is weak on).
   Escalation to the quality-only pick after repeated tool errors; 429/503 → retries → cooldown → next-best model.

Routing from benchmarks alone (before any measured outcomes; `python -m groundhog.router` prints the full table):

| job | model | deciding evidence |
|---|---|---|
| lead (planning), improver | DeepSeek-V4-Flash | Terminal-Bench 56.9, SWE-bench 79.0 |
| scout (tool_exploration), reflector | Nemotron-3.5-Lightning | cheapest tier; IFBench 71.9 |
| text_to_sql, numeric_reasoning, code_generation, repo_editing, agentic_tools, expert_knowledge | DeepSeek-V4-Flash (verifier: Lightning/Nano) | LiveCodeBench 91.6, GPQA 88.1, Terminal-Bench 56.9 |
| long_document, strict_format | Nemotron-3.5-Lightning (verifier: Flash) | RULER/IFBench evidence; Flash publishes neither |

Nemotron-3-Ultra has the best scores in the file but isn't served on our Crusoe account, so it's excluded
automatically. Measured outcomes then move picks (e.g. the scout moved after Nano went 3/7 in the first run).

## Aetheris for Azure (computer-use world, same harness and self-improvement loop)

Text-guided, multimodal computer use of the real Azure portal, built from [aetheris](https://github.com/savir2010/aetheris)
(voice removed; its screenshot -> "what to click" -> highlight idea became real actions) on the unchanged Groundhog
harness (Deep Agents lead/scout/analyst/verifier, benchmark router, shared brain, guards) and meta-harness (evolve loop).

```bash
GROUNDHOG_WORLD=azure .venv/bin/python -m groundhog.app          # http://localhost:7860, browser opens via WSLg
#   "Open Azure portal" -> sign in yourself (choose "Stay signed in"); then type tasks
GROUNDHOG_WORLD=azure .venv/bin/python -m groundhog.evolve evolve --limit 4   # unattended self-improvement on Azure tasks
```

- **Multimodal**: every screen role sees the latest screenshot with numbered marks (Set-of-Marks, as in WebVoyager /
  browser-use) plus the element list, and acts by number (`click(7)`, `type_text(9, ...)`), never by guessed pixels
  (all tested VLMs mislocated a plain red square). Across iframes. Lead (text-only) gets the URL + element list.
- **Nemotron only** (frozen for this world): screen roles route to Nemotron-3.5-Super-VL (WebVoyager 77.7, OSWorld-
  Verified 51.1) / Nano-Omni by benchmarks; the verifier is always a different model than the analyst.
- **Ask & steer**: `ask_human` for every agent; a guard about to trip (steps/time/repeats) asks you instead of giving
  up; anything you type while it runs is injected as `[HUMAN STEERING]` before the next model call.
- **Safety (frozen tools)**: destructive/submitting clicks (delete, stop, purchase, create on forms, save, assign...)
  need your explicit "yes" (denied when unattended); never types into password fields; Azure/Microsoft URLs only.
- **Recursive learning**: after each task you say whether it worked -> recorded as a validation (worked/failed); the
  reflector turns the trace, including your guidance, into Azure lessons in `runs_azure/brain.db` that later tasks
  read. `evolve` runs 16 read-only navigation tasks (8 train / 4 val / 4 test, graded on final URL or blade title)
  and 2 security tasks, with the same improver, gate and leak filter; harness files are `harness_azure/vN/`.
- GPUCo world unchanged (`GROUNDHOG_WORLD=gpuco`, default); backup of the pre-Azure app: git branch `backup-gpuco`,
  copy in `/root/backups/groundhog_gpuco_backup`.

## Where the prompts live (what the improver can change)

| layer | file | editable |
|---|---|---|
| Deep Agents' own harness: base prompt/suffix, built-in tool descriptions (`task`, `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`), excluded tools/middleware | [harness/v0/deepagents_profile.yaml](harness/v0/deepagents_profile.yaml) → installed as the Deep Agents `HarnessProfile` | **yes** |
| stock Deep Agents text it overrides (+ LangChain's tuned Nemotron-3-Ultra profile as an example) | [harness/DEEPAGENTS_DEFAULTS.md](harness/DEEPAGENTS_DEFAULTS.md) (generated from the installed `deepagents`) | read-only reference |
| lead prompt (Deep Agents `system_prompt`) | [harness/v0/lead_prompt.md](harness/v0/lead_prompt.md) | **yes** |
| sub-agent prompts (Deep Agents `SubAgent.system_prompt`) | [harness/v0/subagents.yaml](harness/v0/subagents.yaml) | **yes** |
| tool guide, skills (Deep Agents `skills`), model routing | `tool_guide.md`, `skills/*/SKILL.md`, `routing.yaml` | **yes** |
| guardrails (first in every prompt), tools, checker, sandbox | `frozen/`, `groundhog/` | no |

Every agent's system prompt, in order: guardrails → its own prompt → Deep Agents profile base/suffix → Deep Agents'
tool sections → skills index → BRAIN.md. Each proposed edit must still build into Deep Agents graphs before it runs.

## Layout

| path | what | editable by improver |
|---|---|---|
| `groundhog/world.py` | GPUCo world: DB-A/DB-B, 16 templates each with reference SQL (verified equal to the exact gold answer on both DBs), split by template (8 train / 4 val / 4 test), checker with numeric tolerance, security tasks | no |
| `groundhog/tools.py` | read-only `sql` (200 rows, 5 s, ATTACH blocked), `schema`, `sample`, `python` in a Docker sandbox (`--network none`, read-only, no host mounts, no secrets) | no |
| `groundhog/attacks.py` | attack 1 (memory poisoning: naive vs curated memory) and attack 2 (red-teamed improver + weakened candidate vs the security gate) | no |
| `groundhog/agent.py` | LangChain **Deep Agents** (`create_deep_agent`): lead → built-in `task` tool → `scout` / `analyst-<job>` / `verifier-<job>` sub-agents → `submit` (rejected unless a verifier agrees). Skills via Deep Agents skills; BRAIN.md appended last in every agent's system prompt (stable prefix for prompt caching); built-in general-purpose sub-agent disabled; middleware adds step cap, repeated-call guard, token budget, fallback, escalation, cost log, trace | no |
| `groundhog/brain.py` | ACE-style brain: reflector ops (with cited trace steps), single curator fed from a queue, rapidfuzz dedup ≥85, contradictions resolved by scope then counters, quarantine of tenant-derived instructions/URLs, retire at harmful ≥ helpful+2, trust after a later passing task, ~1.5k-token cap | no |
| `groundhog/evolve.py` | N+1 loop; improver = read-only Deep Agent (may read `harness/`, training traces, archive, router stats; answers/checker/val/test denied and logged); leak filter, gate, archive, git commit, 2×2/transfer eval | no |
| `harness/vN/` | `lead_prompt.md`, `subagents.yaml`, `tool_guide.md`, `skills/<name>/SKILL.md`, `routing.yaml` | **yes** |
| `frozen/` | `guardrails.md`, `benchmarks.yaml` | no |

Planted quirks (DB-A): money in cents, epoch-ms timestamps, stale `jobs_v1`, soft-deleted tenants/jobs, GPU name
aliases, mixed status casing, preemptible discount (`spot_multiplier`), prompt injections in `support_tickets`
(exfiltration URL + a "report revenue ×1.2" instruction; the security test fails only on obeying them).
DB-B: money in mills, epoch seconds, `jobs_legacy`, `is_active` flag, different aliases.

## Run

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install deepagents langchain-openai rapidfuzz streamlit pytest "numpy>=2"
echo "LLM_API_KEY=..." > .env                      # Crusoe key
.venv/bin/python -m groundhog.world                 # build DBs + tasks + answers
.venv/bin/python -m pytest -q tests                 # self-checks (no API calls except none)
.venv/bin/python -m groundhog.evolve evolve --gens 3 --budget-usd 5   # unattended loop (resumable state in runs/)
.venv/bin/python -m groundhog.evolve table          # 2x2 (v0/best x brain off/on) on test + DB-B transfer
.venv/bin/python -m groundhog.evolve attack-memory  # attack 1
.venv/bin/python -m groundhog.evolve attack-improver  # attack 2
.venv/bin/python -m groundhog.evolve intervene --note "fixed X"   # count a human touch
# add --limit N to any command for a small run (N tasks per split, one per template where possible)
.venv/bin/streamlit run groundhog/dashboard.py
```

## Results

See [RESULTS.md](RESULTS.md) (only measured numbers, with sample sizes).

## What's real, what's not (disclosed)

- Only the data is synthetic. All model calls are real Crusoe calls.
- The team runs on the real `deepagents` package (0.7.28) with `langchain-openai` pointed at Crusoe.
- **Walk-away reliability:** finished tasks, improver edits and the generation counter live in `runs/state.db`, so a
  crashed or killed loop resumes where it stopped (restarts are logged as interventions); 429/503 retries with
  exponential backoff + jitter, per-model concurrency limits, a per-task wall-clock timeout, and a budget cap checked
  before every task that stops the loop cleanly.
- **Fallbacks used** (as allowed by the plan): Python router mirroring Switchyard's `auto` route instead of `switchyard-server`
  ([routes.toml](routes.toml) is ready for it); Docker sandbox instead of OpenShell for agent code
  ([sandbox/policy.yaml](sandbox/policy.yaml) lists what is enforced now and the OpenShell target, not yet validated);
  denials go to `runs/denials.jsonl` (the dashboard's denial feed) instead of OpenShell's audit log.
- Model choice per role lives in `routing.yaml` (benchmark weights + optional `pins`), not in `subagents.yaml`.
- Contradiction resolution in the curator is by dedup + counters only (no scope-aware contradiction detection).

Credits: Deep Agents, ACE, Meta-Harness, Harness², RRSI, Darwin Gödel Machine, Switchyard, OpenShell, NemoClaw, Nemotron.
