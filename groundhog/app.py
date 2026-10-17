"""Aetheris for Azure: text-guided, multimodal computer use on the Groundhog harness (voice replaced by text).

    GROUNDHOG_WORLD=azure python -m groundhog.app        ->  http://localhost:7860

Type a task; while it runs, anything you type steers the team; when an agent asks (stuck, off track, approval for a
risky click) your next message is the answer. After each task you confirm whether it worked: that verdict is recorded
as a validation (worked/failed) and the reflector turns the trace (including your guidance) into Azure lessons in the
shared brain, which every later task reads. The unattended self-improvement loop is the same as before:
    GROUNDHOG_WORLD=azure python -m groundhog.evolve evolve --limit 4
"""
import os
import threading
import time

os.environ.setdefault("GROUNDHOG_WORLD", "azure")

from flask import Flask, jsonify, request, send_file  # noqa: E402

from .agent import Team, load_harness  # noqa: E402
from .brain import Brain  # noqa: E402
from .config import HARNESS, RUNS, world  # noqa: E402
from .evolve import current_version, learn, remember_validations  # noqa: E402
from .human import Human  # noqa: E402
from .llm import LLM  # noqa: E402
from .router import Router  # noqa: E402

W, app, human, llm = world(), Flask(__name__), Human(), LLM()
brain = Brain(RUNS / "brain.db")
state = {"busy": False}


def run_task(text: str) -> None:
    state["busy"] = True
    try:
        v = current_version()
        h = load_harness(HARNESS / f"v{v}")
        task = dict(id=f"live-{int(time.time())}", template="live", question=text)
        human.emit({"role": "system", "kind": "task", "text": f"Task started on harness v{v}: {text}"})
        used = brain.ids_in_render()
        res = Team(h, brain.render(), llm, trace_dir=RUNS / "traces" / "live", human=human).run(task)
        res["brain_used"] = used
        verdict = human.ask(f"I finished with answer {res['answer']!r} (now at {res['final'].get('url')}). "
                            "Did I do it correctly? Reply yes, or no + what was wrong.") or ""
        res["passed"] = verdict.strip().lower().startswith("y")
        res["trace"].append(dict(step=len(res["trace"]), role="human", kind="verdict", text=verdict))
        learn(brain, llm, Router(h["routing"]), [res])
        remember_validations(brain, [res], "live")
        human.emit({"role": "system", "kind": "learned", "text": "Brain updated: " + "; ".join(res.get("brain_events", []))[:600]})
    except Exception as e:  # keep the session alive; the error is shown in the chat
        human.emit({"role": "system", "kind": "error", "text": f"{type(e).__name__}: {e}"[:500]})
    finally:
        state["busy"] = False


@app.post("/send")
def send():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify(ok=False)
    if human.waiting or state["busy"]:
        return jsonify(ok=True, kind=human.say(text))
    human.emit({"role": "human", "kind": "task", "text": text})
    threading.Thread(target=run_task, args=(text,), daemon=True, name="task").start()
    return jsonify(ok=True, kind="task")


@app.post("/open")
def open_portal():
    """Opens the portal in the visible browser so the human can sign in there."""
    W.BROWSER.run(lambda: (W.BROWSER.page.goto(W.HOME, timeout=60_000), W.BROWSER.observe()))
    human.emit({"role": "system", "kind": "info", "text": "Opened the Azure portal in the browser window. Sign in there if asked."})
    return jsonify(ok=True)


@app.get("/observe")
def observe_now():
    o = W.BROWSER.run(W.BROWSER.observe)
    return jsonify(url=o["url"], title=o["title"], elements=o["elements"][:3000])


@app.get("/events")
def events():
    after = int(request.args.get("after", -1))
    return jsonify(events=[e for e in human.events if e["n"] > after], busy=state["busy"], waiting=human.waiting)


@app.get("/screen.jpg")
def screen():
    p = RUNS / "screens" / "latest.jpg"
    return send_file(p, mimetype="image/jpeg", max_age=0) if p.exists() else ("", 204)


@app.get("/brain")
def brain_md():
    return brain.render() or "(empty)", 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Aetheris for Azure</title><style>
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}#l{width:40%;display:flex;flex-direction:column;border-right:1px solid #ddd}
#log{flex:1;overflow:auto;padding:10px;font-size:13px}.e{margin:4px 0;padding:6px 8px;border-radius:6px;white-space:pre-wrap}
.human{background:#e8f0fe}.question{background:#fff3cd;font-weight:600}.error{background:#f8d7da}.system{background:#eef7ee}
.tool,.task,.final,.think,.steer,.escalate,.fallback,.reroute,.guard{background:#f6f6f6}#r{flex:1;padding:10px;overflow:auto}
form{display:flex;padding:8px;gap:6px;border-top:1px solid #ddd}input{flex:1;padding:8px}#st{padding:4px 10px;font-size:12px;color:#555}
img{max-width:100%;border:1px solid #ccc}</style></head><body>
<div id="l"><div id="st"></div><div id="log"></div>
<form id="f"><input id="t" placeholder="Give a task, steer while it runs, or answer a question" autocomplete="off"><button>Send</button></form>
<div style="padding:0 8px 8px"><button onclick="fetch('/open',{method:'POST'})">Open Azure portal (sign in)</button>
<a href="/brain" target="_blank">BRAIN.md</a></div></div>
<div id="r"><img id="s" src="/screen.jpg"></div>
<script>
let after=-1;const log=document.getElementById('log');
function cls(e){return e.kind==='question'?'question':e.role==='human'?'human':e.role==='system'?(e.kind==='error'?'error':'system'):(e.kind||'')}
function txt(e){let who=e.role+(e.model?' ('+e.model.split('/').pop()+')':'');let body=e.text||'';
 if(e.tool)body=e.tool+' '+JSON.stringify(e.args||{})+(e.result?'\\n-> '+e.result:'');return who+' · '+(e.kind||'')+'\\n'+body}
async function poll(){try{const r=await fetch('/events?after='+after);const d=await r.json();
 for(const e of d.events){after=e.n;const div=document.createElement('div');div.className='e '+cls(e);div.textContent=txt(e);log.appendChild(div);log.scrollTop=1e9}
 document.getElementById('st').textContent=d.waiting?'Waiting for your answer: '+d.waiting:(d.busy?'Working... (type to steer)':'Idle: type a task');
 document.getElementById('s').src='/screen.jpg?t='+Date.now()}catch(e){} setTimeout(poll,1500)}
document.getElementById('f').onsubmit=async ev=>{ev.preventDefault();const t=document.getElementById('t');if(!t.value)return;
 await fetch('/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t.value})});t.value=''};
poll();</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), threaded=True)
