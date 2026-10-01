#!/usr/bin/env python3
"""P09.5 — the blind review page, and the only thing that will hold the labels.

The one design rule that matters here is that the person must not be able to see what the
machine thinks. Not the class, not the yaw, not the score, not the direction, not which clip an
event came from beyond a number, and not which of the three classes it was drawn from. If any of
that were on the page the label would be contaminated by it, and the whole point of the exercise
is to stop one automatic rule checking another.

So the server holds two things apart:

    what the page is given      a number, a clip, and where to start and stop playing it
    what the page is never given  the algorithmic class, the score, the direction, the proxy
                                  label, the pair status, the clip's name in the sample

The hidden fields sit in `review_set.json` and are joined to the answers only in
`p095_evaluate.py`, after the last label is in. The page has no route that returns them, and this
is enforced by construction rather than by a flag: `/api/set` builds its reply from an explicit
list of public keys, so adding a field to the set file cannot accidentally expose it.

The second pass is not announced
--------------------------------
Ten of the sixty come back after the first pass ends, without being marked as repeats. Self
agreement is the only measurement available of how much of this task is objectively ambiguous: a
person who labels the same move differently on two viewings has told us the ceiling, whatever the
machine scores.

Usage:
    PYTHONPATH=. python scripts/p095_review_server.py
    then open http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
CLIPS = ROOT / "output/p095/clips"
LABELS = DATA / "human_labels.json"
SET = DATA / "review_set.json"

# the only keys the page ever receives about an event
PUBLIC_KEYS = ("event_id", "clip_file", "narrow", "wide", "cut_ok")

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P09.5 — разметка LOOK / TURN</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;background:#0e1116;color:#e8ecf5;
      font:16px/1.45 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:14px 18px 30px}
 h1{font-size:17px;margin:0 0 2px;font-weight:650}
 .sub{color:#94a0b8;font-size:12px;margin-bottom:10px}
 .defs{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px}
 .def{background:#151a23;border:1px solid #263041;border-radius:9px;padding:8px 10px}
 .def b{display:block;font-size:13px;margin-bottom:2px}
 .def.look b{color:#6fd3ff}.def.turn b{color:#6ee7a8}.def.unclear b{color:#ffc46b}
 .def span{color:#a9b4c8;font-size:11.5px;line-height:1.35}
 .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
 .num{font-size:18px;font-weight:700;letter-spacing:.3px}
 .prog{color:#94a0b8;font-size:12.5px}
 .bar{height:5px;background:#1d2430;border-radius:3px;overflow:hidden;margin-bottom:9px}
 .bar>div{height:100%;background:#3d7dd8;width:0;transition:width .2s}
 .stage{background:#000;border-radius:10px;overflow:hidden;border:1px solid #263041;
        margin:0 auto;max-width:760px}
 video{display:block;width:100%;max-height:44vh;background:#000;object-fit:contain}
 .ctl{display:flex;gap:8px;margin:9px 0 12px;flex-wrap:wrap;justify-content:center}
 button{font:inherit;font-weight:600;border-radius:8px;border:1px solid #2c374a;
        background:#1a212c;color:#dbe3f2;padding:8px 13px;cursor:pointer;font-size:14px}
 button:hover{background:#222b39}
 button.wide{background:#1d2c3d;border-color:#2f4a68;color:#9fd0ff}
 .answers{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;max-width:760px;
          margin:0 auto}
 .ans{padding:16px 10px;font-size:18px;border-width:2px;border-radius:10px}
 .ans.look{border-color:#2b6f92;color:#a9e6ff}
 .ans.look:hover{background:#12303f}
 .ans.turn{border-color:#2c7a55;color:#a9f0cd}
 .ans.turn:hover{background:#123326}
 .ans.unclear{border-color:#8a6a2a;color:#ffdca6}
 .ans.unclear:hover{background:#332912}
 .done{background:#151a23;border:1px solid #263041;border-radius:12px;padding:26px;text-align:center}
 .hint{color:#7c879c;font-size:11.5px;margin-top:8px;text-align:center}
 kbd{background:#1d2430;border:1px solid #2c374a;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head><body><div class="wrap">
<h1>Поворот тела или только взгляд?</h1>
<div class="sub">Смотрите клип и отвечайте только на один вопрос. Направление (влево/вправо)
 не спрашивается, и что думает система — не показывается.</div>

<div class="defs">
 <div class="def look"><b>LOOK</b><span>Камера/голова повернулась, но движение в итоге
  продолжается примерно по прежнему направлению. Виден возврат взгляда.</span></div>
 <div class="def turn"><b>TURN</b><span>Человек реально меняет направление движения и
  продолжает двигаться в новом направлении.</span></div>
 <div class="def unclear"><b>UNCLEAR</b><span>По видео невозможно уверенно понять,
  повернулось тело или только камера. Выбирать наугад не нужно.</span></div>
</div>

<div class="top"><div class="num" id="num">…</div><div class="prog" id="prog"></div></div>
<div class="bar"><div id="fill"></div></div>

<div class="stage"><video id="v" playsinline muted preload="auto"></video></div>
<div class="ctl">
 <button onclick="replay()">Повторить</button>
 <button class="wide" onclick="wider()" id="wideBtn">Показать шире</button>
 <button onclick="narrower()" id="narrowBtn" style="display:none">Вернуть обычное окно</button>
 <span class="hint" id="winhint"></span>
</div>

<div class="answers" id="ans">
 <button class="ans look"    onclick="answer('LOOK')">LOOK</button>
 <button class="ans turn"    onclick="answer('TURN')">TURN</button>
 <button class="ans unclear" onclick="answer('UNCLEAR')">UNCLEAR</button>
</div>
<div class="hint">Клавиши: <kbd>1</kbd> LOOK &nbsp; <kbd>2</kbd> TURN &nbsp; <kbd>3</kbd> UNCLEAR
 &nbsp;·&nbsp; <kbd>Space</kbd> повторить &nbsp;·&nbsp; <kbd>W</kbd> шире</div>

<div class="done" id="done" style="display:none"></div>
</div>
<script>
let S=null, idx=0, t0=0, wide=false;
const v=document.getElementById('v');
const WIN=__WINTEXT__;

async function load(){
  S=await (await fetch('/api/state')).json();
  if(S.phase==='done'){ return finish(); }
  idx=S.index;
  const e=S.event;
  document.getElementById('num').textContent =
    S.phase==='round2' ? `Событие ${idx+1} из ${S.n_total}`
                       : `Событие ${idx+1} из ${S.n_total}`;
  document.getElementById('prog').textContent = S.progress_text;
  // the bar counts the current pass only. A grand total would tell the person that a second,
  // shorter pass is coming and how long it is, which is not theirs to know.
  const doneThisPass = S.phase==='round2' ? idx : idx;
  document.getElementById('fill').style.width = (100*doneThisPass/S.n_total)+'%';
  wide=false; setWindow();
  v.src=e.clip_file; v.load();
  t0=Date.now();
  v.onloadedmetadata=()=>{ v.currentTime=e.narrow[0]; v.play().catch(()=>{}); };
  v.ontimeupdate=()=>{ const e=S.event;
    if(!wide && v.currentTime>=e.narrow[1]) v.pause(); };
}

function setWindow(){
  const e=S.event;
  document.getElementById('wideBtn').style.display = wide?'none':'inline-block';
  document.getElementById('narrowBtn').style.display = wide?'inline-block':'none';
  document.getElementById('winhint').textContent = wide ? WIN.wide : WIN.narrow;
}
function replay(){ const e=S.event; v.currentTime = wide?e.wide[0]:e.narrow[0];
  v.play().catch(()=>{}); }
function wider(){ wide=true; setWindow(); v.currentTime=0; v.play().catch(()=>{}); }
function narrower(){ wide=false; setWindow(); replay(); }

async function answer(label){
  const dur=(Date.now()-t0)/1000;
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({event_id:S.event.event_id, round:S.phase==='round2'?2:1,
      human_label:label, review_duration:Math.round(dur*10)/10,
      used_extended_window:wide})});
  v.pause(); load();
}
function finish(){
  for(const id of ['v','ans','wideBtn','narrowBtn','winhint','fill'])
    document.getElementById(id).style.display='none';
  document.getElementById('num').textContent='';
  document.getElementById('prog').textContent='';
  const d=document.getElementById('done'); d.style.display='block';
  d.innerHTML='<div style="font-size:19px;font-weight:650;margin-bottom:8px">Готово — '+
    S.n_answered+' ответов</div><div class="hint">Метки сохранены. Алгоритмические данные '+
    'теперь можно присоединить: scripts/p095_evaluate.py</div>';
}
document.addEventListener('keydown',e=>{
  if(document.getElementById('done').style.display!=='none') return;
  if(e.key==='1') answer('LOOK');
  else if(e.key==='2') answer('TURN');
  else if(e.key==='3') answer('UNCLEAR');
  else if(e.key===' '){e.preventDefault(); replay();}
  else if(e.key==='w'||e.key==='W'||e.key==='ц'||e.key==='Ц') wider();
});
load();
</script></body></html>"""


def read_labels() -> list[dict]:
    if not LABELS.exists():
        return []
    try:
        return json.loads(LABELS.read_text(encoding="utf-8")).get("labels", [])
    except Exception:
        return []


def write_labels(labels: list[dict]) -> None:
    LABELS.write_text(json.dumps({"labels": labels}, indent=2, ensure_ascii=False),
                      encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    set_payload: dict = {}

    def log_message(self, fmt, *a):        # keep the console readable
        return

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _public(self, ev: dict) -> dict:
        """Only the keys on the list. Anything added to the set file stays out of the page."""
        return {k: ev[k] for k in PUBLIC_KEYS if k in ev}

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            body = self.page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if p == "/api/state":
            events = self.set_payload["events"]
            repeats = self.set_payload.get("repeat_round", [])
            labels = read_labels()
            first = [l for l in labels if l.get("round", 1) == 1]
            second = [l for l in labels if l.get("round", 1) == 2]
            if len(first) < len(events):
                nxt = events[len(first)]
                return self._json({"phase": "round1", "index": len(first),
                                   "n_total": len(events), "n_answered": len(first),
                                   "progress_text": f"первый проход · отвечено {len(first)} "
                                                    f"из {len(events)}",
                                   "event": self._public(nxt)})
            if len(second) < len(repeats):
                by_id = {e["event_id"]: e for e in events}
                nxt = by_id[repeats[len(second)]]
                return self._json({"phase": "round2", "index": len(second),
                                   "n_total": len(repeats),
                                   "n_answered": len(events) + len(second),
                                   "progress_text": f"{len(events)} готово · сейчас "
                                                    f"{len(second)} из {len(repeats)}",
                                   "event": self._public(nxt)})
            return self._json({"phase": "done", "n_answered": len(labels)})

        if p.startswith("/clips/"):
            f = (CLIPS / Path(p).name).resolve()
            if not str(f).startswith(str(CLIPS.resolve())) or not f.exists():
                self.send_error(404)
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/label":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            rec = json.loads(self.rfile.read(n).decode())
        except Exception:
            return self._json({"ok": False, "why": "bad json"}, 400)
        if rec.get("human_label") not in ("LOOK", "TURN", "UNCLEAR"):
            return self._json({"ok": False, "why": "bad label"}, 400)
        labels = read_labels()
        # one label per event per pass; a second pass is a separate record on purpose
        labels = [l for l in labels
                  if not (l["event_id"] == rec["event_id"]
                          and l.get("round", 1) == rec.get("round", 1))]
        labels.append({"event_id": rec["event_id"], "round": rec.get("round", 1),
                       "human_label": rec["human_label"],
                       "review_duration_s": rec.get("review_duration"),
                       "used_extended_window": bool(rec.get("used_extended_window")),
                       "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        write_labels(labels)
        self._json({"ok": True, "n": len(labels)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not SET.exists():
        print("нет data/p095/review_set.json — сначала scripts/p095_build_set.py")
        return 1
    Handler.set_payload = json.loads(SET.read_text(encoding="utf-8"))
    wf = Handler.set_payload.get("window_first", {"before_s": 3, "after_s": 20})
    ww = Handler.set_payload.get("window_wide", {"before_s": 5, "after_s": 25})
    wintext = {"narrow": f"окно: {wf['before_s']:.0f} с до, {wf['after_s']:.0f} с после",
               "wide": f"окно: {ww['before_s']:.0f} с до, {ww['after_s']:.0f} с после"}
    Handler.page = PAGE.replace("__WINTEXT__", json.dumps(wintext, ensure_ascii=False))
    n = len(Handler.set_payload["events"])
    reps = len(Handler.set_payload.get("repeat_round", []))
    print("P09.5 — слепая разметка LOOK / TURN")
    print(f"  событий: {n}, затем {reps} повторных показов")
    print(f"  {wintext['narrow']}  ·  шире: {wintext['wide']}")
    print(f"  клипы: {CLIPS}  ({len(list(CLIPS.glob('*.mp4')))} файлов)")
    print(f"  метки пишутся в {LABELS}")
    print(f"  страница: http://{args.host}:{args.port}")
    print("  страница НЕ получает ни класса, ни score, ни направления")
    print("  остановить: Ctrl+C")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
