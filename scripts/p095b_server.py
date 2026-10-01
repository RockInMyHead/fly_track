#!/usr/bin/env python3
"""P09.5B — the review page for the second pass, in the language of the outcome.

The only thing that changed from the first pass is what the page says. The definitions of glance
and turn are gone, and so are the two words themselves: a person is asked whether the walker
carried on the same way or ended up going another way, which is something that can be watched. The
internal names are still LOOK and TURN, and they are recorded under those names, because that is
what the rest of the pipeline speaks — they are simply not shown, since they drag attention to
heads and bodies rather than to where the walk ended up.

Three buttons, and the third is a real answer. The first pass produced no UNCLEAR at all on a
question that is genuinely hard, which is a sign the page was pushing toward a decision. Here the
way out is stated as plainly as the other two.

Usage:
    PYTHONPATH=. python scripts/p095b_server.py
    PYTHONPATH=. python scripts/p095b_standalone.py     # no server at all
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
SETV2 = DATA / "review_set_v2.json"
LABELS = DATA / "human_labels_v2.json"

PUBLIC_KEYS = ("event_id", "clip_file", "narrow", "wide")

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка: в ту же сторону или в другую</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;background:#0e1116;color:#e8ecf5;
      font:16px/1.45 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}
 .wrap{max-width:1020px;margin:0 auto;padding:14px 18px 30px}
 h1{font-size:20px;margin:0 0 8px;font-weight:650;line-height:1.3}
 .q{background:#141a24;border:1px solid #2a3547;border-radius:10px;padding:12px 16px;
    margin-bottom:12px;font-size:16.5px}
 .q b{color:#ffd98a}
 .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
 .num{font-size:17px;font-weight:700}
 .prog{color:#94a0b8;font-size:12.5px}
 .bar{height:5px;background:#1d2430;border-radius:3px;overflow:hidden;margin-bottom:10px}
 .bar>div{height:100%;background:#3d7dd8;width:0;transition:width .2s}
 .stage{background:#000;border-radius:10px;overflow:hidden;border:1px solid #263041;
        margin:0 auto;max-width:780px}
 video{display:block;width:100%;max-height:46vh;background:#000;object-fit:contain}
 .ctl{display:flex;gap:8px;margin:9px 0 12px;flex-wrap:wrap;justify-content:center}
 button{font:inherit;font-weight:600;border-radius:8px;border:1px solid #2c374a;
        background:#1a212c;color:#dbe3f2;padding:8px 13px;cursor:pointer;font-size:14px}
 button:hover{background:#222b39}
 button.wide{background:#1d2c3d;border-color:#2f4a68;color:#9fd0ff}
 .answers{display:grid;grid-template-columns:1fr 1fr 1fr;gap:9px;max-width:780px;margin:0 auto}
 .ans{padding:22px 12px;font-size:17px;border-width:2px;border-radius:10px;line-height:1.25}
 .ans.same{border-color:#2c7a55;color:#a9f0cd}.ans.same:hover{background:#123326}
 .ans.other{border-color:#2b6f92;color:#a9e6ff}.ans.other:hover{background:#12303f}
 .ans.dunno{border-color:#8a6a2a;color:#ffdca6}.ans.dunno:hover{background:#332912}
 .ans.stood{border-color:#6b7280;color:#cfd6e4;grid-column:1 / -1;padding:13px}
 .ans.stood:hover{background:#262d38}
 .done{background:#151a23;border:1px solid #263041;border-radius:12px;padding:22px;
       text-align:center;margin-top:14px}
 .hint{color:#7c879c;font-size:11.5px;margin-top:8px;text-align:center}
 .exp{background:#1d3a2b;border-color:#2f6b4c;color:#a9f0cd;font-size:15px;padding:11px 18px}
 kbd{background:#1d2430;border:1px solid #2c374a;border-radius:4px;padding:1px 6px;font-size:12px}
 textarea{width:100%;height:110px;background:#0b0e13;color:#9fe8b5;border:1px solid #2c374a;
          border-radius:8px;padding:9px;font-family:ui-monospace,Menlo,monospace;font-size:11px}
</style></head><body><div class="wrap">
<h1>Куда человек пошёл после отмеченного момента?</h1>
<div class="q">После отмеченного момента человек <b>продолжил идти в прежнем направлении</b>
 или <b>окончательно пошёл в новом</b>?</div>
<div class="hint" style="text-align:left;margin:0 0 10px">
 Смотрите, куда уходит коридор и как он меняется. Вопрос про результат движения, а не про то,
 что повернулось. Если по видео непонятно — так и отвечайте, это нормальный ответ.</div>

<div class="top"><div class="num" id="num">…</div><div class="prog" id="prog"></div></div>
<div class="bar"><div id="fill"></div></div>

<div class="stage"><video id="v" playsinline muted preload="auto"></video></div>
<div class="ctl">
 <button onclick="replay()">Показать ещё раз</button>
 <button class="wide" id="wideBtn" onclick="wider()">Показать шире</button>
 <button id="narrowBtn" onclick="narrower()" style="display:none">Вернуть короткое окно</button>
 <span class="hint" id="winhint"></span>
</div>

<div class="answers" id="ans">
 <button class="ans same"   onclick="answer('SAME')">В ТУ ЖЕ СТОРОНУ</button>
 <button class="ans other"  onclick="answer('OTHER')">В ДРУГУЮ СТОРОНУ</button>
 <button class="ans dunno"  onclick="answer('DUNNO')">НЕ МОГУ ПОНЯТЬ</button>
 <button class="ans stood"  onclick="answer('STOOD')">ЧЕЛОВЕК НЕ ШЁЛ (крутился на месте)</button>
</div>
<div class="hint">Клавиши: <kbd>1</kbd> в ту же сторону &nbsp; <kbd>2</kbd> в другую &nbsp;
 <kbd>3</kbd> не могу понять &nbsp;·&nbsp; <kbd>4</kbd> не шёл &nbsp;·&nbsp;
 <kbd>Space</kbd> ещё раз &nbsp;·&nbsp; <kbd>W</kbd> шире</div>

<div class="done" id="done" style="display:none"></div>
</div>
<script>
const SET = __SET__;
const WIN = __WINTEXT__;
const KEY = 'p095b_labels_v2';
let idx = 0, wide = false, t0 = 0, labels = [];
try { labels = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch(e) { labels = []; }
const v = document.getElementById('v');

function current(){
  const first = labels.filter(l => l.round === 1), second = labels.filter(l => l.round === 2);
  if (first.length < SET.events.length)
    return {phase:'round1', ev: SET.events[first.length], n: first.length, total: SET.events.length};
  if (second.length < SET.repeat_round.length){
    const id = SET.repeat_round[second.length];
    return {phase:'round2', ev: SET.events.find(e => e.event_id === id),
            n: second.length, total: SET.repeat_round.length};
  }
  return {phase:'done'};
}
function render(){
  const c = current();
  if (c.phase === 'done'){
    for (const id of ['v','ans','wideBtn','narrowBtn','winhint','fill']) {
      const el = document.getElementById(id); if (el) el.style.display = 'none';
    }
    document.getElementById('num').textContent = '';
    document.getElementById('prog').textContent = '';
    const d = document.getElementById('done'); d.style.display = 'block';
    d.innerHTML = '<div style="font-size:19px;font-weight:650;margin-bottom:6px">Готово — '
      + labels.length + ' ответов</div>'
      + '<div class="hint" style="margin-bottom:12px">Скачайте файл и передайте его для оценки.'
      + '</div><button class="exp" onclick="download()">Скачать метки (JSON)</button>'
      + '<div style="margin-top:14px"><textarea id="ta" readonly>'
      + JSON.stringify({labels: labels}) + '</textarea></div>'
      + '<div class="hint">Если скачивание не сработало — выделите текст выше и сохраните в файл '
      + '<b>human_labels_v2.json</b> в папку <b>data/p095/</b></div>';
    return;
  }
  const e = c.ev;
  document.getElementById('num').textContent = 'Событие ' + (c.n + 1) + ' из ' + c.total;
  document.getElementById('prog').textContent = c.phase === 'round2'
    ? ('основной проход готов · ' + c.n + ' из ' + c.total)
    : ('отвечено ' + c.n + ' из ' + c.total);
  document.getElementById('fill').style.width = (100 * c.n / c.total) + '%';
  wide = false; setWindow();
  v.src = e.clip_file; v.load();
  t0 = Date.now();
  v.onloadedmetadata = () => { v.currentTime = e.narrow[0]; v.play().catch(()=>{}); };
  v.ontimeupdate = () => { if (!wide && v.currentTime >= e.narrow[1]) v.pause(); };
}
function setWindow(){
  document.getElementById('wideBtn').style.display = wide ? 'none' : 'inline-block';
  document.getElementById('narrowBtn').style.display = wide ? 'inline-block' : 'none';
  document.getElementById('winhint').textContent = wide ? WIN.wide : WIN.narrow;
}
function replay(){ const e = current().ev;
  v.currentTime = wide ? e.wide[0] : e.narrow[0]; v.play().catch(()=>{}); }
function wider(){ wide = true; setWindow(); v.currentTime = 0; v.play().catch(()=>{}); }
function narrower(){ wide = false; setWindow(); replay(); }
function answer(label){
  const c = current();
  const dur = Math.round((Date.now() - t0) / 100) / 10;
  const round = c.phase === 'round2' ? 2 : 1;
  labels = labels.filter(l => !(l.event_id === c.ev.event_id && l.round === round));
  labels.push({event_id: c.ev.event_id, round: round, human_label: label,
               review_duration_s: dur, used_extended_window: wide,
               saved_at: new Date().toISOString().slice(0,19)});
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch(e){}
  v.pause(); render();
}
function download(){
  const blob = new Blob([JSON.stringify({labels: labels}, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'human_labels_v2.json';
  document.body.appendChild(a); a.click(); a.remove();
}
document.addEventListener('keydown', e => {
  if (current().phase === 'done') return;
  if (e.key === '1') answer('SAME');
  else if (e.key === '2') answer('OTHER');
  else if (e.key === '3') answer('DUNNO');
  else if (e.key === '4') answer('STOOD');
  else if (e.key === ' '){ e.preventDefault(); replay(); }
  else if (e.key === 'w' || e.key === 'W' || e.key === 'ц' || e.key === 'Ц') wider();
});
render();
</script></body></html>"""


def wintext() -> dict:
    s = json.loads(SETV2.read_text(encoding="utf-8"))
    wf = s.get("window_first", {"before_s": 3, "after_s": 7})
    ww = s.get("window_wide", {"before_s": 5, "after_s": 25})
    return {"narrow": f"окно: {wf['before_s']:.0f} с до, {wf['after_s']:.0f} с после",
            "wide": f"окно: {ww['before_s']:.0f} с до, {ww['after_s']:.0f} с после"}


def build_page() -> str:
    s = json.loads(SETV2.read_text(encoding="utf-8"))
    events = [{k: e[k] for k in PUBLIC_KEYS if k in e} for e in s["events"]]
    payload = {"events": events, "repeat_round": s["repeat_round"]}
    return (PAGE.replace("__SET__", json.dumps(payload, ensure_ascii=False))
                .replace("__WINTEXT__", json.dumps(wintext(), ensure_ascii=False)))


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
    payload: dict = {}
    page: str = ""

    def log_message(self, fmt, *a):
        return

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
            events = self.payload["events"]
            reps = self.payload.get("repeat_round", [])
            lab = read_labels()
            first = [l for l in lab if l.get("round", 1) == 1]
            second = [l for l in lab if l.get("round", 1) == 2]
            pub = lambda e: {k: e[k] for k in PUBLIC_KEYS if k in e}
            if len(first) < len(events):
                nxt = events[len(first)]
                return self._json({"phase": "round1", "index": len(first),
                                   "n_total": len(events), "n_answered": len(first),
                                   "progress_text": f"отвечено {len(first)} из {len(events)}",
                                   "event": pub(nxt)})
            if len(second) < len(reps):
                by = {e["event_id"]: e for e in events}
                nxt = by[reps[len(second)]]
                return self._json({"phase": "round2", "index": len(second),
                                   "n_total": len(reps),
                                   "n_answered": len(events) + len(second),
                                   "progress_text": f"{len(events)} готово · сейчас "
                                                    f"{len(second)} из {len(reps)}",
                                   "event": pub(nxt)})
            return self._json({"phase": "done", "n_answered": len(lab)})
        if p.startswith("/clips/"):
            f = (CLIPS / Path(p).name).resolve()
            if not str(f).startswith(str(CLIPS.resolve())) or not f.exists():
                self.send_error(404)
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
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
            return self._json({"ok": False}, 400)
        # The page speaks in tokens because the words glance and turn were steering the answer.
        # The tokens are translated here, at the boundary, so the rest of the pipeline goes on
        # reading LOOK / TURN / UNCLEAR and nothing downstream needs to know about the change.
        TOKENS = {"SAME": "LOOK", "OTHER": "TURN", "DUNNO": "UNCLEAR",
                  "STOOD": "NO_LOCOMOTION"}
        token = rec.get("human_label")
        if token not in TOKENS:
            return self._json({"ok": False, "why": "bad label"}, 400)
        rec = dict(rec, human_label=TOKENS[token])
        lab = read_labels()
        lab = [l for l in lab if not (l["event_id"] == rec["event_id"]
                                      and l.get("round", 1) == rec.get("round", 1))]
        lab.append({"event_id": rec["event_id"], "round": rec.get("round", 1),
                    "human_label": rec["human_label"],
                    "review_duration_s": rec.get("review_duration"),
                    "used_extended_window": bool(rec.get("used_extended_window")),
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        write_labels(lab)
        self._json({"ok": True, "n": len(lab)})


def main() -> int:
    global SETV2, LABELS, CLIPS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8775)
    ap.add_argument("--host", default="0.0.0.0")
    # Paths are arguments rather than constants because a set for a new clip has to be served and
    # saved separately. With the round-one paths baked in, serving a second clip would have written
    # its answers over the first round's labels — the same mistake as an output path that is only
    # decorative.
    ap.add_argument("--set", default=None, help="набор событий (по умолчанию review_set_v2.json)")
    ap.add_argument("--labels", default=None, help="куда писать метки")
    ap.add_argument("--clips", default=None, help="каталог с нарезанными клипами")
    args = ap.parse_args()
    if args.set:
        SETV2 = Path(args.set) if str(args.set).startswith("/") else ROOT / args.set
    if args.labels:
        LABELS = Path(args.labels) if str(args.labels).startswith("/") else ROOT / args.labels
    if args.clips:
        CLIPS = Path(args.clips) if str(args.clips).startswith("/") else ROOT / args.clips
    if not SETV2.exists():
        print("нет data/p095/review_set_v2.json — сначала scripts/p095b_build.py")
        return 1
    Handler.payload = json.loads(SETV2.read_text(encoding="utf-8"))
    Handler.page = build_page()
    n, reps = len(Handler.payload["events"]), len(Handler.payload["repeat_round"])
    print("P09.5B — повторная разметка, исправленный вопрос")
    print(f"  вопрос: «продолжил идти в прежнем направлении или окончательно пошёл в новом?»")
    print(f"  кнопки: В ТУ ЖЕ СТОРОНУ / В ДРУГУЮ СТОРОНУ / НЕ МОГУ ПОНЯТЬ / ЧЕЛОВЕК НЕ ШЁЛ")
    print(f"  слова LOOK и TURN на странице отсутствуют")
    print(f"  событий {n} (+{reps} повторных показов), порядок перемешан заново")
    print(f"  {wintext()['narrow']}  ·  шире: {wintext()['wide']}")
    print(f"  метки пишутся в {LABELS}")
    print(f"  первый проход не тронут: {DATA / 'human_labels_v1.json'}")
    print(f"  страница: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
