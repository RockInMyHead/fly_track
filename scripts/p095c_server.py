#!/usr/bin/env python3
"""P09.5C — the reviewer draws the path instead of choosing a word.

Every attempt so far has asked a person to convert what they saw into a label. That conversion is
where the trouble came from: the first pass defined glance and turn through the mechanism — the
head turned, the body did not — which is the distinction the machine itself cannot make; the
second pass asked about the outcome in words, and a person watching a featureless corridor still
has to decide what "went another way" means when the view swings.

Drawing removes the conversion. A person who has watched the clip knows where the walker went, and
can draw it. From the drawn path the same quantity P07.4 applied to the camera signal is computed
directly:

    total turning   the sum of the heading changes along the drawn path
    net turning     the heading at the end minus the heading at the start

    net small against total   the path went out and came back   -> the glance case
    net most of total         the path ended somewhere new     -> the turn case

That is the definition, measured on the shape rather than on an impression of the video. Nothing
verbal is left for the reviewer to interpret, and nothing about the machine is shown.

The drawing is kept as it was drawn
-----------------------------------
The raw polyline is the record. The verdict is derived from it and shown for confirmation, but the
points are what is stored, so the classification can be recomputed later — with different edges,
or not at all — without asking anyone to look at the clip again. A threshold on the ratio would be
a new number chosen by nobody, so the ratio itself is stored and no verdict is frozen into it.

The orientation is fixed so the drawings are comparable
--------------------------------------------------------
Every canvas starts with the walker at the bottom centre heading straight up, with a small grey
arrow showing that. It does not matter which way the corridor really ran: only the change of
heading is read off the drawing, and change is invariant to how the sheet is turned.

Usage:
    PYTHONPATH=. python scripts/p095c_server.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
CLIPS = ROOT / "output/p095/clips"
SETV2 = DATA / "review_set_v2.json"
LABELS = DATA / "human_labels_v2.json"

PUBLIC_KEYS = ("event_id", "clip_file", "narrow", "wide")

PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка: нарисуйте путь</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;background:#0e1116;color:#e8ecf5;
      font:15px/1.4 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}
 .wrap{max-width:1400px;margin:0 auto;padding:12px 16px 26px}
 h1{font-size:18px;margin:0 0 6px;font-weight:650}
 .task{background:#141a24;border:1px solid #2a3547;border-radius:9px;padding:9px 13px;
       margin-bottom:10px;font-size:14.5px}
 .task b{color:#ffd98a}
 .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}
 .num{font-size:16px;font-weight:700}
 .prog{color:#94a0b8;font-size:12px}
 .bar{height:5px;background:#1d2430;border-radius:3px;overflow:hidden;margin-bottom:9px}
 .bar>div{height:100%;background:#3d7dd8;width:0;transition:width .2s}
 .cols{display:grid;grid-template-columns:1.15fr 1fr;gap:12px;align-items:start}
 .stage{background:#000;border-radius:9px;overflow:hidden;border:1px solid #263041}
 video{display:block;width:100%;max-height:52vh;background:#000;object-fit:contain}
 .ctl{display:flex;gap:7px;margin:7px 0 0;flex-wrap:wrap;align-items:center}
 button{font:inherit;font-weight:600;border-radius:7px;border:1px solid #2c374a;
        background:#1a212c;color:#dbe3f2;padding:7px 12px;cursor:pointer;font-size:13.5px}
 button:hover{background:#222b39}
 button.wide{background:#1d2c3d;border-color:#2f4a68;color:#9fd0ff}
 .panel{background:#141a24;border:1px solid #2a3547;border-radius:9px;padding:9px}
 .canvaswrap{position:relative;width:100%;aspect-ratio:1/1;max-height:52vh;margin:0 auto;
             max-width:52vh}
 canvas{width:100%;height:100%;display:block;background:#0b0e13;border-radius:7px;
        border:1px solid #263041;cursor:crosshair;touch-action:none}
 .legend{color:#8b96ab;font-size:11.5px;margin-top:6px;line-height:1.5}
 .verdict{margin-top:8px;padding:9px 11px;border-radius:8px;font-size:13.5px;
          background:#101720;border:1px solid #263041}
 .verdict b{font-size:15px}
 .ok{color:#8ff0b5}.warn{color:#ffd98a}.bad{color:#ff9f7a}
 .btns{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:9px}
 .btns button{padding:11px 8px;font-size:14px}
 .b-same{border-color:#2c7a55;color:#a9f0cd}.b-same:hover{background:#123326}
 .b-other{border-color:#2b6f92;color:#a9e6ff}.b-other:hover{background:#12303f}
 .b-stood{border-color:#6b7280;color:#cfd6e4}
 .b-dunno{border-color:#8a6a2a;color:#ffdca6}
 .done{background:#151a23;border:1px solid #263041;border-radius:11px;padding:20px;
       text-align:center;margin-top:12px}
 .hint{color:#7c879c;font-size:11.5px;margin-top:6px}
 .exp{background:#1d3a2b;border-color:#2f6b4c;color:#a9f0cd;font-size:14px;padding:10px 16px}
 kbd{background:#1d2430;border:1px solid #2c374a;border-radius:4px;padding:1px 5px;font-size:11px}
 textarea{width:100%;height:96px;background:#0b0e13;color:#9fe8b5;border:1px solid #2c374a;
          border-radius:7px;padding:8px;font-family:ui-monospace,Menlo,monospace;font-size:11px}
</style></head><body><div class="wrap">
<h1>Нарисуйте путь человека</h1>
<div class="task">Смотрите клип и рисуйте справа, <b>куда человек шёл</b>: от точки старта
 (серый кружок внизу) и дальше — как он двигался. Линия должна передавать форму пути:
 если человек вышел и вернулся — линия вернётся; если свернул и пошёл туда — линия уйдёт вбок
 и там останется.</div>

<div class="top"><div class="num" id="num">…</div><div class="prog" id="prog"></div></div>
<div class="bar"><div id="fill"></div></div>

<div class="cols">
 <div>
  <div class="stage"><video id="v" playsinline muted preload="auto"></video></div>
  <div class="ctl">
   <button onclick="replay()">Ещё раз</button>
   <button class="wide" id="wideBtn" onclick="wider()">Показать шире</button>
   <button id="narrowBtn" onclick="narrower()" style="display:none">Короткое окно</button>
   <span class="hint" id="winhint"></span>
  </div>
 </div>
 <div class="panel">
  <div class="canvaswrap"><canvas id="cv" width="700" height="700"></canvas></div>
  <div class="ctl">
   <button onclick="undoStroke()">Убрать штрих</button>
   <button onclick="clearAll()">Стереть всё</button>
   <span class="hint" id="drawhint">рисуйте мышью или пальцем</span>
  </div>
  <div class="legend">Серый кружок — где человек был в начале. Серая стрелка — куда он тогда
   смотрел. Рисуйте от него и дальше. Длина и повороты важны, точный масштаб — нет.</div>
  <div class="verdict" id="verdict">Нарисуйте путь, и я покажу, что из него следует.</div>
  <div class="btns">
   <button class="b-same"  onclick="answer('LOOK')">В ТУ ЖЕ СТОРОНУ</button>
   <button class="b-other" onclick="answer('TURN')">В ДРУГУЮ СТОРОНУ</button>
   <button class="b-stood" onclick="answer('NO_LOCOMOTION')">НЕ ШЁЛ / КРУТИЛСЯ</button>
   <button class="b-dunno" onclick="answer('UNCLEAR')">НЕ МОГУ ПОНЯТЬ</button>
  </div>
  <div class="hint">Кнопки нужны только если рисунок не выражает ответ. Обычно достаточно
   нарисовать и нажать <kbd>Enter</kbd>.</div>
 </div>
</div>

<div class="done" id="done" style="display:none"></div>
</div>
<script>
const SET = __SET__;
const WIN = __WINTEXT__;
const KEY = 'p095c_labels_v2';
let labels = [];
try { labels = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch(e) { labels = []; }
let strokes = [], cur = null, drawing = false, wide = false, t0 = 0, derived = null;

const v = document.getElementById('v');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');

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

/* ---------- drawing ---------- */
function resize(){
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function norm(ev){
  const r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width, 1 - (ev.clientY - r.top) / r.height];
}
function draw(){
  const W = cv.width / (window.devicePixelRatio||1), H = cv.height / (window.devicePixelRatio||1);
  ctx.clearRect(0,0,W,H);
  // grid
  ctx.strokeStyle = '#161d28'; ctx.lineWidth = 1;
  for (let i=1;i<10;i++){
    ctx.beginPath(); ctx.moveTo(W*i/10,0); ctx.lineTo(W*i/10,H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,H*i/10); ctx.lineTo(W,H*i/10); ctx.stroke();
  }
  // start marker at bottom centre, heading up
  const sx = W*0.5, sy = H*0.84;
  ctx.fillStyle = '#7d8798';
  ctx.beginPath(); ctx.arc(sx, sy, 6, 0, 2*Math.PI); ctx.fill();
  ctx.strokeStyle = '#7d8798'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(sx, sy-10); ctx.lineTo(sx, sy-46); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx-6, sy-34); ctx.lineTo(sx, sy-47); ctx.lineTo(sx+6, sy-34);
  ctx.stroke();
  ctx.fillStyle = '#5c6679'; ctx.font = '12px -apple-system,sans-serif';
  ctx.fillText('старт', sx+10, sy+4);
  // strokes
  ctx.strokeStyle = '#6fd3ff'; ctx.lineWidth = 3.5; ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  const all = cur ? strokes.concat([cur]) : strokes;
  for (const st of all){
    if (st.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(st[0][0]*W, (1-st[0][1])*H);
    for (let i=1;i<st.length;i++) ctx.lineTo(st[i][0]*W, (1-st[i][1])*H);
    ctx.stroke();
  }
  if (all.length && all[0].length){
    const last = all[all.length-1][all[all.length-1].length-1];
    ctx.fillStyle = '#6fd3ff';
    ctx.beginPath(); ctx.arc(last[0]*W, (1-last[1])*H, 5, 0, 2*Math.PI); ctx.fill();
  }
}
function allPoints(){
  const out = [];
  for (const st of strokes) for (const p of st) out.push(p);
  return out;
}
/* the same reading P07.4 applied to the camera: net heading over total turning */
function analyse(){
  const P = allPoints();
  if (P.length < 6) return null;
  // resample by arc length so a dense stroke does not weigh more than a sparse one
  const seg = []; let total = 0;
  for (let i=1;i<P.length;i++){
    const d = Math.hypot(P[i][0]-P[i-1][0], P[i][1]-P[i-1][1]);
    if (d > 1e-6){ seg.push({a:P[i-1], b:P[i], d}); total += d; }
  }
  if (total < 0.02 || seg.length < 3) return null;
  // Resample by arc length. The first point is NOT seeded separately: seeding it duplicates
  // the point at arc length zero, and two coincident points give atan2(0,0) = 0, which inserts
  // a phantom 90 degree corner at the start of every path — a straight line came out as a turn.
  const N = 60, pts = [];
  const targets = [];
  for (let k=0;k<=N;k++) targets.push(total*k/N);
  let idx = 0;
  let walked = 0;
  for (const s of seg){
    let t0s = walked, t1s = walked + s.d;
    while (idx < targets.length && targets[idx] <= t1s){
      const f = (targets[idx]-t0s)/Math.max(s.d,1e-9);
      pts.push([s.a[0]+(s.b[0]-s.a[0])*f, s.a[1]+(s.b[1]-s.a[1])*f]);
      idx++;
    }
    walked = t1s;
  }
  const ang = [];
  for (let i=1;i<pts.length;i++){
    const dx = pts[i][0]-pts[i-1][0], dy = pts[i][1]-pts[i-1][1];
    if (Math.hypot(dx,dy) < 1e-9) continue;      // coincident resampled points carry no heading
    ang.push(Math.atan2(dy, dx));
  }
  let turnTotal = 0;
  for (let i=1;i<ang.length;i++){
    let d = ang[i]-ang[i-1];
    while (d > Math.PI) d -= 2*Math.PI;
    while (d < -Math.PI) d += 2*Math.PI;
    turnTotal += Math.abs(d);
  }
  if (ang.length < 3) return null;
  let net = ang[ang.length-1]-ang[0];
  while (net > Math.PI) net -= 2*Math.PI;
  while (net < -Math.PI) net += 2*Math.PI;
  net = Math.abs(net);
  return {turn_total_deg: turnTotal*180/Math.PI, turn_net_deg: net*180/Math.PI,
          ratio: turnTotal > 1e-6 ? net/turnTotal : 1,
          n_points: pts.length, arc: total};
}
function updateVerdict(){
  const a = analyse();
  const el = document.getElementById('verdict');
  if (!a){
    derived = null;
    el.className = 'verdict';
    el.innerHTML = 'Нарисуйте путь, и я покажу, что из него следует.';
    return;
  }
  derived = a;
  // The question is where the path ENDED UP relative to where it started heading, so the
  // quantity that decides is the net change of heading. A straight path has no turning at all,
  // which means the direction did not change; an earlier version divided net by total there and
  // got 0/0, read as 1.0, and called a straight line a turn.
  const net = a.turn_net_deg, tot = a.turn_total_deg;
  let txt, cls;
  if (net <= 35){
    txt = '<b class="ok">ПУТЬ КОНЧИЛСЯ ТАМ ЖЕ, КУДА ШЁЛ</b> — направление не сменилось';
    cls = 'ok';
  } else if (net >= 70){
    txt = '<b class="warn">ПУТЬ УШЁЛ В ДРУГУЮ СТОРОНУ</b> — направление сменилось';
    cls = 'warn';
  } else {
    txt = '<b class="bad">НЕОДНОЗНАЧНО</b> — направление сменилось мало';
    cls = 'bad';
  }
  el.className = 'verdict ' + cls;
  el.innerHTML = txt +
    '<div class="legend">итоговое направление отличается от начального на ' +
    net.toFixed(0) + '°; всего по пути накручено ' + tot.toFixed(0) + '°' +
    (tot > 1e-6 ? ' (отношение ' + a.ratio.toFixed(2) + ')' : '') +
    '<br>границы: до 35° — туда же, от 70° — в другую сторону</div>';
}
function suggest(){
  if (!derived) return null;
  if (derived.turn_net_deg <= 35) return 'LOOK';
  if (derived.turn_net_deg >= 70) return 'TURN';
  return null;
}

/* ---------- pointer ---------- */
cv.addEventListener('pointerdown', e => {
  cv.setPointerCapture(e.pointerId);
  drawing = true; cur = [norm(e)]; draw();
});
cv.addEventListener('pointermove', e => {
  if (!drawing) return;
  const p = norm(e);
  const last = cur[cur.length-1];
  if (Math.hypot(p[0]-last[0], p[1]-last[1]) < 0.004) return;
  cur.push(p); draw(); updateVerdict();
});
cv.addEventListener('pointerup', () => {
  if (!drawing) return;
  drawing = false;
  if (cur && cur.length > 1) strokes.push(cur);
  cur = null; draw(); updateVerdict();
});
function undoStroke(){ if (strokes.length){ strokes.pop(); draw(); updateVerdict(); } }
function clearAll(){ strokes = []; cur = null; draw(); updateVerdict(); }

/* ---------- clip ---------- */
function render(){
  const c = current();
  if (c.phase === 'done'){
    for (const id of ['v','wideBtn','narrowBtn','winhint','fill']){
      const el = document.getElementById(id); if (el) el.style.display = 'none';
    }
    document.getElementById('num').textContent = '';
    document.getElementById('prog').textContent = '';
    const d = document.getElementById('done'); d.style.display = 'block';
    d.innerHTML = '<div style="font-size:18px;font-weight:650;margin-bottom:6px">Готово — '
      + labels.length + ' ответов</div><div class="hint" style="margin-bottom:11px">'
      + 'Скачайте файл и передайте его для оценки.</div>'
      + '<button class="exp" onclick="download()">Скачать метки (JSON)</button>'
      + '<div style="margin-top:12px"><textarea id="ta" readonly>'
      + JSON.stringify({labels: labels}) + '</textarea></div>';
    return;
  }
  const e = c.ev;
  document.getElementById('num').textContent = 'Событие ' + (c.n + 1) + ' из ' + c.total;
  document.getElementById('prog').textContent = c.phase === 'round2'
    ? ('основной проход готов · ' + c.n + ' из ' + c.total)
    : ('отвечено ' + c.n + ' из ' + c.total);
  document.getElementById('fill').style.width = (100 * c.n / c.total) + '%';
  strokes = []; cur = null; wide = false; derived = null;
  setWindow(); draw(); updateVerdict();
  v.src = e.clip_file; v.load(); t0 = Date.now();
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

/* ---------- saving ---------- */
function answer(label){
  const c = current();
  const round = c.phase === 'round2' ? 2 : 1;
  const pts = allPoints();
  const a = derived || analyse();
  const rec = {event_id: c.ev.event_id, round: round,
               human_label: label,
               label_source: label === suggest() ? 'derived_from_drawing' : 'button',
               drawing: strokes.map(st => st.map(p => [+p[0].toFixed(4), +p[1].toFixed(4)])),
               drawing_points: pts.length,
               derived_ratio: a ? +a.ratio.toFixed(4) : null,
               derived_turn_total_deg: a ? +a.turn_total_deg.toFixed(1) : null,
               derived_turn_net_deg: a ? +a.turn_net_deg.toFixed(1) : null,
               review_duration_s: Math.round((Date.now()-t0)/100)/10,
               used_extended_window: wide,
               saved_at: new Date().toISOString().slice(0,19)};
  labels = labels.filter(l => !(l.event_id === c.ev.event_id && l.round === round));
  labels.push(rec);
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch(e){}
  // The record goes to the server as well as to local storage. The first version of this page
  // only wrote local storage, so every drawing stayed in the browser and the data file stayed
  // empty — the work was done and none of it was on disk. Both places, every time.
  send(rec);
  v.pause(); render();
}
function send(rec){
  return fetch('/api/label', {method:'POST', headers:{'Content-Type':'application/json'},
                             body: JSON.stringify(rec)})
    .catch(() => { const d = document.getElementById('drawhint');
                   if (d) d.textContent = 'нет связи с сервером — метка сохранена в браузере'; });
}
/* Anything already in this browser from before the fix is pushed to the server on load, so a
   session that was labelled while the page was broken does not have to be redone. Records the
   server already has are sent again harmlessly: it keys them by event and pass. */
async function sync(){
  const local = labels.slice();
  if (!local.length) return;
  // Sequential, not Promise.all. Six parallel posts to a server that reads and rewrites one
  // file interleaved their reads and writes and left the file unparseable — the records were
  // fine, the file was not. One at a time is slower and correct.
  let ok = 0;
  for (const r of local){
    try { const res = await send(r); if (res && res.ok) ok++; } catch(e){}
  }
  const d = document.getElementById('drawhint');
  if (d) d.textContent = 'синхронизировано с сервером: ' + ok + ' из ' + local.length;
}
function download(){
  const blob = new Blob([JSON.stringify({labels: labels}, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'human_labels_v2.json';
  document.body.appendChild(a); a.click(); a.remove();
}
/* Enter takes the suggestion when the drawing supports one; otherwise nothing happens and the
   person has to choose a button, which is the honest outcome for an ambiguous path */
document.addEventListener('keydown', ev => {
  if (current().phase === 'done') return;
  if (ev.key === 'Enter'){
    const s = suggest();
    if (s) answer(s);
    else document.getElementById('verdict').innerHTML +=
      '<div class="legend bad">Рисунок не даёт однозначного ответа — нажмите кнопку.</div>';
  }
  else if (ev.key === ' '){ ev.preventDefault(); replay(); }
  else if (ev.key === 'w' || ev.key === 'W' || ev.key === 'ц' || ev.key === 'Ц') wider();
  else if (ev.key === 'Backspace'){ ev.preventDefault(); undoStroke(); }
});
window.addEventListener('resize', resize);
resize(); render(); sync();
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
    with _FILE_LOCK:
        if not LABELS.exists():
            return []
        try:
            return json.loads(LABELS.read_text(encoding="utf-8")).get("labels", [])
        except Exception:
            return []


# ThreadingHTTPServer runs handlers concurrently, and every save is a read-modify-write of one
# file. Two at once and the file is truncated or spliced. The lock serialises handlers, and the
# write goes to a temporary file that is then renamed, so a reader never sees a half-written file
# and an interrupted write cannot destroy what was there.
_FILE_LOCK = threading.Lock()


def write_labels(labels: list[dict]) -> None:
    with _FILE_LOCK:
        payload = json.dumps({"labels": labels}, indent=2, ensure_ascii=False)
        tmp = LABELS.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, LABELS)


ALLOWED = ("LOOK", "TURN", "UNCLEAR", "NO_LOCOMOTION")


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
            events, reps = self.payload["events"], self.payload.get("repeat_round", [])
            lab = read_labels()
            first = [l for l in lab if l.get("round", 1) == 1]
            second = [l for l in lab if l.get("round", 1) == 2]
            pub = lambda e: {k: e[k] for k in PUBLIC_KEYS if k in e}
            if len(first) < len(events):
                return self._json({"phase": "round1", "index": len(first),
                                   "n_total": len(events), "n_answered": len(first),
                                   "progress_text": f"отвечено {len(first)} из {len(events)}",
                                   "event": pub(events[len(first)])})
            if len(second) < len(reps):
                by = {e["event_id"]: e for e in events}
                return self._json({"phase": "round2", "index": len(second),
                                   "n_total": len(reps),
                                   "n_answered": len(events) + len(second),
                                   "progress_text": f"{len(events)} готово · сейчас "
                                                    f"{len(second)} из {len(reps)}",
                                   "event": pub(by[reps[len(second)]])})
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
            return self._json({"ok": False, "why": "bad json"}, 400)
        # The page speaks in LOOK / TURN / UNCLEAR / NO_LOCOMOTION already, because the label here
        # is derived from a drawn shape rather than chosen between two described words. The
        # vocabulary is internal to the file and is never explained on screen.
        if rec.get("human_label") not in ALLOWED:
            return self._json({"ok": False, "why": "bad label"}, 400)
        lab = read_labels()
        lab = [l for l in lab if not (l["event_id"] == rec["event_id"]
                                      and l.get("round", 1) == rec.get("round", 1))]
        lab.append(rec)
        write_labels(lab)
        self._json({"ok": True, "n": len(lab)})


def main() -> int:
    global SETV2, LABELS, CLIPS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8775)
    ap.add_argument("--host", default="0.0.0.0")
    # Paths as arguments for the same reason as in the other rounds: a set for a new clip has to be
    # served and saved separately, and a script whose output path is decorative will overwrite the
    # round the earlier phases were measured on.
    ap.add_argument("--set", default=None, help="набор событий")
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
        print(f"нет набора {SETV2} — сначала scripts/p095b_build.py")
        return 1
    Handler.payload = json.loads(SETV2.read_text(encoding="utf-8"))
    Handler.page = build_page()
    n = len(Handler.payload["events"])
    reps = len(Handler.payload["repeat_round"])
    print("P09.5C — разметка рисованием пути")
    print(f"  слева клип, справа холст; ответ выводится из формы нарисованного пути")
    print(f"  формула та же, что P07.4 применял к сигналу камеры: нетто / полный поворот")
    print(f"  событий {n} (+{reps} повторных показов), порядок тот же, что в P09.5B")
    print(f"  {wintext()['narrow']}  ·  шире: {wintext()['wide']}")
    print(f"  метки пишутся в {LABELS} (сырой рисунок сохраняется целиком)")
    print(f"  страница: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
