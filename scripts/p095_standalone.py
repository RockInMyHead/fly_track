#!/usr/bin/env python3
"""P09.5 — one self-contained page for the review, with no server behind it.

The server version has a dependency worth removing: it runs in a shell that belongs to some
other session, and we have watched that kind of process die three times in this project. Worse,
the person doing the review cannot restart it if it does — they would be told the thing they were
about to use is gone.

So this writes a single HTML file that opens by double-clicking, plays the clips from a relative
folder, keeps the answers in the browser's own storage, and offers a button that downloads them
as JSON. Nothing to start, nothing to keep alive, nothing that can die mid-review.

The blindness is the same and it is structural rather than declared: the file is built from an
explicit list of public fields, so the algorithmic class, the score, the direction and the pair
status are not in the document at all. Not hidden in it — absent from it. A curious person
opening the page source would find a number, a clip name and two offsets, and nothing else.

Answers survive a reload because they are written to localStorage after every click, and a repeat
does not overwrite its earlier answer: the two passes are separate records, which is what makes
the self-agreement check possible.

Usage:
    PYTHONPATH=. python scripts/p095_standalone.py
    open output/p095/review.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
OUT = ROOT / "output/p095"

# the only fields that reach the document
PUBLIC = ("event_id", "clip_file", "narrow", "wide")

TEMPLATE = """<!doctype html>
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
 .num{font-size:18px;font-weight:700}
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
 .answers{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;max-width:760px;margin:0 auto}
 .ans{padding:16px 10px;font-size:18px;border-width:2px;border-radius:10px}
 .ans.look{border-color:#2b6f92;color:#a9e6ff}.ans.look:hover{background:#12303f}
 .ans.turn{border-color:#2c7a55;color:#a9f0cd}.ans.turn:hover{background:#123326}
 .ans.unclear{border-color:#8a6a2a;color:#ffdca6}.ans.unclear:hover{background:#332912}
 .done{background:#151a23;border:1px solid #263041;border-radius:12px;padding:22px;
       text-align:center;margin-top:14px}
 .hint{color:#7c879c;font-size:11.5px;margin-top:8px;text-align:center}
 .exp{background:#1d3a2b;border-color:#2f6b4c;color:#a9f0cd;font-size:15px;padding:11px 18px}
 kbd{background:#1d2430;border:1px solid #2c374a;border-radius:4px;padding:1px 6px;font-size:12px}
 textarea{width:100%;height:120px;background:#0b0e13;color:#9fe8b5;border:1px solid #2c374a;
          border-radius:8px;padding:9px;font-family:ui-monospace,Menlo,monospace;font-size:11px}
</style></head><body><div class="wrap">
<h1>Поворот тела или только взгляд?</h1>
<div class="sub">Отвечайте только по видео. Направление не спрашивается, и что думает
 система — не показывается нигде на этой странице.</div>

<div class="defs">
 <div class="def look"><b>LOOK</b><span>Камера/голова повернулась, но движение в итоге
  продолжается примерно по прежнему направлению. Виден возврат взгляда.</span></div>
 <div class="def turn"><b>TURN</b><span>Человек реально меняет направление движения и
  продолжает двигаться в новом направлении.</span></div>
 <div class="def unclear"><b>UNCLEAR</b><span>По видео невозможно уверенно понять,
  повернулось тело или только камера. Угадывать не нужно.</span></div>
</div>

<div class="top"><div class="num" id="num">…</div><div class="prog" id="prog"></div></div>
<div class="bar"><div id="fill"></div></div>

<div class="stage"><video id="v" playsinline muted preload="auto"></video></div>
<div class="ctl">
 <button onclick="replay()">Повторить</button>
 <button class="wide" id="wideBtn" onclick="wider()">Показать шире</button>
 <button id="narrowBtn" onclick="narrower()" style="display:none">Вернуть обычное окно</button>
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
const SET = __SET__;
const KEY = 'p095_labels_v1';
let idx = 0, wide = false, t0 = 0, labels = [];
try { labels = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch(e) { labels = []; }
const v = document.getElementById('v');
const WIN = __WINTEXT__;

function saved(round){
  return labels.filter(l => l.round === round);
}
function current(){
  const first = saved(1), second = saved(2);
  if (first.length < SET.events.length)
    return {phase:'round1', ev: SET.events[first.length], n: first.length, total: SET.events.length};
  if (second.length < SET.repeat_round.length){
    const id = SET.repeat_round[second.length];
    const ev = SET.events.find(e => e.event_id === id);
    return {phase:'round2', ev, n: second.length, total: SET.repeat_round.length};
  }
  return {phase:'done'};
}

function render(){
  const c = current();
  if (c.phase === 'done'){
    for (const id of ['v','ans','wideBtn','narrowBtn','winhint','fill'])
      document.getElementById(id).style.display = 'none';
    document.getElementById('num').textContent = '';
    document.getElementById('prog').textContent = '';
    const d = document.getElementById('done');
    d.style.display = 'block';
    d.innerHTML = '<div style="font-size:19px;font-weight:650;margin-bottom:6px">Готово — '
      + labels.length + ' ответов</div>'
      + '<div class="hint" style="margin-bottom:12px">Скачайте файл и передайте его для оценки.</div>'
      + '<button class="exp" onclick="download()">Скачать метки (JSON)</button>'
      + '<div style="margin-top:14px"><textarea id="ta" readonly>'
      + JSON.stringify({labels: labels}) + '</textarea></div>'
      + '<div class="hint">Если скачивание не сработало — выделите текст выше и сохраните '
      + 'в файл <b>human_labels.json</b> в папку <b>data/p095/</b></div>';
    return;
  }
  const e = c.ev;
  document.getElementById('num').textContent = (c.phase === 'round2' ? 'Показ ' : 'Событие ')
    + (c.n + 1) + ' из ' + c.total;
  document.getElementById('prog').textContent = c.phase === 'round2'
    ? ('первый проход готов · ' + c.n + ' из ' + c.total)
    : ('отвечено ' + c.n + ' из ' + c.total);
  document.getElementById('fill').style.width = (100 * c.n / c.total) + '%';
  wide = false; setWindow();
  v.src = e.clip_file; v.load();
  t0 = Date.now();
  v.onloadedmetadata = () => { v.currentTime = e.narrow[0]; v.play().catch(()=>{}); };
  v.ontimeupdate = () => {
    if (!wide && v.currentTime >= e.narrow[1]) v.pause();
  };
}
function setWindow(){
  const c = current();
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
  a.href = URL.createObjectURL(blob); a.download = 'human_labels.json';
  document.body.appendChild(a); a.click(); a.remove();
}
document.addEventListener('keydown', e => {
  if (current().phase === 'done') return;
  if (e.key === '1') answer('LOOK');
  else if (e.key === '2') answer('TURN');
  else if (e.key === '3') answer('UNCLEAR');
  else if (e.key === ' '){ e.preventDefault(); replay(); }
  else if (e.key === 'w' || e.key === 'W' || e.key === 'ц' || e.key === 'Ц') wider();
});
render();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "review.html"))
    args = ap.parse_args()
    s = json.loads((DATA / "review_set.json").read_text(encoding="utf-8"))
    events = [{k: e[k] for k in PUBLIC if k in e} for e in s["events"]]
    payload = {"events": events, "repeat_round": s["repeat_round"]}
    # the algorithmic fields are not stripped here and kept somewhere else: they are simply never
    # read, so the document cannot contain them however the set file grows
    wf = s.get("window_first", {"before_s": 3, "after_s": 20})
    ww = s.get("window_wide", {"before_s": 5, "after_s": 25})
    wintext = {"narrow": f"окно: {wf['before_s']:.0f} с до, {wf['after_s']:.0f} с после",
               "wide": f"окно: {ww['before_s']:.0f} с до, {ww['after_s']:.0f} с после"}
    doc = (TEMPLATE.replace("__SET__", json.dumps(payload, ensure_ascii=False))
                   .replace("__WINTEXT__", json.dumps(wintext, ensure_ascii=False)))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(doc, encoding="utf-8")

    leaked = [k for k in ("algo_class_b", "algo_score_b", "algo_class_a", "algo_score_a",
                          "algo_direction", "proxy_label", "pair_status", "yaw")
              if k in doc]
    print("P09.5 — автономная страница разметки (без сервера)")
    print(f"  событий: {len(events)}, повторов: {len(s['repeat_round'])}")
    print(f"  алгоритмических полей в файле: {leaked or 'ни одного'}")
    print(f"  размер: {dest.stat().st_size / 1024:.0f} КБ")
    print(f"  записано: {dest}")
    print()
    print("  открыть: двойной клик по файлу, или")
    print(f"    open {dest}")
    print("  метки хранятся в браузере и скачиваются кнопкой в конце")
    print("  скачанный human_labels.json положить в data/p095/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
