#!/usr/bin/env python3
"""
A live viewer: the clip on the left, the MaleCNS trajectory on the right.

The trajectory is drawn as the video plays, so a dot marks where the fly thinks it is at
the moment being shown. That makes it possible to see *when* the path goes wrong rather
than only that it ends up wrong, which is what the static figures could not show.

The trajectory data is written into the page rather than fetched, so the page works from
the annotation server without needing new endpoints. Only the video is loaded over HTTP,
from the media directory the server already exposes.

Which clips appear
------------------
Any clip in the registry that has a trajectory from `p072_trajectory.py`. Adding a clip
means running the chain and regenerating this page; nothing here is clip-specific.

Two things worth knowing while watching
---------------------------------------
The trajectory is in arbitrary units and each clip is scaled to its own spread, so the
shape is comparable between clips but the numbers are not metres. The note under the
canvas says so, because a clean-looking path is easy to mistake for a correct one.

Usage:
    PYTHONPATH=. python scripts/p07_live_viewer.py
    PYTHONPATH=. python scripts/p07_live_viewer.py --video VID00006
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "output/p072"
EVENTS = ROOT / "output/p072"
MEDIA = ROOT / "webapp/media"
OUT = ROOT / "webapp/flycam.html"

# clip -> (media file, trajectory csv). The media file must exist for the clip to appear.
CLIPS = [
    ("VID00001", "VID00001_fixed.mp4"),
    ("VID00002", "VID00002_fixed.mp4"),
    ("VID00005", "VID00005_fixed.mp4"),
    ("VID00006", "VID00006_fixed.mp4"),
    ("VID00009", "VID00009_fixed.mp4"),
]


def load_clip(name: str, media: str) -> dict | None:
    traj = TRAJ / f"trajectory_{name}.csv"
    if not traj.exists() or not (MEDIA / media).exists():
        return None
    rows = list(csv.DictReader(traj.open()))
    if not rows:
        return None
    t = [round(float(r["t"]), 3) for r in rows]
    x = [round(float(r["x"]), 5) for r in rows]
    y = [round(float(r["y"]), 5) for r in rows]
    # decimate for the browser: 60k points would be drawn every frame for no gain, and
    # at 12 points per second the curve is visually identical
    stride = max(1, len(rows) // 6000)
    idx = list(range(0, len(rows), stride))
    if idx[-1] != len(rows) - 1:
        idx.append(len(rows) - 1)
    out = {
        "name": name, "media": media,
        "t": [t[i] for i in idx], "x": [x[i] for i in idx], "y": [y[i] for i in idx],
        "duration": t[-1],
        "events": [],
    }
    ev = EVENTS / f"events_{name}.csv"
    if ev.exists():
        for e in csv.DictReader(ev.open()):
            out["events"].append({
                "t": round(float(e["t_start"]), 2),
                "dir": e["direction"],
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="clip to open first")
    args = ap.parse_args()

    clips = [c for c in (load_clip(n, m) for n, m in CLIPS) if c]
    if not clips:
        print("нет ни одной траектории — сначала прогоните p072_trajectory.py")
        return

    first = args.video or clips[0]["name"]
    payload = json.dumps(clips, ensure_ascii=False)

    html = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>MaleCNS — видео и траектория</title>
<style>
  :root{--bg:#0f1319;--panel:#161b23;--line:#252c37;--fg:#dde3ec;--dim:#7b8798;
        --left:#2ecc71;--right:#b06ae0;--path:#4aa3ff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  header{display:flex;align-items:center;gap:14px;padding:10px 16px;
         border-bottom:1px solid var(--line);flex-wrap:wrap}
  h1{font-size:15px;margin:0;font-weight:600}
  select,button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
                border-radius:6px;padding:5px 9px;font:inherit}
  button{cursor:pointer}
  button:hover{border-color:#3a4250}
  .wrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,0.85fr);
        gap:14px;padding:14px;height:calc(100vh - 55px)}
  @media (max-width:900px){.wrap{grid-template-columns:1fr;height:auto}}
  .pane{background:var(--panel);border:1px solid var(--line);border-radius:10px;
        padding:10px;display:flex;flex-direction:column;min-height:0}
  video{width:100%;border-radius:6px;background:#000;display:block}
  .canvasbox{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
  canvas{background:#0b0e13;border-radius:6px;max-width:100%;max-height:100%;
         width:auto;height:auto}
  .row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px}
  input[type=range]{flex:1;min-width:120px;accent-color:var(--path)}
  .stat{font-variant-numeric:tabular-nums;color:var(--dim);font-size:12px}
  .note{color:var(--dim);font-size:11.5px;margin-top:8px;line-height:1.4}
  kbd{background:#0b0e13;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
      font:11px ui-monospace,monospace}
</style></head><body>
<header>
  <h1>MaleCNS: видео и траектория</h1>
  <select id="clip"></select>
  <button id="play">▶ play</button>
  <label class="stat">скорость
    <select id="rate">
      <option>0.25</option><option>0.5</option><option selected>1</option>
      <option>2</option><option>4</option><option>8</option>
    </select>
  </label>
  <span class="stat" id="clock">0.0 / 0.0 с</span>
</header>
<div class="wrap">
  <div class="pane">
    <video id="vid" preload="auto" playsinline></video>
    <div class="row">
      <input type="range" id="seek" min="0" max="100" value="0" step="0.01">
    </div>
    <div class="note">← → шаг 5 с · <kbd>space</kbd> пауза · клик по полосе — переход
    </div>
  </div>
  <div class="pane">
    <div class="canvasbox"><canvas id="map" width="720" height="720"></canvas></div>
    <div class="row">
      <span class="stat" id="pos">x — · y —</span>
      <span class="stat" id="turn">поворот —</span>
    </div>
    <div class="note">Синяя линия — путь, построенный MaleCNS с момента старта.
      Оранжевая точка — где муха по своему счёту находится в показанный момент.
      Зелёные метки — повороты LEFT, фиолетовые — RIGHT.
      Масштаб условный: единицы произвольные, абсолютного масштаба у одного ролика нет.
    </div>
  </div>
</div>
<script>
const CLIPS = __PAYLOAD__;
let clip = null, pts = [], view = null;

const vid = document.getElementById("vid");
const map = document.getElementById("map");
const ctx = map.getContext("2d");
const sel = document.getElementById("clip");
const rate = document.getElementById("rate");
const seek = document.getElementById("seek");
const clock = document.getElementById("clock");
const posEl = document.getElementById("pos");
const turnEl = document.getElementById("turn");
const playBtn = document.getElementById("play");

CLIPS.forEach(c => {
  const o = document.createElement("option");
  o.value = c.name;
  o.textContent = c.name + "  (" + c.duration.toFixed(0) + " с, " + c.events.length + " поворотов)";
  sel.appendChild(o);
});
sel.value = "__FIRST__";

function pick(name){
  clip = CLIPS.find(c => c.name === name);
  pts = clip.t.map((t,i) => ({t, x: clip.x[i], y: clip.y[i]}));
  // fixed view per clip so the map never jumps while playing
  let x0 = Math.min(...pts.map(p=>p.x)), x1 = Math.max(...pts.map(p=>p.x));
  let y0 = Math.min(...pts.map(p=>p.y)), y1 = Math.max(...pts.map(p=>p.y));
  const pad = 0.08 * Math.max(x1-x0, y1-y0, 1e-9);
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const span = Math.max(x1-x0, y1-y0);
  view = {cx:(x0+x1)/2, cy:(y0+y1)/2, span};
  vid.src = "/media/" + clip.media;
  vid.playbackRate = parseFloat(rate.value);
  draw();
}

function toPx(x,y){
  const s = (Math.min(map.width, map.height) * 0.90) / view.span;
  return [map.width/2 + (x-view.cx)*s, map.height/2 - (y-view.cy)*s];
}

function headIndex(tNow){
  // last point at or before the shown time; the trajectory arrays are sorted by time
  let lo = 0, hi = pts.length - 1, r = 0;
  while (lo <= hi){
    const mid = (lo+hi) >> 1;
    if (pts[mid].t <= tNow){ r = mid; lo = mid+1; } else hi = mid-1;
  }
  return r;
}

function draw(){
  const tNow = vid.currentTime || 0;
  const head = headIndex(tNow);
  ctx.clearRect(0,0,map.width,map.height);

  // grid at fixed fractions, so movement is readable
  ctx.strokeStyle = "#1a2029"; ctx.lineWidth = 1;
  for (let k=1;k<8;k++){
    const u = map.width*k/8, v = map.height*k/8;
    ctx.beginPath(); ctx.moveTo(u,0); ctx.lineTo(u,map.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,v); ctx.lineTo(map.width,v); ctx.stroke();
  }

  // the path taken so far
  if (head > 1){
    ctx.strokeStyle = "#4aa3ff"; ctx.lineWidth = 2.2;
    ctx.lineJoin = "round"; ctx.beginPath();
    const [sx,sy] = toPx(pts[0].x, pts[0].y);
    ctx.moveTo(sx,sy);
    for (let i=1;i<=head;i++){
      const [px,py] = toPx(pts[i].x, pts[i].y);
      ctx.lineTo(px,py);
    }
    ctx.stroke();
  }

  // turns already passed
  for (const e of clip.events){
    if (e.t > tNow) continue;
    const i = headIndex(e.t);
    const [ex,ey] = toPx(pts[i].x, pts[i].y);
    ctx.fillStyle = e.dir === "LEFT" ? "#2ecc71" : "#b06ae0";
    ctx.beginPath(); ctx.arc(ex,ey,4.5,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();
  }

  // start marker
  const [ax,ay] = toPx(pts[0].x, pts[0].y);
  ctx.fillStyle = "#2ecc71";
  ctx.beginPath(); ctx.arc(ax,ay,7,0,Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#000"; ctx.stroke();

  // where the fly thinks it is right now
  const [hx,hy] = toPx(pts[head].x, pts[head].y);
  ctx.fillStyle = "#ff9f43";
  ctx.beginPath(); ctx.arc(hx,hy,7.5,0,Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5; ctx.stroke();

  clock.textContent = tNow.toFixed(1) + " / " + clip.duration.toFixed(1) + " с";
  posEl.textContent = "x " + pts[head].x.toFixed(2) + " · y " + pts[head].y.toFixed(2);
  const last = [...clip.events].reverse().find(e => e.t <= tNow);
  turnEl.textContent = last
    ? ("последний поворот: " + (last.dir === "LEFT" ? "LEFT" : "RIGHT") + " в " + last.t.toFixed(1) + " с")
    : "поворотов пока не было";
  if (!seek.dataset.busy) seek.value = (tNow / clip.duration * 100);
}

sel.onchange = () => pick(sel.value);
rate.onchange = () => { vid.playbackRate = parseFloat(rate.value); };
playBtn.onclick = () => { vid.paused ? vid.play() : vid.pause(); };
vid.onplay  = () => playBtn.textContent = "❚❚ пауза";
vid.onpause = () => playBtn.textContent = "▶ play";
vid.ontimeupdate = draw;
vid.onloadedmetadata = draw;
seek.oninput = () => {
  seek.dataset.busy = "1";
  vid.currentTime = clip.duration * seek.value / 100;
  draw();
};
seek.onchange = () => { delete seek.dataset.busy; };

document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space"){ e.preventDefault(); vid.paused ? vid.play() : vid.pause(); }
  if (e.code === "ArrowRight") vid.currentTime = Math.min(clip.duration, vid.currentTime + 5);
  if (e.code === "ArrowLeft")  vid.currentTime = Math.max(0, vid.currentTime - 5);
});

pick(sel.value);
</script>
</body></html>
"""
    html = html.replace("__PAYLOAD__", payload).replace("__FIRST__", first)
    OUT.write_text(html, encoding="utf-8")

    print("P07 — живой просмотр: видео слева, траектория справа")
    print(f"  роликов в странице: {len(clips)}")
    for c in clips:
        print(f"    {c['name']:>10s}  {c['duration']:6.1f} с, "
              f"{len(c['t']):5d} точек, {len(c['events']):3d} поворотов")
    print(f"\n  открыт первым: {first}")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"\n  страница доступна по адресу сервера разметки: /flycam.html")


if __name__ == "__main__":
    main()
