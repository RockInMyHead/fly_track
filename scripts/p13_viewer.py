#!/usr/bin/env python3
"""P13 — watch the walk: video on the left, the route drawn beside it, kept in step.

The point of the page is to make the two things that were built separately agree in front of you.
The video is a chunk as the camera recorded it. The route is what the tracker reconstructed for
that same chunk, in the same seconds. When the picture and the line disagree, you can see it.

    left      the chunk, playing
    right     the floor plan, the route of this chunk drawn as the video advances, and a marker
              showing where the walker is at the current second
    background  chunks already watched, dimmed, so the route accumulates into the whole walk

The route is not interpolated for display only: the tracker wrote a point every 20 ms and the page
thins those to one per 0.15 s, which is finer than the eye can follow and small enough to send.

Moving to the next chunk keeps the clock: the next video starts at its own zero, which is where the
previous chunk's file ended, so stopping and starting is what the camera did.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/p13_viewer.py --port 8780
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
MEDIA = ROOT / "webapp/media"
PLAN = ROOT / "data/p08/plan.png"
ANCHOR = OUT / "CHAIN_ANCHOR.json"
CACHE = OUT / "traj_cache"

PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>P13 — просмотр: видео и маршрут</title>
<style>
 html,body{margin:0;height:100%;background:#0d0f14;color:#e6e9f2;overflow:hidden;
   font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 #top{height:52px;display:flex;align-items:center;gap:14px;padding:0 16px;
   border-bottom:1px solid #232838;background:#12151d}
 #top b{font-size:15px}
 select,button{background:#1b2030;color:#e6e9f2;border:1px solid #2c3346;border-radius:7px;
   padding:6px 11px;font-size:13px;cursor:pointer}
 button:hover{background:#232a3d}
 #wrap{display:flex;height:calc(100% - 52px)}
 #left{flex:1.25;min-width:0;background:#000;position:relative}
 video{width:100%;height:100%;object-fit:contain;background:#000}
 #right{flex:1;min-width:0;position:relative;background:#fff;border-left:1px solid #232838}
 canvas{display:block;width:100%;height:100%}
 #meta{position:absolute;left:12px;top:10px;background:rgba(13,15,20,.86);border-radius:8px;
   padding:8px 12px;font-size:12.5px;line-height:1.6;pointer-events:none}
 #meta b{color:#7dd3fc}
 .k{color:#8b93a7}
 #note{position:absolute;inset:0;display:none;place-items:center;text-align:center;
   background:#0d0f14;color:#c8cede;font-size:15px;padding:40px}
 #note.on{display:grid}
 #hint{position:absolute;left:12px;bottom:10px;background:rgba(13,15,20,.86);border-radius:8px;
   padding:6px 10px;font-size:12px;color:#9aa5bd}
</style></head><body>
<div id="top">
  <b>P13</b>
  <span class="k">кусок</span>
  <select id="pick"></select>
  <button id="prev">◀ пред</button>
  <button id="next">след ▶</button>
  <span class="k" id="clock">—</span>
  <span style="flex:1"></span>
  <label class="k"><input type="checkbox" id="carry" checked> показывать пройденное</label>
  <span class="k" id="vstate"></span>
</div>
<div id="wrap">
  <div id="left">
    <video id="v" controls preload="auto"></video>
    <div id="note"></div>
    <div id="hint">пробел — пауза · ←/→ — 1 с · shift+←/→ — 10 с · [ ] — кусок</div>
  </div>
  <div id="right">
    <canvas id="cv"></canvas>
    <div id="meta"></div>
  </div>
</div>
<script>
const CHUNKS = __CHUNKS__;
let cur = 0, PLAN=null, W=5298, H=3743, view={sc:1,ox:0,oy:0};
let TRAJ=null, CARRY=[];      // this chunk, and the ones already shown
const v=document.getElementById('v'), cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const img=new Image();
img.onload=()=>{PLAN=img;W=img.naturalWidth;H=img.naturalHeight;fit();draw();};
img.src='/api/plan';

function fit(){
  const box=cv.parentElement, dpr=window.devicePixelRatio||1;
  cv.width=box.clientWidth*dpr; cv.height=box.clientHeight*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const s=Math.min(box.clientWidth/W, box.clientHeight/H)*0.97;
  view={sc:s,ox:(box.clientWidth-s*W)/2,oy:(box.clientHeight-s*H)/2};
}
function sc(x,y){return [view.ox+view.sc*x, view.oy+view.sc*y];}

function draw(){
  const box=cv.parentElement, cw=box.clientWidth, ch=box.clientHeight;
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,cw,ch);
  if(PLAN) ctx.drawImage(PLAN, view.ox, view.oy, view.sc*W, view.sc*H);

  // everything watched before, faint, so the route reads as one walk
  if(document.getElementById('carry').checked){
    ctx.strokeStyle='rgba(60,90,150,.28)'; ctx.lineWidth=2;
    for(const seg of CARRY){
      if(!seg.x.length) continue; ctx.beginPath();
      for(let k=0;k<seg.x.length;k++){const p=sc(seg.x[k],seg.y[k]); k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);}
      ctx.stroke();
    }
  }
  if(!TRAJ||!TRAJ.x.length){ meta(); return; }

  const t=v.currentTime||0;
  // how far along the route the walker is at this second
  let i=0; while(i<TRAJ.t.length && TRAJ.t[i]<=t) i++;
  i=Math.max(1, Math.min(i, TRAJ.x.length-1));

  ctx.strokeStyle='#1d4ed8'; ctx.lineWidth=4; ctx.lineJoin='round'; ctx.lineCap='round';
  ctx.beginPath();
  for(let k=0;k<i;k++){const p=sc(TRAJ.x[k],TRAJ.y[k]); k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);}
  ctx.stroke();

  // the part still ahead, dashed and pale: the tracker's whole route, not yet walked
  ctx.setLineDash([6,7]); ctx.strokeStyle='rgba(29,78,216,.32)'; ctx.lineWidth=2;
  ctx.beginPath();
  for(let k=i;k<TRAJ.x.length;k++){const p=sc(TRAJ.x[k],TRAJ.y[k]); k===i?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]);}
  ctx.stroke(); ctx.setLineDash([]);

  // where the walker is now, interpolated between the two nearest points
  const a=Math.max(0,i-2), b=Math.min(TRAJ.x.length-1,i);
  const ta=TRAJ.t[a]||0, tb=TRAJ.t[b]||ta+1;
  const f=tb>ta?Math.max(0,Math.min(1,(t-ta)/(tb-ta))):0;
  const mx=TRAJ.x[a]+(TRAJ.x[b]-TRAJ.x[a])*f, my=TRAJ.y[a]+(TRAJ.y[b]-TRAJ.y[a])*f;
  const [px,py]=sc(mx,my);
  ctx.fillStyle='#dc2626'; ctx.strokeStyle='#fff'; ctx.lineWidth=2.5;
  ctx.beginPath(); ctx.arc(px,py,9,0,7); ctx.fill(); ctx.stroke();

  const [sx,sy]=sc(TRAJ.x[0],TRAJ.y[0]);
  ctx.fillStyle='#137a3f'; ctx.beginPath(); ctx.arc(sx,sy,7,0,7); ctx.fill();
  ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();

  meta(i,t);
}

function meta(i,t){
  const c=CHUNKS[cur]||{};
  const el=document.getElementById('meta');
  const edge=(TRAJ&&TRAJ.edge&&TRAJ.edge[Math.max(0,i-1)])||'—';
  el.innerHTML =
    `<b>${c.video||''}</b><br>`+
    `<span class="k">время</span> ${fmt(t)} <span class="k">из</span> ${fmt(c.duration||0)}<br>`+
    `<span class="k">проход</span> ${TRAJ?edge:'нет данных'}<br>`+
    `<span class="k">точек пройдено</span> ${TRAJ?i+' / '+TRAJ.x.length:'—'}`;
}
function fmt(s){s=Math.max(0,s||0);const m=Math.floor(s/60),x=Math.floor(s%60);
  return m+':'+String(x).padStart(2,'0');}

cv.addEventListener('wheel',ev=>{
  ev.preventDefault();
  const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  const bx=(mx-view.ox)/view.sc, by=(my-view.oy)/view.sc;
  view.sc*= ev.deltaY<0?1.15:1/1.15;
  view.ox=mx-bx*view.sc; view.oy=my-by*view.sc; draw();
},{passive:false});
let drag=null;
cv.addEventListener('mousedown',e=>drag={x:e.clientX,y:e.clientY,ox:view.ox,oy:view.oy});
window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{if(!drag)return;
  view.ox=drag.ox+(e.clientX-drag.x); view.oy=drag.oy+(e.clientY-drag.y); draw();});
window.addEventListener('resize',()=>{fit();draw();});

async function load(i, keepTime){
  cur=Math.max(0,Math.min(CHUNKS.length-1,i));
  const c=CHUNKS[cur];
  document.getElementById('pick').value=c.video;
  const note=document.getElementById('note');
  const vs=document.getElementById('vstate');
  if(!c.video_ready){
    note.className='on';
    note.innerHTML=`видео для <b>${c.video}</b> ещё готовится<br><span class="k">`+
      `перекодирование идёт в фоне, маршрут для этого куска уже есть</span>`;
    vs.textContent='видео готовится';
  } else {
    note.className=''; vs.textContent='';
    const same = v.dataset.video===c.video;
    if(!same){ v.src='/media/'+c.video; v.dataset.video=c.video; }
    if(!keepTime) v.currentTime=0;
  }
  TRAJ=null;
  try{
    const r=await fetch('/api/traj/'+c.video); TRAJ=await r.json();
  }catch(e){ TRAJ=null; }
  await rebuildCarry();
  document.getElementById('clock').textContent =
    `кусок ${cur+1} из ${CHUNKS.length}`;
  draw();
}
async function rebuildCarry(){
  CARRY=[];
  for(let k=0;k<cur;k++){
    try{const r=await fetch('/api/traj/'+CHUNKS[k].video+'?thin=8'); CARRY.push(await r.json());}catch(e){}
  }
}
v.addEventListener('timeupdate',draw);
v.addEventListener('play',draw);
v.addEventListener('ended',()=>{ if(cur<CHUNKS.length-1) load(cur+1,false); });

document.getElementById('pick').addEventListener('change',e=>{
  const i=CHUNKS.findIndex(c=>c.video===e.target.value); if(i>=0) load(i,false);
});
document.getElementById('prev').onclick=()=>load(cur-1,false);
document.getElementById('next').onclick=()=>load(cur+1,false);
document.getElementById('carry').onchange=draw;
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT')return;
  if(e.code==='Space'){e.preventDefault(); v.paused?v.play():v.pause();}
  else if(e.key==='ArrowLeft'){v.currentTime-=e.shiftKey?10:1; draw();}
  else if(e.key==='ArrowRight'){v.currentTime+=e.shiftKey?10:1; draw();}
  else if(e.key==='['){load(cur-1,false);} else if(e.key===']'){load(cur+1,false);}
});

(async function start(){
  const sel=document.getElementById('pick');
  sel.innerHTML=CHUNKS.map(c=>`<option value="${c.video}">${c.video}`+
    (c.video_ready?'':' (видео готовится)')+`</option>`).join('');
  const first=CHUNKS.findIndex(c=>c.video_ready);
  await load(first<0?0:first,false);
  setInterval(()=>{ if(v.paused) draw(); },1000);
})();
</script></body></html>"""


def thin(video: str, step: int) -> dict:
    """The trajectory of one chunk, thinned to `step` rows. Cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{video}_{step}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    src = OUT / f"runs/{video}/graph_trajectory.csv"
    if not src.exists():
        return {"t": [], "x": [], "y": [], "edge": []}
    t, x, y, e = [], [], [], []
    with src.open() as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i % step:
                continue
            t.append(round(float(row["t"]), 2))
            x.append(round(float(row["x_px"]), 1))
            y.append(round(float(row["y_px"]), 1))
            e.append(row["edge"])
    d = {"t": t, "x": x, "y": y, "edge": e}
    p.write_text(json.dumps(d), encoding="utf-8")
    return d


def chunks() -> list[dict]:
    order = json.loads(ANCHOR.read_text(encoding="utf-8"))["order"]
    out = []
    for v in order:
        traj = OUT / f"runs/{v}/graph_trajectory.csv"
        dur = 0.0
        if traj.exists():
            try:
                with traj.open() as fh:
                    fh.readline()
                    dur = float([ln for ln in fh if ln.strip()][-1].split(",")[0])
            except Exception:
                pass
        prev = MEDIA / f"{v}_fixed.mp4"
        ready = False
        if prev.exists():
            js = CACHE / f"{v}_dur.json"
            if js.exists():
                try:
                    have = float(json.loads(js.read_text())["seconds"])
                except Exception:
                    have = None
            else:
                have = None
            # the file's own length, read once, so a stale preview is not offered as good
            if have is None:
                try:
                    import subprocess
                    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                        "format=duration", "-of", "csv=p=0", str(prev)],
                                       capture_output=True, text=True, timeout=60).stdout.strip()
                    have = float(o) if o else None
                except Exception:
                    have = None
                # Cache write is a convenience and must not be able to discard the measurement:
                # an earlier version wrote the file inside the same try, so when the cache
                # directory did not exist yet the exception threw away a perfectly good duration
                # and every chunk reported "video not ready".
                if have:
                    try:
                        CACHE.mkdir(parents=True, exist_ok=True)
                        js.write_text(json.dumps({"seconds": have}), encoding="utf-8")
                    except Exception:
                        pass
            ready = bool(have and dur and abs(have - dur) <= 4.0)
        out.append({"video": v, "duration": round(dur, 1), "video_ready": ready})
    return out


class H(BaseHTTPRequestHandler):
    server_version = "p13viewer/1.0"

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, o):
        self._send(json.dumps(o, ensure_ascii=False, default=str).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        p = self.path.split("?", 1)[0]
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        if p in ("/", "/index.html", "/p13v"):
            page = PAGE.replace("__CHUNKS__", json.dumps(chunks(), ensure_ascii=False))
            return self._send(page.encode(), "text/html; charset=utf-8")
        if p.startswith("/api/traj/"):
            m = re.fullmatch(r"/api/traj/(VID\d+)", p)
            if not m:
                return self._send(b"bad", "text/plain", 400)
            step = 7
            mm = re.search(r"thin=(\d+)", q)
            if mm:
                step = max(1, int(mm.group(1)))
            return self._json(thin(m.group(1), step))
        if p == "/api/plan":
            if PLAN.exists():
                return self._send(PLAN.read_bytes(), "image/png")
            return self._send(b"no plan", "text/plain", 404)
        if p.startswith("/media/"):
            m = re.fullmatch(r"/media/(VID\d+)", p)
            if not m:
                return self._send(b"bad", "text/plain", 400)
            f = MEDIA / f"{m.group(1)}_fixed.mp4"
            if not f.exists():
                return self._send(b"not ready", "text/plain", 404)
            return self._range(f)
        if re.fullmatch(r"/[A-Za-z0-9_.\-]+", p):
            f = (ROOT / p.lstrip("/")).resolve()
            if str(f).startswith(str(ROOT)) and f.is_file():
                return self._send(f.read_bytes(), "application/octet-stream")
        return self._send(b"not found", "text/plain", 404)

    def _range(self, f: Path):
        """Range support: without it the video cannot be seeked, which defeats the page."""
        size = f.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
        start, end = max(0, min(start, size - 1)), max(0, min(end, size - 1))
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with f.open("rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"P13 просмотр  http://{a.host}:{a.port}/")
    cs = chunks()
    print(f"  кусков {len(cs)}, видео готово {sum(1 for c in cs if c['video_ready'])}")
    for c in cs:
        if not c["video_ready"]:
            print(f"    ждёт видео: {c['video']}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
