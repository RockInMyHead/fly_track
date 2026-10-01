#!/usr/bin/env python3
"""P13 — the live view: the route growing on the plan while the queue works through the chunks.

Serves one page and one endpoint. The page redraws the plan and the accumulated route every two
seconds, so a walk of several hours can be watched rather than waited out.

    GET /p13            the page
    GET /api/p13/state  progress, the per-chunk table, and the route so far
    GET /api/p13/plan   the floor plan
    GET /api/p13/video  the chunk being worked on, if it has been prepared for the browser

Nothing is computed here. The queue writes output/p13/state.json and this reads it, so the picture
cannot disagree with the work.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/p13_live.py --port 8766
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
STATE = OUT / "state.json"
LIVE = OUT / "live_trajectory.csv"
PLAN = ROOT / "data/p08/plan.png"

PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>P13 — маршрут в реальном времени</title>
<style>
 html,body{margin:0;height:100%;background:#0f1117;color:#e6e9f2;
   font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 #wrap{display:flex;height:100%}
 #left{flex:1;position:relative;min-width:0;background:#fff}
 canvas{display:block;width:100%;height:100%}
 #right{width:390px;flex:none;border-left:1px solid #232838;padding:16px 18px;overflow:auto;
   background:#12151d}
 h1{font-size:16px;margin:0 0 4px}
 h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#8b93a7;margin:18px 0 8px}
 .bar{height:12px;border-radius:6px;background:#232838;overflow:hidden;margin:10px 0 6px}
 .bar>i{display:block;height:100%;background:linear-gradient(90deg,#3ddc84,#3d7dd8);
   transition:width .4s}
 .big{font-size:26px;font-weight:700;letter-spacing:.01em}
 .muted{color:#8b93a7}
 table{width:100%;border-collapse:collapse;font-size:12.5px}
 td{padding:4px 2px;border-bottom:1px solid #1d2230}
 td:first-child{white-space:nowrap}
 .ok{color:#3ddc84}.err{color:#ff6b6b}.run{color:#ffd60a}.wait{color:#6b7280}
 .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
   background:#232838;color:#9aa5bd}
 #err{background:#2a1012;border:1px solid #7f1d1d;border-radius:8px;padding:8px 10px;
   font-size:12px;margin-top:8px;white-space:pre-wrap}
</style></head><body>
<div id="wrap">
  <div id="left"><canvas id="cv"></canvas></div>
  <div id="right">
    <h1>P13 — проход по кускам записи</h1>
    <div class="muted" id="sub">состояние читается каждые 2 секунды</div>
    <div class="bar"><i id="pb" style="width:0%"></i></div>
    <div><span class="big" id="cnt">0 / 0</span> <span class="muted">кусков</span></div>
    <div class="muted" id="times" style="margin-top:6px"></div>
    <h2>Сейчас</h2>
    <div id="cur" class="muted">—</div>
    <h2>Цепочка стартов</h2>
    <div id="chain" class="muted" style="font-size:12px">—</div>
    <h2>Куски</h2>
    <table id="tbl"></table>
    <div id="err" style="display:none"></div>
  </div>
</div>
<script>
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let plan=null, W=5298, H=3743, view={sc:1,ox:0,oy:0};
const img=new Image();
img.onload=()=>{plan=img;W=img.naturalWidth;H=img.naturalHeight;fit();draw();};
img.src='/api/p13/plan?t='+Date.now();
let ROUTE=null, COLORS=['#1d4ed8','#dc2626','#059669','#d97706','#7c3aed','#0891b2',
  '#be185d','#4d7c0f','#b45309','#0f766e','#9333ea','#b91c1c'];

function fit(){
  const box=cv.parentElement;
  const dpr=window.devicePixelRatio||1;
  cv.width=box.clientWidth*dpr; cv.height=box.clientHeight*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const s=Math.min(box.clientWidth/W, box.clientHeight/H)*0.97;
  view={sc:s, ox:(box.clientWidth-s*W)/2, oy:(box.clientHeight-s*H)/2};
}
function sc(x,y){return [view.ox+view.sc*x, view.oy+view.sc*y];}

function draw(){
  const box=cv.parentElement, cw=box.clientWidth, ch=box.clientHeight;
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,cw,ch);
  if(plan) ctx.drawImage(plan, view.ox, view.oy, view.sc*W, view.sc*H);
  if(ROUTE && ROUTE.segments){
    const segs=ROUTE.segments;
    segs.forEach((seg,i)=>{
      const col=COLORS[i%COLORS.length];
      ctx.strokeStyle=col; ctx.lineWidth=3.0; ctx.lineJoin='round';
      ctx.beginPath();
      const n=seg.x.length;
      for(let k=0;k<n;k++){const [px,py]=sc(seg.x[k],seg.y[k]); k?ctx.lineTo(px,py):ctx.moveTo(px,py);}
      ctx.stroke();
      // a small tick where one recording hands over to the next, so the seams are visible
      // without cluttering the picture with a marker per chunk
      if(i){const [jx,jy]=sc(seg.x[0],seg.y[0]);
        ctx.fillStyle='#fff'; ctx.strokeStyle=col; ctx.lineWidth=2;
        ctx.beginPath(); ctx.arc(jx,jy,5,0,7); ctx.fill(); ctx.stroke();}
    });
    // ONE start and ONE end for the whole route. A marker per chunk made the picture unreadable
    // and, worse, made it look as though the walk had many beginnings.
    const f=segs[0], l=segs[segs.length-1];
    if(f&&f.x.length){
      const [sx,sy]=sc(f.x[0],f.y[0]);
      ctx.fillStyle='#137a3f'; ctx.strokeStyle='#fff'; ctx.lineWidth=3;
      ctx.beginPath(); ctx.arc(sx,sy,11,0,7); ctx.fill(); ctx.stroke();
      ctx.fillStyle='#0d5c2e'; ctx.font='bold 15px -apple-system,sans-serif';
      ctx.fillText('СТАРТ  T12 → J11', sx+18, sy-12);
    }
    if(l&&l.x.length){
      const [ex,ey]=sc(l.x[l.x.length-1],l.y[l.y.length-1]);
      ctx.fillStyle='#c2410c'; ctx.strokeStyle='#fff'; ctx.lineWidth=3;
      ctx.beginPath(); ctx.rect(ex-9,ey-9,18,18); ctx.fill(); ctx.stroke();
      ctx.fillStyle='#9a3412'; ctx.font='bold 14px -apple-system,sans-serif';
      ctx.fillText('КОНЕЦ', ex+16, ey+26);
    }
  }
  // progress box
  ctx.fillStyle='rgba(15,17,23,.82)'; ctx.fillRect(10,10,268,26);
  ctx.fillStyle='#e6e9f2'; ctx.font='13px -apple-system,sans-serif';
  ctx.fillText(ROUTE?(ROUTE.text||''):'ждём данные…', 18, 28);
}
window.addEventListener('resize',()=>{fit();draw();});
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
window.addEventListener('mousemove',e=>{
  if(!drag)return; view.ox=drag.ox+(e.clientX-drag.x); view.oy=drag.oy+(e.clientY-drag.y); draw();
});

function hms(s){s=Math.max(0,Math.round(s));
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
  return (h?h+'ч ':'')+(h||m?String(m).padStart(2,'0')+'м ':'')+String(x).padStart(2,'0')+'с';}

async function tick(){
  try{
    const r=await fetch('/api/p13/state?t='+Date.now());
    const j=await r.json();
    document.getElementById('cnt').textContent=(j.done||0)+' / '+(j.total||0);
    document.getElementById('pb').style.width=(100*(j.done||0)/Math.max(j.total||1,1))+'%';
    let t='прошло '+(j.elapsed_s?hms(j.elapsed_s):'0с');
    if(j.eta_s) t+=' · осталось ~'+hms(j.eta_s);
    document.getElementById('times').textContent=t;
    document.getElementById('cur').innerHTML = j.current
      ? '<span class="pill">▶ '+j.current+'</span>' : '<span class="muted">—</span>';
    document.getElementById('chain').innerHTML = (j.chain&&j.chain.length)
      ? j.chain.slice(-8).map(c=>c.after+' → '+c.next_start_edge+' ('+c.next_start_from+'→…)')
          .join('<br>') : '—';
    const tbody=(j.videos||[]).map(v=>{
      const cls={'готово':'ok','ошибка':'err','в работе':'run'}[v.status]||'wait';
      const st=v.start?(v.start.from+'→'+v.start.to):'';
      const en=v.end?(v.end.from+'→'+v.end.to):'';
      return '<tr><td><b>'+v.video+'</b></td><td class="'+cls+'">'+v.status+'</td>'+
        '<td class="muted">'+(v.seconds?Math.round(v.seconds)+'с':'')+'</td>'+
        '<td class="muted" style="font-size:11px">'+(en||st)+'</td></tr>';
    }).join('');
    document.getElementById('tbl').innerHTML=tbody;
    const e=document.getElementById('err');
    if(j.errors&&j.errors.length){e.style.display='block';
      e.textContent=j.errors.map(x=>x.video+' / '+x.step+': '+x.error).join('\n');}
    else e.style.display='none';

    if(j.route&&j.route.segments){
      ROUTE=j.route;
      draw();
    }
  }catch(err){ /* keep the last picture */ }
}
setInterval(tick,2000); tick();
setInterval(()=>{ if(!plan){img.src='/api/p13/plan?t='+Date.now();} },5000);
</script></body></html>"""


def recorded(video: str) -> bool:
    """Are the recordings for this chunk on disk, and newer than its source?

    The dashboard used to show whatever the queue processes had last written to state.json, but
    two queues run at once and both write that file, so each was overwriting the other's progress
    and the total flickered between seven and eight. What is on disk cannot flicker: a chunk counts
    as recorded when its visual drive and its derived route exist and postdate the file they were
    made from. The queue's own state is still used, but only for one thing it alone knows — which
    chunk is being worked on right now.
    """
    src = ROOT / f"data/p01r/{video}.AVI"
    if not src.exists():
        return False
    for p in (ROOT / f"output/p07/yaw_signal_{video}.csv",
              ROOT / f"output/p07/forward_signal_{video}.csv",
              ROOT / f"output/p071/trajectory_{video}.csv"):
        if not p.exists() or p.stat().st_mtime < src.stat().st_mtime:
            return False
    return True


ANCHOR = OUT / "CHAIN_ANCHOR.json"


def read_state() -> dict:
    """Merged view: the anchor owns the order, the chain owns the seams, the disk owns progress.

    The chain writes its own state file only once it has finished a chunk, so at the start of a
    long run there is nothing to read and the dashboard would fall back to a queue's view — seven
    chunks instead of sixteen, and no starting edge. The anchor is written before anything runs and
    carries the order and the first start, so it is read first and the chain's file only refines it.
    """
    c = {}
    try:
        c = json.loads((OUT / "chain_state.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    a = {}
    try:
        a = json.loads(ANCHOR.read_text(encoding="utf-8"))
    except Exception:
        pass
    q = {}
    try:
        q = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        pass

    order = c.get("order") or a.get("order") or []
    # The anchor's start is where the walk began and never changes; the chain's is where the last
    # finished chunk handed over, which after the final chunk is its ending. Showing the second as
    # "the start" made the finished picture claim the walk began at its own end.
    start = a.get("start") or c.get("start") or {}
    handover = c.get("start") or {}
    if not order:
        out = dict(q)
        out.setdefault("videos", [])
        out.setdefault("chain", [])
        out.setdefault("errors", [])
        return out

    done_map = c.get("done") or {}
    current = q.get("current") or c.get("current")
    vids = []
    for v in order:
        d = done_map.get(v)
        is_done = bool(d) or recorded(v)
        vids.append({
            "video": v,
            "status": "готово" if is_done else ("в работе" if v == current else "ожидает"),
            "seconds": 0, "start": None,
            "end": ({"from": d["end_from"], "to": d["end_to"]} if d else None),
        })

    errors = [{"video": "цепочка", "step": "", "error": e} for e in c.get("errors", [])]
    for e in q.get("errors", []):
        errors.append(e)
    return {
        "videos": vids,
        "total": len(order),
        "done": sum(1 for v in vids if v["status"] == "готово"),
        "current": current,
        "chain": [{"after": h["after"], "next_start_edge": h["next_start_edge"],
                   "next_start_from": h["next_start_from"]} for h in c.get("handovers", [])],
        "chain_start": start,
        "current_handover": handover,
        "chain_order": order,
        "errors": errors,
        "updated_at": c.get("updated_at") or a.get("anchor_written", ""),
    }


def build_route(state: dict) -> dict:
    """Group the accumulated route by chunk, so each chunk can be drawn in its own colour."""
    segs = []
    if not LIVE.exists():
        return {"segments": [], "text": "маршрут пуст"}
    cur, buf = None, None
    with LIVE.open() as fh:
        for row in csv.DictReader(fh):
            v = row["video"]
            if v != cur:
                if buf and buf["x"]:
                    segs.append(buf)
                cur, buf = v, {"video": v, "x": [], "y": []}
            try:
                buf["x"].append(float(row["x_px"]))
                buf["y"].append(float(row["y_px"]))
            except ValueError:
                continue
    if buf and buf["x"]:
        segs.append(buf)
    # thin each segment so the payload stays small
    for s in segs:
        step = max(1, len(s["x"]) // 1200)
        s["x"], s["y"] = s["x"][::step], s["y"][::step]
    n = sum(len(s["x"]) for s in segs)
    txt = f"кусков нарисовано {len(segs)}, точек {n}"
    return {"segments": segs, "text": txt}


class H(BaseHTTPRequestHandler):
    server_version = "p13live/1.0"

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj, ensure_ascii=False, default=str).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        p = self.path.split("?", 1)[0]
        if p in ("/p13", "/p13.html", "/"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if p == "/api/p13/state":
            st = read_state()
            st["route"] = build_route(st)
            return self._json(st)
        if p == "/api/p13/plan":
            if PLAN.exists():
                return self._send(PLAN.read_bytes(), "image/png")
            return self._send(b"no plan", "text/plain", 404)
        if p == "/api/p13/video":
            st = read_state()
            cur = st.get("current")
            if cur:
                f = ROOT / f"webapp/media/{cur}_fixed.mp4"
                if f.exists():
                    return self._send(f.read_bytes(), "video/mp4")
            return self._send(b"no video", "text/plain", 404)
        if re.fullmatch(r"/[A-Za-z0-9_.\-]+", p):
            f = (ROOT / p.lstrip("/")).resolve()
            if str(f).startswith(str(ROOT)) and f.is_file():
                return self._send(f.read_bytes(), "application/octet-stream")
        return self._send(b"not found", "text/plain", 404)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"P13 live  http://{a.host}:{a.port}/p13")
    print(f"  состояние: {STATE}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
