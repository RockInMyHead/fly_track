#!/usr/bin/env python3
"""P13.1B — the blind STOP/MOVE review page, and the matrix it produces.

The reviewer sees one clip and three buttons: the person is walking, or standing, or it cannot be
told. Nothing else is on the page — not the gate's class, not the motion value, not the tracker's
speed, not even which recording the clip came from. The server does not send those fields, so they
cannot leak through the page source, and the naming is random.

Why the blindness is worth the trouble: the point of this pass is to find out whether the quantity
the gate is built on actually tracks the thing it is supposed to track. If the reviewer could see the
classification, the answer would drift toward agreeing with it, and the one number that matters —
how often the gate freezes real walking — would come out too low.

Scoring, once the answers are in:

    gate STOP, person standing     the gate was right to freeze
    gate STOP, person walking      FALSE STOP: real path was thrown away. The number that decides.
    gate MOVE, person walking      the gate was right to allow movement
    gate MOVE, person standing     the gate missed a stop, which costs a little invented path
    unclear, either way            reported separately and left out of the four counts

Usage:
    PYTHONPATH=. .venv/bin/python scripts/p13_1b_server.py --port 8790
    PYTHONPATH=. .venv/bin/python scripts/p13_1b_server.py --score
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
DATA = ROOT / "data/p13"

# Defaults for the first review round. Overridable so the same page serves later rounds without
# being copied: a second copy of this file would be a second implementation of blindness, and
# blindness is the one thing that must not drift between rounds.
CLIPS = OUT / "p13_1b_clips"
MANIFEST = OUT / "p13_1b_manifest.json"
KEY = DATA / "p13_1b_key.json"
ANSWERS = OUT / "p13_1b_answers.json"


def use_round(prefix: str) -> None:
    """Point the page at another review round's files: clips, manifest, key, answers."""
    global CLIPS, MANIFEST, KEY, ANSWERS
    CLIPS = OUT / f"{prefix}_clips"
    MANIFEST = OUT / f"{prefix}_manifest.json"
    KEY = DATA / f"{prefix}_key.json"
    ANSWERS = OUT / f"{prefix}_answers.json"

PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Проверка клипов</title>
<style>
 html,body{margin:0;height:100%;background:#0d0f14;color:#e6e9f2;
   font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 #box{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;
   gap:16px;padding:0 16px;box-sizing:border-box}
 video{max-width:min(1100px,92vw);max-height:56vh;background:#000;border-radius:8px}
 .q{font-size:20px;font-weight:600;text-align:center}
 .q b{color:#ffd60a}
 #wrap{width:min(1100px,92vw)}
 #bar{position:relative;height:10px;background:#232838;border-radius:5px;margin-top:8px}
 #fill{position:absolute;left:0;top:0;bottom:0;background:#3d7dd8;border-radius:5px;width:0}
 #tick{position:absolute;top:-6px;bottom:-6px;width:3px;background:#ffd60a;
   border-radius:2px;transform:translateX(-1px)}
 #barLbl{display:flex;justify-content:space-between;font-size:12.5px;color:#8b93a7;margin-top:5px}
 .btns{display:flex;gap:14px;flex-wrap:wrap;justify-content:center}
 button{font-size:18px;padding:14px 26px;border-radius:10px;border:1px solid #2c3346;
   background:#1b2030;color:#e6e9f2;cursor:pointer}
 button:hover{background:#242c40}
 #walks{border-color:#2f6f45} #stands{border-color:#7a4a20} #unclear{border-color:#3a4256}
 #again{font-size:14px;padding:8px 16px;color:#9aa5bd}
 .top{position:fixed;top:0;left:0;right:0;height:5px;background:#1b2030}
 .top>i{display:block;height:100%;background:#3d7dd8;width:0;transition:width .2s}
 #meta{position:fixed;bottom:10px;width:100%;text-align:center;color:#8b93a7;font-size:13px}
 kbd{background:#1b2030;border:1px solid #2c3346;border-radius:5px;padding:1px 6px;font-size:12.5px}
 #done{display:none;text-align:center;font-size:18px;line-height:2}
</style></head><body>
<div class="top"><i id="pb"></i></div>
<div id="box">
  <div class="q">В <b>отмеченный момент</b> человек идёт или стоит?</div>
  <video id="v" loop muted playsinline></video>
  <div id="wrap">
    <div id="bar"><div id="fill"></div><div id="tick"></div></div>
    <div id="barLbl"><span>0 с</span><span id="mid">отмеченный момент — середина клипа</span>
      <span id="durLbl">5 с</span></div>
  </div>
  <div class="btns">
    <button id="walks"  onclick="ans('WALKS')">ИДЁТ</button>
    <button id="stands" onclick="ans('STANDS')">СТОИТ</button>
    <button id="unclear" onclick="ans('UNCLEAR')">НЕЯСНО</button>
    <button id="again" onclick="toMoment()">ещё раз с момента</button>
  </div>
  <div id="done">
    <div>Готово. Спасибо.</div>
    <div id="summary" style="color:#8b93a7"></div>
  </div>
</div>
<div id="meta"><kbd>1</kbd> идёт · <kbd>2</kbd> стоит · <kbd>3</kbd> неясно ·
  <kbd>пробел</kbd> повтор · <kbd>←</kbd> назад · <span id="cnt"></span></div>
<script>
let CLIPS=[], i=0, MOMENT=2.5, DUR=5, marked=false;
const v=document.getElementById('v');
const fill=document.getElementById('fill'), tick=document.getElementById('tick');

function toMoment(){
  // Play again from a little before the marked moment, so the reviewer sees it in context.
  v.currentTime=Math.max(0, MOMENT-1.2); marked=false; v.play().catch(()=>{});
  document.getElementById('mid').style.color='#8b93a7';
}
function show(){
  if(i>=CLIPS.length){
    document.querySelector('.q').style.display='none';
    v.style.display='none'; document.querySelector('.btns').style.display='none';
    document.getElementById('wrap').style.display='none';
    document.getElementById('done').style.display='block';
    fetch('/api/done').then(r=>r.json()).then(j=>{
      document.getElementById('summary').textContent=(j.n||0)+' размечено';
    });
    return;
  }
  const c=CLIPS[i];
  MOMENT = (c.moment_in_clip!=null)? c.moment_in_clip : 2.5;
  DUR = (c.clip_s!=null)? c.clip_s : 5;
  DUR = 5;
  tick.style.left = (100*MOMENT/DUR)+'%';
  document.getElementById('durLbl').textContent = DUR.toFixed(1)+' с';
  document.getElementById('mid').textContent =
    'отмеченный момент: ' + MOMENT.toFixed(1) + ' с';
  resetMark();
  v.src='/clip/'+c.file;
  v.load();
  v.play().catch(()=>{});
  document.getElementById('pb').style.width=(100*i/CLIPS.length)+'%';
  document.getElementById('cnt').textContent='клип '+(i+1)+' из '+CLIPS.length;
}
// The clip is always the same length and the moment is always the middle, so the page can stop
// there exactly. Nothing is drawn on the video: a mark appearing at the instant being judged could
// be read as a cue about the answer.
//
// Two bugs were found here by testing, and both are worth naming.
//
// The first: `timeupdate` fires about four times a second, so waiting for it to notice the crossing
// left the pause up to five frames past the moment. Hence the seek to MOMENT after pausing, so the
// frame on screen is the one being judged.
//
// The second, and the reason for the `armed` flag: after `src` changes the element keeps reporting
// the *previous* clip's time until the new data arrives. The check therefore fired on a stale value
// and paused at 0.03 seconds while marking the moment as reached. Widening the accepted range does
// not fix that, because a stale value can sit anywhere. Arming on having seen a time *below* the
// moment does: only a clip genuinely playing from its start can satisfy it, and no leftover count
// can. The flag is cleared when a new clip's metadata arrives.
let armed=false;
function resetMark(){
  marked=false; armed=false;
  document.getElementById('mid').style.color='#8b93a7';
  document.getElementById('fill').style.width='0';
}
v.addEventListener('loadedmetadata',()=>{ try{ v.currentTime=0; }catch(e){} resetMark(); });
v.addEventListener('loadeddata',resetMark);
v.addEventListener('timeupdate',()=>{
  if(DUR) fill.style.width=(100*Math.min(v.currentTime,DUR)/DUR)+'%';
  if(marked) return;
  if(v.currentTime < MOMENT){ armed=true; return; }
  if(armed){                                   // crossed the moment while genuinely playing
    marked=true;
    v.pause();
    try { v.currentTime = MOMENT; } catch(e) {}
    document.getElementById('mid').style.color='#ffd60a';
  }
});
async function ans(a){
  if(i>=CLIPS.length) return;
  await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:CLIPS[i].id,answer:a})});
  i++; show();
}
document.addEventListener('keydown',e=>{
  if(e.key==='1') ans('WALKS');
  else if(e.key==='2') ans('STANDS');
  else if(e.key==='3') ans('UNCLEAR');
  else if(e.code==='Space'){ e.preventDefault(); toMoment(); }
  else if(e.key==='ArrowLeft'&&i>0){ i--; show(); }
});
(async function(){
  const r=await fetch('/api/clips'); const j=await r.json();
  CLIPS=j.clips||[]; DUR=j.clip_s||5;
  show();
})();
</script></body></html>"""


def clips_payload() -> dict:
    """Only what the page needs to play and mark: no class, no recording, no motion value.

    The moment inside the clip is sent because the page has to pause there, and it is the same for
    every clip — the middle of five seconds — so it carries nothing about the answer. The run length
    and the margin, which would, stay in the key.
    """
    if not MANIFEST.exists():
        return {"clips": [], "clip_s": 5.0}
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "clips": [{"id": c["id"], "file": c["file"],
                   "moment_in_clip": c.get("moment_in_clip")} for c in m["clips"]],
        "clip_s": m.get("clip_s", 5.0),
    }


def score() -> dict:
    key = {k["id"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))["items"]}
    ans = {}
    if ANSWERS.exists():
        ans = json.loads(ANSWERS.read_text(encoding="utf-8")).get("answers", {})
    cells = {("STOP", "STANDS"): 0, ("STOP", "WALKS"): 0,
             ("MOVE", "WALKS"): 0, ("MOVE", "STANDS"): 0,
             ("STOP", "UNCLEAR"): 0, ("MOVE", "UNCLEAR"): 0}
    margins = {"FALSE_STOP": [], "TRUE_STOP": [], "TRUE_MOVE": [], "FALSE_MOVE": []}
    for cid, k in key.items():
        a = ans.get(cid)
        if not a:
            continue
        cells[(k["gate"], a)] = cells.get((k["gate"], a), 0) + 1
        if k["gate"] == "STOP":
            margins["TRUE_STOP" if a == "STANDS" else "FALSE_STOP"].append(k["margin"]) \
                if a in ("STANDS", "WALKS") else None
        else:
            margins["TRUE_MOVE" if a == "WALKS" else "FALSE_MOVE"].append(k["margin"]) \
                if a in ("WALKS", "STANDS") else None

    n_stop = cells[("STOP", "STANDS")] + cells[("STOP", "WALKS")]
    n_move = cells[("MOVE", "WALKS")] + cells[("MOVE", "STANDS")]
    return {
        "n_answered": len(ans), "n_total": len(key),
        "matrix": {f"{g}|{a}": v for (g, a), v in cells.items()},
        "gate_STOP_person_stands": cells[("STOP", "STANDS")],
        "gate_STOP_person_walks_FALSE_STOP": cells[("STOP", "WALKS")],
        "gate_MOVE_person_walks": cells[("MOVE", "WALKS")],
        "gate_MOVE_person_stands": cells[("MOVE", "STANDS")],
        "unclear": cells[("STOP", "UNCLEAR")] + cells[("MOVE", "UNCLEAR")],
        "false_stop_rate": (cells[("STOP", "WALKS")] / n_stop) if n_stop else None,
        "false_move_rate": (cells[("MOVE", "STANDS")] / n_move) if n_move else None,
        "margin_medians": {k: (sorted(v)[len(v) // 2] if v else None) for k, v in margins.items()},
        "margin_note": (
            "margin — насколько окно сидит внутри или снаружи порога: отрицательное значит "
            "внутри STOP. Если ложные STOP липнут к нулю, ошибка пороговая, а не смысловая"
        ),
    }


class H(BaseHTTPRequestHandler):
    server_version = "p13_1b/1.0"

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
        if p in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if p == "/api/clips":
            return self._json(clips_payload())
        if p == "/api/done":
            s = score()
            return self._json({"n": s["n_answered"]})
        if p.startswith("/clip/"):
            name = Path(p).name
            # One letter and three digits, which is what both rounds name their clips: `w001` in
            # P13.1B and `c001` in P13.1C. It was pinned to `w` at first, and the second round's
            # clips were refused as malformed — a hardcoded prefix is a second place where the two
            # rounds can drift apart.
            if not re.fullmatch(r"[a-z]\d{3}\.mp4", name):
                return self._send(b"bad", "text/plain", 400)
            f = CLIPS / name
            if not f.exists():
                return self._send(b"no clip", "text/plain", 404)
            return self._send(f.read_bytes(), "video/mp4")
        return self._send(b"not found", "text/plain", 404)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:
            body = {}
        if self.path.split("?", 1)[0] == "/api/answer":
            cid, a = str(body.get("id", "")), str(body.get("answer", ""))
            if a not in ("WALKS", "STANDS", "UNCLEAR"):
                return self._json({"ok": False, "error": "unknown answer"})
            doc = {"answers": {}}
            if ANSWERS.exists():
                try:
                    doc = json.loads(ANSWERS.read_text(encoding="utf-8"))
                except Exception:
                    pass
            doc.setdefault("answers", {})[cid] = a
            ANSWERS.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            return self._json({"ok": True, "n": len(doc["answers"])})
        return self._json({"ok": False}, 404)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--round", default=None,
                    help="префикс раунда разметки, например p13_1c; по умолчанию p13_1b")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    if a.round:
        use_round(a.round)

    if a.score:
        s = score()
        print("=" * 84)
        print("P13.1B — ИТОГ СЛЕПОЙ ПРОВЕРКИ")
        print("=" * 84)
        print(f"  размечено {s['n_answered']} из {s['n_total']}")
        print()
        print(f"  gate STOP → человек СТОИТ : {s['gate_STOP_person_stands']}")
        print(f"  gate STOP → человек ИДЁТ  : {s['gate_STOP_person_walks_FALSE_STOP']}"
              + ("   <-- FALSE STOP" if s['gate_STOP_person_walks_FALSE_STOP'] else ""))
        print(f"  gate MOVE → человек ИДЁТ  : {s['gate_MOVE_person_walks']}")
        print(f"  gate MOVE → человек СТОИТ : {s['gate_MOVE_person_stands']}")
        print(f"  неясно                    : {s['unclear']}")
        print()
        if s["false_stop_rate"] is not None:
            print(f"  FALSE STOP: {100*s['false_stop_rate']:.0f}% окон класса STOP "
                  f"оказались настоящей ходьбой")
            print("     это доля настоящего пути, которую гейт выбрасывает")
        if s["false_move_rate"] is not None:
            print(f"  пропущенный STOP: {100*s['false_move_rate']:.0f}% окон класса MOVE "
                  f"оказались стоянием")
        print()
        print(f"  медианы margin: {s['margin_medians']}")
        print(f"  {s['margin_note']}")
        (OUT / "p13_1b_score.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\n  записано: {OUT/'p13_1b_score.json'}")
        return 0

    m = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    print(f"P13.1B  http://{a.host}:{a.port}/")
    print(f"  клипов в манифесте: {len(m.get('clips', []))}")
    print("  на странице нет ни класса, ни признака движения, ни скорости")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
