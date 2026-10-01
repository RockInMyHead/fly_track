#!/usr/bin/env python3
"""Run every camera chunk with FINAL TRACKER V2, as one walk.

VID00002 starts where the person said the walk starts. Each later chunk starts
at the point the previous chunk's line actually ended: same edge, same
direction, same distance along that edge. A chunk that the tracker refuses is
skipped and the next one still continues from the last point that was drawn.

    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v2_batch.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "output/p13/CHAIN_ANCHOR.json"
OUT = ROOT / "output/final_tracker_v2"
PY = sys.executable


def main() -> int:
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    order = list(anchor["order"])
    start = {
        "edge": anchor["start"]["edge"],
        "from": anchor["start"]["from"],
        "progress_m": 0.0,
    }
    print("FINAL TRACKER V2 — все ролики одной цепочкой")
    print(f"  порядок: {' → '.join(order)}")
    print(f"  старт: {start['from']} по {start['edge']}")
    print()

    seams = []
    t0 = time.time()
    for clip in order:
        out_dir = OUT / clip
        cmd = [
            PY, "scripts/final_tracker_v2.py",
            "--video", clip,
            "--start-edge", start["edge"],
            "--start-from", start["from"],
            "--start-progress", str(start["progress_m"]),
            "--out", str(out_dir),
        ]
        print(f"  {clip}: от {start['from']} по {start['edge']}, "
              f"{start['progress_m']:.2f} м", flush=True)
        t_clip = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        wall_s = time.time() - t_clip
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip().splitlines()
            why = err[-1] if err else f"код {r.returncode}"
            print(f"    пропуск за {wall_s:.1f} с: {why}")
            seams.append({"clip": clip, "ok": False, "why": why, "wall_s": round(wall_s, 2),
                          "started_from": dict(start)})
            continue
        rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        end = rep["end"]
        video_s = float(rep.get("stats", {}).get("samples") or 0)
        rate = video_s / wall_s if wall_s > 0 else 0.0
        print(f"    конец {end['from']} → {end['to']} по {end['edge']}, "
              f"{end['progress_m']:.2f} м; {video_s:.0f} с видео за {wall_s:.1f} с "
              f"({rate:.0f}× реального времени)")
        seams.append({
            "clip": clip, "ok": True,
            "started_from": dict(start),
            "ended_at": end,
            "video_s": round(video_s, 1),
            "wall_s": round(wall_s, 2),
            "realtime_x": round(rate, 1),
        })
        start = {"edge": end["edge"], "from": end["from"], "progress_m": end["progress_m"]}

    doc = {
        "phase": "FINAL TRACKER V2 chain",
        "order": order,
        "anchor": anchor["start"],
        "seams": seams,
        "elapsed_s": round(time.time() - t0, 1),
    }
    ok_rows = [s for s in seams if s["ok"]]
    video_s = sum(s["video_s"] for s in ok_rows)
    wall_ok = sum(s["wall_s"] for s in ok_rows)
    rate = video_s / wall_ok if wall_ok else 0.0
    doc["video_s"] = round(video_s, 1)
    doc["wall_ok_s"] = round(wall_ok, 2)
    doc["realtime_x"] = round(rate, 1)
    (OUT / "chain.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  готово {len(ok_rows)}/{len(order)} за {doc['elapsed_s']:.0f} с")
    print(f"  видео {video_s:.0f} с обработано за {wall_ok:.1f} с — "
          f"{rate:.0f}× реального времени, {wall_ok / max(len(ok_rows), 1):.1f} с на ролик")
    print(f"  стыки: {OUT / 'chain.json'}")
    return 0 if ok_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
