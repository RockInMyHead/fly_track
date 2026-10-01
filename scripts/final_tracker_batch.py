#!/usr/bin/env python3
"""FINAL TRACKER V1 — trajectory.png (and CSVs) for every clip with a P13 start.

Skips clips missing brain, yaw, or run_start.json. Does not tune parameters.

    PYTHONPATH=. .venv/bin/python scripts/final_tracker_batch.py
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_batch.py --only VID00007 VID00011
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/final_tracker"
RUNS = ROOT / "output/p13/runs"
PY = sys.executable


def discover() -> list[str]:
    if not RUNS.exists():
        return []
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        if (d / "run_start.json").exists():
            out.append(d.name)
    return out


def start_of(clip: str) -> tuple[str, str] | None:
    p = RUNS / clip / "run_start.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("edge") or not d.get("from"):
        return None
    return d["edge"], d["from"]


def ready(clip: str) -> list[str]:
    miss = []
    if not (ROOT / f"output/p10/brain_{clip}.npz").exists():
        miss.append("brain")
    if not (ROOT / f"output/p07/yaw_signal_{clip}.csv").exists():
        miss.append("yaw")
    return miss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="ограничить список")
    ap.add_argument("--skip-existing", action="store_true",
                    help="не перезаписывать, если trajectory.png уже есть")
    a = ap.parse_args()

    clips = a.only if a.only else discover()
    print("=" * 100)
    print("FINAL TRACKER V1 — ПАКЕТНАЯ ОТРИСОВКА ТРАЕКТОРИЙ")
    print("=" * 100)
    print(f"  кусков в очереди: {len(clips)}")
    print()

    rows = []
    t0 = time.time()
    for clip in clips:
        miss = ready(clip)
        if miss:
            print(f"  {clip}: пропуск — нет {', '.join(miss)}")
            rows.append({"clip": clip, "ok": False, "why": f"нет {miss}"})
            continue
        st = start_of(clip)
        if st is None:
            print(f"  {clip}: пропуск — run_start.json пуст")
            rows.append({"clip": clip, "ok": False, "why": "нет старта"})
            continue
        edge, frm = st
        out_dir = OUT / clip
        if a.skip_existing and (out_dir / "trajectory.png").exists():
            print(f"  {clip}: уже есть trajectory.png — пропуск")
            rows.append({"clip": clip, "ok": True, "skipped": True})
            continue

        print(f"  {clip}: старт {frm} → по {edge} …", flush=True)
        cmd = [PY, "scripts/final_tracker.py", "--video", clip,
               "--graph", "data/p08/graph.json",
               "--start-edge", edge, "--start-from", frm,
               "--out", str(out_dir), "--quiet"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
        rep = out_dir / "report.json"
        png = out_dir / "trajectory.png"
        if r.returncode != 0 or not rep.exists():
            print(f"    ОШИБКА код {r.returncode}")
            tail = (r.stdout or "") + (r.stderr or "")
            if tail:
                print("    " + tail.strip().splitlines()[-1][:120])
            rows.append({"clip": clip, "ok": False, "returncode": r.returncode})
            continue
        d = json.loads(rep.read_text(encoding="utf-8"))
        s = d["stats"]
        print(f"    OK  {s['route_meters']} м, png={'да' if png.exists() else 'нет'}, "
              f"teleport={s['self_checks']['teleports']}")
        rows.append({
            "clip": clip, "ok": True,
            "meters": s["route_meters"],
            "png": png.exists(),
            "start": {"edge": edge, "from": frm},
        })

    ok = sum(1 for r in rows if r.get("ok"))
    doc = {"phase": "FINAL TRACKER V1 batch", "clips": rows,
           "ok": ok, "total": len(rows), "elapsed_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "BATCH.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  готово: {ok}/{len(rows)}  за {doc['elapsed_s']} с")
    print(f"  сводка: {OUT / 'BATCH.json'}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
