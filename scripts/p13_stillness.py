#!/usr/bin/env python3
"""P13 — how much of the reconstructed route was walked while the person stood still.

The question this answers came from watching the video: between fourteen and seventeen minutes of
VID00002 the person is standing, and the plan shows him crossing the building. That is not a
rounding error, it is the tracker inventing a route, and the cause is measurable.

WHAT THE TRACKER WALKS ON
-------------------------
`p08_graph_tracker` advances a hypothesis with

    step = min(pace * relative_speed(t), max_pace) * dt

and `relative_speed` is the forward signal divided by *its own median over the clip*, floored at
zero. Two things follow, and both are defects rather than choices:

    the division is by the clip's own median, so the number says how this second compares with a
    typical second of the same recording, never whether the person is moving at all

    the forward signal carries almost no information about movement. Measured here against the
    actual frame-to-frame change of the video, the correlation is -0.06 over the whole of VID00002,
    and the signal is *higher* during a two-minute standstill than during the walking either side
    of it. P10 had already found this quantity to be below chance at telling MOVE from NO_MOVE
    (AUC 0.318), but that was read as a weak readout, not as the thing the route is paced by.

So the walker moves at roughly half pace at all times. Junctions are reached that were never
visited, decisions are taken there, and the route grows while the camera is still.

WHAT THIS SCRIPT MEASURES
-------------------------
For every chunk: the frame-to-frame change of the video at 2 Hz, the distance the route covers in
the seconds where that change is above a stillness threshold, and the distance it covers in the
seconds where it is below. The threshold is not fitted: standing reads 6 to 7 on this measure and
walking 27 to 40, so the gap is wide enough that the number is a description rather than a choice,
and a second threshold is reported alongside so the effect can be seen to be insensitive to it.

Usage:
    PYTHONPATH=. python scripts/p13_stillness.py
    PYTHONPATH=. python scripts/p13_stillness.py --only VID00002
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
MEDIA = ROOT / "webapp/media"
TMP = Path("/tmp/p13_still")

# the separation is wide on this measure, so the answer does not hinge on where in the gap the line
# is drawn; two values are reported and both are shown
THRESHOLDS = (10.0, 14.0)


def frame_motion(video: Path, fps: int = 2, width: int = 200) -> tuple[np.ndarray, float]:
    """Frame-to-frame absolute difference, at `fps`, for the whole clip."""
    TMP.mkdir(parents=True, exist_ok=True)
    for f in TMP.glob("*.jpg"):
        f.unlink()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"fps={fps},scale={width}:-2", "-q:v", "5",
                    str(TMP / "f_%05d.jpg")], capture_output=True)
    files = sorted(TMP.glob("f_*.jpg"))
    prev, out = None, []
    for f in files:
        g = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY).astype(np.float64)
        out.append(float(np.mean(np.abs(g - prev))) if prev is not None else 0.0)
        prev = g
    for f in TMP.glob("*.jpg"):
        f.unlink()
    return np.asarray(out), 1.0 / fps


def route(video: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = OUT / f"runs/{video}/graph_trajectory.csv"
    if not p.exists():
        return np.zeros(0), np.zeros(0), np.zeros(0)
    t, x, y = [], [], []
    with p.open() as fh:
        for r in csv.DictReader(fh):
            t.append(float(r["t"]))
            x.append(float(r["x_px"]))
            y.append(float(r["y_px"]))
    return np.asarray(t), np.asarray(x), np.asarray(y)


def analyse(video: str) -> dict | None:
    prev = MEDIA / f"{video}_fixed.mp4"
    if not prev.exists():
        return None
    t, x, y = route(video)
    if len(t) < 10:
        return None
    mot, dt = frame_motion(prev)
    mot_t = np.arange(len(mot)) * dt

    d = np.hypot(np.diff(x), np.diff(y))
    mid = (t[:-1] + t[1:]) / 2.0
    m = np.interp(mid, mot_t, mot)
    total = float(d.sum())

    # how well the tracker's own speed signal tracks real motion, as the cause and not the symptom
    sp_path = ROOT / f"output/p071/trajectory_{video}.csv"
    corr = None
    if sp_path.exists():
        sp_t, sp_v = [], []
        with sp_path.open() as fh:
            for r in csv.DictReader(fh):
                sp_t.append(float(r["t"]))
                sp_v.append(float(r["speed"]))
        sp = np.interp(mot_t, sp_t, sp_v)
        if len(sp) > 30 and sp.std() > 1e-9 and mot.std() > 1e-9:
            corr = float(np.corrcoef(mot, sp)[0, 1])

    row = {"video": video, "seconds": float(t[-1]), "route_m": total * 0.049629,
           "speed_vs_motion_corr": corr,
           "median_motion": float(np.median(mot)),
           "motion_p25": float(np.percentile(mot, 25)),
           "motion_p75": float(np.percentile(mot, 75))}
    for th in THRESHOLDS:
        moving = m > th
        tag = str(th).replace(".0", "")
        row[f"still_m_{tag}"] = float((d * (~moving)).sum()) * 0.049629
        row[f"still_frac_{tag}"] = float((d * (~moving)).sum() / total) if total else 0.0
        row[f"still_time_{tag}"] = float((~moving).mean())
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    order = json.loads((OUT / "CHAIN_ANCHOR.json").read_text(encoding="utf-8"))["order"]
    todo = [a.only] if a.only else order

    print("=" * 100)
    print("P13 — сколько маршрута пройдено, пока человек стоял")
    print("=" * 100)
    print(f"  пороги покоя: {THRESHOLDS} (покой читается как 6–7, ходьба 27–40)")
    print()
    print(f"  {'кусок':<10}{'маршрут, м':>11}{'стоя, м':>9}{'стоя, %':>9}"
          f"{'время стоя':>11}{'корр. скор/движ':>17}")
    rows = []
    for v in todo:
        r = analyse(v)
        if r is None:
            print(f"  {v:<10}{'нет видео или маршрута':>47}")
            continue
        rows.append(r)
        c = r["speed_vs_motion_corr"]
        print(f"  {v:<10}{r['route_m']:>11.0f}{r['still_m_10']:>9.0f}"
              f"{100*r['still_frac_10']:>8.0f}%{100*r['still_time_10']:>10.0f}%"
              f"{(f'{c:+.2f}' if c is not None else '—'):>17}")
    print()
    if rows:
        tot = sum(r["route_m"] for r in rows)
        still = sum(r["still_m_10"] for r in rows)
        corrs = [r["speed_vs_motion_corr"] for r in rows if r["speed_vs_motion_corr"] is not None]
        print(f"  ИТОГО по {len(rows)} кускам: маршрут {tot:.0f} м, из них стоя {still:.0f} м "
              f"({100*still/max(tot,1):.0f}%)")
        if corrs:
            print(f"  корреляция скорости трекера с движением: медиана {np.median(corrs):+.2f} "
                  f"(0 = никакой связи)")
        print()
        print("  Это не погрешность округления. Маршрут растёт, пока камера стоит, потому что")
        print("  трекер двигается по сигналу, не связанному с движением. На развилках, куда он")
        print("  так попадает, решения принимаются о местах, где человек не был.")
    (Path(a.out) / "stillness_report.json").write_text(
        json.dumps({"thresholds": list(THRESHOLDS), "chunks": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  записано: {Path(a.out)/'stillness_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
