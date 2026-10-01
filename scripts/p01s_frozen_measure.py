#!/usr/bin/env python3
"""
P0.1S — apply the frozen measurement to every labelled event and record the numbers.

Uses `fly_vo/content_motion.py`, the single frozen definition:

    window     exactly [start, end] as declared in the event file
    area       central 30-70% rows
    frames     native video frame rate
    field      Lucas-Kanade, 16 x 8 cells
    per frame  spatial median, spatial mean, spatial mean|dx|, seed coverage
    per window content_dx = median over time of the spatial median
               coherence  = |content_dx| / median over time of spatial mean|dx|
               reliable   = median over time of seed coverage

NO threshold is applied and no event is called pass or fail. The point of this run
is to have one table of numbers that everything later refers back to. In particular
no coherence cut-off is chosen here: a cut-off would have to be picked after seeing
results, so it waits for the blind labels.

The report also records how much the answer moves when the window is shifted, and
how it changes if the mean is used instead of the median — as descriptive columns,
not as a verdict.

Usage:
    PYTHONPATH=. python scripts/p01s_frozen_measure.py
    PYTHONPATH=. python scripts/p01s_frozen_measure.py --all-events
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.content_motion import DEFAULT_FPS, measure_window

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
SCORED = ("left_65", "left_81", "left_108", "right_67", "right_210", "right_216", "fwd_140")
SHIFT_S = 1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-events", action="store_true",
                    help="include MIXED/REJECT events as well")
    ap.add_argument("-o", "--output", default="output/p01s_frozen_measure")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    events = [e for e in doc["events"]
              if args.all_events or e["id"] in SCORED]

    print("P0.1S — frozen measurement")
    print("  content_dx > 0 : content moved RIGHT  ->  camera rotated LEFT")
    print("  content_dx < 0 : content moved LEFT   ->  camera rotated RIGHT")
    print("  no threshold applied anywhere in this table\n")
    print(f"  {'event':10s} {'label':10s} {'t0':>6s} {'t1':>6s} {'N':>4s} "
          f"{'content_dx':>11s} {'mean_dx':>9s} {'coherence':>10s} {'reliable':>9s} "
          f"{'activity':>9s} {'content':>8s} {'camera':>7s}")

    rows = []
    for e in events:
        eid = e["id"]
        t0, t1 = float(e["start"]), float(e["end"])
        m = measure_window(str(VIDEO), t0, t1, eid)

        # sensitivity columns: how much the number moves under a shift of the window,
        # and whether the mean and the median agree. Recorded, not judged.
        t0s = max(0.0, t0 - SHIFT_S)
        t1s = t1 - SHIFT_S
        m_shift = measure_window(str(VIDEO), t0s, t1s, eid + "_shifted")
        agree = np.sign(m.content_dx) == np.sign(m.mean_dx)

        rec = m.as_dict()
        rec.update({
            "old_label": e["label"],
            "shifted_window": [t0s, t1s],
            "content_dx_shifted": m_shift.content_dx,
            "window_shift_changes_sign": bool(
                np.sign(m.content_dx) != np.sign(m_shift.content_dx)
            ),
            "mean_and_median_agree": bool(agree),
            "per_frame_median": m.per_frame_median,
            "per_frame_coverage": m.per_frame_coverage,
        })
        rows.append(rec)
        print(f"  {eid:10s} {e['label']:10s} {t0:6.1f} {t1:6.1f} {m.n_frames:4d} "
              f"{m.content_dx:+11.3f} {m.mean_dx:+9.3f} {m.coherence:10.2f} "
              f"{m.reliable_fraction:9.2f} {m.local_activity:9.3f} "
              f"{m.content_direction:>8s} {m.camera_direction:>7s}")

    print(f"\n  descriptive columns, not verdicts:")
    print(f"  {'event':10s} {'window-shift':>13s} {'mean vs median':>15s} "
          f"{'coherence':>10s} {'reliable':>9s}")
    for r in rows:
        print(f"  {r['event']:10s} "
              f"{'SIGN FLIPS' if r['window_shift_changes_sign'] else 'sign stays':>13s} "
              f"{'agree' if r['mean_and_median_agree'] else 'DISAGREE':>15s} "
              f"{r['coherence']:10.2f} {r['reliable_fraction']:9.2f}")

    # spread of a single event's per-frame medians, which is where the instability lives
    print(f"\n  per-frame spread of content motion inside each window (px/frame):")
    print(f"  {'event':10s} {'p10':>7s} {'p50':>7s} {'p90':>7s} {'iqr':>7s} "
          f"{'frac frames positive':>20s}")
    for r in rows:
        s = np.asarray(r["per_frame_median"])
        if not len(s):
            continue
        print(f"  {r['event']:10s} {np.percentile(s, 10):+7.3f} {np.percentile(s, 50):+7.3f} "
              f"{np.percentile(s, 90):+7.3f} "
              f"{np.percentile(s, 75) - np.percentile(s, 25):7.3f} "
              f"{(s > 0).mean():20.0%}")

    (out / "frozen_measurements.json").write_text(
        json.dumps({
            "definition": {
                "window": "exactly [start, end] from the event file",
                "area": "central 30-70% rows",
                "fps": DEFAULT_FPS,
                "field": "Lucas-Kanade, 16x8",
                "content_dx": "median over time of the spatial median of cell dx",
                "coherence": "|content_dx| / median over time of spatial mean|dx|",
                "reliable_fraction": "median over time of seed coverage",
                "sign": "content_dx < 0 -> camera rotated RIGHT",
                "threshold": "NONE — deliberately, until the blind labels exist",
            },
            "events": rows,
        }, indent=2), encoding="utf-8")

    with (out / "frozen_measurements.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "old_label", "t0", "t1", "n_frames", "content_dx", "mean_dx",
                    "coherence", "reliable_fraction", "local_activity",
                    "content_direction", "camera_direction_if_yaw",
                    "content_dx_shifted", "window_shift_changes_sign",
                    "mean_and_median_agree"])
        for r in rows:
            w.writerow([r["event"], r["old_label"], r["t0"], r["t1"], r["n_frames"],
                        f"{r['content_dx']:.5f}", f"{r['mean_dx']:.5f}",
                        f"{r['coherence']:.4f}", f"{r['reliable_fraction']:.4f}",
                        f"{r['local_activity']:.5f}", r["content_direction"],
                        r["camera_direction_if_yaw"],
                        f"{r['content_dx_shifted']:.5f}",
                        r["window_shift_changes_sign"], r["mean_and_median_agree"]])

    print(f"\n  old '3/6' is retired: it mixed several windows, several aggregators and "
          f"at least one wrong label, so it never measured one quantity.")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
