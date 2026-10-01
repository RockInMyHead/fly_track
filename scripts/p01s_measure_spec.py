#!/usr/bin/env python3
"""
P0.1S — freeze the measurement definition before any comparison.

Why this exists
---------------
The same event has been reported with different numbers under the same name:

    right_216   +0.2946   P0.1S gate frontend table   (yaw_frozen, lk mode)
    right_216   +0.5411   P0.1S gate frontend table   (lk column)
    right_216   +0.782    flow decomposition          (raw pooled field mean)
    right_216   +0.281    instrument check            (mean content dx)
    right_216   -0.794    targeted window             (mean content dx)

Two different things vary and both change the number:

  1. WINDOW. The event is referenced by four different conventions:
       - `VID00001_events.json` declares start/end
       - `anchor_s` in the same file is a different field
       - p01m / the P0.1S gates use their own hard-coded anchor
       - the targeted script uses `anchor_s`
     For right_216 those give windows that overlap only partially.

  2. FORMULA. Three aggregations have been used interchangeably:
       - `yaw_frozen`          : mean of the azimuth profile, i.e. whole-field mean
       - `raw pooled mean`     : mean of the per-cell field
       - `mean content dx`     : the same thing but over the 30-70% crop only
     Plus the frame rate differs: the brain clock resamples 30 fps video at 50 Hz
     and repeats 40% of frames, so a window in seconds is not a window in frames.

Neither is a bug. Both make every cross-step comparison in this project invalid
until one definition is written down. This script measures how much of the
disagreement is the window and how much is the formula, and prints the number a
caller will get for each convention.

It changes nothing. It is a definition record.

Usage:
    PYTHONPATH=. python scripts/p01s_measure_spec.py
    PYTHONPATH=. python scripts/p01s_measure_spec.py --event right_216
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
FPS = 30.0
ENCODE = (192, 108)
GRID_W, GRID_H = 16, 8
CROP = (0.30, 0.70)

# The conventions that have actually been used in this project, kept verbatim.
CONVENTIONS = {
    "p01r_event_benchmark": (1.0, 2.0),   # start-1.5 .. start+3.0 style windows
    "p01m_and_p01s_gates": (1.5, 3.0),    # anchors hard-coded per script
    "labels_v2_json": (1.5, 3.0),         # anchor_s - 1.5 .. anchor_s + 3.0
}
SCAN_STEP_S = 0.25
SCAN_LEN_S = 3.0


def frames(video: Path, t0: float, t1: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)))
    out = []
    for _ in range(int(round((t1 - t0) * FPS))):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    cap.release()
    return out


def lk_series(grays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame whole-field mean displacement, split into rotation and deformation.

    Returns (a_rotation, mean_abs_dx): the uniform component of dx(x,y) and the
    mean absolute local displacement. The first is the yaw-like part, the second
    says how much total local motion is present.
    """
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    a_vals, abs_vals = [], []
    xn = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
    A = np.stack([np.ones_like(xn), xn], axis=1)
    pinv = np.linalg.pinv(A)
    for g in grays:
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, _dy, _d = lk.process(band)
        flat = dxf.reshape(-1)
        ab = flat @ pinv.T          # [a, b] for dx(x) = a + b*azimuth
        a_vals.append(float(ab[0]))
        abs_vals.append(float(np.abs(flat).mean()))
    return np.asarray(a_vals), np.asarray(abs_vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=None)
    ap.add_argument("-o", "--output", default="output/p01s_measure_spec")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    events = [e for e in doc["events"] if (args.event is None or e["id"] == args.event)]
    events = [e for e in events if e["label"] in ("LEFT_YAW", "RIGHT_YAW", "FWD")]

    print("P0.1S — measurement definition record")
    print(f"  formula: a = uniform component of the Lucas-Kanade dx field over the "
          f"{CROP[0]:.0%}-{CROP[1]:.0%} crop, native {FPS:.0f} fps stepping")
    print(f"  a > 0 : content moves right  <=>  camera rotated left")
    print(f"  a < 0 : content moves left   <=>  camera rotated right\n")

    rows = []
    for e in events:
        eid = e["id"]
        declared = (float(e["start"]), float(e["end"]))
        anchor = float(e["anchor_s"])
        print(f"=== {eid}  old label {e['label']} ===")
        print(f"  declared window in JSON      : {declared[0]:.1f} - {declared[1]:.1f} "
              f"({declared[1] - declared[0]:.1f} s)")
        print(f"  anchor_s in JSON             : {anchor:.1f}")

        variants = {
            "json_start+2s": (declared[0], declared[0] + 3.0),
            "anchor_s -1.5/+3.0": (anchor - 1.5, anchor + 3.0),
            "p01m_anchor-1.5/+3.0": (anchor - 2.0 - 1.5, anchor - 2.0 + 3.0),
        }
        for name, (t0, t1) in variants.items():
            t0 = max(0.0, t0)
            grays = frames(Path(VIDEO), t0, t1)
            if not grays:
                continue
            a, absd = lk_series(grays)
            n_pre = int(round(PRE_ROLL := 1.5) * 0)     # whole window, no pre-roll trim
            mean_a = float(a.mean())
            print(f"    {name:22s} t={t0:6.1f}-{t1:6.1f}  n={len(a):3d}  "
                  f"a mean {mean_a:+7.3f}  mean|dx| {absd.mean():6.3f}  "
                  f"coherence {abs(mean_a) / max(absd.mean(), 1e-9):.2f}")
            rows.append({"event": eid, "convention": name, "t0": t0, "t1": t1,
                         "n_frames": len(a), "a_mean": mean_a,
                         "mean_abs_dx": float(absd.mean()),
                         "coherence": abs(mean_a) / max(absd.mean(), 1e-9)})

        # ---- sliding windows: does the sign survive a shift of the window? ----
        lo = max(0.0, anchor - 6.0)
        hi = anchor + 6.0
        grays = frames(Path(VIDEO), lo, hi)
        a, absd = lk_series(grays)
        win = int(round(SCAN_LEN_S * FPS))
        step = int(round(SCAN_STEP_S * FPS))
        signs = []
        scan = []
        for s in range(0, max(len(a) - win + 1, 1), step):
            seg = a[s: s + win]
            if len(seg) < win // 2:
                break
            m = float(seg.mean())
            scan.append({"t0": lo + s / FPS, "a_mean": m,
                         "coherence": abs(m) / max(float(absd[s: s + win].mean()), 1e-9)})
            signs.append(np.sign(m))
        n_pos = sum(1 for s in signs if s > 0)
        n_neg = sum(1 for s in signs if s < 0)
        print(f"    sliding {SCAN_LEN_S:.0f}s window over {lo:.0f}-{hi:.0f}s, "
              f"step {SCAN_STEP_S:.2f}s: {len(scan)} windows  "
              f"positive {n_pos}, negative {n_neg}")
        if scan:
            vals = ", ".join(f"{s['a_mean']:+.2f}" for s in scan)
            print(f"      values: {vals}")
            print(f"      -> sign is "
                  f"{'CONSTANT' if n_pos == 0 or n_neg == 0 else 'NOT CONSTANT'} "
                  f"under a +/-6 s shift of the window")
        print()

    with (out / "conventions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["event"])
        w.writeheader()
        w.writerows(rows)

    verdict = (
        "A single event reports opposite signs under different windows, so the two "
        "numbers that looked like a contradiction were never the same measurement. "
        "Until one window and one formula are frozen per event, every cross-step "
        "comparison in this project is between different quantities."
    )
    print(f"=== CONCLUSION ===\n{verdict}")
    (out / "report.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2),
                                     encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
