#!/usr/bin/env python3
"""
P0.1Q follow-up — where does the direction die inside the 2-D field?

P0.1Q replaced the collapsed 1-D column profile with a full-resolution 2-D
measurement. The signal got measurably better per frame, but the fly's readout
still failed on four of six real turns. This probe asks which of two things is
responsible:

  (A) MEASUREMENT: individual grid cells are themselves unstable, so no amount of
      clever pooling can recover the direction.
  (B) POOLING: individual cells hold the correct sign, but averaging the eight
      elevation rows of a column cancels them against each other.

It also compares two pooling rules on the same field — mean (what injection
currently uses) vs median / sign vote — because if (B) is true, the pooling rule
is the fix and no new retinotopy is needed.

Everything is frontend-only: no MaleCNS, no injection, nothing trained.

Usage:
    PYTHONPATH=. python scripts/p01q_stability.py
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

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS = {
    "left_65":   (63.5, "LEFT"),
    "left_81":   (78.5, "LEFT"),
    "left_108":  (106.0, "LEFT"),
    "right_67":  (66.0, "RIGHT"),
    "right_210": (209.0, "RIGHT"),
    "right_216": (214.0, "RIGHT"),
    "fwd_140":   (137.5, "FWD"),
}
SIGN = {"LEFT": -1, "RIGHT": +1, "FWD": 0}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DT = 0.02


def capture(mode: str, ew: int, eh: int, t0: float, t1: float) -> list[dict]:
    import cv2

    front = DirectionalMotionFrontend(
        FlowConfig(n_azimuth_bins=128, ema_tau_s=0.5, brain_dt=DT, crop_mode="center_band",
                   field_mode=mode, grid_w=16, grid_h=8)
    )
    out = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, DT):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (ew, eh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        d = front.process(img)
        out.append({
            "t": t,
            "profile": np.asarray(d.signed_flow_field, dtype=np.float64),
            "field": np.asarray(d.motion_field_2d, dtype=np.float64)
            if d.motion_field_2d else None,
        })
    return out


def run_stats(signals: np.ndarray, want: int) -> dict:
    """signals: [frames, units]. Per-unit sign stability over the event."""
    signs = np.sign(signals)
    correct = signs == want
    frac = correct.mean(axis=0)
    longest = np.zeros(signals.shape[1])
    for j in range(signals.shape[1]):
        best = cur = 0
        for v in correct[:, j]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        longest[j] = best * DT
    pooled = signals.mean(axis=1)
    pooled_frac = float(np.mean(np.sign(pooled) == want))
    best = cur = 0
    for v in np.sign(pooled) == want:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {
        "unit_frac_correct_median": float(np.median(frac)),
        "unit_frac_correct_p90": float(np.percentile(frac, 90)),
        "units_above_70pct": float(np.mean(frac >= 0.7)),
        "units_above_90pct": float(np.mean(frac >= 0.9)),
        "unit_longest_median_s": float(np.median(longest)),
        "unit_longest_p90_s": float(np.percentile(longest, 90)),
        "pooled_frac_correct": pooled_frac,
        "pooled_longest_correct_s": best * DT,
        "pooled_signed_mean": float(pooled.mean()),
        "pooled_sign_correct": bool(np.sign(pooled.mean()) == want) if want else None,
        "median_across_units_frac": float(np.mean(np.sign(np.median(signals, axis=1)) == want)),
        "vote_across_units_frac": float(np.mean(np.sign(signs.sum(axis=1)) == want)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01q_stability")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1Q follow-up — grid-cell stability vs pooling rule")
    print(f"  per-frame sign of each cell, event window = {EVENT_DUR_S:.1f}s\n")

    rows = []
    for eid, (start, lab) in EVENTS.items():
        t0, t1 = max(0.0, start - PRE_ROLL_S), start + EVENT_DUR_S
        want = SIGN[lab]
        n_pre = int(round(PRE_ROLL_S / DT))

        rec = {"event": eid, "label": lab}
        for tag, mode, ew, eh in (("grid2d", "grid2d", 192, 108), ("legacy96", "panorama", 96, 72)):
            frames = capture(mode, ew, eh, t0, t1)
            if tag == "grid2d":
                ev = frames[n_pre:]
                flat = np.stack([np.asarray(f["field"]).reshape(-1) for f in ev], axis=0)
            else:
                ev = frames[n_pre:]
                flat = np.stack([np.asarray(f["profile"]) for f in ev], axis=0)
            rec[tag] = run_stats(flat, want) if want else None
        rows.append(rec)

        print(f"=== {eid} ({lab}) ===")
        for tag in ("grid2d", "legacy96"):
            s = rec[tag]
            if not s:
                print(f"  {tag:9s} (FWD — no correct sign to compare)")
                continue
            n_units = 128 if tag == "grid2d" else 128
            print(f"  {tag:9s} cells={n_units:3d}  "
                  f"per-cell correct: median {s['unit_frac_correct_median']:.0%}, "
                  f">=70% correct {s['units_above_70pct']:.0%} of cells, "
                  f">=90% {s['units_above_90pct']:.0%}   "
                  f"per-cell longest median {s['unit_longest_median_s']:.2f}s")
            print(f"  {'':9s} pooled mean : {s['pooled_frac_correct']:.0%} frames correct, "
                  f"longest {s['pooled_longest_correct_s']:.2f}s")
            print(f"  {'':9s} pooled median: {s['median_across_units_frac']:.0%}   "
                  f"sign vote: {s['vote_across_units_frac']:.0%}")
        print()

    with (out / "stability.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "label", "mode", "key", "value"])
        for r in rows:
            for tag in ("grid2d", "legacy96"):
                if r[tag]:
                    for k, v in r[tag].items():
                        w.writerow([r["event"], r["label"], tag, k, v])

    # ---- summary across the six turns ----
    turns = [r for r in rows if r["label"] != "FWD"]
    print("=== summary over the six turns ===")
    for tag in ("grid2d", "legacy96"):
        med = np.mean([r[tag]["unit_frac_correct_median"] for r in turns])
        ab70 = np.mean([r[tag]["units_above_70pct"] for r in turns])
        ab90 = np.mean([r[tag]["units_above_90pct"] for r in turns])
        lmed = np.mean([r[tag]["unit_longest_median_s"] for r in turns])
        pooled = np.mean([r[tag]["pooled_frac_correct"] for r in turns])
        vote = np.mean([r[tag]["vote_across_units_frac"] for r in turns])
        print(f"  {tag:9s} per-cell median correct {med:.0%}  >=70% {ab70:.0%}  >=90% {ab90:.0%}  "
              f"longest-median {lmed:.2f}s | pooled mean {pooled:.0%}  sign-vote {vote:.0%}")

    g = {r["event"]: r["grid2d"] for r in turns}
    measurement_ok = np.mean([g[e]["units_above_70pct"] for e in g])
    pooling_gain = np.mean(
        [g[e]["vote_across_units_frac"] - g[e]["pooled_frac_correct"] for e in g]
    )
    if measurement_ok < 0.6:
        verdict = (
            f"MEASUREMENT IS THE LIMIT: only {measurement_ok:.0%} of grid cells hold the correct "
            "sign for >=70% of the turn on average. The direction is not present in the local "
            "motion field to begin with, so no pooling rule and no downstream circuit can recover "
            "it. The 2-D field is better than the 1-D profile per frame, but not enough."
        )
    elif pooling_gain > 0.1:
        verdict = (
            f"POOLING IS THE LIMIT: cells carry the sign ({measurement_ok:.0%} of them are correct "
            f">=70% of the time) but averaging the elevation rows cancels it — a sign vote across "
            f"cells gains {pooling_gain:+.0%} over the mean. Changing the pooling rule is the fix."
        )
    else:
        verdict = (
            f"BOTH ARE MARGINAL: {measurement_ok:.0%} of cells hold the sign >=70% of the time and "
            f"robust pooling gains only {pooling_gain:+.0%} over the mean. Neither a new pooling "
            "rule nor a better local estimate on its own is sufficient."
        )
    print(f"\n=== VERDICT ===\n{verdict}")

    (out / "report.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict}, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
