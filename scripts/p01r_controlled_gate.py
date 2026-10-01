#!/usr/bin/env python3
"""
P0.1R+ controlled clip gate — ONE frozen run after blind segmentation.

Requires:
  - data/p01r/controlled.mp4 (or path in controlled_segments.json)
  - data/p01r/controlled_segments.json with blind_review_done=true

Usage:
    PYTHONPATH=. python scripts/p01r_controlled_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.optic_flow import FROZEN_YAW_DETECTORS
from fly_vo.p01r_gate import SNR_THRESHOLD, YAW_NEAR_ZERO, evaluate_controlled_clip, load_controlled_segments
from scripts.p01r_calibrate import run_visual_only

EMA_TAU = 0.5  # frozen — do not change after seeing results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default=str(ROOT / "data/p01r/controlled_segments.json"))
    parser.add_argument("-o", "--output", default="output/p01r_controlled")
    args = parser.parse_args()

    seg_doc = load_controlled_segments(Path(args.segments))
    if not seg_doc.get("blind_review_done"):
        print("FAIL: blind_review_done=false in controlled_segments.json")
        print("Run: PYTHONPATH=. python scripts/p01r_blind_review.py <controlled.mp4>")
        print("Review frames, edit segments, set blind_review_done=true, then re-run.")
        raise SystemExit(2)

    video = Path(seg_doc["video_path"])
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        print(f"FAIL: video not found: {video}")
        print("Record controlled clip per data/p01r/controlled_recording_spec.json")
        raise SystemExit(2)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"P0.1R+ controlled gate: {video.name}")
    print(f"Frozen readout: {FROZEN_YAW_DETECTORS}  weights=1/3  tau={EMA_TAU}  crop=center_band")
    print(f"SNR threshold: {SNR_THRESHOLD}  |yaw| near-zero: {YAW_NEAR_ZERO}")

    rows = run_visual_only(video, "center_band", out / "visual_controlled.csv", EMA_TAU)
    result = evaluate_controlled_clip(rows, seg_doc)

    report = {
        "video": str(video),
        "segments": seg_doc["segments"],
        "frozen_readout": [{"dx": dx, "dt": lag, "weight": 1 / 3} for dx, lag in FROZEN_YAW_DETECTORS],
        "ema_tau": EMA_TAU,
        "snr_threshold": SNR_THRESHOLD,
        "eval": result,
    }
    (out / "controlled_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fm = result.get("fwd_medians", {})
    print(f"\nForward medians: FWD_1={fm.get('fwd_1', 0):+.4f}  FWD_2={fm.get('fwd_2', 0):+.4f}  FWD_3={fm.get('fwd_3', 0):+.4f}")
    print(f"FWD spread: {result.get('fwd_spread')}  global_fwd_separated: {result.get('global_fwd_separated')}")

    for key in ("left", "right"):
        ev = result.get(key, {})
        if "reason" in ev:
            print(f"\n{key.upper()}: SKIP — {ev['reason']}")
            continue
        print(f"\n{ev['label']} turn {ev['turn']}:")
        print(
            f"  SNR={ev['snr_robust']:.2f}  sign={ev['sign_ok']}  "
            f"pre≈0={ev['pre_near_zero']}  post≈0={ev['post_near_zero']}  "
            f"fwd_cohesion={ev['fwd_cohesion_ok']}  PASS={ev['pass']}"
        )
        print(f"  yaw: pre={ev['yaw_pre']:+.4f}  turn={ev['yaw_turn']:+.4f}  post={ev['yaw_post']:+.4f}")

    left_pass = result.get("left", {}).get("pass", False)
    right_pass = result.get("right", {}).get("pass", False)
    overall = result.get("pass", False)

    print(f"\n{'PASS' if overall else 'FAIL'} P0.1R+ CONTROLLED GATE")
    print(f"  LEFT:  {'PASS' if left_pass else 'FAIL'}")
    print(f"  RIGHT: {'PASS' if right_pass else 'FAIL'}")

    if left_pass and right_pass:
        print("\n→ P0.1R+ CLOSED. Proceed to P0.2 (descending neuron tracing).")
    elif left_pass and not right_pass:
        print("\n→ Asymmetric signed pathway — compare T4/T5 L/R, retinal mapping, detector polarity.")
    elif not left_pass and not right_pass:
        print("\n→ Both fail on clean clip — revisit frontend (51-70 may have been special case).")

    print(f"Wrote {out}/controlled_gate_report.json")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
