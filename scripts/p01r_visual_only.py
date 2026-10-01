#!/usr/bin/env python3
"""Fast pre-brain diagnostic CSV for P0.1R (no MaleCNS)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.p01r_calibrate import evaluate_visual, run_visual_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        nargs="?",
        default="/Users/artem/Downloads/VID00001_51-70.mp4",
    )
    parser.add_argument("-o", "--output", default="output/p01r_VID00001_51-70")
    parser.add_argument("--crop", choices=["full_frame", "center_band", "both"], default="both")
    parser.add_argument("--ema-tau", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.output)
    video = Path(args.video)
    modes = ["full_frame", "center_band"] if args.crop == "both" else [args.crop]

    for mode in modes:
        rows = run_visual_only(video, mode, out / f"visual_{mode}.csv", args.ema_tau)
        ev = evaluate_visual(rows)
        print(
            f"{mode}: window={ev.get('turn_window')}  "
            f"yaw_ratio={ev.get('yaw_ratio', 0):.2f}  "
            f"sign={ev.get('sign_ok')}  "
            f"coh={ev.get('coherence_turn_mean', 0):.2f}  "
            f"oracle={ev.get('oracle_turn_mean', 0):+.5f}  "
            f"PASS={ev.get('pass')}"
        )
    print(f"Wrote {out}/visual_*.csv")


if __name__ == "__main__":
    main()
