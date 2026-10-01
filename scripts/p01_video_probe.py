#!/usr/bin/env python3
"""
P0.1R video probe — pre-brain diagnostics + neural rates (no x/y/θ).

Usage:
    PYTHONPATH=. python scripts/p01_video_probe.py
    PYTHONPATH=. python scripts/p01r_calibrate.py   # full calibration + gain sweep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        nargs="?",
        default="/Users/artem/Downloads/VID00001_51-70.mp4",
    )
    parser.add_argument("-o", "--output", default="output/p01r_VID00001_51-70")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "p01r_calibrate.py"),
        args.video,
        "-o",
        args.output,
    ]
    subprocess.run(cmd, cwd=str(ROOT), env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)})


if __name__ == "__main__":
    main()
