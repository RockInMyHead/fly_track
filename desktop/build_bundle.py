#!/usr/bin/env python3
"""Assemble the app/ folder of an installed Fly Track from this checkout.

    python desktop/build_bundle.py --snapshot          refresh desktop/seed from data/ and output/
    python desktop/build_bundle.py --dest DIR          code + seed -> DIR (an empty app/ folder)
        [--malecns DIR]                                copy the brain files from DIR
        [--download-malecns]                           or fetch them (flybrain release, sha256-checked)

data/ and output/ are not in git; desktop/seed holds the few shared files the chain
needs on a machine that has nothing but a new camera video.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "desktop" / "seed"

SEED_FILES = [
    "data/p08/graph.json",
    "data/p08/plan.png",
    "data/final_tracker/FROZEN_V1.json",
    "data/final_tracker_v3/FROZEN_STAND.json",
    "data/p01r/t4_azimuth.npz",
    "data/p10/FROZEN_P10.json",
    "output/p053_threshold/targets.csv",
    "output/p05_layer_trace/groups.json",
    "output/p051_signal_break/group_metrics.csv",
    "output/p061_synthetic/synthetic_seed64.npz",
    "output/p061_synthetic/synthetic_seed65.npz",
]
CODE_DIRS = {
    "fly_vo": ("*.py",),
    "scripts": ("*.py",),
    "webapp": ("*.py", "*.html", "*.js", "*.css"),
    "desktop": ("*.py",),
    "desktop/assets": ("*",),
}
EMPTY_DIRS = ["data/p01r", "data/app", "webapp/media", "output/final_tracker/cache"]
MALECNS = ("brain.npz", "weights.npz")


def snapshot() -> None:
    for rel in SEED_FILES:
        src = ROOT / rel
        if not src.exists():
            sys.exit(f"нет {rel}")
        dst = SEED / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  seed {rel}")


def build(dest: Path, malecns: Path | None, download: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for d, patterns in CODE_DIRS.items():
        src = ROOT / d
        if not src.exists():
            continue
        for pat in patterns:
            for f in src.glob(pat):
                if f.is_file() and "__pycache__" not in f.parts:
                    out = dest / d / f.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out)
    for rel in SEED_FILES:
        src = SEED / rel
        if not src.exists():
            sys.exit(f"в desktop/seed нет {rel}: сначала --snapshot")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    for rel in EMPTY_DIRS:
        (dest / rel).mkdir(parents=True, exist_ok=True)
    brain = dest / "data" / "malecns"
    brain.mkdir(parents=True, exist_ok=True)
    if malecns:
        for n in MALECNS:
            shutil.copy2(malecns / n, brain / n)
    elif download:
        from flybrain import data as fbdata
        fbdata.download(brain)
    missing = [n for n in MALECNS if not (brain / n).exists()]
    print(f"собрано {dest}" + (f"  (нет мозга: {', '.join(missing)})" if missing else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--dest")
    ap.add_argument("--malecns")
    ap.add_argument("--download-malecns", action="store_true")
    a = ap.parse_args()
    if a.snapshot:
        snapshot()
    if a.dest:
        build(Path(a.dest), Path(a.malecns) if a.malecns else None, a.download_malecns)


if __name__ == "__main__":
    main()
