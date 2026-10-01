#!/usr/bin/env python3
"""
P07 step 3 — find the cells that signal forward motion.

The problem with using DNg100 by assumption
-------------------------------------------
It is the obvious candidate for forward, but "obvious" is what got the earlier yaw work
into trouble: a cell that separates real windows may be reading anything that happens to
differ between them. So forward cells are chosen the same way yaw cells were, on a clean
stimulus where only one thing changes.

The stimulus
------------
Each still frame is shown three ways:

    FWD     the image expands away from the centre, as it does when a camera moves ahead
    BACK    the image contracts toward the centre
    STATIC  the image does not change scale

All three share the same mean scale, 1.30, and FWD and BACK move through the same range
(1.05 to 1.55), so brightness, texture, field of view and the size of the change are
identical. The only difference is the sign of the scale change.

The scale is implemented as a resize followed by a centre crop, and the minimum scale is
1.05, so the frame never has to be padded. No border artefact can enter, which a naive
zoom-out would produce as a growing black ring.

What is measured
----------------
    delta_fwd   rate(FWD) - rate(STATIC)
    delta_back  rate(BACK) - rate(STATIC)

A cell is kept as a forward candidate if delta_fwd keeps its sign across frames and
across seeds, and if that sign is positive. `delta_back` is reported alongside, because a
cell that responds to any change of scale rather than to expansion specifically is a
weaker candidate, and the two are only distinguishable with both directions present.

Usage:
    PYTHONPATH=. python scripts/p07_forward_synthetic.py --seed 64
    PYTHONPATH=. python scripts/p07_forward_synthetic.py --seed 65
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import VideoVisualEncoder

from p061_stimulus import N_FRAMES, SETTLE_STEPS, WARM_FRAMES, bright_still, pick_frames

P053_RUN = ROOT / "scripts/p053_threshold_run.py"
OUT = ROOT / "output/p07/forward"
STEPS_PER_CLIP = 24
SCALE_MIN = 1.05
SCALE_MAX = 1.55
SCALE_MID = (SCALE_MIN + SCALE_MAX) / 2


def scaled(frame: np.ndarray, s: float) -> np.ndarray:
    """Resize by `s` and centre-crop, so s above 1 never needs padding."""
    h, w = frame.shape[:2]
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])


def make_scale_clip(frame: np.ndarray, mode: str,
                    n: int = STEPS_PER_CLIP) -> list[np.ndarray]:
    """FWD expands, BACK contracts, STATIC holds. Same range and mean for all three."""
    if mode == "STATIC":
        return [scaled(frame, SCALE_MID) for _ in range(n)]
    lo, hi = (SCALE_MIN, SCALE_MAX) if mode == "FWD" else (SCALE_MAX, SCALE_MIN)
    return [scaled(frame, lo + (hi - lo) * k / max(n - 1, 1)) for k in range(n)]


def load_targets():
    spec = importlib.util.spec_from_file_location("p053_run", P053_RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_targets


def run_clip(engine, encoder, clip, watch) -> np.ndarray:
    counts = np.zeros(len(watch))
    n_used = 0
    mark = np.zeros(engine.brain.n, dtype=bool)
    for k, fr in enumerate(clip):
        eye, inject, _ = encoder.encode_frame(fr)
        fired = engine.step(0.0, eye_drive=eye, inject=inject).fired
        if k >= WARM_FRAMES:
            mark[:] = False
            if len(fired):
                mark[fired] = True
            counts += mark[watch]
            n_used += 1
    return counts / max(n_used, 1) / engine.config.brain_dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=64)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    build_targets = load_targets()
    engine = MaleCNSEngine(FlyVOConfig())
    targets, carriers = build_targets(engine.brain)
    watch = np.array([t["cell"] for t in targets], dtype=np.int64)

    print("P07.3 — синтетический тест движения вперёд")
    print(f"  масштаб: FWD {SCALE_MIN}->{SCALE_MAX}, BACK {SCALE_MAX}->{SCALE_MIN}, "
          f"STATIC {SCALE_MID} постоянный")
    print(f"  у всех трёх одинаковы: средний масштаб, диапазон, яркость, текстура")
    print(f"  кадров {N_FRAMES}, seed {args.seed}, клеток под наблюдением {len(watch)}")

    frames = pick_frames()
    engine.reset(seed=args.seed)
    encoder = VideoVisualEncoder(engine, flow_config=None)
    blank = bright_still(frames[0])

    res = {k: np.zeros((len(frames), len(watch)))
           for k in ("FWD", "BACK", "STATIC")}
    t0 = time.perf_counter()
    for fi, fr in enumerate(frames):
        for mode in ("FWD", "BACK", "STATIC"):
            for _ in range(SETTLE_STEPS):
                eye, inject, _ = encoder.encode_frame(blank)
                engine.step(0.0, eye_drive=eye, inject=inject)
            res[mode][fi] = run_clip(engine, encoder, make_scale_clip(fr, mode), watch)
        if fi % 5 == 0 or fi == len(frames) - 1:
            print(f"    кадр {fi + 1}/{len(frames)}  {time.perf_counter() - t0:.0f}s",
                  flush=True)

    np.savez_compressed(OUT / f"scale_seed{args.seed}.npz",
                        fwd=res["FWD"], back=res["BACK"], static=res["STATIC"],
                        watch=watch)
    path = OUT / f"targets_seed{args.seed}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "cell_type", "side", "carrier_weight"])
        for t in targets:
            w.writerow([t["cell"], t["cell_type"], t["side"], t["carrier_weight"]])
    print(f"\n  записано {OUT}/scale_seed{args.seed}.npz")


if __name__ == "__main__":
    main()
