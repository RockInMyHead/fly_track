#!/usr/bin/env python3
"""
P06.1 — a clean test of direction, with everything but the sign held fixed.

Why the previous tests could not settle this
--------------------------------------------
Every measurement so far compared real windows of a walking clip: a leftward turn window
and a rightward turn window differ not only in the direction of motion but in scenery,
speed, exposure and how busy the frame is. A neuron that separates them may be reading
any of those. The mirror pass removed none of it, because flipping an image reverses the
motion and any left-right asymmetry of the scene at the same time. That is why DNg100,
which receives exactly zero input from the 48 carriers the whole P05 chain was about,
still separated the turns at AUC 0.86.

The stimulus here
-----------------
Each of 25 still frames is presented three ways:

    LEFT    the frame slides left by a fixed step every step
    RIGHT   the same frame slides right by the same step
    STATIC  the frame does not move at all

Everything else is identical by construction: the same source frame, the same duration,
the same step size, the same brightness and texture, the same absolute displacement.
The only difference between LEFT and RIGHT is its sign.

The shift is cyclic, `np.roll`, so no black edge ever enters the frame. A truncated
shift would put a growing dark band on one side, which is a large and asymmetric change
in the image and would confound the very thing being measured.

What is measured
----------------
For every cell, its firing rate in each of the three conditions, and

    delta = rate(RIGHT) - rate(LEFT)

with the sign counted per frame. A cell that reads direction must give the same sign of
delta on frame after frame, across scenes as different as a corridor and a bench. A cell
that reads brightness, texture or scene structure has no reason to.

The three numbers that matter are the sign consistency across frames, the mean delta
against its own spread, and the AUC between LEFT and RIGHT pooled over frames. All three
are reported per cell and per seed. The static condition is the floor: how much a cell
varies with no motion at all, which is the noise any delta must clear.

About the controls
------------------
DNg100, DNa02 and MDN are included, and they are *not* required to fail. DNg100 receives
no input from the P05 carriers, but that says nothing about whether it reads direction
through some other route. If it separates clean LEFT from clean RIGHT here, that is a
result about the connectome, not a broken test. What the controls do is make the primary
result interpretable: they show what this design gives when a cell has no reason to
care, whatever the mechanism.

Two seeds, so a single lucky initialisation cannot decide it.

Usage:
    PYTHONPATH=. python scripts/p061_synthetic_run.py --seed 64
    PYTHONPATH=. python scripts/p061_synthetic_run.py --seed 65
    PYTHONPATH=. python scripts/p061_synthetic_run.py --analyze-only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import VideoVisualEncoder
from fly_vo.video_reader import VideoReader

from p061_stimulus import (CONDITIONS, N_FRAMES, SETTLE_STEPS, SHIFT_PX,
                           STEPS_PER_CLIP, WARM_FRAMES, bright_still, make_clip,
                           pick_frames)

VIDEO1 = ROOT / "data/p01r/VID00001.AVI"
VIDEO2 = ROOT / "data/p01r/VID00002.AVI"
P053_RUN = ROOT / "scripts/p053_threshold_run.py"
OUT = ROOT / "output/p061_synthetic"

# Frozen in data/p05_frozen_targets.json; not re-chosen here.
PRIMARY = [("DNp17", "L"), ("DNp17", "R"), ("DNa07", "L"), ("DNa07", "R"),
           ("DNp26", "L"), ("DNp26", "R"), ("DNp20", "L"), ("DNp20", "R")]
CONTROLS = [("DNa02", "L"), ("DNg100", "L"), ("MDN", "L")]


def load_p053():
    spec = importlib.util.spec_from_file_location("p053_run", P053_RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod






def measure(engine, encoder, cells: list[int], frames: list[np.ndarray],
            label: str, seed: int, frames_note: int) -> dict:
    """Run one clip and return the mean rate per watched cell over the last steps."""
    counts = np.zeros(len(cells))
    n_used = 0
    for k, fr in enumerate(frames):
        eye, inject, _ = encoder.encode_frame(fr)
        if k >= WARM_FRAMES:
            counts += np.isin(cells, engine.step(0.0, eye_drive=eye, inject=inject).fired)
            n_used += 1
    return counts / max(n_used, 1) / engine.config.brain_dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_p053()
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    targets, carriers = mod.build_targets(engine.brain)
    brain = engine.brain
    idx = np.array([t["cell"] for t in targets], dtype=np.int64)

    want = PRIMARY + CONTROLS
    col_of = {}
    for nm, side in want:
        for k, t in enumerate(targets):
            if t["cell_type"] == nm and t["side"] == side:
                col_of[(nm, side)] = k
                break

    print("P06.1 — чистый тест направления")
    print(f"  стимул: {N_FRAMES} неподвижных кадров x (LEFT, RIGHT, STATIC)")
    print(f"  сдвиг {SHIFT_PX} px за шаг, {STEPS_PER_CLIP} шагов, циклический "
          f"(края не появляются)")
    print(f"  прогрев перед окном измерения: {WARM_FRAMES} шагов, "
          f"выключение между клипами: {SETTLE_STEPS} шагов")
    print(f"  seed {args.seed}, нейронов под наблюдением: {len(idx)}")

    if not args.analyze_only:
        print("\n  отбираю кадры...")
        frames = pick_frames()
        diffs = [float(np.abs(frames[i].astype(float)
                              - frames[i + 1].astype(float)).mean())
                 for i in range(min(5, len(frames) - 1))]
        print(f"  отобрано {len(frames)} кадров, разница между соседними "
              f"{np.mean(diffs):.0f}")
        engine.reset(seed=args.seed)
        encoder = VideoVisualEncoder(engine, flow_config=None)
        blank = bright_still(frames[0])

        rows = []
        t0 = time.perf_counter()
        for fi, fr in enumerate(frames):
            rec = {"frame": fi}
            for cond, direction in (("LEFT", -1), ("RIGHT", +1), ("STATIC", 0)):
                # settle with a static field so the previous clip's activity decays
                for _ in range(SETTLE_STEPS):
                    eye, inject, _ = encoder.encode_frame(blank)
                    engine.step(0.0, eye_drive=eye, inject=inject)
                clip = make_clip(fr, direction)
                r = measure(engine, encoder, idx, clip, cond, args.seed, fi)
                rec[cond] = r
            rows.append(rec)
            if fi % 5 == 0 or fi == len(frames) - 1:
                el = time.perf_counter() - t0
                print(f"    кадр {fi + 1}/{len(frames)}  {el:.0f}s", flush=True)

        arr = {}
        for cond in ("LEFT", "RIGHT", "STATIC"):
            arr[cond] = np.stack([r[cond] for r in rows])
        np.savez_compressed(OUT / f"synthetic_seed{args.seed}.npz",
                            left=arr["LEFT"], right=arr["RIGHT"],
                            static=arr["STATIC"], idx=idx)
        print(f"\n  записано {OUT}/synthetic_seed{args.seed}.npz")
    else:
        path = OUT / f"synthetic_seed{args.seed}.npz"
        if not path.exists():
            print(f"  нет {path}")
            return
        d = np.load(path)
        arr = {"LEFT": d["left"], "RIGHT": d["right"], "STATIC": d["static"]}

    print(f"\n=== СРЕДНИЕ ПО КАДРАМ (seed {args.seed}, Гц) ===")
    print(f"  {'клетка':>12s} {'группа':>10s} {'LEFT':>8s} {'RIGHT':>8s} "
          f"{'STATIC':>8s} {'delta':>9s}")
    for nm, side in want:
        k = col_of.get((nm, side))
        if k is None:
            continue
        g = "контроль" if (nm, side) in CONTROLS else "основная"
        L, R, S = arr["LEFT"][:, k], arr["RIGHT"][:, k], arr["STATIC"][:, k]
        print(f"  {nm + ' ' + side:>12s} {g:>10s} {L.mean():8.2f} {R.mean():8.2f} "
              f"{S.mean():8.2f} {(R - L).mean():+9.3f}")

    print(f"\n  файлы для анализа генерирует p061_synthetic_analyze.py")


if __name__ == "__main__":
    main()
