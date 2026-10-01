#!/usr/bin/env python3
"""P09 — screen every descending neuron for direction, on the clean stimulus only.

P08.5 ended with a negative result that is worth restating because it is the reason for
this script. At T50 and T49 the walker turned, and of the four descending pairs the tracker
reads, none pointed the way the walker went. The four were not badly chosen — they were
chosen for the right reason, by P06.1 on a clean stimulus — but four pairs out of a brain
that has 466 descending types and 1340 descending cells is a small sample.

So the question here is not "which cells fit our two failures". That question cannot be
asked without fitting to them. The question is the one P06.1 already asked, asked of
everything: on a stimulus where the only difference between two conditions is the sign of
the motion, which descending cells separate them, reproducibly, on 125 different scenes and
two seeds.

Nothing about the video enters this script. No route, no junction, no label. The frames come
from two clips, but only as scenes — the same 125 stills are shown left, right and still, so
the scene is identical within a frame and cannot explain a LEFT/RIGHT difference.

What gets recorded
------------------
The engine steps the whole brain regardless, so watching 1340 cells costs the same as
watching 46: the firing list is masked once per step and counted. Both sides of every
descending type are taken, plus any cell whose type starts with DN, rather than a curated
list — a curated list is what produced the four that failed.

The metrics are in `p09_synthetic_test.py`; this script only runs the stimulus and writes the
raw per-frame rates, so that the filtering can be argued about separately from the
simulation, and so that a rerun of the filter does not cost another fifteen minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

from p061_stimulus import (SETTLE_STEPS, STEPS_PER_CLIP, WARM_FRAMES, bright_still,
                           make_clip, pick_frames)  # noqa: E402

OUT = ROOT / "output/p09"
P053 = ROOT / "output/p053_threshold/targets.csv"


def dn_inventory(brain) -> tuple[np.ndarray, list[dict]]:
    """Every descending cell, both sides, with the type and side that name it.

    The rule is the cell type's own name: the connectome labels descending neurons with a
    `DN` prefix, and that is the only criterion used. It is deliberately not "cells that look
    promising" — the whole point of the phase is that the promising-looking subset is what
    failed.
    """
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    cells = [c for c in range(brain.n) if str(ct[c]).startswith("DN")]
    cells.sort()
    meta = [{"cell": int(c), "cell_type": str(ct[c]), "side": str(sd[c])} for c in cells]
    return np.array(cells, dtype=np.int64), meta


def read_current_targets() -> set[int]:
    """The 46 cells the tracker reads today, so they can be marked in the table."""
    if not P053.exists():
        return set()
    return {int(r["cell"]) for r in csv.DictReader(P053.open(encoding="utf-8"))}


def run_seed(seed: int, frames: list[np.ndarray], cells: np.ndarray,
             dn_pos: np.ndarray, engine, progress_every: int = 10) -> dict:
    """One pass of the stimulus, returning rates per cell per frame per condition.

    The counts are taken over the last steps of each clip only, after the warm-up, so that
    the measured window is the response to the motion rather than the transient of the clip
    starting. Between clips the field is held still for the settle period, so one condition
    cannot leak into the next.
    """
    engine.reset(seed=seed)
    encoder = VideoVisualEncoder(engine, flow_config=None)
    blank = bright_still(frames[0])
    n_cells = len(cells)
    out = {c: np.zeros((len(frames), n_cells)) for c in ("LEFT", "RIGHT", "STATIC")}
    t0 = time.perf_counter()

    for fi, fr in enumerate(frames):
        for cond, direction in (("LEFT", -1), ("RIGHT", 1), ("STATIC", 0)):
            for _ in range(SETTLE_STEPS):
                eye, inject, _ = encoder.encode_frame(blank)
                engine.step(0.0, eye_drive=eye, inject=inject)
            clip = make_clip(fr, direction)
            counts = np.zeros(n_cells)
            used = 0
            for k, cf in enumerate(clip):
                eye, inject, _ = encoder.encode_frame(cf)
                fired = engine.step(0.0, eye_drive=eye, inject=inject).fired
                if k >= WARM_FRAMES:
                    sel = fired[dn_pos[fired] >= 0]
                    if len(sel):
                        counts += np.bincount(dn_pos[sel], minlength=n_cells)
                    used += 1
            out[cond][fi] = counts / max(used, 1) / engine.config.brain_dt
        if (fi + 1) % progress_every == 0 or fi == len(frames) - 1:
            el = time.perf_counter() - t0
            rate = el / (fi + 1)
            print(f"    кадр {fi + 1}/{len(frames)}  {el:6.0f}с  "
                  f"(осталось ≈ {rate * (len(frames) - fi - 1) / 60:.1f} мин)",
                  flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="64,65")
    ap.add_argument("--frames", type=int, default=125)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    cells, meta = dn_inventory(engine.brain)
    dn_pos = np.full(engine.brain.n, -1, dtype=np.int64)
    dn_pos[cells] = np.arange(len(cells))
    current = read_current_targets()

    from collections import Counter
    types = Counter(m["cell_type"] for m in meta)
    sides = Counter(m["side"] for m in meta)
    print("P09 — скрининг всех нисходящих клеток на чистом стимуле")
    print(f"  мозг: {engine.brain.n} клеток")
    print(f"  нисходящих клеток отобрано: {len(cells)}")
    print(f"  типов: {len(types)}   сторон: {dict(sides)}")
    print(f"  из них клетки текущего трекера: {len(current)} (контроль)")
    print(f"  стимул: {args.frames} неподвижных кадров × (LEFT, RIGHT, STATIC), "
          f"циклический сдвиг, {STEPS_PER_CLIP} шагов, прогрев {WARM_FRAMES}")
    print(f"  seeds: {args.seeds}")
    print()

    frames = pick_frames(args.frames)
    print(f"  кадров получено: {len(frames)}")

    summary = {"n_cells": int(len(cells)), "n_types": len(types),
               "n_current": len(current), "frames": len(frames),
               "seeds": [int(s) for s in args.seeds.split(",")],
               "sides": dict(sides)}
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n  --- seed {seed} ---")
        arr = run_seed(seed, frames, cells, dn_pos, engine)
        np.savez_compressed(out / f"synthetic_raw_seed{seed}.npz",
                            left=arr["LEFT"], right=arr["RIGHT"],
                            static=arr["STATIC"], cells=cells,
                            cell_type=np.array([m["cell_type"] for m in meta]),
                            side=np.array([m["side"] for m in meta]))
        print(f"    записано {out / f'synthetic_raw_seed{seed}.npz'}")

    (out / "screen_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"\nзаписано: {out/'screen_meta.json'}")
    print("метрики считает p09_synthetic_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
