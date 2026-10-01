#!/usr/bin/env python3
"""
P06.2 — which cells carry the direction to DNp17, DNa07, DNp26 and DNp20?

P06.1 established that the direction is readable on a clean stimulus, weakly: median AUC
0.72 against 0.95-1.00 on real windows, which means the real numbers were inflated by
scene differences. Here the question changes from whether to how.

The candidate set is built from the connectome, not from the activity
-------------------------------------------------------------------
2,388 cells project directly onto the 18 target cells (4 types x 2 sides, some with
several cells each). They span 819 types and sit at depths 1 to 3 from T4/T5. The
largest groups are LLPC1, LLPC2, LLPC3, LPC1, LPLC4, LPLC1 and LC22 — and LLPC1 and
LPC1 are exactly what P0.1K had already flagged as direction selective under a mirrored
synthetic stimulus, which is a good sign that the set is the right one.

The stimulus is identical to P06.1's
------------------------------------
Same 25 still frames from both clips, same cyclic shift, same step size, same warm-up
and settle between clips. That matters: the numbers have to be comparable to the
baseline, or a change after ablation could not be attributed to the ablation.

Ablation
--------
`--ablate TYPE[:side]` zeroes every outgoing weight of that type, on a copy of the weight
array so the original is untouched. Cells of that type stop influencing anything
downstream, which is what turning the type off means. The copy is restored after the run
even if the run raises.

Usage:
    PYTHONPATH=. python scripts/p062_path_trace.py --tag base --seed 64
    PYTHONPATH=. python scripts/p062_path_trace.py --tag abl_LLPC1_L --seed 64 --ablate LLPC1:L
    PYTHONPATH=. python scripts/p062_path_trace.py --list-types
"""

from __future__ import annotations

import argparse
import collections
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

from p061_stimulus import N_FRAMES, SETTLE_STEPS, WARM_FRAMES, make_clip, pick_frames, bright_still

OUT = ROOT / "output/p062_path"
TARGET_TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]


def watched_cells(brain) -> tuple[np.ndarray, np.ndarray]:
    """First-order inputs to the target cells, plus the target cells themselves.

    Returns the cell indices and their depth from T4/T5, so the analysis can separate
    what sits directly on T4/T5 from what arrives later.
    """
    ct = np.asarray(brain.cell_type)
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)
    n = brain.n

    tgt = np.array(sorted({int(c) for nm in TARGET_TYPES
                           for c in np.flatnonzero(ct == nm)}), dtype=np.int64)
    pos = np.full(n, -1, np.int64)
    pos[tgt] = np.arange(len(tgt))
    counts = np.diff(ip).astype(np.int64)
    edge_src = np.repeat(np.arange(n, dtype=np.int32), counts)
    hit = np.flatnonzero(pos[ix] >= 0)
    src = np.unique(edge_src[hit])

    watch = np.unique(np.concatenate([src.astype(np.int64), tgt]))

    depth = np.full(n, -1, np.int8)
    seed = np.isin(ct, ("T4a", "T4b", "T5a", "T5b"))
    depth[seed] = 0
    fr = np.flatnonzero(seed)
    for d in range(1, 5):
        a, z = ip[fr], ip[fr + 1]
        cn = z - a
        if cn.sum() == 0:
            break
        off = np.arange(cn.sum()) - np.repeat(np.cumsum(cn) - cn, cn)
        tg = ix[np.repeat(a, cn) + off]
        new = np.unique(tg[depth[tg] < 0])
        depth[new] = d
        fr = new
        if not len(new):
            break
    return watch, depth[watch]


def run_clip(engine, encoder, clip: list[np.ndarray], watch: np.ndarray) -> np.ndarray:
    """Mean firing rate per watched cell over the measured steps."""
    counts = np.zeros(len(watch))
    n_used = 0
    mark = np.zeros(engine.brain.n, dtype=bool)
    for k, fr in enumerate(clip):
        eye, inject, _ = encoder.encode_frame(fr)
        fired = engine.step(0.0, eye_drive=eye, inject=inject).fired
        if k >= WARM_FRAMES:
            # a mask is cheaper than np.isin here: it is one pass over the whole sheet
            # rather than a sort of every spike index for every step
            mark[:] = False
            if len(fired):
                mark[fired] = True
            counts += mark[watch]
            n_used += 1
    return counts / max(n_used, 1) / engine.config.brain_dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--ablate", default=None, help="TYPE or TYPE:side")
    ap.add_argument("--list-types", action="store_true")
    ap.add_argument("--n-frames", type=int, default=N_FRAMES,
                    help="how many still frames; more frames means more independent "
                         "estimates of the effect and is the only way to give a weak "
                         "ablation effect a chance to show")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    brain = engine.brain
    watch, depths = watched_cells(brain)
    print("P06.2 — трассировка пути к целевым нисходящим нейронам")
    print(f"  клеток под наблюдением: {len(watch)} "
          f"(первые входы целевых DN + сами DN)")
    print(f"  по глубинам: "
          f"{dict(collections.Counter(int(x) for x in depths))}")

    ct = np.asarray(brain.cell_type)
    if args.list_types:
        c = collections.Counter(ct[watch].tolist())
        print(f"\n  типов среди наблюдаемых: {len(c)}")
        for t, k in c.most_common(40):
            print(f"    {str(t):>20s} {k:5d}")
        return

    # ---- optional ablation -------------------------------------------------
    original = brain.weights
    ablated = None
    if args.ablate:
        w2 = original.copy()
        ip = np.asarray(brain.indptr)
        specs = []
        total_cells = 0
        total_edges = 0
        # several types may be removed at once, to test whether a signal that survives
        # the removal of any single type is carried redundantly by several
        for item in args.ablate.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            name = parts[0]
            side = parts[1] if len(parts) > 1 else None
            cells = np.flatnonzero((ct == name)
                                   & ((brain.side == side) if side else True))
            if len(cells) == 0:
                specs.append({"spec": item, "found": False})
                continue
            edges = 0
            for c in cells.tolist():
                a, z = ip[c], ip[c + 1]
                if z > a:
                    w2[a:z] = 0.0
                    edges += int(z - a)
            specs.append({"spec": item, "found": True, "cells": int(len(cells)),
                          "edges": edges})
            total_cells += len(cells)
            total_edges += edges
        if total_edges == 0:
            print(f"  АБЛЯЦИЯ {args.ablate}: нечего обнулять, тип не найден")
            return
        brain.weights = w2
        ablated = {"spec": args.ablate, "parts": specs,
                   "cells": total_cells, "edges_zeroed": total_edges}
        print(f"  АБЛЯЦИЯ: {args.ablate}")
        for sp in specs:
            if sp.get("found"):
                print(f"    {sp['spec']}: {sp['cells']} клеток, "
                      f"{sp['edges']} связей обнулено")
            else:
                print(f"    {sp['spec']}: тип не найден")

    try:
        print(f"\n  стимул: {args.n_frames} кадров x (LEFT, RIGHT, STATIC)"
              f"{' (как в P06.1)' if args.n_frames == N_FRAMES else ''}")
        frames = pick_frames(args.n_frames)
        engine.reset(seed=args.seed)
        encoder = VideoVisualEncoder(engine, flow_config=None)
        blank = bright_still(frames[0])

        res = {k: np.zeros((len(frames), len(watch))) for k in
               ("LEFT", "RIGHT", "STATIC")}
        t0 = time.perf_counter()
        for fi, fr in enumerate(frames):
            for cond, direction in (("LEFT", -1), ("RIGHT", +1), ("STATIC", 0)):
                for _ in range(SETTLE_STEPS):
                    eye, inject, _ = encoder.encode_frame(blank)
                    engine.step(0.0, eye_drive=eye, inject=inject)
                res[cond][fi] = run_clip(engine, encoder, make_clip(fr, direction),
                                         watch)
            if fi % 5 == 0 or fi == len(frames) - 1:
                print(f"    кадр {fi + 1}/{len(frames)}  "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

        np.savez_compressed(OUT / f"trace_{args.tag}_seed{args.seed}.npz",
                            left=res["LEFT"], right=res["RIGHT"],
                            static=res["STATIC"], watch=watch, depth=depths)
        meta = {"tag": args.tag, "seed": args.seed, "ablated": ablated,
                "n_watched": int(len(watch)),
                "n_frames": args.n_frames,
                "depth_counts": {str(k): int(v) for k, v in
                                 collections.Counter(int(x) for x in depths).items()}}
        (OUT / f"trace_{args.tag}_seed{args.seed}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  записано {OUT}/trace_{args.tag}_seed{args.seed}.npz")
    finally:
        # the ablation must never outlive the run, even on an exception
        brain.weights = original
        if ablated:
            print("  веса восстановлены")


if __name__ == "__main__":
    main()
