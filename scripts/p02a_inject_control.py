#!/usr/bin/env python3
"""
P0.2A control — does MaleCNS propagate signed T4/T5 L/R asymmetry to DNs?

Pure mechanistic probe: no video, no frontend, no trajectory.
Injects a fixed drive into T4/T5 of ONE side only and measures the
descending-neuron L/R differential.

  cond LEFT  : T4/T5 left  groups injected
  cond RIGHT : T4/T5 right groups injected
  cond NONE  : no inject (baseline asymmetry)

If DN lateralization flips sign between LEFT and RIGHT → the CNS preserves
the signed asymmetry. If not → the visual drive is not directionally
resolvable downstream (at the tested amplitude).

Usage:
    PYTHONPATH=. python scripts/p02a_inject_control.py
    PYTHONPATH=. python scripts/p02a_inject_control.py --strength 0.5 --seeds 5 --steps 150
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import T4_TYPES, T5_TYPES

ROOT_DIR = ROOT


def dn_table(brain):
    sc = np.asarray(brain.superclass)
    idx = np.where(sc == "descending_neuron")[0]
    ct = np.asarray(brain.cell_type)[idx]
    sd = np.asarray(brain.side)[idx]
    slot = np.full(int(sc.shape[0]), -1, dtype=np.int64)
    slot[idx] = np.arange(len(idx), dtype=np.int64)
    return idx, slot, ct, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("-o", "--output", default="output/p02a_dn_trace/inject_control")
    args = ap.parse_args()

    cfg = FlyVOConfig()
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain

    idx, slot, ct_slot, sd_slot = dn_table(brain)
    n_dn = len(idx)

    # T4/T5 groups per side
    groups: dict[str, np.ndarray] = {}
    for t in T4_TYPES + T5_TYPES:
        for side in ("L", "R"):
            parts = []
            for s in ("L", "R"):
                try:
                    c = brain.cells([t], side=s)
                    if s == side and len(c):
                        parts.append(c)
                except Exception:
                    pass
            arr = np.concatenate(parts) if parts else np.array([], dtype=int)
            groups[f"{t}_{side}"] = arr

    def side_cells(side: str) -> np.ndarray:
        arrs = [groups[f"{t}_{side}"] for t in T4_TYPES + T5_TYPES if len(groups[f"{t}_{side}"])]
        return np.concatenate(arrs) if arrs else np.array([], dtype=int)

    left_cells, right_cells = side_cells("L"), side_cells("R")
    print(f"T4/T5 injected population: L={len(left_cells)}  R={len(right_cells)}")

    conditions = {
        "NONE": [],
        "LEFT": [(left_cells, args.strength)],
        "RIGHT": [(right_cells, args.strength)],
    }

    seeds = list(range(1, args.seeds + 1))
    per_cond: dict[str, np.ndarray] = {}
    total_spikes: dict[str, list[int]] = defaultdict(list)

    for cond, inject in conditions.items():
        mat = np.zeros((len(seeds), n_dn), dtype=np.float64)
        for si, seed in enumerate(seeds):
            engine.reset(seed=seed)
            counts = np.zeros(n_dn, dtype=np.int64)
            tot = 0
            for _ in range(args.steps):
                res = engine.step(0.0, eye_drive=None, inject=inject)
                fired = res.fired
                tot += len(fired)
                if len(fired):
                    hits = slot[fired]
                    hits = hits[hits >= 0]
                    if len(hits):
                        np.add.at(counts, hits, 1)
            mat[si] = counts / (args.steps * dt)  # Hz
            total_spikes[cond].append(tot)
        per_cond[cond] = mat
        print(f"  {cond:5s}: total brain spikes/seed mean={np.mean(total_spikes[cond]):.0f}")

    # --- family-level tables ---
    fam_side: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(n_dn):
        fam_side[(str(ct_slot[i]), str(sd_slot[i]))].append(i)

    rows = []
    for ctype in sorted({c for (c, _s) in fam_side if (c, "L") in fam_side and (c, "R") in fam_side}):
        il, ir = fam_side[(ctype, "L")], fam_side[(ctype, "R")]
        def diff(cond: str) -> np.ndarray:
            return per_cond[cond][:, il].sum(axis=1) - per_cond[cond][:, ir].sum(axis=1)
        d_none, d_l, d_r = diff("NONE"), diff("LEFT"), diff("RIGHT")
        defl_l = float(np.median(d_l) - np.median(d_none))
        defl_r = float(np.median(d_r) - np.median(d_none))
        rows.append({
            "cell_type": ctype,
            "n_L": len(il),
            "n_R": len(ir),
            "diff_NONE": float(np.median(d_none)),
            "diff_LEFT": float(np.median(d_l)),
            "diff_RIGHT": float(np.median(d_r)),
            "defl_LEFT": defl_l,
            "defl_RIGHT": defl_r,
            "sign_flip": bool(defl_l * defl_r < 0),
            "consistency_flip": float(np.mean(np.sign(d_l - d_none) != np.sign(d_r - d_none))),
        })

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "families.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    flips = [r for r in rows if r["sign_flip"] and r["consistency_flip"] >= 0.8]
    flips.sort(key=lambda r: -min(abs(r["defl_LEFT"]), abs(r["defl_RIGHT"])))

    report = {
        "probe": "synthetic T4/T5 side-inject control",
        "strength": args.strength,
        "steps": args.steps,
        "seeds": seeds,
        "n_L_cells": int(len(left_cells)),
        "n_R_cells": int(len(right_cells)),
        "brain_spikes_per_seed_mean": {k: float(np.mean(v)) for k, v in total_spikes.items()},
        "n_types_with_signflip": len(flips),
        "top_signflip": flips[:20],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDN types with genuine L/R sign-flip: {len(flips)}")
    for r in flips[:12]:
        print(
            f"  {r['cell_type']:12s} defl_L={r['defl_LEFT']:+.2f} defl_R={r['defl_RIGHT']:+.2f} "
            f"consistency={r['consistency_flip']:.2f}"
        )
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
