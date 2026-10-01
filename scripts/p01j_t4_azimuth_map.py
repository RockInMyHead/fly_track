#!/usr/bin/env python3
"""
Infer per-cell azimuth for T4/T5 (retinotopic optic-lobe cells) from the connectome.

Photoreceptors carry an azimuth (-1 = rear of left eye ... 0 = frontal ... +1 = rear of
right eye). T4/T5 have no azimuth in brain.npz, but each T4/T5 cell is one optic column,
so its position can be recovered from weighted photoreceptor input two hops upstream.

Method (weights use |w|; azimuth is a position, not a sign):
  hop 1: photoreceptor -> ol_intrinsic          az1[i] = Σ|w|·az / Σ|w|
  hop 2: ol_intrinsic  -> T4/T5                 az2[c] = Σ|w|·az1 / Σ|w|

The result is cached to data/p01r/t4_azimuth.npz so downstream code never recomputes it.

Usage:
    PYTHONPATH=. python scripts/p01j_t4_azimuth_map.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.malecns_engine import MaleCNSEngine

T_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
OUT_NPZ = ROOT / "data/p01r/t4_azimuth.npz"
OUT_JSON = ROOT / "data/p01r/t4_azimuth.json"


def infer(brain) -> dict:
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    sc = np.asarray(brain.superclass)
    vis = np.asarray(brain.visual)
    az = np.asarray(brain.azimuth).astype(np.float64)
    indptr, indices, w = brain.indptr, brain.indices, brain.weights

    # ---- hop 1: photoreceptors -> downstream (lamina / medulla intrinsics) ----
    num = np.zeros(brain.n, dtype=np.float64)
    den = np.zeros(brain.n, dtype=np.float64)
    for k, j in enumerate(vis.tolist()):
        wj = w[indptr[j] : indptr[j + 1]]
        ij = indices[indptr[j] : indptr[j + 1]]
        aw = np.abs(wj)
        np.add.at(num, ij, aw * az[k])
        np.add.at(den, ij, aw)

    hop1 = den > 0
    az1 = np.zeros(brain.n, dtype=np.float64)
    az1[hop1] = num[hop1] / den[hop1]

    # ---- hop 2: those neurons -> T4/T5 ----
    src = np.flatnonzero(hop1)
    num2 = np.zeros(brain.n, dtype=np.float64)
    den2 = np.zeros(brain.n, dtype=np.float64)
    for j in src.tolist():
        wj = w[indptr[j] : indptr[j + 1]]
        ij = indices[indptr[j] : indptr[j + 1]]
        aw = np.abs(wj)
        np.add.at(num2, ij, aw * az1[j])
        np.add.at(den2, ij, aw)

    az2 = np.full(brain.n, np.nan, dtype=np.float32)
    ok = den2 > 0
    az2[ok] = (num2[ok] / den2[ok]).astype(np.float32)

    # ---- report ----
    report = {"n_hop1": int(hop1.sum()), "n_t4t5_resolved": 0, "per_type": {}}
    for t in T_TYPES:
        for side in ("L", "R"):
            m = (ct == t) & (sd == side)
            vals = az2[m]
            good = np.isfinite(vals)
            if good.sum():
                report["per_type"][f"{t}_{side}"] = {
                    "n": int(m.sum()),
                    "n_resolved": int(good.sum()),
                    "mean": float(np.mean(vals[good])),
                    "min": float(np.min(vals[good])),
                    "max": float(np.max(vals[good])),
                    "std": float(np.std(vals[good])),
                }
                report["n_t4t5_resolved"] += int(good.sum())

    return {"azimuth": az2, "report": report}


def main() -> None:
    engine = MaleCNSEngine()
    engine._ensure_loaded()
    brain = engine.brain

    res = infer(brain)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        azimuth=res["azimuth"],
        cell_type=np.asarray(brain.cell_type),
        side=np.asarray(brain.side),
    )
    OUT_JSON.write_text(json.dumps(res["report"], indent=2), encoding="utf-8")

    r = res["report"]
    print(f"hop1 neurons with azimuth: {r['n_hop1']}")
    print(f"T4/T5 cells resolved:      {r['n_t4t5_resolved']}")
    print(f"\n{'type':10s} {'n':>5s} {'res':>5s} {'mean_az':>8s} {'min':>7s} {'max':>7s} {'std':>6s}")
    for k, v in r["per_type"].items():
        print(
            f"{k:10s} {v['n']:5d} {v['n_resolved']:5d} {v['mean']:+8.3f} "
            f"{v['min']:+7.2f} {v['max']:+7.2f} {v['std']:6.3f}"
        )
    print(f"\nWrote {OUT_NPZ} / {OUT_JSON}")


if __name__ == "__main__":
    main()
