#!/usr/bin/env python3
"""
P0.1I synthetic smoke test — encoder-realistic.

The old control drove T4_L + T5_L directly, which bypassed _build_inject and
tested a configuration the encoder never produces. This test goes through the
real path:

    synthetic frames -> DirectionalMotionFrontend -> _build_inject -> T4/T5 -> MaleCNS -> DN

Conditions: YAW_LEFT / FORWARD / YAW_RIGHT, 5 seeds.
Checks the signed lateral drive and the DNp01 / DNp04 L-R readout.

Usage:
    PYTHONPATH=. python scripts/p01i_smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import FlowConfig
from fly_vo.p01i_probe import dn_layout, fam_indices, oriented_sign
from fly_vo.visual_encoder import VideoVisualEncoder
from scripts.visual_observability_test import _make_frames

# Expected oriented sign per condition (LEFT=+1, FWD=0, RIGHT=-1).
CONDITIONS = {"YAW_LEFT": +1, "FORWARD": 0, "YAW_RIGHT": -1}
RATE_WINDOW = 12          # trailing steps used for the rate estimate
NEUTRAL_RATIO = 0.25      # |FWD| < ratio * min(|LEFT|, |RIGHT|)
QUARTET = ("DNp01", "DNp04")


def run_synthetic(seeds: list[int], frames_per_class: int = 45) -> dict:
    cfg = FlyVOConfig(use_pathway_fallback=False)
    fc = FlowConfig(crop_mode="center_band", flow_inject_gain=8.0, ema_tau_s=0.5)
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=fc)
    engine._ensure_loaded()

    _idx, slot, ct_slot, sd_slot = dn_layout(engine.brain)
    n_dn = len(ct_slot)
    fams = {t: fam_indices(ct_slot, sd_slot, t) for t in QUARTET}

    synth = _make_frames(n=frames_per_class)
    stim = {c: synth[c] for c in CONDITIONS}

    per_seed: dict[str, dict] = {}
    for cond in CONDITIONS:
        per_seed[cond] = {"inject": [], **{t: [] for t in QUARTET}}
        for seed in seeds:
            encoder.reset()
            engine.reset(seed=seed)
            inj_lr: list[float] = []
            counts = np.zeros((n_dn, len(stim[cond])), dtype=np.int32)
            for i, bgr in enumerate(stim[cond]):
                eye, inject, metrics = encoder.encode_frame(bgr)
                inj_lr.append(float(metrics.get("inject_combined_lr", 0.0)))
                res = engine.step(i * cfg.brain_dt, eye_drive=eye, inject=inject)
                fired = res.fired
                if len(fired):
                    hits = slot[fired]
                    hits = hits[hits >= 0]
                    if len(hits):
                        np.add.at(counts[:, i], hits, 1)
            lo = max(0, len(stim[cond]) - RATE_WINDOW)
            hi = len(stim[cond])
            span = max(hi - lo, 1) * cfg.brain_dt
            per_seed[cond]["inject"].append(float(np.median(inj_lr[lo:])))
            for t in QUARTET:
                il, ir = fams[t]
                l = float(counts[il, lo:hi].sum()) / span if il else 0.0
                r = float(counts[ir, lo:hi].sum()) / span if ir else 0.0
                per_seed[cond][t].append(l - r)

    return {"per_seed": per_seed, "seeds": seeds, "n_dn": n_dn}


def evaluate_synthetic(res: dict) -> dict:
    per_seed = res["per_seed"]
    seeds = res["seeds"]
    checks: dict[str, bool] = {}
    table: dict[str, dict] = {}

    def agg(cond: str, key: str) -> tuple[float, list[float], list[int]]:
        vals = per_seed[cond][key]
        med = float(np.median(vals))
        return med, vals, [oriented_sign(v) for v in vals]

    for key in ("inject",) + QUARTET:
        meds, vals, signs = {}, {}, {}
        for cond, want in CONDITIONS.items():
            m, v, s = agg(cond, key)
            meds[cond], vals[cond], signs[cond] = m, v, s
        table[key] = {"median": meds, "seed_signs": signs, "values": vals}

        if key == "inject":
            ratio = NEUTRAL_RATIO
        else:
            ratio = NEUTRAL_RATIO

        left_ok = all(s == +1 for s in signs["YAW_LEFT"])
        right_ok = all(s == -1 for s in signs["YAW_RIGHT"])
        neutral_ok = abs(meds["FORWARD"]) < ratio * min(
            abs(meds["YAW_LEFT"]), abs(meds["YAW_RIGHT"])
        )
        checks[f"{key}_left_pos"] = left_ok
        checks[f"{key}_right_neg"] = right_ok
        checks[f"{key}_neutral_small"] = neutral_ok
        checks[f"{key}_ALL"] = left_ok and right_ok and neutral_ok

    checks["ALL"] = all(v for k, v in checks.items() if k.endswith("_ALL"))
    return {"checks": checks, "table": table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01i_smoke/smoke_report.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    print("P0.1I synthetic smoke test (encoder-realistic: frames -> frontend -> _build_inject -> CNS)")
    res = run_synthetic(args.seeds)
    ev = evaluate_synthetic(res)

    print(f"\n{'signal':10s} {'LEFT':>10s} {'FWD':>10s} {'RIGHT':>10s}   seed signs L/F/R")
    for key, t in ev["table"].items():
        m = t["median"]
        ss = t["seed_signs"]
        fmt = lambda s: {1: "+", -1: "-", 0: "0"}[s]
        print(
            f"{key:10s} {m['YAW_LEFT']:+10.4f} {m['FORWARD']:+10.4f} {m['YAW_RIGHT']:+10.4f}   "
            + "/".join("".join(fmt(s) for s in ss[c]) for c in CONDITIONS)
        )

    print("\n=== checks ===")
    for k, v in ev["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    report = {"probe": "P0.1I synthetic encoder-realistic smoke test", "seeds": args.seeds, **ev}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n{'PASS' if ev['checks']['ALL'] else 'FAIL'} P0.1I SYNTHETIC SMOKE")
    print(f"Wrote {out}")
    raise SystemExit(0 if ev["checks"]["ALL"] else 1)


if __name__ == "__main__":
    main()
