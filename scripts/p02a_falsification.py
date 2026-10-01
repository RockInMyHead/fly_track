#!/usr/bin/env python3
"""
P0.2A FOLLOW-UP — falsification test (single run, no tuning).

Question:
  Can MaleCNS recover the correct yaw sign on FAILED real events better than
  the scalar yaw_frozen, or does it faithfully reproduce the (wrong) inject?

Frozen:
  - frontend (optic_flow.py) untouched
  - MaleCNS untouched
  - primary DN quartet fixed BEFORE looking at results:
        DNp01, DNp04, DNpe015, DNpe043
  - polarity of each DN type taken from the SYNTHETIC inject control,
    never from real events

Events:
  FAILED            left_81, left_108, right_67, right_210
  POSITIVE CONTROLS left_65, right_216
  NEUTRAL CONTROL   fwd_140

All signs are reported in a single orientation:  LEFT => +1, RIGHT => -1.

  yaw_oriented  = -sign(yaw_frozen)            (yaw_frozen is negative for LEFT)
  inject_oriented =  sign(T4_L-R + T5_L-R)     (LEFT drives L side up)
  dn_oriented   =  sign(dn_L - dn_R) * canonical_polarity

Usage:
    PYTHONPATH=. python scripts/p02a_falsification.py
    PYTHONPATH=. python scripts/p02a_falsification.py --seeds 5
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

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import T4_TYPES, T5_TYPES, VideoVisualEncoder

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
EMA_TAU = 0.5

# Frozen primary quartet — chosen in P0.2A, not re-selected here.
PRIMARY_QUARTET = ("DNp01", "DNp04", "DNpe015", "DNpe043")

# 18 synthetic ∩ real sign-flip types (diagnostics only, polarity not used for verdict).
CROSSREF_TYPES = (
    "DNpe054", "DNp51,DNpe019", "DNge110", "DNpe015", "DNp01", "DNg11", "DNg53",
    "DNpe043", "DNp69", "DNp10", "DNa07", "DNg94", "DNp04", "DNp20", "DNp26",
    "DNp66", "DNge043", "DNg58",
)

EVENTS = {
    # id: (start_s, expected_visual_sign_oriented)   LEFT=+1, RIGHT=-1
    "left_65":   (63.5, +1),
    "right_216": (214.0, -1),
    "fwd_140":   (137.5, 0),
    "left_81":   (78.5, +1),
    "left_108":  (106.0, +1),
    "right_67":  (66.0, -1),
    "right_210": (209.0, -1),
}

FAILED = ("left_81", "left_108", "right_67", "right_210")
CONTROLS = ("left_65", "right_216", "fwd_140")


# ------------------------------------------------------------------ helpers
def dn_layout(brain):
    sc = np.asarray(brain.superclass)
    idx = np.where(sc == "descending_neuron")[0]
    ct = np.asarray(brain.cell_type)[idx]
    sd = np.asarray(brain.side)[idx]
    slot = np.full(int(sc.shape[0]), -1, dtype=np.int64)
    slot[idx] = np.arange(len(idx), dtype=np.int64)
    return idx, slot, ct, sd


def load_synthetic_polarity(path: Path) -> dict[str, int]:
    """canonical polarity: +1 if synthetic LEFT increased L-R diff, else -1."""
    out: dict[str, int] = {}
    if not path.exists():
        raise SystemExit(f"synthetic inject control missing: {path}\nRun scripts/p02a_inject_control.py first")
    for r in csv.DictReader(path.open()):
        out[r["cell_type"]] = 1 if float(r["defl_LEFT"]) > 0 else -1
    return out


def oriented_sign(value: float) -> int:
    if value > 0:
        return +1
    if value < 0:
        return -1
    return 0


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("-o", "--output", default="output/p02a_falsification")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    seeds = list(range(1, args.seeds + 1))
    polarity = load_synthetic_polarity(ROOT / "output/p02a_dn_trace/inject_control/families.csv")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = EMA_TAU
    dt = cfg.brain_dt

    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain
    idx, slot, ct_slot, sd_slot = dn_layout(brain)
    n_dn = len(idx)

    def fam_indices(ctype: str) -> tuple[list[int], list[int]]:
        il = [i for i in range(n_dn) if str(ct_slot[i]) == ctype and str(sd_slot[i]) == "L"]
        ir = [i for i in range(n_dn) if str(ct_slot[i]) == ctype and str(sd_slot[i]) == "R"]
        return il, ir

    quartet_idx = {t: fam_indices(t) for t in PRIMARY_QUARTET}
    crossref_idx = {t: fam_indices(t) for t in CROSSREF_TYPES}

    print(f"Frozen quartet: {PRIMARY_QUARTET}")
    print("Canonical polarity (from synthetic control, frozen before this run):")
    for t in PRIMARY_QUARTET:
        print(f"  {t:10s} polarity={polarity.get(t, 0):+d}  cells L/R={len(quartet_idx[t][0])}/{len(quartet_idx[t][1])}")

    # ---- cache frontend per event (seed-independent) ----
    stages: dict[str, dict] = {}
    for eid, (start, _vis) in EVENTS.items():
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain)
        frames: list[dict] = []
        inject_rows: list[dict[str, float]] = []
        yaws: list[float] = []
        for t, frame in iter_video_at_brain_hz(str(video), t0, t1, dt):
            eye, inject, metrics = enc.encode_frame(frame)
            frames.append({"t": t, "eye": eye, "inject": inject})
            inject_rows.append(metrics["inject"])
            yaws.append(float(metrics.get("yaw_frozen", 0.0)))
        pre_steps = int(round(PRE_ROLL_S / dt))
        pre_steps = min(pre_steps, max(0, len(frames) - 1))

        def side_sum(rows: list[dict], kinds: tuple[str, ...], side: str) -> float:
            return float(np.mean([sum(r.get(f"{k}_{side}", 0.0) for k in kinds) for r in rows]))

        t4_l = side_sum(inject_rows[pre_steps:], T4_TYPES, "L")
        t4_r = side_sum(inject_rows[pre_steps:], T4_TYPES, "R")
        t5_l = side_sum(inject_rows[pre_steps:], T5_TYPES, "L")
        t5_r = side_sum(inject_rows[pre_steps:], T5_TYPES, "R")

        stages[eid] = {
            "frames": frames,
            "pre_steps": pre_steps,
            "t4_lr": t4_l - t4_r,
            "t5_lr": t5_l - t5_r,
            "combined_lr": (t4_l + t5_l) - (t4_r + t5_r),
            "t_range": (t0, t1),
            "yaw_frozen_median": float(np.median(yaws[pre_steps:])) if len(yaws) > pre_steps else 0.0,
        }

    # yaw_frozen captured during the frontend pass above

    # ---- CNS replay, all seeds ----
    results: dict[str, dict] = {}
    for eid, (start, vis_sign) in EVENTS.items():
        frames = stages[eid]["frames"]
        pre_steps = stages[eid]["pre_steps"]
        ev_lo, ev_hi = pre_steps, len(frames)
        dn_diffs: dict[str, list[float]] = defaultdict(list)

        for seed in seeds:
            engine.reset(seed=seed)
            counts = np.zeros((n_dn, len(frames)), dtype=np.int32)
            for i, fr in enumerate(frames):
                res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
                fired = res.fired
                if len(fired):
                    hits = slot[fired]
                    hits = hits[hits >= 0]
                    if len(hits):
                        np.add.at(counts[:, i], hits, 1)
            span = max(ev_hi - ev_lo, 1) * dt
            for ctype in PRIMARY_QUARTET + CROSSREF_TYPES:
                il, ir = (quartet_idx if ctype in PRIMARY_QUARTET else crossref_idx)[ctype]
                if not il and not ir:
                    continue
                l_rate = counts[il, ev_lo:ev_hi].sum() / span if il else 0.0
                r_rate = counts[ir, ev_lo:ev_hi].sum() / span if ir else 0.0
                dn_diffs[ctype].append(float(l_rate) - float(r_rate))

        results[eid] = {
            "expected_visual_sign": vis_sign,
            "yaw_frozen_median": stages[eid]["yaw_frozen_median"],
            "yaw_oriented": -oriented_sign(stages[eid]["yaw_frozen_median"]),
            "t4_lr": stages[eid]["t4_lr"],
            "t5_lr": stages[eid]["t5_lr"],
            "combined_inject_lr": stages[eid]["combined_lr"],
            "inject_oriented": oriented_sign(stages[eid]["combined_lr"]),
            "dn_diffs": {k: v for k, v in dn_diffs.items()},
        }

    # ---- assemble rows ----
    rows: list[dict] = []
    for eid, (start, vis_sign) in EVENTS.items():
        r = results[eid]
        row = {
            "event": eid,
            "group": "failed" if eid in FAILED else "control",
            "t_start": start,
            "expected_visual_sign": vis_sign,
            "yaw_frozen": round(r["yaw_frozen_median"], 5),
            "yaw_oriented": r["yaw_oriented"],
            "yaw_ok": r["yaw_oriented"] == vis_sign,
            "inject_T4_LR": round(r["t4_lr"], 5),
            "inject_T5_LR": round(r["t5_lr"], 5),
            "inject_combined_LR": round(r["combined_inject_lr"], 5),
            "inject_oriented": r["inject_oriented"],
            "inject_ok": r["inject_oriented"] == vis_sign,
        }
        # primary quartet consensus
        quartet_signs: list[int] = []
        for t in PRIMARY_QUARTET:
            diffs = np.array(r["dn_diffs"].get(t, []))
            med = float(np.median(diffs)) if len(diffs) else 0.0
            sgn = oriented_sign(med) * polarity.get(t, 1)
            row[f"{t}_dn_LR"] = round(med, 4)
            row[f"{t}_dn_oriented"] = sgn
            row[f"{t}_dn_seed_signs"] = "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in diffs)
            row[f"{t}_dn_consistency"] = float(np.mean([np.sign(d) == np.sign(med) for d in diffs])) if len(diffs) else 0.0
            if sgn != 0:
                quartet_signs.append(sgn)
        consensus = 0
        if quartet_signs:
            pos = sum(1 for s in quartet_signs if s > 0)
            neg = sum(1 for s in quartet_signs if s < 0)
            consensus = +1 if pos > neg else (-1 if neg > pos else 0)
        row["dn_consensus_oriented"] = consensus
        row["dn_ok"] = consensus == vis_sign
        rows.append(row)

    # ---- write CSVs ----
    with (out / "causal_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-DN detail (quartet + crossref)
    detail: list[dict] = []
    for eid in EVENTS:
        r = results[eid]
        for ctype, diffs in r["dn_diffs"].items():
            arr = np.array(diffs)
            med = float(np.median(arr)) if len(arr) else 0.0
            pol = polarity.get(ctype, 1)
            detail.append({
                "event": eid,
                "group": "failed" if eid in FAILED else "control",
                "expected_visual_sign": r["expected_visual_sign"],
                "in_primary_quartet": ctype in PRIMARY_QUARTET,
                "cell_type": ctype,
                "canonical_polarity": pol,
                "dn_LR_median": round(med, 5),
                "dn_LR_mean": round(float(np.mean(arr)), 5) if len(arr) else 0.0,
                "dn_LR_std": round(float(np.std(arr)), 5) if len(arr) else 0.0,
                "dn_oriented": oriented_sign(med) * pol,
                "seed_signs": "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in diffs),
                "seed_consistency": round(float(np.mean([np.sign(d) == np.sign(med) for d in arr])), 3) if len(arr) else 0.0,
            })
    with (out / "dn_detail.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    # ---- verdict ----
    failed_rows = [r for r in rows if r["group"] == "failed"]
    yaw_hits = sum(1 for r in failed_rows if r["yaw_ok"])
    inj_hits = sum(1 for r in failed_rows if r["inject_ok"])
    dn_hits = sum(1 for r in failed_rows if r["dn_ok"])
    dn_beats_yaw = dn_hits - yaw_hits
    dn_follows_inject = sum(1 for r in failed_rows if r["dn_consensus_oriented"] == r["inject_oriented"])

    if dn_beats_yaw >= 2:
        scenario = "B — MaleCNS denoises: DN sign better than yaw_frozen on >=2/4 failed events"
    elif dn_follows_inject == len(failed_rows):
        scenario = "A — MaleCNS is faithful: DN sign == inject sign on 4/4 failed events (bottleneck is upstream)"
    else:
        scenario = "A' — MaleCNS reproduces inject (no denoising); bottleneck upstream"

    # seed consistency for quartet on failed events
    seed_cons = []
    for r in failed_rows:
        for t in PRIMARY_QUARTET:
            seed_cons.append(r[f"{t}_dn_consistency"])
    mean_seed_cons = float(np.mean(seed_cons)) if seed_cons else 0.0

    report = {
        "probe": "P0.2A follow-up falsification test (single run, frozen)",
        "frozen": {
            "frontend": "unchanged",
            "malecns": "unchanged",
            "primary_quartet": list(PRIMARY_QUARTET),
            "polarity_source": "synthetic inject control (never real events)",
            "ema_tau": EMA_TAU,
            "pre_roll_s": PRE_ROLL_S,
            "event_dur_s": EVENT_DUR_S,
            "seeds": seeds,
        },
        "causal_table": rows,
        "failed_event_accuracy": {
            "yaw_frozen": f"{yaw_hits}/{len(failed_rows)}",
            "inject": f"{inj_hits}/{len(failed_rows)}",
            "DN_consensus": f"{dn_hits}/{len(failed_rows)}",
        },
        "dn_minus_yaw_hits": dn_beats_yaw,
        "dn_equals_inject_count": f"{dn_follows_inject}/{len(failed_rows)}",
        "mean_seed_consistency_quartet_failed": round(mean_seed_cons, 3),
        "scenario": scenario,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ---- console ----
    def s(sign: int) -> str:
        return {1: "+", -1: "-", 0: "0"}[sign]

    print("\n=== CAUSAL TABLE (all signs oriented: LEFT=+, RIGHT=-, FWD=0) ===")
    print(f"{'event':10s} {'grp':7s} {'exp':>4s} {'yaw_f':>7s} {'yaw':>4s} {'T4_LR':>8s} {'T5_LR':>8s} {'inj':>4s} {'DN':>4s}")
    for r in rows:
        print(
            f"{r['event']:10s} {r['group']:7s} {s(r['expected_visual_sign']):>4s} "
            f"{r['yaw_frozen']:+7.4f} {s(r['yaw_oriented']):>4s} "
            f"{r['inject_T4_LR']:+8.4f} {r['inject_T5_LR']:+8.4f} "
            f"{s(r['inject_oriented']):>4s} {s(r['dn_consensus_oriented']):>4s}"
        )

    print("\n=== PER-DN (failed events) ===")
    for r in failed_rows:
        parts = []
        for t in PRIMARY_QUARTET:
            parts.append(f"{t}={s(r[f'{t}_dn_oriented'])}({r[f'{t}_dn_consistency']:.1f})")
        print(f"  {r['event']:10s} exp={s(r['expected_visual_sign'])}  " + "  ".join(parts))

    print("\n=== FAILED-EVENT ACCURACY ===")
    print(f"  yaw_frozen:   {yaw_hits}/{len(failed_rows)}")
    print(f"  inject:       {inj_hits}/{len(failed_rows)}")
    print(f"  DN consensus: {dn_hits}/{len(failed_rows)}")
    print(f"  DN == inject: {dn_follows_inject}/{len(failed_rows)}")
    print(f"  quartet seed consistency (failed): {mean_seed_cons:.2f}")
    print(f"\nSCENARIO: {scenario}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
