#!/usr/bin/env python3
"""
P0.1J GATE — does the connectome derive yaw direction from local motion alone?

The encoder no longer computes a LEFT/RIGHT decision. It injects only a local
opponent motion field into T4a/b and T5a/b (see fly_vo/local_motion_inject.py).
This gate asks what the brain does with it:

  A  injection invariants (algebra)
  B  synthetic injection sanity      — opponent index tracks motion direction
  C  synthetic DN selectivity        — does any DN separate LEFT from RIGHT?
  D  real events                     — left_65 / right_216 / fwd_140 + failed four
  E  verdict                         — emergent yaw selectivity, or not

Reported separately, and never conflated:
  yaw_frozen            the old pooled observer (logged, not injected)
  opponent index        our local motion code entering the brain
  DN response           what the connectome produced

Usage:
    PYTHONPATH=. python scripts/p01j_gate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import FlowConfig
from fly_vo.p01i_probe import (
    capture_event,
    dn_layout,
    fam_indices,
    oriented_sign,
    replay_dn,
)
from fly_vo.visual_encoder import VideoVisualEncoder
from scripts.visual_observability_test import _make_frames

SYNTH_CONDITIONS = {"YAW_LEFT": +1, "FORWARD": 0, "YAW_RIGHT": -1}
SYNTH_STEPS = 90
RATE_WINDOW = 24

REAL_EVENTS = {
    "left_65":   (63.5, +1, "control"),
    "right_216": (214.0, -1, "control"),
    "fwd_140":   (137.5, 0, "neutral"),
    "left_81":   (78.5, +1, "failed"),
    "left_108":  (106.0, +1, "failed"),
    "right_67":  (66.0, -1, "failed"),
    "right_210": (209.0, -1, "failed"),
}

SEP_MIN = 2.0
CONSISTENCY_MIN = 0.8


# ------------------------------------------------------------------ helpers
def _robust_sigma(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _sep(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt((_robust_sigma(a) ** 2 + _robust_sigma(b) ** 2) / 2.0)
    if s < 1e-9:
        return 0.0
    return abs(float(np.median(a)) - float(np.median(b))) / s


# ------------------------------------------------------------------ A
def section_a() -> dict:
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests/test_local_opponent_invariants.py")],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    lines = (r.stdout or r.stderr).strip().splitlines()
    print("\n=== A. Injection invariants ===")
    for ln in lines:
        print(f"  {ln}")
    return {"pass": r.returncode == 0, "output": lines}


# ------------------------------------------------------------------ B
def run_synthetic(seeds: list[int]) -> dict:
    cfg = FlyVOConfig(use_pathway_fallback=False)
    fc = FlowConfig(crop_mode="center_band", flow_inject_gain=8.0, ema_tau_s=0.5)
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=fc)
    engine._ensure_loaded()

    _idx, slot, ct_slot, sd_slot = dn_layout(engine.brain)
    n_dn = len(ct_slot)
    synth = _make_frames(n=SYNTH_STEPS)

    out: dict[str, dict] = {}
    for cond in SYNTH_CONDITIONS:
        frames_bgr = synth[cond]
        opp, yaw, dn_counts = [], [], []
        for seed in seeds:
            encoder.reset()
            engine.reset(seed=seed)
            inj_lr, yaws = [], []
            counts = np.zeros((n_dn, len(frames_bgr)), dtype=np.int32)
            for i, bgr in enumerate(frames_bgr):
                eye, inject, metrics = encoder.encode_frame(bgr)
                inj_lr.append(float(metrics["local_opponent_index"]))
                yaws.append(float(metrics["yaw_frozen"]))
                res = engine.step(i * cfg.brain_dt, eye_drive=eye, inject=inject)
                fired = res.fired
                if len(fired):
                    hits = slot[fired]
                    hits = hits[hits >= 0]
                    if len(hits):
                        np.add.at(counts[:, i], hits, 1)
            opp.append(float(np.mean(inj_lr[-RATE_WINDOW:])))
            yaw.append(float(np.mean(yaws[-RATE_WINDOW:])))
            dn_counts.append(counts)
        lo, hi = len(frames_bgr) - RATE_WINDOW, len(frames_bgr)
        span = (hi - lo) * cfg.brain_dt
        rates = np.stack([c[:, lo:hi].sum(axis=1) / span for c in dn_counts])  # [seeds, n_dn]
        out[cond] = {
            "opponent_index": opp,
            "yaw_frozen": yaw,
            "dn_rates": rates,
            "dn_counts": dn_counts,
        }
    return {"result": out, "n_dn": n_dn, "ct": ct_slot, "sd": sd_slot, "config": cfg}


# ------------------------------------------------------------------ C
def section_c(syn: dict, seeds: list[int]) -> dict:
    res = syn["result"]
    ct, sd, n_dn = syn["ct"], syn["sd"], syn["n_dn"]

    print("\n=== B. Synthetic injection sanity ===")
    opp = {c: float(np.mean(res[c]["opponent_index"])) for c in SYNTH_CONDITIONS}
    yaw = {c: float(np.mean(res[c]["yaw_frozen"])) for c in SYNTH_CONDITIONS}
    for c in SYNTH_CONDITIONS:
        print(f"  {c:10s} opponent_index={opp[c]:+.4f}   yaw_frozen={yaw[c]:+.4f}")

    # The opponent code must (i) agree in sign with the frozen frontend's own
    # convention, (ii) be antisymmetric between LEFT and RIGHT, (iii) be near zero
    # when there is no rotation. The absolute sign is a stated convention, so it is
    # checked against yaw_frozen rather than hardcoded here.
    agree_left = oriented_sign(opp["YAW_LEFT"]) == oriented_sign(yaw["YAW_LEFT"])
    agree_right = oriented_sign(opp["YAW_RIGHT"]) == oriented_sign(yaw["YAW_RIGHT"])
    antisym = oriented_sign(opp["YAW_LEFT"]) == -oriented_sign(opp["YAW_RIGHT"])
    fwd_small = abs(opp["FORWARD"]) < 0.25 * min(abs(opp["YAW_LEFT"]), abs(opp["YAW_RIGHT"]))
    b_ok = agree_left and agree_right and antisym and fwd_small
    print(
        f"  [{'PASS' if b_ok else 'FAIL'}] tracks motion direction "
        f"(agree_L={agree_left} agree_R={agree_right} antisym={antisym} fwd_near_zero={fwd_small})"
    )

    # ---- per-DN-cell yaw selectivity: rate_LEFT vs rate_RIGHT ----
    L = res["YAW_LEFT"]["dn_rates"]
    R = res["YAW_RIGHT"]["dn_rates"]
    F = res["FORWARD"]["dn_rates"]

    print("\n=== C. Synthetic DN yaw selectivity (LEFT vs RIGHT) ===")
    rows = []
    for i in range(n_dn):
        lo, hi = L[:, i], R[:, i]
        sep = _sep(lo, hi)
        cons = float(np.mean(np.sign(lo - hi))) if len(lo) == len(hi) else 0.0
        m_l, m_r, m_f = float(np.median(lo)), float(np.median(hi)), float(np.median(F[:, i]))
        yaw_index = (m_l - m_r) / (m_l + m_r + 1e-9)
        rows.append({
            "cell_index": int(i),
            "cell_type": str(ct[i]),
            "side": str(sd[i]),
            "rate_LEFT": m_l,
            "rate_FWD": m_f,
            "rate_RIGHT": m_r,
            "yaw_index": yaw_index,
            "sep_LEFT_RIGHT": round(sep, 3),
            "consistency": round(cons, 3),
            "seed_signs": "".join("+" if v > 0 else ("-" if v < 0 else "0") for v in (lo - hi)),
        })

    cand = [
        r for r in rows
        if r["sep_LEFT_RIGHT"] >= SEP_MIN and r["consistency"] >= CONSISTENCY_MIN
    ]
    cand.sort(key=lambda r: -r["sep_LEFT_RIGHT"])
    print(f"  DN cells with LEFT/RIGHT separation >= {SEP_MIN} and consistency >= {CONSISTENCY_MIN}: {len(cand)} / {n_dn}")
    for r in cand[:12]:
        print(
            f"    {r['cell_type']:12s} {r['side']}  L={r['rate_LEFT']:7.2f} F={r['rate_FWD']:7.2f} "
            f"R={r['rate_RIGHT']:7.2f}  yaw_idx={r['yaw_index']:+.3f} sep={r['sep_LEFT_RIGHT']:.2f} {r['seed_signs']}"
        )

    # family-level (both sides pooled)
    fam: dict[str, list[int]] = {}
    for i in range(n_dn):
        fam.setdefault(str(ct[i]), []).append(i)
    fam_rows = []
    for ctype, idxs in fam.items():
        lo = L[:, idxs].sum(axis=1)
        hi_ = R[:, idxs].sum(axis=1)
        fo = F[:, idxs].sum(axis=1)
        sep = _sep(lo, hi_)
        cons = float(np.mean(np.sign(lo - hi_)))
        m_l, m_r = float(np.median(lo)), float(np.median(hi_))
        fam_rows.append({
            "cell_type": ctype,
            "n_cells": len(idxs),
            "rate_LEFT": m_l,
            "rate_FWD": float(np.median(fo)),
            "rate_RIGHT": m_r,
            "yaw_index": (m_l - m_r) / (m_l + m_r + 1e-9),
            "sep_LEFT_RIGHT": round(sep, 3),
            "consistency": round(cons, 3),
        })
    fam_cand = [r for r in fam_rows if r["sep_LEFT_RIGHT"] >= SEP_MIN and r["consistency"] >= CONSISTENCY_MIN]
    fam_cand.sort(key=lambda r: -r["sep_LEFT_RIGHT"])
    print(f"\n  DN *types* passing the same criterion: {len(fam_cand)} / {len(fam_rows)}")
    for r in fam_cand[:12]:
        print(
            f"    {r['cell_type']:14s} n={r['n_cells']:3d}  L={r['rate_LEFT']:7.2f} F={r['rate_FWD']:7.2f} "
            f"R={r['rate_RIGHT']:7.2f}  yaw_idx={r['yaw_index']:+.3f} sep={r['sep_LEFT_RIGHT']:.2f}"
        )

    # ---- permutation null: how many types pass by chance? ----
    # Each seed is an independent run, so randomly swapping the LEFT/RIGHT label
    # per seed gives a valid null for the across-seed comparison. The full
    # criterion (separation AND consistency) is permuted, not just separation.
    rng = np.random.default_rng(0)
    n_perm = 200
    null_counts = []
    n_cells_passing_obs = 0
    for i in range(n_dn):
        seps = np.asarray(rows[i]["sep_LEFT_RIGHT"])
        cons = np.asarray(rows[i]["consistency"])
        if seps >= SEP_MIN and cons >= CONSISTENCY_MIN:
            n_cells_passing_obs += 1

    for p in range(n_perm):
        swap = rng.random(len(seeds)) < 0.5
        lp = L.copy()
        rp = R.copy()
        lp[swap], rp[swap] = R[swap], L[swap]
        cnt = 0
        for i in range(n_dn):
            if _sep(lp[:, i], rp[:, i]) < SEP_MIN:
                continue
            if float(np.mean(np.sign(lp[:, i] - rp[:, i]))) >= CONSISTENCY_MIN:
                cnt += 1
        null_counts.append(cnt)
    null_counts_arr = np.asarray(null_counts)
    p_value = float(np.mean(null_counts_arr >= n_cells_passing_obs))
    print(
        f"\n  permutation null (200 draws, full criterion): "
        f"mean passing = {null_counts_arr.mean():.2f} "
        f"(95th pct {np.percentile(null_counts_arr, 95):.0f}), "
        f"observed = {n_cells_passing_obs}, p = {p_value:.3f}"
    )

    return {
        "b_pass": b_ok,
        "opponent_index": opp,
        "yaw_frozen": yaw,
        "dn_cells": rows,
        "dn_families": fam_rows,
        "n_dn_cells_passing": len(cand),
        "n_dn_types_passing": len(fam_cand),
        "permutation_null": {
            "n_perm": n_perm,
            "mean_passing": float(null_counts_arr.mean()),
            "p95_passing": float(np.percentile(null_counts_arr, 95)),
            "observed_passing": n_cells_passing_obs,
            "p_value": p_value,
        },
        "top_cells": cand[:25],
        "top_types": fam_cand[:25],
    }


def section_drive_check(seeds: list[int], syn: dict) -> dict:
    """Is the failure amplitude or selectivity? Compare DN drive vs no inject."""
    cfg = syn["config"]
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    _idx, slot, ct_slot, sd_slot = dn_layout(engine.brain)
    dn_idx = np.flatnonzero(np.asarray(ct_slot) == "DNp01")

    fc = FlowConfig(crop_mode="center_band", flow_inject_gain=8.0, ema_tau_s=0.5)
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=fc)
    frames = _make_frames(n=SYNTH_STEPS)["YAW_LEFT"]
    warm = SYNTH_STEPS - RATE_WINDOW

    print("\n=== D0. Does the drive reach the DNs at all? (synthetic YAW_LEFT) ===")
    out = {}
    for mode in ("no_inject", "local_opponent"):
        total = 0
        for seed in seeds:
            encoder.reset()
            engine.reset(seed=seed)
            for i, bgr in enumerate(frames):
                eye, inject, _m = encoder.encode_frame(bgr)
                res = engine.step(
                    i * cfg.brain_dt,
                    eye_drive=eye,
                    inject=(inject if mode == "local_opponent" else []),
                )
                if i < warm:
                    continue
                fired = set(np.asarray(res.fired).tolist())
                total += sum(1 for j in dn_idx if j in fired)
        span = RATE_WINDOW * len(seeds) * cfg.brain_dt
        out[mode] = total / span
        print(f"  DNp01 rate [{mode:14s}] = {out[mode]:7.2f} Hz")
    out["ratio"] = out["local_opponent"] / max(out["no_inject"], 1e-9)
    print(f"  drive reaches DNs: {'YES' if out['ratio'] > 5 else 'NO'} ({out['ratio']:.0f}x over baseline)")
    return out


# ------------------------------------------------------------------ D
def section_d(seeds: list[int], video: Path, syn: dict) -> dict:
    cfg = syn["config"]
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    _idx, slot, ct_slot, sd_slot = dn_layout(engine.brain)
    n_dn = len(ct_slot)

    # Follow the DN types that showed synthetic yaw selectivity, when there are any.
    top_types = [r["cell_type"] for r in syn.get("top_types_by_sep", [])][:6]

    print("\n=== D. Real events ===")
    rows = []
    for eid, (start, vis, group) in REAL_EVENTS.items():
        cap = capture_event(video, start, cfg)
        frames = cap["frames"]
        ev_lo, ev_hi = cap["n_pre"], len(frames)
        span = max(ev_hi - ev_lo, 1) * dt
        counts = [
            replay_dn(engine, frames, slot, n_dn, seed) for seed in seeds
        ]
        rates = np.stack([c[:, ev_lo:ev_hi].sum(axis=1) / span for c in counts])  # [seeds, n_dn]
        opp = float(np.median([f["metrics"].get("local_opponent_index", 0.0) for f in frames[ev_lo:]]))
        yaw = cap["yaw_frozen"]

        row = {
            "event": eid,
            "group": group,
            "expected_visual_sign": vis,
            "yaw_frozen": round(yaw, 5),
            "opponent_index": round(opp, 5),
            "opponent_sign": oriented_sign(opp),
            "dn_total_rate": float(np.median(rates.sum(axis=1))),
        }
        for t in top_types:
            il, ir = fam_indices(ct_slot, sd_slot, t)
            if not il and not ir:
                continue
            r = rates[:, il].sum(axis=1) if il else np.zeros(len(seeds))
            row[f"{t}_rate"] = round(float(np.median(r)), 4)

        # Lateral differential of the previously-lateralised quartet. With no
        # left/right asymmetry injected, any residual must come from the connectome.
        for t in ("DNp01", "DNp04"):
            il, ir = fam_indices(ct_slot, sd_slot, t)
            if not il or not ir:
                continue
            l = rates[:, il].sum(axis=1)
            r = rates[:, ir].sum(axis=1)
            d = l - r
            row[f"{t}_LR"] = round(float(np.median(d)), 4)
            row[f"{t}_LR_seed_signs"] = "".join(
                "+" if v > 0 else ("-" if v < 0 else "0") for v in d
            )
            row[f"{t}_L"] = round(float(np.median(l)), 4)
            row[f"{t}_R"] = round(float(np.median(r)), 4)

        rows.append(row)
        print(
            f"  {eid:10s} {group:8s} yaw={yaw:+.4f} opp={opp:+.4f} "
            f"dn_total={row['dn_total_rate']:8.1f}  "
            f"DNp01_LR={row.get('DNp01_LR', 0):+7.3f}  DNp04_LR={row.get('DNp04_LR', 0):+7.3f}"
        )

    return {"rows": rows, "top_types_used": top_types}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("-o", "--output", default="output/p01j_gate")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    print("P0.1J GATE — connectome-derived yaw from local opponent motion")

    a = section_a()
    syn = run_synthetic(args.seeds)
    c = section_c(syn, args.seeds)
    # feed the selected types into section D
    syn["top_types_by_sep"] = c["top_types"]
    d0 = section_drive_check(args.seeds, syn)
    d = section_d(args.seeds, video, syn)

    # ---- purity check: nothing pooled enters the brain ----
    pur = (ROOT / "fly_vo/local_motion_inject.py").read_text(encoding="utf-8")
    purity_ok = all(
        k not in pur.split('"""')[2] if len(pur.split('"""')) > 2 else True
        for k in ("yaw_frozen",)
    )

    report = {
        "probe": "P0.1J — connectome-derived yaw from local opponent motion",
        "what_changed": "fly_vo/visual_encoder.py + fly_vo/local_motion_inject.py (injection only)",
        "not_changed": [
            "optic_flow.py", "frozen readout", "tau", "detector weights",
            "blind labels", "trajectory", "dopamine", "motion_readout", "MaleCNS dynamics",
        ],
        "seeds": args.seeds,
        "A_invariants": a["pass"],
        "B_injection_sanity": c["b_pass"],
        "C_synthetic": {
            "n_dn_cells_passing": c["n_dn_cells_passing"],
            "n_dn_types_passing": c["n_dn_types_passing"],
            "top_cells": c["top_cells"],
            "top_types": c["top_types"],
        },
        "D0_drive_reaches_dns": d0,
        "D_real_events": d["rows"],
        "D_types_followed": d["top_types_used"],
        "A_output": a["output"],
    }

    with (out / "dn_synthetic_selectivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(c["dn_cells"][0].keys()))
        w.writeheader()
        w.writerows(c["dn_cells"])
    with (out / "dn_types_selectivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(c["dn_families"][0].keys()))
        w.writeheader()
        w.writerows(c["dn_families"])
    with (out / "real_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d["rows"][0].keys()))
        w.writeheader()
        w.writerows(d["rows"])

    # ---- verdict ----
    perm = c["permutation_null"]
    significant = perm["p_value"] < 0.05
    robust = significant and perm["observed_passing"] >= 3
    if robust:
        verdict = (
            f"ESTABLISHED YAW SELECTIVITY: {perm['observed_passing']} DN cells separate LEFT from "
            f"RIGHT beyond chance (permutation p={perm['p_value']:.3f})"
        )
    elif significant:
        verdict = (
            f"MARGINAL: only {perm['observed_passing']} of {syn['n_dn']} DN cells pass "
            f"(permutation p={perm['p_value']:.3f}), on low spike counts. Not enough to claim a "
            "working yaw readout."
        )
    else:
        verdict = (
            f"NO ESTABLISHED YAW SELECTIVITY: {perm['observed_passing']} DN cells pass the screen, "
            f"which is at chance level (null mean {perm['mean_passing']:.2f}, "
            f"permutation p={perm['p_value']:.3f}). The connectome did not turn local opponent "
            "motion into a LEFT/RIGHT signal at the descending-neuron level."
        )
    report["verdict"] = verdict
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== GATE ===")
    print(f"  [{'PASS' if a['pass'] else 'FAIL'}] A injection invariants")
    print(f"  [{'PASS' if c['b_pass'] else 'FAIL'}] B synthetic injection sanity")
    print(f"  [{'PASS' if d0['ratio'] > 5 else 'FAIL'}] D0 drive reaches DNs "
          f"(DNp01 {d0['no_inject']:.2f} -> {d0['local_opponent']:.2f} Hz, {d0['ratio']:.0f}x)")
    print(f"  [{'YES' if significant else 'NO '}] C emergent DN yaw selectivity "
          f"(p={c['permutation_null']['p_value']:.3f}, n={c['permutation_null']['observed_passing']})")
    print(f"\n{verdict}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
