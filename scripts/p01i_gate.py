#!/usr/bin/env python3
"""
P0.1I GATE — isolate the signed yaw injection path.

PASS here means fidelity along

    yaw_frozen  ->  inject  ->  DNp01 / DNp04

It does NOT mean visual-label accuracy, and it does NOT close P0.1R+.
Events whose yaw_frozen is already wrong must stay wrong downstream — that is
the correct outcome, and it localises the remaining error to video -> yaw_frozen.

Sections:
  A  algebra invariants (tests/test_inject_invariants.py)
  B  synthetic encoder-realistic smoke test (frames -> _build_inject -> CNS)
  C  real positive controls: left_65, right_216
  D  real neutral control:   fwd_140
  E  full causal table over the frozen seven events (labels unchanged)
  F  fidelity rates: inject==yaw, DNp01==inject, DNp04==inject

Usage:
    PYTHONPATH=. python scripts/p01i_gate.py
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
from fly_vo.p01i_probe import (
    capture_event,
    dn_layout,
    fam_indices,
    load_canonical_polarity,
    oriented_sign,
    replay_dn,
)
from scripts.p01i_smoke_test import evaluate_synthetic, run_synthetic

# Frozen event set — blind labels, unchanged.
EVENTS = {
    "left_65":   (63.5, +1, "control"),
    "right_216": (214.0, -1, "control"),
    "fwd_140":   (137.5, 0, "neutral"),
    "left_81":   (78.5, +1, "failed"),
    "left_108":  (106.0, +1, "failed"),
    "right_67":  (66.0, -1, "failed"),
    "right_210": (209.0, -1, "failed"),
}
QUARTET = ("DNp01", "DNp04")
NEUTRAL_RATIO = 0.25
POLARITY_CSV = ROOT / "output/p02a_dn_trace/inject_control/families.csv"


def _sign_str(s: int) -> str:
    return {1: "+", -1: "-", 0: "0"}[s]


def section_a() -> dict:
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests/test_inject_invariants.py")],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    tail = (r.stdout or r.stderr).strip().splitlines()
    ok = r.returncode == 0
    print("\n=== A. Algebra invariants ===")
    for line in tail:
        print(f"  {line}")
    return {"pass": ok, "output": tail}


def section_b(seeds: list[int]) -> dict:
    print("\n=== B. Synthetic smoke (encoder-realistic) ===")
    res = run_synthetic(seeds)
    ev = evaluate_synthetic(res)
    for k, v in ev["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    m = ev["table"]["inject"]["median"]
    print(f"  inject medians: LEFT={m['YAW_LEFT']:+.4f}  FWD={m['FORWARD']:+.4f}  RIGHT={m['YAW_RIGHT']:+.4f}")
    for t in QUARTET:
        mt = ev["table"][t]["median"]
        print(f"  {t:8s} medians: LEFT={mt['YAW_LEFT']:+.3f}  FWD={mt['FORWARD']:+.3f}  RIGHT={mt['YAW_RIGHT']:+.3f}")
    return {"pass": bool(ev["checks"]["ALL"]), **ev}


def section_cde(seeds: list[int], video: Path) -> dict:
    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    dt = cfg.brain_dt

    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    _idx, slot, ct_slot, sd_slot = dn_layout(engine.brain)
    n_dn = len(ct_slot)
    fams = {t: fam_indices(ct_slot, sd_slot, t) for t in QUARTET}
    polarity = load_canonical_polarity(POLARITY_CSV)

    rows: list[dict] = []
    detail: list[dict] = []
    for eid, (start, vis_sign, group) in EVENTS.items():
        cap = capture_event(video, start, cfg)
        frames = cap["frames"]
        ev_lo, ev_hi = cap["n_pre"], len(frames)

        dn_diffs: dict[str, list[float]] = {t: [] for t in QUARTET}
        for seed in seeds:
            counts = replay_dn(engine, frames, slot, n_dn, seed)
            for t in QUARTET:
                il, ir = fams[t]
                span = max(ev_hi - ev_lo, 1) * dt
                l = float(counts[il, ev_lo:ev_hi].sum()) / span if il else 0.0
                r = float(counts[ir, ev_lo:ev_hi].sum()) / span if ir else 0.0
                dn_diffs[t].append(l - r)

        yaw_oriented = -oriented_sign(cap["yaw_frozen"])
        inj_oriented = oriented_sign(cap["combined_lr"])

        row = {
            "event": eid,
            "group": group,
            "expected_visual_sign": vis_sign,
            "yaw_frozen": round(cap["yaw_frozen"], 5),
            "yaw_oriented": yaw_oriented,
            "inject_T4_LR": round(cap["t4_lr"], 5),
            "inject_T5_LR": round(cap["t5_lr"], 5),
            "inject_combined_LR": round(cap["combined_lr"], 5),
            "inject_scene_common": round(cap["scene_common"], 5),
            "inject_oriented": inj_oriented,
            "inject_matches_yaw": inj_oriented == yaw_oriented,
            "inject_matches_label": inj_oriented == vis_sign,
        }
        for t in QUARTET:
            arr = np.array(dn_diffs[t])
            med = float(np.median(arr)) if len(arr) else 0.0
            pol = polarity.get(t, 1)
            row[f"{t}_dn_LR"] = round(med, 4)
            row[f"{t}_dn_oriented"] = oriented_sign(med) * pol
            row[f"{t}_matches_inject"] = (oriented_sign(med) * pol) == inj_oriented
            row[f"{t}_seed_signs"] = "".join(_sign_str(oriented_sign(v)) for v in arr)
            row[f"{t}_seed_consistency"] = round(
                float(np.mean([np.sign(v) == np.sign(med) for v in arr])), 3
            ) if len(arr) else 0.0
        rows.append(row)

        for t in QUARTET:
            arr = np.array(dn_diffs[t])
            detail.append({
                "event": eid,
                "group": group,
                "cell_type": t,
                "canonical_polarity": polarity.get(t, 1),
                "n_L": len(fams[t][0]),
                "n_R": len(fams[t][1]),
                "dn_LR_median": round(float(np.median(arr)), 5) if len(arr) else 0.0,
                "dn_LR_std": round(float(np.std(arr)), 5) if len(arr) else 0.0,
                "seed_signs": "".join(_sign_str(oriented_sign(v)) for v in arr),
                "seed_consistency": round(float(np.mean([np.sign(v) == np.sign(np.median(arr)) for v in arr])), 3) if len(arr) else 0.0,
            })

    # ---- C: positive controls ----
    print("\n=== C. Real positive controls ===")
    c_ok = True
    for r in rows:
        if r["group"] != "control":
            continue
        good = r["inject_matches_yaw"] and r["inject_matches_label"] and all(
            r[f"{t}_matches_inject"] for t in QUARTET
        )
        c_ok = c_ok and good
        print(
            f"  [{'PASS' if good else 'FAIL'}] {r['event']:10s} label={_sign_str(r['expected_visual_sign'])} "
            f"yaw={_sign_str(r['yaw_oriented'])} inject={_sign_str(r['inject_oriented'])} "
            f"DNp01={_sign_str(r['DNp01_dn_oriented'])} DNp04={_sign_str(r['DNp04_dn_oriented'])}"
        )

    # ---- D: neutral ----
    print("\n=== D. Real neutral control (relative) ===")
    by = {r["event"]: r for r in rows}
    ctrl_mags = [abs(by["left_65"]["inject_combined_LR"]), abs(by["right_216"]["inject_combined_LR"])]
    fwd_mag = abs(by["fwd_140"]["inject_combined_LR"])
    d_ok = fwd_mag < NEUTRAL_RATIO * min(ctrl_mags)
    print(f"  |inject| FWD={fwd_mag:.4f}  LEFT={ctrl_mags[0]:.4f}  RIGHT={ctrl_mags[1]:.4f}")
    print(f"  ratio FWD/min(turn) = {fwd_mag / max(min(ctrl_mags), 1e-12):.3f}  (need < {NEUTRAL_RATIO})")
    print(f"  [{'PASS' if d_ok else 'FAIL'}] neutral substantially smaller than both turns")

    # ---- E: causal table ----
    print("\n=== E. Causal table (labels unchanged) ===")
    print(f"{'event':10s} {'grp':8s} {'label':>5s} {'yaw':>5s} {'inject':>7s} {'DNp01':>6s} {'DNp04':>6s}")
    for r in rows:
        print(
            f"{r['event']:10s} {r['group']:8s} {_sign_str(r['expected_visual_sign']):>5s} "
            f"{_sign_str(r['yaw_oriented']):>5s} {_sign_str(r['inject_oriented']):>7s} "
            f"{_sign_str(r['DNp01_dn_oriented']):>6s} {_sign_str(r['DNp04_dn_oriented']):>6s}   "
            f"({r['inject_combined_LR']:+.4f})"
        )

    # ---- F: fidelity ----
    inject_vs_yaw = sum(1 for r in rows if r["inject_matches_yaw"])
    dn01_vs_inject = sum(1 for r in rows if r["DNp01_matches_inject"])
    dn04_vs_inject = sum(1 for r in rows if r["DNp04_matches_inject"])
    n = len(rows)
    inject_vs_yaw_rate = inject_vs_yaw / n
    dn01_rate = dn01_vs_inject / n
    dn04_rate = dn04_vs_inject / n
    f_ok = inject_vs_yaw_rate >= 0.95 and dn01_rate >= 0.95 and dn04_rate >= 0.95

    print("\n=== F. Fidelity ===")
    print(f"  inject == yaw_frozen       {inject_vs_yaw}/{n}  ({inject_vs_yaw_rate:.0%})")
    print(f"  DNp01  == inject           {dn01_vs_inject}/{n}  ({dn01_rate:.0%})")
    print(f"  DNp04  == inject           {dn04_vs_inject}/{n}  ({dn04_rate:.0%})")
    print(f"  [{'PASS' if f_ok else 'FAIL'}] fidelity >= 95% on all three")

    label_acc = sum(1 for r in rows if r["inject_matches_label"]) / n
    print(f"  (informational) inject == visual label: {label_acc:.0%} — NOT part of the P0.1I gate")

    return {
        "rows": rows,
        "detail": detail,
        "section_c": c_ok,
        "section_d": d_ok,
        "section_f": f_ok,
        "fidelity": {
            "inject_vs_yaw": f"{inject_vs_yaw}/{n}",
            "dnp01_vs_inject": f"{dn01_vs_inject}/{n}",
            "dnp04_vs_inject": f"{dn04_vs_inject}/{n}",
            "inject_vs_visual_label_informational": round(label_acc, 3),
        },
        "neutral_ratio": round(fwd_mag / max(min(ctrl_mags), 1e-12), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("-o", "--output", default="output/p01i_gate")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    print("P0.1I GATE — signed yaw injection fidelity (frozen frontend, no retune)")
    a = section_a()
    b = section_b(args.seeds)
    cde = section_cde(args.seeds, video)

    with (out / "causal_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cde["rows"][0].keys()))
        w.writeheader()
        w.writerows(cde["rows"])
    with (out / "dn_detail.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cde["detail"][0].keys()))
        w.writeheader()
        w.writerows(cde["detail"])

    gate = {
        "A_algebra": a["pass"],
        "B_synthetic": b["pass"],
        "C_real_positive_controls": cde["section_c"],
        "D_real_neutral_relative": cde["section_d"],
        "F_fidelity": cde["section_f"],
    }
    report = {
        "probe": "P0.1I — isolate signed yaw injection",
        "scope": "yaw_frozen -> inject -> DN fidelity ONLY (does not close P0.1R+)",
        "frozen": {
            "optic_flow": "unchanged",
            "frozen_readout": "unchanged",
            "labels": "unchanged",
            "changed_file": "fly_vo/visual_encoder.py (_build_inject only)",
            "seeds": args.seeds,
            "neutral_ratio_threshold": NEUTRAL_RATIO,
        },
        "gate": gate,
        "fidelity": cde["fidelity"],
        "neutral_ratio": cde["neutral_ratio"],
        "A_output": a["output"],
        "B_medians": {k: v["median"] for k, v in b["table"].items()},
        "B_table_full": b["table"],
        "E_causal_table": cde["rows"],
    }
    report["pass"] = all(gate.values())
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== GATE ===")
    for k, v in gate.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n{'PASS' if report['pass'] else 'FAIL'} P0.1I GATE")
    print("Note: P0.1R+ is NOT closed by this patch.")
    print(f"Wrote {out}/")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
