#!/usr/bin/env python3
"""
Inject decomposition — why is the yaw lateral drive so weak?

Re-derives the _build_inject strengths analytically (frozen formulas, no code change)
and separates the contribution of each term:

  T4 branch:  base = 0.55 * yaw_h   + 0.30 * exp_h   + 0.15 * max(0, +f)
  T5 branch:  base = 0.50 * off_yaw + 0.30 * exp_h   + 0.20 * max(0, -f)

with
  T4:  yaw_h   = max(0, -yaw) on L ;  max(0, +yaw) on R
  T5:  off_yaw = max(0, +yaw) on L ;  max(0, -yaw) on R     <-- opposite lateral convention

For a LEFT turn (yaw < 0, Y = |yaw|):
  T4 puts +0.55*Y on the L side
  T5 puts +0.50*Y on the R side
  => net yaw lateralization = (0.55 - 0.50) * Y = 0.05 * Y   (21x attenuation)

If T5 used the SAME lateral convention as T4 the net would be 1.05 * Y.

Usage:
    PYTHONPATH=. python scripts/p02a_inject_decomposition.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig, tanh_drive
from fly_vo.visual_encoder import T4_TYPES, T5_TYPES

EMA_TAU = 0.5
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
ENCODE_W, ENCODE_H = 96, 72

EVENTS = {
    "left_65":   (63.5, +1),
    "right_216": (214.0, -1),
    "fwd_140":   (137.5, 0),
    "left_81":   (78.5, +1),
    "left_108":  (106.0, +1),
    "right_67":  (66.0, -1),
    "right_210": (209.0, -1),
}


def resize_gray(frame):
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (ENCODE_W, ENCODE_H), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    video = ROOT / "data/p01r/VID00001.AVI"
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    cfg = FlyVOConfig()
    fcfg = FlowConfig(crop_mode="center_band", ema_tau_s=EMA_TAU, brain_dt=cfg.brain_dt)
    g = cfg.visual_gain
    fg = fcfg.flow_inject_gain

    rows: list[dict] = []
    for eid, (start, exp_sign) in EVENTS.items():
        front = DirectionalMotionFrontend(fcfg)
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        pre_steps = int(round(PRE_ROLL_S / cfg.brain_dt))

        yaw_terms: list[float] = []
        exp_terms: list[float] = []
        flow_terms: list[float] = []
        t4lr: list[float] = []
        t5lr: list[float] = []
        cf_net: list[float] = []

        for i, (_t, frame) in enumerate(iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt)):
            img = resize_gray(frame).astype(np.float32) / 255.0
            d = front.process(img)
            if i < pre_steps:
                continue

            yaw = tanh_drive(fg, d.yaw_signal)
            exp = tanh_drive(fg, d.expansion_signal)
            flow_l = tanh_drive(fg, d.flow_L)
            flow_r = tanh_drive(fg, d.flow_R)

            # --- T4 branch (4 subtypes) ---
            t4_l = 4 * g * (0.55 * max(0.0, -yaw) + 0.30 * max(0.0, -exp) + 0.15 * max(0.0, flow_l))
            t4_r = 4 * g * (0.55 * max(0.0, yaw) + 0.30 * max(0.0, exp) + 0.15 * max(0.0, flow_r))

            # --- T5 branch, AS IMPLEMENTED (opposite lateral convention) ---
            t5_l_impl = 4 * g * (0.50 * max(0.0, yaw) + 0.30 * max(0.0, exp) + 0.20 * max(0.0, -flow_l))
            t5_r_impl = 4 * g * (0.50 * max(0.0, -yaw) + 0.30 * max(0.0, -exp) + 0.20 * max(0.0, -flow_r))

            # --- T5 branch, SAME lateral convention as T4 (counterfactual, analytical only) ---
            t5_l_fix = 4 * g * (0.50 * max(0.0, -yaw) + 0.30 * max(0.0, -exp) + 0.20 * max(0.0, -flow_l))
            t5_r_fix = 4 * g * (0.50 * max(0.0, yaw) + 0.30 * max(0.0, exp) + 0.20 * max(0.0, -flow_r))

            t4lr.append(t4_l - t4_r)
            t5lr.append(t5_l_impl - t5_r_impl)
            cf_net.append((t4_l + t5_l_fix) - (t4_r + t5_r_fix))

            # yaw-only lateral contribution, in the same 4*g units
            yaw_net = 4 * g * (
                (0.55 * max(0.0, -yaw) + 0.50 * max(0.0, yaw))
                - (0.55 * max(0.0, yaw) + 0.50 * max(0.0, -yaw))
            )
            exp_net = 4 * g * (0.30 * max(0.0, -exp) - 0.30 * max(0.0, exp))
            flow_net = 4 * g * (0.15 * max(0.0, flow_l) - 0.15 * max(0.0, flow_r))
            yaw_terms.append(yaw_net)
            exp_terms.append(exp_net)
            flow_terms.append(flow_net)

        def med(x: list[float]) -> float:
            return float(np.median(x)) if x else 0.0

        a, b, c = med(yaw_terms), med(exp_terms), med(flow_terms)
        rows.append({
            "event": eid,
            "group": "failed" if eid in ("left_81", "left_108", "right_67", "right_210") else "control",
            "expected_visual_sign": exp_sign,
            "T4_LR": round(med(t4lr), 4),
            "T5_LR_implemented": round(med(t5lr), 4),
            "implemented_net": round(med(t4lr) + med(t5lr), 4),
            "counterfactual_net_same_convention": round(med(cf_net), 4),
            "term_yaw_only": round(a, 5),
            "term_expansion": round(b, 5),
            "term_hemifield_flow": round(c, 5),
            "yaw_term_share": round(abs(a) / (abs(a) + abs(b) + abs(c) + 1e-12), 3),
        })

    out = ROOT / "output/p02a_falsification"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inject_decomposition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'event':10s} {'exp':>4s} {'T4_LR':>8s} {'T5_LR':>8s} {'impl_net':>9s} {'cf_net':>8s} {'yaw%':>6s}")
    for r in rows:
        s = {1: "+", -1: "-", 0: "0"}[r["expected_visual_sign"]]
        print(
            f"{r['event']:10s} {s:>4s} {r['T4_LR']:+8.4f} {r['T5_LR_implemented']:+8.4f} "
            f"{r['implemented_net']:+9.4f} {r['counterfactual_net_same_convention']:+8.4f} "
            f"{r['yaw_term_share']*100:5.0f}%"
        )

    fa = [r for r in rows if r["group"] == "failed"]
    ctrl = [r for r in rows if r["group"] == "control" and r["expected_visual_sign"] != 0]

    def acc(key: str, subset: list[dict]) -> str:
        hits = sum(1 for r in subset if np.sign(r[key]) == r["expected_visual_sign"])
        return f"{hits}/{len(subset)}"

    summary = {
        "note": "counterfactual column is analytical only — _build_inject was NOT modified",
        "implemented_formula": {
            "T4": "g*(0.55*yaw_h + 0.30*exp_h + 0.15*max(0,+f)), yaw_h= max(0,-yaw) on L, max(0,+yaw) on R",
            "T5": "g*(0.50*off_yaw + 0.30*exp_h + 0.20*max(0,-f)), off_yaw= max(0,+yaw) on L, max(0,-yaw) on R",
            "yaw_weights": {"T4": 0.55, "T5": 0.50},
            "lateral_convention_match": False,
            "net_yaw_lateralization_factor": 0.05,
            "net_if_same_convention": 1.05,
            "attenuation_vs_ideal": round(1.05 / 0.05, 1),
        },
        "accuracy_clean_controls": {
            "T4_LR_alone": acc("T4_LR", ctrl),
            "implemented_net": acc("implemented_net", ctrl),
            "counterfactual_net": acc("counterfactual_net_same_convention", ctrl),
        },
        "accuracy_failed_events": {
            "T4_LR_alone": acc("T4_LR", fa),
            "implemented_net": acc("implemented_net", fa),
            "counterfactual_net": acc("counterfactual_net_same_convention", fa),
        },
        "term_share_fwd": {
            r["event"]: {"yaw": r["term_yaw_only"], "expansion": r["term_expansion"], "flow": r["term_hemifield_flow"]}
            for r in rows if r["expected_visual_sign"] == 0
        },
        "rows": rows,
    }
    (out / "inject_decomposition.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nclean-control accuracy:")
    for k, v in summary["accuracy_clean_controls"].items():
        print(f"  {k:24s} {v}")
    print("failed-event accuracy:")
    for k, v in summary["accuracy_failed_events"].items():
        print(f"  {k:24s} {v}")
    print(f"\nWrote {out}/inject_decomposition.csv / .json")


if __name__ == "__main__":
    main()
