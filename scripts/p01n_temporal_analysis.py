#!/usr/bin/env python3
"""
P0.1N temporal analysis — is m(u) weak, or just inconsistent in time?

The structure analysis rules out blur, low detail and gradient orientation. But
two different summaries of the same field disagree strongly for left_108:

  median over time of |m(u)|     : left_65 0.194   left_108 0.221   (left_108 larger)
  |median over time of m(u)|     : left_65 0.049   left_108 0.016   (left_108 smaller)

That can only happen if the sign of m(u) flips during the window. This script
measures exactly that: per-bin temporal sign consistency, and the pooled
direction as a function of time.

Measurement only. optic_flow.py unchanged.

Usage:
    PYTHONPATH=. python scripts/p01n_temporal_analysis.py
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
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

EVENTS = {
    "left_65":   (63.5, +1, "good LEFT"),
    "left_81":   (78.5, +1, "mid LEFT"),
    "left_108":  (106.0, +1, "weak LEFT"),
    "right_67":  (66.0, -1, "mislabeled? RIGHT"),
    "right_210": (209.0, -1, "weak RIGHT"),
    "right_216": (214.0, -1, "good RIGHT"),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
SMOOTH = 25  # 0.5 s at 50 Hz


def main() -> None:
    out = ROOT / "output/p01n_field"
    out.mkdir(parents=True, exist_ok=True)
    video = ROOT / "data/p01r/VID00001.AVI"
    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    fc = FlowConfig(crop_mode="center_band", ema_tau_s=cfg.ema_tau_s, brain_dt=cfg.brain_dt)

    print("P0.1N temporal analysis — sign stability of m(u) inside each window\n")

    rows = []
    series: dict[str, dict] = {}
    for eid, (start, lab, desc) in EVENTS.items():
        front = DirectionalMotionFrontend(fc)
        enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain, flow_config=fc)
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        n_pre = int(round(PRE_ROLL_S / cfg.brain_dt))

        fields, times, opp = [], [], []
        for i, (t, frame) in enumerate(iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt)):
            _e, _i2, m = enc.encode_frame(frame)
            if i < n_pre:
                continue
            f = np.asarray(enc._last_flow.signed_flow_field, dtype=np.float64)
            fields.append(f)
            times.append(t)
            opp.append(float(m["local_opponent_index"]))

        n_bins = max(len(f) for f in fields)
        mat = np.full((len(fields), n_bins), np.nan)
        for i, f in enumerate(fields):
            mat[i, :len(f)] = f
        times = np.asarray(times)
        opp = np.asarray(opp)

        # per-bin temporal sign consistency
        valid = np.isfinite(mat)
        sgn = np.sign(mat)
        n_pos = np.nansum(sgn > 0, axis=0)
        n_neg = np.nansum(sgn < 0, axis=0)
        total = np.maximum(n_pos + n_neg, 1)
        consistency = np.maximum(n_pos, n_neg) / total

        med_field = np.nanmedian(mat, axis=0)
        abs_med_field = np.abs(med_field)
        med_abs_field = np.nanmedian(np.abs(mat), axis=0)

        # local_opponent_index is positive for RIGHT (see P0.1J), so the sign that
        # matches a LEFT label is negative. Map to "correct" accordingly.
        k = np.ones(SMOOTH) / SMOOTH
        opp_s = np.convolve(opp, k, mode="same") if len(opp) > SMOOTH else opp
        correct = (np.sign(opp_s) == -lab) & (np.abs(opp_s) > 1e-9)
        frac_correct = float(np.mean(correct))
        best = cur = 0
        for c in correct:
            cur = cur + 1 if c else 0
            best = max(best, cur)

        # chance level for per-bin sign consistency: if the sign were a fair coin,
        # max(n_pos,n_neg)/(n_pos+n_neg) would sit around 0.5 + O(1/sqrt(n)).
        n_nz = (n_pos + n_neg).astype(np.float64)
        rng = np.random.default_rng(0)
        null_cons = []
        for _ in range(200):
            flips = rng.random((mat.shape[0], mat.shape[1])) < 0.5
            p = np.where(flips, n_pos, n_neg)
            q = np.where(flips, n_neg, n_pos)
            t = np.maximum(p + q, 1)
            null_cons.append(float(np.nanmedian(np.maximum(p, q) / t)))
        null_median = float(np.mean(null_cons))

        rows.append({
            "event": eid,
            "label": lab,
            "description": desc,
            "steps": len(opp),
            "opponent_median": float(np.median(opp)),
            "opponent_smoothed_min": float(np.min(opp_s)),
            "opponent_smoothed_max": float(np.max(opp_s)),
            "opponent_sign_flips": int(np.sum(np.diff(np.sign(opp_s)) != 0)),
            "frac_time_correct_sign": frac_correct,
            "longest_correct_run_s": best * cfg.brain_dt,
            "bin_sign_consistency_median": float(np.nanmedian(consistency)),
            "bin_consistency_chance_level": null_median,
            "bins_consistent_gt_0p8": int(np.nansum(consistency > 0.8)),
            "bins_consistent_above_chance": int(np.nansum(consistency > null_median + 0.15)),
            "bins_total": int(np.sum(valid.any(axis=0))),
            "med_abs_field": float(np.nanmedian(med_abs_field)),
            "median_abs_per_frame": float(np.nanmedian(med_abs_field)),
            "time_med_then_abs": float(np.nanmedian(abs_med_field)),
        })
        series[eid] = {
            "times": times.tolist(),
            "opp_smoothed": opp_s.tolist(),
            "consistency": consistency.tolist(),
            "med_field": med_field.tolist(),
        }

        print(f"=== {eid} ({desc}), label={'LEFT' if lab > 0 else 'RIGHT'} ===")
        print(f"  smoothed opponent range: {np.min(opp_s):+.4f} .. {np.max(opp_s):+.4f}")
        print(f"  sign flips over window:  {rows[-1]['opponent_sign_flips']}")
        print(f"  time with correct sign:  {frac_correct:.0%}")
        print(f"  longest correct run:     {rows[-1]['longest_correct_run_s']:.2f}s of {len(opp) * cfg.brain_dt:.2f}s")
        print(f"  bin sign consistency:    median {rows[-1]['bin_sign_consistency_median']:.2f} "
              f"(chance {null_median:.2f}), "
              f"{rows[-1]['bins_consistent_above_chance']}/{rows[-1]['bins_total']} bins well above chance")
        print(f"  |field| per frame:       {rows[-1]['median_abs_per_frame']:.4f}")
        print(f"  |time-median field|:     {rows[-1]['time_med_then_abs']:.4f}  "
              f"(ratio {rows[-1]['time_med_then_abs'] / (rows[-1]['median_abs_per_frame'] + 1e-9):.3f})")
        print()

    print("=== summary: temporal consistency ranking ===")
    ranked = sorted(rows, key=lambda r: -r["frac_time_correct_sign"])
    print(f"{'event':10s} {'label':>6s} {'time_correct':>13s} {'longest_run':>12s} "
          f"{'bin_consist':>12s} {'chance':>8s} {'bins>chance':>12s} {'|t-med|/|frame|':>16s}")
    for r in ranked:
        ratio = r["time_med_then_abs"] / (r["median_abs_per_frame"] + 1e-9)
        print(f"{r['event']:10s} {('LEFT' if r['label'] > 0 else 'RIGHT'):>6s} "
              f"{r['frac_time_correct_sign']:13.0%} {r['longest_correct_run_s']:11.2f}s "
              f"{r['bin_sign_consistency_median']:12.2f} {r['bin_consistency_chance_level']:8.2f} "
              f"{r['bins_consistent_above_chance']:>5d}/{r['bins_total']:<5d} {ratio:16.3f}")

    with (out / "temporal_analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "temporal_series.json").write_text(json.dumps(series), encoding="utf-8")
    (out / "temporal_analysis.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}/temporal_analysis.csv / temporal_series.json")


if __name__ == "__main__":
    main()
