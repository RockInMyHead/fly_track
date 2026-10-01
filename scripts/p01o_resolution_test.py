#!/usr/bin/env python3
"""
P0.1O follow-up — is the azimuth resolution mismatch the cause?

The normaliser ablation (A per-bin / B mean-only / C global) was a clean null:
all three give 3/6, left_108 recovers in none of them.

That directed attention at the sampling itself. Measured directly:

    encode_width = 96   ->   n_azimuth_bins = 128

build_azimuth_panorama assigns each image column to a bin, so with more bins than
columns 32 of the 128 bins (25%) never receive a column and stay exactly zero,
in a fixed periodic comb (every 4th bin). 26% of the field positions that feed the
readout are therefore structurally zero, and the two hemifields do not contain the
same number of holes.

This script varies ONLY resolve/encode_width and n_azimuth_bins. No code changes:
both are FlyVOConfig / FlowConfig parameters.

  W=96,  NB=128   current      -> 25% of bins permanently zero
  W=128, NB=128   one col/bin  -> 0%  (same angular size per dx: 4/128 = 3/96)
  W=192, NB=128   oversampled  -> 0%  (each bin averages 1.5 columns)

Usage:
    PYTHONPATH=. python scripts/p01o_resolution_test.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.optic_flow import FlowConfig, build_azimuth_panorama
from fly_vo.visual_encoder import VideoVisualEncoder
from fly_vo.malecns_engine import MaleCNSEngine

EVENTS = {
    "left_65":   (63.5, -1),
    "left_81":   (78.5, -1),
    "left_108":  (106.0, -1),
    "right_67":  (66.0, +1),
    "right_210": (209.0, +1),
    "right_216": (214.0, +1),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0

CONFIGS = [
    ("W96_NB128_current", 96, 72, 128),
    ("W128_NB128_onecol", 128, 96, 128),
    ("W192_NB128_over", 192, 144, 128),
]


def holes(width: int, n_bins: int) -> int:
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    b = np.clip(np.floor((xs + 1.0) * 0.5 * (n_bins - 1)).astype(int), 0, n_bins - 1)
    return n_bins - len(np.unique(b))


def chance_consistency(n_pos: np.ndarray, n_neg: np.ndarray, trials: int = 200) -> float:
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(trials):
        flip = rng.random(len(n_pos)) < 0.5
        p = np.where(flip, n_pos, n_neg)
        q = np.where(flip, n_neg, n_pos)
        t = np.maximum(p + q, 1)
        vals.append(float(np.nanmedian(np.maximum(p, q) / t)))
    return float(np.mean(vals))


def run(video: Path, base: FlyVOConfig, label: str, w: int, h: int, nb: int) -> dict:
    cfg = FlyVOConfig()
    cfg.encode_width = w
    cfg.encode_height = h
    cfg.ema_tau_s = base.ema_tau_s
    fc = FlowConfig(crop_mode="center_band", ema_tau_s=cfg.ema_tau_s, brain_dt=cfg.brain_dt)
    fc.n_azimuth_bins = nb
    n_pre = int(round(PRE_ROLL_S / cfg.brain_dt))
    n_holes = holes(w, nb)

    out: dict[str, dict] = {}
    for eid, (start, lab) in EVENTS.items():
        enc = VideoVisualEncoder(
            MaleCNSEngine(cfg), gain=cfg.visual_gain, n_azimuth_bins=nb, flow_config=fc
        )
        fields, yaws = [], []
        for i, (_t, frame) in enumerate(
            iter_video_at_brain_hz(str(video), max(0.0, start - PRE_ROLL_S), start + EVENT_DUR_S, cfg.brain_dt)
        ):
            _e, _inj, m = enc.encode_frame(frame)
            if i < n_pre:
                continue
            fields.append(np.asarray(enc._last_flow.signed_flow_field, dtype=np.float64))
            yaws.append(float(m["yaw_frozen"]))
        if not fields:
            continue
        n_bins = min(len(f) for f in fields)
        mat = np.stack([f[:n_bins] for f in fields])
        y_med = float(np.median(yaws))

        n_pos = np.nansum(mat > 0, axis=0)
        n_neg = np.nansum(mat < 0, axis=0)
        cons = np.maximum(n_pos, n_neg) / np.maximum(n_pos + n_neg, 1)
        ch = chance_consistency(n_pos, n_neg)

        mu_t = np.nanmean(mat, axis=0)
        sd_t = np.nanstd(mat, axis=0)
        field_snr = float(np.nanmean(np.abs(mu_t)) / (np.nanmean(sd_t) + 1e-12))

        cohs = []
        for row in mat:
            a, b = row[:-1], row[1:]
            m_ = np.isfinite(a) & np.isfinite(b)
            if m_.sum() > 4 and a[m_].std() > 1e-12 and b[m_].std() > 1e-12:
                cohs.append(float(np.corrcoef(a[m_], b[m_])[0, 1]))

        out[eid] = {
            "config": label,
            "encode_width": w,
            "n_azimuth_bins": nb,
            "empty_bins": n_holes,
            "empty_fraction": n_holes / nb,
            "label": lab,
            "yaw_frozen_median": y_med,
            "sign_ok": bool(np.sign(y_med) == lab),
            "bin_consistency_median": float(np.nanmedian(cons)),
            "bin_consistency_chance": ch,
            "bins_above_chance": int(np.nansum(cons > ch + 0.15)),
            "bins_total": int(n_bins),
            "field_snr": field_snr,
            "adjacent_bin_coherence": float(np.nanmedian(cohs)) if cohs else float("nan"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("-o", "--output", default="output/p01o_resolution")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    base = FlyVOConfig()
    base.ema_tau_s = 0.5

    print("P0.1O follow-up — azimuth sampling: 96 columns into 128 bins")
    print("only encode_width / n_azimuth_bins vary; optic_flow.py unchanged\n")
    print(f"{'config':22s} {'cols':>5s} {'bins':>5s} {'empty bins':>12s} {'empty %':>8s}")

    results = {}
    for label, w, h, nb in CONFIGS:
        results[label] = run(video, base, label, w, h, nb)
        e = results[label]
        r0 = next(iter(e.values()))
        print(f"{label:22s} {w:5d} {nb:5d} {r0['empty_bins']:12d} {r0['empty_fraction']:8.0%}")

    print("\n" + "=" * 104)
    print(f"{'event':10s} {'lbl':>4s} | " + " | ".join(f"{lbl:^28s}" for lbl, *_ in CONFIGS))
    print(f"{'':10s} {'':>4s} | " + " | ".join(f"{'yaw':>9s} {'sign':>4s} {'coh':>6s} {'snr':>5s}" for _ in CONFIGS))
    print("-" * 104)
    for eid in EVENTS:
        lab = "L" if EVENTS[eid][1] < 0 else "R"
        cells = []
        for lbl, *_ in CONFIGS:
            r = results[lbl][eid]
            cells.append(
                f"{r['yaw_frozen_median']:+9.4f} {('yes' if r['sign_ok'] else 'NO'):>4s} "
                f"{r['adjacent_bin_coherence']:+6.3f} {r['field_snr']:5.3f}"
            )
        print(f"{eid:10s} {lab:>4s} | " + " | ".join(cells))

    print("\n" + "=" * 104)
    summary = {}
    for lbl, *_ in CONFIGS:
        rs = results[lbl]
        ok = sum(1 for r in rs.values() if r["sign_ok"])
        coh = np.nanmean([r["adjacent_bin_coherence"] for r in rs.values()])
        snr = np.nanmean([r["field_snr"] for r in rs.values()])
        bins = sum(r["bins_above_chance"] for r in rs.values())
        tot = sum(r["bins_total"] for r in rs.values())
        summary[lbl] = {"sign_ok": ok, "coherence": coh, "field_snr": snr, "bins": bins, "tot": tot}
        print(
            f"{lbl:22s} signs {ok}/6   bins above chance {bins}/{tot}   "
            f"mean coherence {coh:+.3f}   mean field SNR {snr:.3f}"
        )

    print("\nleft_108 detail:")
    for lbl, *_ in CONFIGS:
        r = results[lbl]["left_108"]
        print(
            f"  {lbl:22s} yaw={r['yaw_frozen_median']:+.4f} sign_ok={r['sign_ok']} "
            f"coherence={r['adjacent_bin_coherence']:+.3f} snr={r['field_snr']:.3f}"
        )

    a = summary["W96_NB128_current"]
    fixed = [lbl for lbl, *_ in CONFIGS[1:] if results[lbl]["left_108"]["sign_ok"]]
    improved = [lbl for lbl, *_ in CONFIGS[1:] if summary[lbl]["sign_ok"] > a["sign_ok"]]
    kept = all(
        results[lbl][e]["sign_ok"] for lbl, *_ in CONFIGS[1:] for e in ("left_65", "right_216")
    )

    if fixed and improved and kept:
        verdict = (
            f"CONFIRMED: removing the empty-bin comb recovers left_108 ({', '.join(fixed)}) and "
            f"raises sign accuracy from {a['sign_ok']}/6 to "
            f"{max(summary[l]['sign_ok'] for l in summary if l != 'W96_NB128_current')}/6, "
            "without breaking left_65 or right_216. The 96-into-128 binning was the defect."
        )
    elif fixed:
        verdict = (
            f"PARTIAL: {', '.join(fixed)} recovers left_108 but overall accuracy does not improve "
            f"(current {a['sign_ok']}/6; best {max(summary[l]['sign_ok'] for l in summary):}/6)."
        )
    else:
        verdict = (
            f"NOT CONFIRMED: no configuration recovers left_108 (current {a['sign_ok']}/6, "
            f"alternatives {[summary[l]['sign_ok'] for l, *_ in CONFIGS[1:]]}). "
            "The empty-bin comb is a real defect but not the cause of the failure."
        )

    report = {
        "probe": "P0.1O follow-up — azimuth sampling resolution",
        "note": "only encode_width and n_azimuth_bins vary; optic_flow.py unchanged",
        "configs": {lbl: {"encode_width": w, "n_azimuth_bins": nb} for lbl, w, _h, nb in CONFIGS},
        "results": results,
        "summary": summary,
        "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows = [r for lbl, *_ in CONFIGS for r in results[lbl].values()]
    with (out / "results.csv").open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)

    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
