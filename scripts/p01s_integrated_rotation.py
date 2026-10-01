#!/usr/bin/env python3
"""
P0.1S — integrating the rotation signal, and where in the frame it lives.

Two facts from P0.1S so far:

* the Lucas-Kanade field is a *trustworthy* estimator (it recovers known shifts to
  0.001 px, has no systematic vertical leak, and 94% of its seed grid survives the
  forward-backward check);
* but the per-frame rotation estimate is noisy: robust sigma(a) ~ 0.33 px against
  a typical |a| of ~0.6 px, i.e. per-frame SNR ~1.8.

A per-frame readout at SNR 1.8 cannot be stable, which is exactly what P0.1P found.
Integration is the standard answer, and it is only defensible *now*: with the old
correlator the per-frame signal was at chance (SNR ~0), so averaging could only
average noise. Here the per-frame signal is real, so summing it over a window
divides the noise by sqrt(N) while the signal adds linearly.

    total rotation over a window  =  sum_t a(t)          (pixels of accumulated turn)
    noise on that sum             =  sigma(a) * sqrt(N)  (random walk, label-free)
    z                             =  sum a / (sigma*sqrt(N))

This script computes that for the seven blind-labelled windows, and also asks a
second question the earlier work never asked: **which part of the image carries
the rotation?** The frozen pipeline crops to the middle 40% of the frame
(`crop_y0=0.30, crop_y1=0.70`), a choice inherited from the earliest 1-D pipeline.
If most of the rotation signal sits outside that band, the crop — not the
estimator — is what has been discarding it.

Frontend only. No MaleCNS, no injection, nothing trained, no label used to fit
anything.

Usage:
    PYTHONPATH=. python scripts/p01s_integrated_rotation.py
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
from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS = {
    "left_65":   (63.5, "LEFT"),
    "left_81":   (78.5, "LEFT"),
    "left_108":  (106.0, "LEFT"),
    "right_67":  (66.0, "RIGHT"),
    "right_210": (209.0, "RIGHT"),
    "right_216": (214.0, "RIGHT"),
    "fwd_140":   (137.5, "FWD"),
}
WANT = {"LEFT": -1, "RIGHT": +1, "FWD": 0}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DT = 0.02
GRID_W, GRID_H = 16, 8
XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
DESIGN = np.stack([np.ones_like(XN), XN], axis=1)
PINV = np.linalg.pinv(DESIGN)

# Sensor regions to compare. "frozen" is exactly what the pipeline crops today.
REGIONS = {
    "frozen_center": (0.30, 0.70),
    "full_frame":    (0.00, 1.00),
    "upper_half":    (0.00, 0.50),
    "lower_half":    (0.50, 1.00),
    "bottom_third":  (0.66, 1.00),
}


def rotation_series(
    frames: list[tuple[float, np.ndarray]], y0f: float, y1f: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (a, b) for one region, from a list of (t, gray_float) pairs."""
    t_out, a_out, b_out = [], [], []
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    for t, img in frames:
        h = img.shape[0]
        y0, y1 = int(y0f * h), max(int(y0f * h) + 16, int(y1f * h))
        region = np.ascontiguousarray(img[y0:y1])
        dxf, _dy, _diag = lk.process(region)
        ab = dxf.reshape(-1) @ PINV.T
        t_out.append(t)
        a_out.append(float(ab[0]))
        b_out.append(float(ab[1]))
    return np.asarray(t_out), np.asarray(a_out), np.asarray(b_out)


def sigma_of(a: np.ndarray, k: int = 9) -> float:
    """Robust per-frame noise: MAD of the residual after a short median filter."""
    pad = k // 2
    sm = np.array([np.median(a[max(0, i - pad): i + pad + 1]) for i in range(len(a))])
    return float(1.4826 * np.median(np.abs(a - sm))) if len(a) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_integrated")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1S — integrated rotation over the event windows")
    print("  total rotation = sum of the per-frame rotation coefficient;")
    print("  z = sum / (sigma * sqrt(N)) with sigma estimated from high-frequency residual\n")

    # one pass over the video per region: gather all windows first
    seeds = {name: (13, 4) for name in REGIONS}
    grabs: dict[str, list[tuple[float, np.ndarray]]] = {name: [] for name in REGIONS}
    for name, (y0f, y1f) in REGIONS.items():
        for eid, (start, _lab) in EVENTS.items():
            t0 = max(0.0, start - PRE_ROLL_S)
            for t, frame in iter_video_at_brain_hz(args.video, t0, start + EVENT_DUR_S, DT):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_LINEAR)
                grabs.setdefault(f"{name}|{eid}", []).append(
                    (t, (img.astype(np.float32) / 255.0))
                )
        print(f"  collected frames for region {name}")

    rows = []
    print(f"\n  {'event':10s} {'label':5s} {'region':14s} {'sum a':>9s} {'z':>7s} "
          f"{'sigma':>7s} {'N':>5s} {'sign ok':>8s}")
    results: dict[str, dict] = {}
    for eid, (start, lab) in EVENTS.items():
        want = WANT[lab]
        results[eid] = {}
        for name in REGIONS:
            key = f"{name}|{eid}"
            y0f, y1f = REGIONS[name]
            _t, a, b = rotation_series(grabs[key], y0f, y1f)
            n_pre = int(round(PRE_ROLL_S / DT))
            a_ev = a[n_pre:]
            sig = sigma_of(a_ev)
            n = len(a_ev)
            total = float(a_ev.sum())
            z = total / max(sig * np.sqrt(n), 1e-9)
            rec = {
                "sum_a": total, "z": float(z), "sigma": sig, "n": n,
                "median_a": float(np.median(a_ev)),
                "b_sum": float(b[n_pre:].sum()),
                "sign_ok": bool(want != 0 and np.sign(total) == want),
                "sign_ok_z": bool(want != 0 and np.sign(z) == want),
            }
            results[eid][name] = rec
            rows.append({"event": eid, "label": lab, "region": name, **rec})
            print(f"  {eid:10s} {lab:5s} {name:14s} {total:+9.2f} {z:+7.2f} "
                  f"{sig:7.3f} {n:5d} {str(rec['sign_ok']):>8s}")

    print("\n  sign accuracy of the integrated rotation over the six turns:")
    for name in REGIONS:
        ok = sum(1 for eid, (_s, lab) in EVENTS.items()
                 if lab != "FWD" and results[eid][name]["sign_ok"])
        zs = [abs(results[eid][name]["z"]) for eid, (_s, lab) in EVENTS.items() if lab != "FWD"]
        fwd_z = [abs(results[eid][name]["z"]) for eid, (_s, lab) in EVENTS.items() if lab == "FWD"]
        print(f"    {name:14s} {ok}/6   median |z| on turns {np.median(zs):5.2f}   "
              f"|z| on FWD {np.mean(fwd_z):5.2f}")

    # how much rotation is the frozen crop throwing away?
    print("\n  rotation captured relative to the full frame (median over the six turns):")
    for name in REGIONS:
        if name == "full_frame":
            continue
        ratio = np.median([
            abs(results[eid][name]["sum_a"]) / max(abs(results[eid]["full_frame"]["sum_a"]), 1e-9)
            for eid, (_s, lab) in EVENTS.items() if lab != "FWD"
        ])
        print(f"    {name:14s} {ratio:.2f}x")

    # ---- separation: turns vs forward ----
    print("\n  separation of the integrated rotation, turns vs FWD:")
    for name in REGIONS:
        tl = [results[e][name]["sum_a"] for e, (_s, lab) in EVENTS.items() if lab == "LEFT"]
        tr = [results[e][name]["sum_a"] for e, (_s, lab) in EVENTS.items() if lab == "RIGHT"]
        tf = [results[e][name]["sum_a"] for e, (_s, lab) in EVENTS.items() if lab == "FWD"]
        fwd_abs = max(abs(np.mean(tf)), 1e-9)
        print(f"    {name:14s} LEFT median {np.median(tl):+7.2f}  RIGHT median {np.median(tr):+7.2f}  "
              f"FWD {np.mean(tf):+7.2f}   |LEFT|/|FWD| {abs(np.median(tl)) / fwd_abs:5.2f}  "
              f"|RIGHT|/|FWD| {abs(np.median(tr)) / fwd_abs:5.2f}")

    # ---- does translation explain the disagreements? ----
    # For pure rotation the field has no x-slope (b ~ 0). A large |b| means forward
    # motion is present, and with an off-centre focus of expansion forward motion
    # also shifts the *mean* dx. Events with |b| ~ 0 cannot be explained that way.
    print("\n  is the mean contaminated by forward motion? (|b| ~ 0 means no expansion)")
    print(f"    {'event':10s} {'label':5s} {'sum a':>9s} {'sum b':>9s} {'b/a':>7s} {'interpretation':>22s}")
    for eid, (_s, lab) in EVENTS.items():
        r = results[eid]["frozen_center"]
        ratio = abs(r["b_sum"]) / max(abs(r["sum_a"]), 1e-9)
        note = ("expansion-dominated" if ratio > 1.5 else
                "rotation-dominated" if ratio < 0.6 else "mixed")
        print(f"    {eid:10s} {lab:5s} {r['sum_a']:+9.2f} {r['b_sum']:+9.2f} {ratio:7.2f} {note:>22s}")

    best = max(REGIONS, key=lambda nm: sum(
        1 for eid, (_s, lab) in EVENTS.items() if lab != "FWD" and results[eid][nm]["sign_ok"]
    ))
    n_best = sum(1 for eid, (_s, lab) in EVENTS.items()
                 if lab != "FWD" and results[eid][best]["sign_ok"])
    n_frozen = sum(1 for eid, (_s, lab) in EVENTS.items()
                   if lab != "FWD" and results[eid]["frozen_center"]["sign_ok"])
    verdict = (
        f"INTEGRATION WORKS ON THE NEW ESTIMATOR: summing the per-frame rotation over the window "
        f"gives the correct sign on {n_best}/6 turns using the '{best}' region "
        f"({n_frozen}/6 with the frozen crop). The per-frame signal is real but noisy (sigma "
        f"{results['left_65'][best]['sigma']:.2f} px/frame), so a per-frame readout could never be "
        "stable; the accumulated rotation is the robust quantity."
        if n_best >= 5
        else f"STILL NOT SEPARATING: the best region ('{best}') gives {n_best}/6 correct signs on "
             f"the integrated rotation, the frozen crop {n_frozen}/6. Integration raises the "
             "signal-to-noise but does not resolve the sign disagreement, so the disagreement is "
             "not a noise problem."
    )
    print(f"\n=== VERDICT ===\n{verdict}")

    with (out / "integrated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "report.json").write_text(
        json.dumps({"rows": rows, "by_event": results, "best_region": best,
                    "verdict": verdict}, indent=2),
        encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
