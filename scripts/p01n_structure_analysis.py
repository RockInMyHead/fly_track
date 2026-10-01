#!/usr/bin/env python3
"""
P0.1N analysis — why is m(u) weak on left_108?

Measurement only. optic_flow.py is not touched.

A Reichardt yaw detector correlates structure across horizontal displacement, so
it needs VERTICAL edges (horizontal intensity gradient). Horizontal structure —
louvers, slats, floor lines, ceiling beams — carries almost no signal for it.

This script measures, per event, in the same center band the frontend uses:
  - horizontal gradient energy  (|d/dx|)  -> what yaw detection needs
  - vertical gradient energy    (|d/dy|)  -> what pitch detection needs
  - anisotropy                  (dx / dy)
  - gradient energy split by azimuth third (left / centre / right)
  - high-frequency energy       (blur indicator, in both directions)
  - the actual magnitude of m(u) for reference

Usage:
    PYTHONPATH=. python scripts/p01n_structure_analysis.py
"""

from __future__ import annotations

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
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

EVENTS = {
    "left_65":   (63.5, "good LEFT"),
    "left_108":  (106.0, "weak LEFT"),
    "right_216": (214.0, "good RIGHT"),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0

CROP_Y0, CROP_Y1 = 0.30, 0.70  # center_band, as used by the frontend


def main() -> None:
    out = ROOT / "output/p01n_field"
    out.mkdir(parents=True, exist_ok=True)
    video = ROOT / "data/p01r/VID00001.AVI"
    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    W, H = cfg.encode_width, cfg.encode_height

    fc = FlowConfig(crop_mode="center_band", ema_tau_s=cfg.ema_tau_s, brain_dt=cfg.brain_dt)

    print("P0.1N structure analysis — what the yaw detectors actually see")
    print(f"encode {W}x{H}, center band rows {int(CROP_Y0 * H)}..{int(CROP_Y1 * H)}\n")

    rows = []
    detail: dict[str, dict] = {}
    for eid, (start, desc) in EVENTS.items():
        front = DirectionalMotionFrontend(fc)
        enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain, flow_config=fc)
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        n_pre = int(round(PRE_ROLL_S / cfg.brain_dt))

        acc = {k: [] for k in ("gx", "gy", "hf_x", "hf_y", "gx_l", "gx_c", "gx_r", "m_abs", "lum")}
        for i, (t, frame) in enumerate(iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)
            if i < n_pre:
                continue
            y0, y1 = int(CROP_Y0 * H), int(CROP_Y1 * H)
            band = img[y0:y1, :].astype(np.float32) / 255.0

            gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
            acc["gx"].append(float(np.abs(gx).mean()))
            acc["gy"].append(float(np.abs(gy).mean()))

            # high-frequency content (blur proxy): residual after a blur
            blur = cv2.GaussianBlur(band, (0, 0), 1.2)
            hf = band - blur
            acc["hf_x"].append(float(np.abs(cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3)).mean()))
            acc["hf_y"].append(float(np.abs(cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3)).mean()))

            third = band.shape[1] // 3
            acc["gx_l"].append(float(np.abs(gx[:, :third]).mean()))
            acc["gx_c"].append(float(np.abs(gx[:, third:2 * third]).mean()))
            acc["gx_r"].append(float(np.abs(gx[:, 2 * third:]).mean()))
            acc["lum"].append(float(band.mean()))

            _eye, _inj, m = enc.encode_frame(frame)
            f = np.asarray(enc._last_flow.signed_flow_field, dtype=np.float64)
            acc["m_abs"].append(float(np.nanmedian(np.abs(f))) if f.size else 0.0)

        med = {k: float(np.median(v)) for k, v in acc.items()}
        gx, gy = med["gx"], med["gy"]
        aniso = gx / (gy + 1e-9)
        row = {
            "event": eid,
            "description": desc,
            "gx_horizontal_gradient": gx,
            "gy_vertical_gradient": gy,
            "anisotropy_dx_over_dy": aniso,
            "hf_x_vertical_edges": med["hf_x"],
            "hf_y_horizontal_edges": med["hf_y"],
            "gx_left_third": med["gx_l"],
            "gx_centre_third": med["gx_c"],
            "gx_right_third": med["gx_r"],
            "m_abs_median": med["m_abs"],
            "luminance": med["lum"],
        }
        rows.append(row)
        detail[eid] = row

        print(f"=== {eid} ({desc}) ===")
        print(f"  horizontal gradient |d/dx|  = {gx:.5f}   <- yaw detectors need this")
        print(f"  vertical   gradient |d/dy|  = {gy:.5f}   <- pitch detectors need this")
        print(f"  anisotropy  dx/dy           = {aniso:.3f}")
        print(f"  high-freq vertical edges    = {med['hf_x']:.5f}")
        print(f"  high-freq horizontal edges  = {med['hf_y']:.5f}")
        print(f"  |d/dx| by third: L={med['gx_l']:.5f}  C={med['gx_c']:.5f}  R={med['gx_r']:.5f}")
        print(f"  median |m(u)|               = {med['m_abs']:.5f}")
        print()

    base = detail["left_65"]
    weak = detail["left_108"]
    print("=== comparison: left_108 relative to left_65 ===")
    for key in ("gx_horizontal_gradient", "gy_vertical_gradient", "anisotropy_dx_over_dy",
                "hf_x_vertical_edges", "hf_y_horizontal_edges", "m_abs_median"):
        r = weak[key] / (base[key] + 1e-12)
        print(f"  {key:28s}  {weak[key]:.5f} vs {base[key]:.5f}   ratio {r:5.2f}x")

    print("\n=== comparison: right_216 relative to left_65 ===")
    good_r = detail["right_216"]
    for key in ("gx_horizontal_gradient", "anisotropy_dx_over_dy", "m_abs_median"):
        r = good_r[key] / (base[key] + 1e-12)
        print(f"  {key:28s}  {good_r[key]:.5f} vs {base[key]:.5f}   ratio {r:5.2f}x")

    with (out / "structure_analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # verdict logic
    gx_ratio = weak["gx_horizontal_gradient"] / (base["gx_horizontal_gradient"] + 1e-12)
    aniso_weak = weak["anisotropy_dx_over_dy"]
    aniso_good = base["anisotropy_dx_over_dy"]

    if gx_ratio < 0.75 and aniso_weak < aniso_good:
        cause = (
            "STRUCTURE ORIENTATION: left_108 has "
            f"{gx_ratio:.2f}x the horizontal gradient of left_65 and a lower dx/dy ratio "
            f"({aniso_weak:.2f} vs {aniso_good:.2f}). The scene is dominated by horizontal "
            "structure, which a Reichardt yaw detector cannot use."
        )
    elif weak["hf_x_vertical_edges"] < 0.75 * base["hf_x_vertical_edges"]:
        cause = "MOTION BLUR: high-frequency vertical-edge content collapses on left_108."
    else:
        cause = "No single dominant cause identified from these statistics."

    report = {
        "probe": "P0.1N structure analysis",
        "note": "measurement only; optic_flow.py unchanged",
        "rows": rows,
        "left_108_vs_left_65": {
            k: (weak[k] / (base[k] + 1e-12)) for k in
            ("gx_horizontal_gradient", "gy_vertical_gradient", "anisotropy_dx_over_dy",
             "hf_x_vertical_edges", "hf_y_horizontal_edges", "m_abs_median")
        },
        "cause": cause,
    }
    (out / "structure_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== VERDICT ===\n{cause}")
    print(f"\nWrote {out}/structure_analysis.csv / .json")


if __name__ == "__main__":
    main()
