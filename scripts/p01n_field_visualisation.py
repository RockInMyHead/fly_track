#!/usr/bin/env python3
"""
P0.1N — look at m(u) directly. No code changes, no brain.

Takes three events (a good left turn, a weak left turn, a good right turn) and
plots the signed local motion field m(u) across the 128 horizontal regions, plus
the raw frames and the perpendicular (vertical) motion, so the reason for the
weak signal can be seen rather than inferred.

Outputs, per event:
  <event>_overview.png   mean m(u) +- spread, overlaid for the three events
  <event>_heatmap.png    m(u) vs time
  <event>_frames.png     sample frames across the window
  <event>_profile.csv    numeric profile, per bin

Usage:
    PYTHONPATH=. python scripts/p01n_field_visualisation.py
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
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

EVENTS = {
    "left_65":   (63.5, +1, "good LEFT"),
    "left_108":  (106.0, +1, "weak LEFT"),
    "right_216": (214.0, -1, "good RIGHT"),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
SCALE = 3.0  # upscale factor for saved images


def capture_event(video: Path, start: float, cfg: FlyVOConfig) -> dict:
    """Frames, signed field, and per-frame statistics across one event window."""
    fc = FlowConfig(crop_mode="center_band", ema_tau_s=0.5, brain_dt=cfg.brain_dt)
    front = DirectionalMotionFrontend(fc)
    enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain, flow_config=fc)

    t0 = max(0.0, start - PRE_ROLL_S)
    t1 = start + EVENT_DUR_S
    fields, times, frames, stats = [], [], [], []
    for t, frame in iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt):
        d = front.process(
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (cfg.encode_width, cfg.encode_height),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            / 255.0
        )
        f = np.asarray(d.signed_flow_field, dtype=np.float64)
        fields.append(f)
        times.append(t)
        frames.append(frame)
        _, _, m = enc.encode_frame(frame)
        stats.append({
            "t": t,
            "opponent": m["local_opponent_index"],
            "mean_lum": d.mean_luminance,
            "contrast": d.contrast,
            "frame_diff": d.frame_diff_mean,
            "motion_energy": d.motion_energy,
            "expansion": d.expansion_signal,
            "yaw_frozen": d.yaw_frozen,
            "motion_field": f,
        })

    n_pre = min(int(round(PRE_ROLL_S / cfg.brain_dt)), max(0, len(fields) - 1))
    n_bins = max((len(f) for f in fields), default=0)
    mat = np.full((len(fields), n_bins), np.nan)
    for i, f in enumerate(fields):
        mat[i, : len(f)] = f
    return {
        "times": np.asarray(times),
        "fields": mat,
        "frames": frames,
        "stats": stats,
        "n_pre": n_pre,
        "n_bins": n_bins,
    }


def _save_scaled(path: Path, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cv2.imwrite(str(path), cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST))


def _finite(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    out[~np.isfinite(out)] = fill
    return out


def plot_profile(path: Path, profiles: dict[str, np.ndarray], spreads: dict[str, np.ndarray]) -> None:
    """Bar-style profile of mean m(u) per bin for each event."""
    h, w = 420, 1280
    img = np.full((h, w, 3), 255, np.uint8)
    colours = {"left_65": (200, 60, 60), "left_108": (40, 160, 40), "right_216": (180, 120, 30)}

    mid = h // 2
    cv2.line(img, (60, mid), (w - 20, mid), (150, 150, 150), 1)
    scale = 160.0
    n = len(_finite(profiles[next(iter(profiles))]))

    for name, prof in profiles.items():
        col = colours.get(name, (0, 0, 0))
        prof = _finite(prof)
        spread = _finite(spreads[name]) if name in spreads else None
        xs = np.linspace(60, w - 20, n)
        pts = [(int(x), int(mid - np.clip(v * scale, -190, 190))) for x, v in zip(xs, prof)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, 2)
        if spread is not None:
            up = [(int(x), int(mid - np.clip((v + s) * scale, -190, 190))) for x, v, s in zip(xs, prof, spread)]
            dn = [(int(x), int(mid - np.clip((v - s) * scale, -190, 190))) for x, v, s in zip(xs, prof, spread)]
            for i in range(len(up) - 1):
                cv2.line(img, up[i], up[i + 1], col, 1)
                cv2.line(img, dn[i], dn[i + 1], col, 1)

    cv2.putText(img, "m(u) mean across window   (up = leftward motion)", (60, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "left edge (azimuth -1)", (30, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.putText(img, "right edge (azimuth +1)", (w - 260, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    y = 46
    for name in profiles:
        cv2.line(img, (w - 300, y - 5), (w - 270, y - 5), colours[name], 3)
        cv2.putText(img, name, (w - 260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y += 20
    cv2.imwrite(str(path), img)


def plot_heatmap(path: Path, data: dict, title: str) -> None:
    mat = data["fields"][data["n_pre"]:]
    if mat.size == 0:
        return
    finite = mat[np.isfinite(mat)]
    lim = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0
    lim = max(lim, 1e-6)
    norm = np.clip(mat / lim, -1, 1)
    rgb = np.zeros((*norm.shape, 3), np.uint8)
    pos = norm > 0
    neg = norm < 0
    rgb[pos] = np.stack([
        255 - (norm[pos] * 255).astype(np.uint8),
        (255 * (1 - norm[pos])).astype(np.uint8),
        (255 * (1 - norm[pos])).astype(np.uint8),
    ], axis=1)
    rgb[neg] = np.stack([
        (255 * (1 + norm[neg])).astype(np.uint8),
        (255 * (1 + norm[neg])).astype(np.uint8),
        255 - (-norm[neg] * 255).astype(np.uint8),
    ], axis=1)
    rgb[~np.isfinite(mat)] = 255
    rgb = cv2.resize(rgb, (norm.shape[1] * 8, norm.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
    cv2.putText(rgb, f"{title}  m(u) vs time  (red = leftward, blue = rightward)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(rgb, "azimuth -1", (8, rgb.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(rgb, "azimuth +1", (rgb.shape[1] - 110, rgb.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(path), rgb)


def plot_frames(path: Path, data: dict, title: str) -> None:
    idx = [data["n_pre"] + i for i in range(0, len(data["frames"]) - data["n_pre"], max(1, (len(data["frames"]) - data["n_pre"]) // 5))][:6]
    imgs = [data["frames"][i] for i in idx if i < len(data["frames"])]
    if not imgs:
        return
    imgs = [cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2)) for im in imgs]
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs) + 6 * (len(imgs) - 1)
    canvas = np.full((h + 30, w, 3), 240, np.uint8)
    x = 0
    for im, i in zip(imgs, idx):
        canvas[30:30 + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, f"t={data['times'][i]:.1f}s", (x + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x += im.shape[1] + 6
    cv2.putText(canvas, title, (w - 260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("-o", "--output", default="output/p01n_field")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5

    data = {}
    for eid, (start, lab, desc) in EVENTS.items():
        d = capture_event(video, start, cfg)
        data[eid] = d
        ev = d["fields"][d["n_pre"]:]
        prof = np.nanmedian(ev, axis=0)
        spread = np.nanpercentile(ev, 75, axis=0) - np.nanpercentile(ev, 25, axis=0)
        plot_heatmap(out / f"{eid}_heatmap.png", d, f"{eid} ({desc})")
        plot_frames(out / f"{eid}_frames.png", d, f"{eid} ({desc})")
        with (out / f"{eid}_profile.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin", "azimuth", "median_m", "iqr_m"])
            n = len(prof)
            for i in range(n):
                az = -1.0 + 2.0 * i / max(n - 1, 1)
                w.writerow([i, round(az, 4), float(prof[i]), float(spread[i])])
        stats = d["stats"][d["n_pre"]:]
        print(f"\n=== {eid} ({desc}) ===")
        print(f"  bins={d['n_bins']}  window_steps={len(stats)}")
        for key in ("opponent", "yaw_frozen", "mean_lum", "contrast", "frame_diff", "motion_energy"):
            vals = np.array([s[key] for s in stats])
            print(f"  {key:15s} median={np.median(vals):+9.4f}  min={vals.min():+9.4f}  max={vals.max():+9.4f}")
        neg = int(np.nansum(prof < 0)); pos = int(np.nansum(prof > 0)); zero = int(np.nansum(np.abs(prof) < 1e-6))
        print(f"  bins with m<0 (leftward): {neg:3d}   m>0 (rightward): {pos:3d}   m≈0: {zero:3d}")
        print(f"  |m| median={np.nanmedian(np.abs(prof)):.4f}  p90={np.nanpercentile(np.abs(prof), 90):.4f}")

    profiles = {e: np.nanmedian(d["fields"][d["n_pre"]:], axis=0) for e, d in data.items()}
    spreads = {
        e: np.nanpercentile(d["fields"][d["n_pre"]:], 75, axis=0)
        - np.nanpercentile(d["fields"][d["n_pre"]:], 25, axis=0)
        for e, d in data.items()
    }
    plot_profile(out / "profile_compare.png", profiles, spreads)

    report = {
        "probe": "P0.1N — visualise m(u); no code changes",
        "events": {k: {"start": v[0], "label": v[1], "description": v[2]} for k, v in EVENTS.items()},
        "per_event": {
            e: {
                "bins": d["n_bins"],
                "bins_leftward": int(np.nansum(profiles[e] < 0)),
                "bins_rightward": int(np.nansum(profiles[e] > 0)),
                "abs_m_median": float(np.nanmedian(np.abs(profiles[e]))),
                "abs_m_p90": float(np.nanpercentile(np.abs(profiles[e]), 90)),
                "opponent_index_median": float(np.median([s["opponent"] for s in d["stats"][d["n_pre"]:]])),
                "frame_diff_median": float(np.median([s["frame_diff"] for s in d["stats"][d["n_pre"]:]])),
                "contrast_median": float(np.median([s["contrast"] for s in d["stats"][d["n_pre"]:]])),
                "motion_energy_median": float(np.median([s["motion_energy"] for s in d["stats"][d["n_pre"]:]])),
            }
            for e, d in data.items()
        },
        "files": sorted(p.name for p in out.glob("*.png")),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
