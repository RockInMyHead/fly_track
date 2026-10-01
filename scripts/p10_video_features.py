#!/usr/bin/env python3
"""P10 level 1 — what the raw video itself contains, as fixed geometric numbers.

No network is trained and nothing here looks at the brain. Each clip is read frame by frame and
a dense optical flow is fitted between consecutive frames. From that flow field, eight fixed
quantities are derived, none of them invented for this dataset:

    dx, dy            where the image content moved, horizontally and vertically
    divergence        how much the image expands about its own centre (gradient of the flow)
    curl              how much the image turns as a whole (rotation of the flow)
    radial_slope      regression of horizontal flow on horizontal position — the fingerprint of
                      moving into the scene rather than turning inside it
    coherence         how much of the motion agrees on a single direction instead of cancelling
    opposition        right-half minus left-half horizontal flow, normalised — the yaw signature
    parallax          how differently the near band and the far band move. Turning moves the
                      whole image by the same amount regardless of distance; translating does not.
    energy            mean flow magnitude

The physical reason this level might succeed where the brain channels failed is that expansion and
parallax are the two things a single camera does record differently for "walked to the end and
came back" against "turned around": walking produces divergence and depth-dependent flow, and
turning produces neither, no matter which way the head is pointing.

Everything is reported twice — over the whole frame and over the central band, because the bottom
of a head-mounted frame may contain the wearer. Both variants are kept so the choice cannot be
made after seeing which one separates the categories.

Usage:
    PYTHONPATH=. python scripts/p10_video_features.py [--video VID00001] [--out output/p10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_MAG = 1e-6

# Feature names, in the order they are computed. Kept explicit so the analysis cannot silently
# grow a feature set after the split was chosen.
FEATURES = (
    "dx", "dy", "abs_dx", "abs_dy",
    "divergence", "curl", "radial_slope", "tang_slope",
    "coherence", "opposition", "parallax", "energy",
)


def _grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(field)
    return gx, gy


def features_of(flow: np.ndarray, y0: int, y1: int) -> dict[str, float]:
    """Fixed geometric summary of one flow field, on rows y0..y1."""
    sub = flow[y0:y1]
    u, v = sub[..., 0], sub[..., 1]
    h, w = u.shape
    if h < 4 or w < 4:
        return {k: 0.0 for k in FEATURES}

    gx_u, gy_u = _grad(u)
    gx_v, gy_v = _grad(v)

    energy = float(np.mean(np.hypot(u, v)))
    mean_u, mean_v = float(np.mean(u)), float(np.mean(v))

    # Coherence: how much of the field survives averaging. A uniform drift keeps its full length;
    # opposing halves cancel to nothing, which is what happens when the image turns about a point.
    coherence = float(np.hypot(mean_u, mean_v) / (energy + MIN_MAG))

    mid = w // 2
    left_u = float(np.mean(u[:, :mid]))
    right_u = float(np.mean(u[:, mid:]))
    opposition = float((right_u - left_u) / (abs(right_u) + abs(left_u) + MIN_MAG))

    # Parallax: the vertical spread of horizontal motion between the top and bottom third. A turn
    # adds the same displacement everywhere; a translation adds more where the scene is nearer.
    third = max(h // 3, 1)
    top_mag = float(np.mean(np.hypot(u[:third], v[:third])))
    bot_mag = float(np.mean(np.hypot(u[-third:], v[-third:])))
    parallax = float((top_mag - bot_mag) / (top_mag + bot_mag + MIN_MAG))

    # Radial slope: fit u = a * x_norm + b. a > 0 means the image fans outwards (walking in),
    # a ~ 0 with a large uniform offset means the image slides (turning).
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    u_col = np.mean(u, axis=0)
    a_rad = float(np.polyfit(xs, u_col, 1)[0]) if w > 2 else 0.0
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    v_row = np.mean(v, axis=1)
    a_tan = float(np.polyfit(ys, v_row, 1)[0]) if h > 2 else 0.0

    return {
        "dx": mean_u,
        "dy": mean_v,
        "abs_dx": float(np.mean(np.abs(u))),
        "abs_dy": float(np.mean(np.abs(v))),
        "divergence": float(np.mean(gx_u + gy_v)),
        "curl": float(np.mean(gx_v - gy_u)),
        "radial_slope": a_rad,
        "tang_slope": a_tan,
        "coherence": coherence,
        "opposition": opposition,
        "parallax": parallax,
        "energy": energy,
    }


def compute(clip: Path, width: int, height: int, band: tuple[float, float]) -> dict:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"не открыть {clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    series: dict[str, list[float]] = {}
    ts: list[float] = []
    prev: np.ndarray | None = None
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (width, height), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32) / 255.0
        t = i / fps
        i += 1
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, g, None, pyr_scale=0.5, levels=3, winsize=21,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            full = features_of(flow, 0, height)
            y0, y1 = int(band[0] * height), max(int(band[1] * height), int(band[0] * height) + 4)
            ctr = features_of(flow, y0, y1)
            for k, val in full.items():
                series.setdefault(f"f_{k}", []).append(val)
            for k, val in ctr.items():
                series.setdefault(f"c_{k}", []).append(val)
            ts.append(t)
        prev = g
    cap.release()

    out = {"t": np.asarray(ts, np.float32), "fps": np.float32(fps), "width": width,
           "height": height}
    for k, vals in series.items():
        out[k] = np.asarray(vals, np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "output/p10"))
    ap.add_argument("--video", default=None, help="обработать только один клип")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--band", type=float, nargs=2, default=(0.30, 0.70))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    by_clip: dict[str, list[dict]] = {}
    for l in frozen["labels"]:
        by_clip.setdefault(l["clip"], []).append(l)

    todo = [args.video] if args.video else sorted(by_clip)
    print("=" * 96)
    print("P10 уровень 1 — геометрические признаки сырого видео")
    print("=" * 96)
    print(f"  клипов: {len(todo)}  кадр {args.width}x{args.height}  полоса {tuple(args.band)}")
    print()
    for v in todo:
        evs = by_clip[v]
        sources = {e["clip_file"] for e in evs}
        dests = []
        t0 = time.perf_counter()
        for rel in sorted(sources):
            clip = ROOT / "output/p095" / rel
            dest = out / f"video_feat_{Path(rel).stem}.npz"
            dests.append(dest)
            if dest.exists() and not args.force:
                print(f"  {Path(rel).stem}: уже есть, пропуск")
                continue
            if not clip.exists():
                print(f"  {Path(rel).stem}: НЕТ ФАЙЛА {clip}")
                continue
            d = compute(clip, args.width, args.height, tuple(args.band))
            np.savez_compressed(dest, **d)
            print(f"  {Path(rel).stem}: {len(d['t'])} кадров, "
                  f"{sum(1 for k in d if k not in ('t','fps','width','height'))} признаков, "
                  f"{time.perf_counter()-t0:.1f} с -> {dest.name}")
    print()
    print("  готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
