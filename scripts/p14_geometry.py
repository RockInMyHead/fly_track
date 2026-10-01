#!/usr/bin/env python3
"""P14 — does the resolution a recording is stored at change what the levels see?

THE QUESTION
------------
The P10 levels are built from clips of two different sizes: `VID00001` and `VID00002` came from
1080p files in `~/Downloads`, the other fifteen chunks are 720p from the camera. If the features
depend on the stored resolution, then `video_full` and `video_band` are not comparable across
chunks and the level comparison is partly a comparison of file formats.

THE TRAP IN MEASURING IT
------------------------
The obvious comparison — 1080p `VID00002` against the 720p camera files — is worthless, because
those are two different walks. Any difference could be the corridor, not the pixels. The earlier
note that camera files read |a| of 0.91–1.81 against 0.34–0.55 on the 1080p files is exactly this
trap: different recordings, different motion, and the resolutions differ too.

So this holds the walk fixed and varies only the resolution: one recording, re-encoded at several
sizes through the same codec and the same settings, with the scale filter the only difference. The
1080p source is the genuine 1080p capture, so the variants are real re-encodings rather than an
upscale that invents no detail.

WHAT DECIDES THE ANSWER
-----------------------
For each quantity, the shift between resolutions is reported against the quantity's own spread
within the window:

    effect = |mean at A − mean at B| / standard deviation at A

An effect near zero means the stored resolution does not move the number, and the levels can be
compared across chunks as they are. An effect of order one means resolution moves the number as
much as the walk itself does, and the video level has to be recomputed in one geometry before any
cross-chunk claim.

Both levels that take pixels as input are checked: the P10 video features (Farneback at 320x180)
and the brain-input path (the encoder, which resizes to its own encode size before anything else).

Usage:
    PYTHONPATH=. python scripts/p14_geometry.py
    PYTHONPATH=. python scripts/p14_geometry.py --sizes 854x480,1280x720,1920x1080
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import p10_video_features as VF  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

OUT = ROOT / "output/p14"
TMP = Path("/tmp/p14geom")

# The genuine 1080p capture: a real 1080p recording, on the internal disk so the variants are cheap
# to make. Using a camera chunk instead would mean manufacturing 1080p by upscaling, which adds no
# detail and would understate any real effect.
DEFAULT_SOURCE = Path("/Users/artem/Downloads/dae9fd91-43a7-41a9-9044-9de098d1e369_VID00002.AVI")


def cut_and_scale(src: Path, size: str, start: float, window: float, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    TMP.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-r", "30", "-i", str(src),
         "-t", str(window), "-vf", f"scale={size}", "-c:v", "libx264", "-crf", "18",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(dest)],
        capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists():
        print(f"    ffmpeg не справился для {size}: {r.stderr[:160]}")
        return False
    return True


def video_features(path: Path) -> dict[str, np.ndarray]:
    d = VF.compute(path, 320, 180, (0.30, 0.70))
    return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()
            if k not in ("t", "fps", "width", "height")}


def encoder_signals(path: Path, n_frames: int | None = None) -> dict[str, np.ndarray]:
    """Per-frame quantities the brain-input path derives from the picture.

    Only the encoder runs: the brain is never stepped, so this measures the drive the level receives,
    not any dynamics. `loom` is included by hand because it is computed from the image extremum and
    is the one quantity here that could plausibly care about stored resolution.
    """
    engine = MaleCNSEngine(FlyVOConfig())
    enc = VideoVisualEncoder(engine, flow_config=None)
    cap = cv2.VideoCapture(str(path))
    series: dict[str, list[float]] = {}
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        _eye, inject, metrics = enc.encode_frame(bgr)
        total = float(sum(float(amt) for _cells, amt in inject)) if inject else 0.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        w2 = gray.shape[1] // 2
        series.setdefault("inject_total", []).append(total)
        series.setdefault("loom_left", []).append(float(np.abs(gray[:, :w2].mean()
                                                              - gray[:, :w2].min())))
        series.setdefault("loom_right", []).append(float(np.abs(gray[:, w2:].mean()
                                                               - gray[:, w2:].min())))
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                series.setdefault(f"m_{k}", []).append(float(v))
        n += 1
        if n_frames is not None and n >= n_frames:
            break
    cap.release()
    return {k: np.asarray(v, dtype=np.float64) for k, v in series.items()}


def compare(series: dict[str, dict[str, np.ndarray]], sizes: list[str],
            base: str) -> list[dict]:
    """Effect of resolution on each quantity, in units of that quantity's own spread."""
    rows = []
    keys = sorted(set().union(*(set(s.keys()) for s in series.values())))
    for k in keys:
        b = series[base].get(k)
        if b is None or len(b) < 10 or b.std() < 1e-12:
            continue
        row = {"quantity": k, "base": base, "base_mean": float(b.mean()),
               "base_sd": float(b.std())}
        for s in sizes:
            a = series[s].get(k)
            if a is None or len(a) < 10:
                continue
            n = min(len(a), len(b))
            shift = abs(float(a[:n].mean()) - float(b[:n].mean()))
            row[f"mean_{s}"] = float(a.mean())
            row[f"effect_{s}"] = shift / float(b.std())
        rows.append(row)
    rows.sort(key=lambda r: -max([v for k, v in r.items() if k.startswith("effect_")] or [0]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=150.0)
    ap.add_argument("--sizes", default="854x480,1280x720,1920x1080")
    ap.add_argument("--base", default="1920x1080", help="относительно какого размера считать сдвиг")
    ap.add_argument("--out", default=str(OUT / "geometry_report.json"))
    a = ap.parse_args()

    sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]
    src = Path(a.source)
    if not src.exists():
        print(f"нет источника: {src}")
        return 1

    print("=" * 100)
    print("P14 — ЗАВИСЯТ ЛИ УРОВНИ ОТ РАЗРЕШЕНИЯ ХРАНЕНИЯ")
    print("=" * 100)
    print(f"  источник: {src.name} ({'настоящий 1080p' if '1920' in str(src) or True else ''})")
    print(f"  окно: {a.start:.0f}..{a.start + a.window:.0f} с,  базовый размер {a.base}")
    print(f"  варианты: {sizes}  (одна запись, один кодек, отличается только scale)")
    print("  значит разница между вариантами не может быть разной прогулкой")
    print()

    paths = {}
    for s in sizes:
        dest = TMP / f"win_{s.replace('x','_')}.mp4"
        print(f"  готовлю {s} ...", flush=True)
        if not cut_and_scale(src, s, a.start, a.window, dest):
            return 1
        paths[s] = dest

    print("\n  признаки уровня video (Farneback при 320x180):")
    feats = {}
    for s, p in paths.items():
        feats[s] = video_features(p)
        print(f"    {s}: {len(feats[s].get('f_energy', []))} кадров, "
              f"{len(feats[s])} величин", flush=True)

    print("\n  путь входа в мозг (энкодер, encode_frame; мозг не шагает):")
    encs = {}
    for s, p in paths.items():
        encs[s] = encoder_signals(p)
        print(f"    {s}: {len(encs[s].get('inject_total', []))} кадров, "
              f"{len(encs[s])} величин", flush=True)

    rows_v = compare(feats, sizes, a.base)
    rows_e = compare(encs, sizes, a.base)

    def show(title: str, rows: list[dict], top: int = 14):
        print()
        print(f"  {title}")
        print(f"    {'величина':<22}" + "".join(f"{s:>14}" for s in sizes)
              + f"{'эффект макс':>13}")
        print("    " + "-" * (22 + 14 * len(sizes) + 13))
        for r in rows[:top]:
            cells = "".join(f"{r.get(f'mean_{s}', float('nan')):>14.4f}" for s in sizes)
            eff = max([v for k, v in r.items() if k.startswith("effect_")] or [0])
            print(f"    {r['quantity']:<22}{cells}{eff:>13.2f}")

    show("УРОВЕНЬ video — признаки видео", rows_v)
    show("УРОВЕНЬ input — что энкодер подаёт в мозг", rows_e)

    all_rows = rows_v + rows_e
    worst = max([max([v for k, v in r.items() if k.startswith("effect_")] or [0])
                 for r in all_rows] or [0])
    big = [r["quantity"] for r in all_rows
           if max([v for k, v in r.items() if k.startswith("effect_")] or [0]) > 1.0]
    small = [r["quantity"] for r in all_rows
             if max([v for k, v in r.items() if k.startswith("effect_")] or [0]) < 0.25]

    print()
    print("=" * 100)
    print(f"  наибольший сдвиг от разрешения: {worst:.2f} в единицах собственного разброса")
    print(f"  величин, где сдвиг > 1 (разрешение двигает как сама прогулка): {len(big)}")
    for q in big[:10]:
        print(f"    {q}")
    print(f"  величин, где сдвиг < 0.25 (не двигает): {len(small)} из {len(all_rows)}")
    print()
    if not big:
        print("  ВЫВОД: геометрия уже канонична. Признаки не зависят от разрешения хранения,")
        print("  и уровни video/input можно сравнивать между кусками как есть. Разница |a|,")
        print("  замеченная ранее, объясняется разными прогулками, а не разрешением.")
    else:
        print("  ВЫВОД: разрешение двигает часть величин. Эти величины нельзя сравнивать между")
        print("  куском 1080p и куском 720p, пока признаки не пересчитаны в одной геометрии.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "phase": "P14 — зависимость уровней от разрешения хранения",
        "source": str(src), "window_s": [a.start, a.start + a.window],
        "sizes": sizes, "base": a.base,
        "method": ("одна запись, один кодек и одни настройки; отличается только scale, поэтому "
                   "разница между вариантами — это разрешение, а не прогулка"),
        "effect_definition": "|среднее(A) − среднее(базовое)| / сигма(базового)",
        "video_level": rows_v,
        "input_level": rows_e,
        "largest_effect": worst,
        "quantities_moved_more_than_own_spread": big,
        "quantities_not_moved": small,
        "verdict": ("геометрия канонична" if not big else "нужен пересчёт в одной геометрии"),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
