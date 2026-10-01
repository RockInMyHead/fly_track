#!/usr/bin/env python3
"""Settle which file a preview was actually built from, by decoding forward only.

WHY THIS EXISTS
---------------
Almost every preview in `webapp/media` turned out to disagree with the camera file of the same
name. The question that matters is not "do they disagree" but "what is this preview instead of".

The evidence points at the legacy material: `VID00002_fixed.mp4` is dated 20 September 09:28, seven
minutes after `~/Downloads/dae9fd91-..._VID00002.AVI` was written on 20 September 09:21, and the
legacy file is 3325 MB against the camera's 2123 MB. So this asks the file directly: compare the
preview with each candidate and let it say which one it is.

WHY DECODE FORWARD
------------------
`cap.set(CAP_PROP_POS_FRAMES, k)` reports the requested frame back, but returns a different
picture: sampled that way the camera's own recording scored +0.148 against a forward-decoded copy
of itself. An AVI whose index does not match its data cannot be seeked, and this one cannot. A
check that seeks is not merely coarse, it is wrong, and it is wrong in the direction of declaring
matches to be mismatches. So frames are read in order and the ones wanted are kept.

Comparing a prefix rather than the whole file keeps this affordable: the camera runs at about 95
frames/s over USB, so a minute of footage costs about twenty seconds, which is enough to decide
identity, while a full chunk would cost ten minutes.

Usage:
    PYTHONPATH=. python scripts/audit_identity.py --video VID00002 \
        --candidates "/Volumes/NO NAME/DCIM/VID00002.AVI,/Users/artem/Downloads/dae9fd91-43a7-41a9-9044-9de098d1e369_VID00002.AVI"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "webapp/media"
OUT = ROOT / "output/p13"

SAME_ABOVE = 0.80
SHARP = 0.25     # best must beat the surrounding offsets by this much to count as a real match


def samples(path: Path, stride: int, count: int) -> np.ndarray:
    """Read forward, keeping every `stride`-th frame, until `count` samples or end of file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"не открыть: {path}")
    out, k = [], 0
    while len(out) < count:
        ok, im = cap.read()
        if not ok:
            break
        if k % stride == 0:
            out.append(cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY))
        k += 1
    cap.release()
    return np.stack(out) if out else np.zeros((0, 90, 160), np.uint8)


def z(a: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)


def score_against(ref: np.ndarray, cand: np.ndarray, window: int):
    """For each reference sample, the best partner in the candidate and where it sits."""
    zr, zc = z(ref), z(cand)
    pairs = []
    for i in range(len(ref)):
        s = zc @ zr[i] / zc.shape[1]
        lo, hi = max(0, i - window), min(len(cand), i + window + 1)
        if lo >= hi:
            continue
        seg = s[lo:hi]
        j = lo + int(np.argmax(seg))
        pairs.append((i, j, float(s[j])))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--candidates", required=True, help="пути через запятую, первый — с камеры")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--n", type=int, default=120, help="сколько точек с превью")
    ap.add_argument("--window", type=int, default=40, help="окно поиска в точках выборки")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    prev = MEDIA / f"{a.video}_fixed.mp4"
    cands = [Path(p.strip()) for p in a.candidates.split(",") if p.strip()]

    print("=" * 100)
    print(f"ИЗ ЧЕГО СОБРАНО ПРЕВЬЮ: {a.video}_fixed.mp4")
    print("=" * 100)
    print(f"  превью: {prev.name}  {prev.stat().st_size / 1048576:.0f} МБ  "
          f"создано {__import__('datetime').datetime.fromtimestamp(prev.stat().st_mtime):%Y-%m-%d %H:%M}")
    print()
    web = samples(prev, a.stride, a.n)
    print(f"  точек превью: {len(web)}")

    results = []
    for c in cands:
        if not c.exists():
            print(f"\n  {c.name}: файла нет")
            continue
        cam = samples(c, a.stride, len(web))
        pairs = score_against(web, cam, a.window)
        rr = np.array([p[2] for p in pairs])
        best = float(np.median(rr))
        # background: how well the same reference frame matches frames far away in the candidate
        others = []
        zc = z(cam)
        for i in range(0, len(web), max(1, len(web) // 20)):
            s = zc @ z(web)[i] / zc.shape[1]
            far = np.abs(np.arange(len(cam)) - i) > a.window
            if far.any():
                others.append(float(np.median(s[far])))
        bg = float(np.median(others)) if others else 0.0
        verdict = ("ТА ЖЕ ЗАПИСЬ" if (best >= SAME_ABOVE and best - bg >= SHARP)
                   else "не она")
        results.append({"candidate": str(c), "size_mb": c.stat().st_size / 1048576,
                        "median_best": best, "background": bg, "verdict": verdict})
        print()
        print(f"  {c.name}")
        print(f"    размер {c.stat().st_size / 1048576:.0f} МБ")
        print(f"    медиана лучшего совпадения: {best:+.3f}")
        print(f"    фон (то же кадр против далёких кадров): {bg:+.3f}")
        print(f"    разрыв: {best - bg:+.3f}  (нужно ≥{SHARP} для 'та же запись')")
        print(f"    ВЕРДИКТ: {verdict}")

    print()
    print("=" * 100)
    winners = [r for r in results if r["verdict"] == "ТА ЖЕ ЗАПИСЬ"]
    if len(winners) == 1:
        print(f"  ПРЕВЬЮ СОБРАНО ИЗ: {Path(winners[0]['candidate']).name}")
    elif not winners:
        print("  Ни один кандидат не подошёл — источник ещё не найден.")
    else:
        print(f"  Подошло несколько кандидатов: {[Path(w['candidate']).name for w in winners]}")

    if a.out:
        Path(a.out).write_text(json.dumps({"video": a.video, "results": results},
                                          ensure_ascii=False, indent=2, default=float),
                               encoding="utf-8")
        print(f"\n  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
