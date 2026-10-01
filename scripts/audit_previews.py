#!/usr/bin/env python3
"""Audit every browser preview against the camera file it claims to be.

WHY THIS EXISTS
---------------
`webapp/media/{v}_fixed.mp4` is what the review pages play and what both annotation rounds cut
their clips from, and `p13_stillness.frame_motion` measures movement from it. So if a preview is a
different recording than the camera's file of the same name, then the movement values and the human
labels are about footage nobody recorded under that number.

That is exactly what happened. `VID00002_fixed.mp4` is dated 20 September, from before this work
began, and it is a different recording: frames at the same timestamp correlate between −0.21 and
+0.18 with the camera's, and one shows a corridor while the other shows shop shelving.

THE MISTAKE THAT LET IT THROUGH
-------------------------------
`p13_preview.py` decided whether a preview needed rebuilding by comparing its **duration** with the
chunk's trajectory. Both recordings are about 1228 seconds, so the check passed. Duration is not
identity — the same family of error as `copy2` keeping the source's timestamps, a stale yaw file
passing an existence test, and a truncated copy passing a size test.

WHAT THIS MEASURES
------------------
For each chunk, a frame from the preview and a frame from the camera file at the same instant, and
the normalised correlation between them. Matching recordings give about +0.9; unrelated ones about
0. Both bands are reported so the verdict is not a single number away from the boundary.

Usage:
    PYTHONPATH=. python scripts/audit_previews.py
    PYTHONPATH=. python scripts/audit_previews.py --times 120,600,1100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERA = Path("/Volumes/NO NAME/DCIM")
MEDIA = ROOT / "webapp/media"
OUT = ROOT / "output/p13"
TMP = Path("/tmp/preview_audit")

MATCH_ABOVE = 0.6      # judged the same recording
DIFFER_BELOW = 0.3     # judged a different recording


def frame(path: Path, t: float) -> np.ndarray | None:
    dst = TMP / f"{path.stem}_{int(t)}.jpg"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t}", "-i", str(path),
                        "-frames:v", "1", "-q:v", "3", str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        return None
    im = cv2.imread(str(dst))
    dst.unlink(missing_ok=True)
    if im is None:
        return None
    return cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float64)


def compare(a: Path, b: Path, times: list[float]) -> tuple[float | None, list[float]]:
    vals = []
    for t in times:
        fa, fb = frame(a, t), frame(b, t)
        if fa is None or fb is None:
            continue
        za = (fa - fa.mean()) / (fa.std() + 1e-9)
        zb = (fb - fb.mean()) / (fb.std() + 1e-9)
        vals.append(float((za * zb).mean()))
    if not vals:
        return None, []
    return float(np.median(vals)), vals


def duration(p: Path) -> float | None:
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(p)], capture_output=True, text=True,
                           timeout=60).stdout.strip()
        return float(o) if o else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--times", default="120,600,1100")
    ap.add_argument("--out", default=str(OUT / "preview_audit.json"))
    a = ap.parse_args()
    times = [float(x) for x in a.times.split(",") if x.strip()]
    TMP.mkdir(parents=True, exist_ok=True)

    cams = sorted(p.stem for p in CAMERA.glob("VID*.AVI"))
    previews = sorted(p.stem.replace("_fixed", "") for p in MEDIA.glob("VID*_fixed.mp4"))
    todo = [v for v in cams if v in previews]

    print("=" * 100)
    print("АУДИТ ПРЕВЬЮ: совпадает ли то, что играет браузер, с файлом камеры")
    print("=" * 100)
    print(f"  на камере: {len(cams)} файлов, превью есть для {len(todo)}")
    print(f"  моменты для сравнения: {times}")
    print(f"  совпадение: корреляция ≥{MATCH_ABOVE} — та же запись; ≤{DIFFER_BELOW} — другая")
    print()
    print(f"  {'кусок':<10}{'превью':>9}{'камера':>9}{'корреляция':>12}{'вердикт':>16}")
    print("  " + "-" * 58)

    rows = []
    for v in todo:
        prev = MEDIA / f"{v}_fixed.mp4"
        cam = CAMERA / f"{v}.AVI"
        med, vals = compare(prev, cam, times)
        dp, dc = duration(prev), duration(cam)
        if med is None:
            verd = "не сравнить"
        elif med >= MATCH_ABOVE:
            verd = "та же запись"
        elif med <= DIFFER_BELOW:
            verd = "ДРУГАЯ ЗАПИСЬ"
        else:
            verd = "неясно"
        rows.append({"video": v, "preview_s": dp, "camera_s": dc,
                     "corr_median": med, "corr_all": vals, "verdict": verd})
        print(f"  {v:<10}{(dp or 0):>9.0f}{(dc or 0):>9.0f}"
              f"{(med if med is not None else float('nan')):>12.3f}{verd:>16}")

    print()
    c = Counter(r["verdict"] for r in rows)
    print(f"  ИТОГО: {dict(c)}")
    bad = [r["video"] for r in rows if r["verdict"] == "ДРУГАЯ ЗАПИСЬ"]
    if bad:
        print()
        print(f"  ПРЕВЬЮ ОТ ДРУГОЙ ЗАПИСИ: {bad}")
        print("  Эти файлы играет страница разметки, и из них нарезаны клипы")

        print("  обоих раундов разметки, а признак движения измерялся по ним же.")
    (Path(a.out)).write_text(json.dumps({
        "phase": "аудит превью", "times": times,
        "match_above": MATCH_ABOVE, "differ_below": DIFFER_BELOW,
        "rows": rows, "wrong": bad,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
