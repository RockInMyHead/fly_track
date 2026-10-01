#!/usr/bin/env python3
"""Find, for every preview, which file it was really built from.

`webapp/media/{v}_fixed.mp4` is what the review pages play and what the movement signal was
measured on, so if it holds another recording than the camera's {v}.AVI then everything derived
from it belongs to a different walk.

This looks for candidates to blame rather than only confirming the mismatch: for each chunk it
tests the camera file, the legacy material found in `~/Downloads` and `data/p01r`, and reports
which one the preview actually is. Decoding is forward-only because seeking into these AVIs returns
the wrong picture while reporting the position asked for.

Usage:
    PYTHONPATH=. python scripts/audit_source_of_previews.py
    PYTHONPATH=. python scripts/audit_source_of_previews.py --stride 30 --n 90
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_identity import samples, z, score_against, SAME_ABOVE, SHARP  # noqa: E402

CAMERA = Path("/Volumes/NO NAME/DCIM")
MEDIA = ROOT / "webapp/media"
LEGACY_DIRS = [Path("/Users/artem/Downloads"), ROOT / "data/p01r"]
OUT = ROOT / "output/p13"


def legacy_candidates(video: str) -> list[Path]:
    """Files somewhere in the legacy directories whose name carries this chunk's number."""
    num = video.replace("VID", "")
    out = []
    for d in LEGACY_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.upper() not in (".AVI", ".MP4", ".MOV"):
                continue
            if p.is_symlink() or not p.is_file():
                continue
            name = p.name
            if f"VID{num}" in name or name.endswith(f"VID{num}.AVI"):
                out.append(p)
    return out


def judge(web: np.ndarray, cand: np.ndarray, window: int) -> tuple[float, float]:
    pairs = score_against(web, cand, window)
    if not pairs:
        return 0.0, 0.0
    best = float(np.median([p[2] for p in pairs]))
    zc, zw = z(cand), z(web)
    bg = []
    step = max(1, len(web) // 20)
    for i in range(0, len(web), step):
        s = zc @ zw[i] / zc.shape[1]
        far = np.abs(np.arange(len(cand)) - i) > window
        if far.any():
            bg.append(float(np.median(s[far])))
    return best, (float(np.median(bg)) if bg else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--out", default=str(OUT / "preview_sources.json"))
    a = ap.parse_args()

    videos = sorted(p.stem.replace("_fixed", "") for p in MEDIA.glob("VID*_fixed.mp4"))
    print("=" * 104)
    print("ИСТОЧНИК КАЖДОГО ПРЕВЬЮ: с камеры или из старого материала")
    print("=" * 104)
    print(f"  {'кусок':<10}{'превью создано':>17}{'камера':>10}{'старое':>10}{'источник':>28}")
    print("  " + "-" * 75)

    rows = []
    for v in videos:
        prev = MEDIA / f"{v}_fixed.mp4"
        mtime = dt.datetime.fromtimestamp(prev.stat().st_mtime)
        web = samples(prev, a.stride, a.n)
        row = {"video": v, "preview_mb": prev.stat().st_size / 1048576,
               "preview_born": mtime.strftime("%Y-%m-%d %H:%M"), "tests": []}

        cam_p = CAMERA / f"{v}.AVI"
        cam_best = cam_bg = None
        if cam_p.exists():
            cam = samples(cam_p, a.stride, len(web))
            cam_best, cam_bg = judge(web, cam, a.window)
            row["tests"].append({"candidate": str(cam_p), "best": cam_best, "bg": cam_bg})

        leg_best = leg_bg = None
        leg_p = None
        for c in legacy_candidates(v):
            leg = samples(c, a.stride, len(web))
            b, g = judge(web, leg, a.window)
            row["tests"].append({"candidate": str(c), "best": b, "bg": g})
            if leg_best is None or b > leg_best:
                leg_best, leg_bg, leg_p = b, g, c

        def hit(b, g):
            return b is not None and b >= SAME_ABOVE and (b - g) >= SHARP

        if hit(cam_best, cam_bg):
            src = "камера"
        elif hit(leg_best, leg_bg):
            src = leg_p.name[:26]
        else:
            src = "не найден"
        row["source"] = src
        row["camera_best"] = cam_best
        row["legacy_best"] = leg_best
        rows.append(row)

        fc = f"{cam_best:+.3f}" if cam_best is not None else "—"
        fl = f"{leg_best:+.3f}" if leg_best is not None else "—"
        print(f"  {v:<10}{mtime:%Y-%m-%d %H:%M:>17}{fc:>10}{fl:>10}{src:>28}", flush=True)

    print()
    print("=" * 104)
    good = [r["video"] for r in rows if r["source"] == "камера"]
    wrong = [r for r in rows if r["source"] not in ("камера", "не найден")]
    unknown = [r["video"] for r in rows if r["source"] == "не найден"]
    print(f"  С КАМЕРЫ ({len(good)}): {good}")
    print(f"  ИЗ СТАРОГО МАТЕРИАЛА ({len(wrong)}):")
    for r in wrong:
        print(f"    {r['video']}: {r['source']}  (превью создано {r['preview_born']})")
    if unknown:
        print(f"  ИСТОЧНИК НЕ НАЙДЕН ({len(unknown)}): {unknown}")
    print()
    print("  Всё, что считалось по превью из старого материала, относится к другому проходу —")
    print("  включая признак движения и обе раздачи на разметку.")

    Path(a.out).write_text(json.dumps({"rows": rows, "from_camera": good,
                                       "from_legacy": [r["video"] for r in wrong],
                                       "unknown": unknown},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"\n  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
