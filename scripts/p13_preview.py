#!/usr/bin/env python3
"""P13 — preview copies of the chunks, small enough to play in a browser.

The camera writes MJPEG in an AVI, which no browser plays, so each chunk needs a transcode. Full
quality is not needed for watching a route: 640x360 at crf 30 comes out around 63 megabytes for
twenty minutes, against about 550 for the full-size version, and reads clearly enough to see where
the walker is.

Two things this checks rather than assumes. A preview whose length disagrees with the chunk's
trajectory is stale and is rebuilt — `VID00006_fixed.mp4` was 250 seconds while its chunk's route
covered 1228, because it had been made from an older clip that shared its name. And a preview is
never overwritten while it is correct, so the two hours of transcoding are paid once.

Usage:
    PYTHONPATH=. python scripts/p13_preview.py            # everything missing
    PYTHONPATH=. python scripts/p13_preview.py --only VID00003
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAMERA = Path("/Volumes/NO NAME/DCIM")
MEDIA = ROOT / "webapp/media"
OUT = ROOT / "output/p13"
ANCHOR = OUT / "CHAIN_ANCHOR.json"


def traj_end(video: str) -> float | None:
    p = OUT / f"runs/{video}/graph_trajectory.csv"
    if not p.exists():
        return None
    try:
        with p.open() as fh:
            fh.readline()
            last = [ln for ln in fh if ln.strip()][-1]
        return float(last.split(",")[0])
    except Exception:
        return None


def preview_seconds(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def ok(video: str) -> bool:
    """Is the preview present and the right length for this chunk?"""
    p = MEDIA / f"{video}_fixed.mp4"
    if not p.exists():
        return False
    have, want = preview_seconds(p), traj_end(video)
    if have is None or want is None:
        return False
    return abs(have - want) <= 4.0


def make(video: str, height: int, crf: int) -> bool:
    src = CAMERA / f"{video}.AVI"
    if not src.exists():
        print(f"  {video}: нет источника на камере")
        return False
    MEDIA.mkdir(parents=True, exist_ok=True)
    dest = MEDIA / f"{video}_fixed.mp4"
    tmp = MEDIA / f"{video}_building.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", "30", "-i", str(src),
           "-vf", f"scale={int(height * 16 / 9 / 2) * 2}:{height}",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        print(f"  {video}: ffmpeg не справился: {r.stderr[:200]}")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    print(f"  {video}: готов за {time.time() - t0:.0f} с, "
          f"{dest.stat().st_size / 1e6:.0f} МБ, {preview_seconds(dest):.0f} с", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--crf", type=int, default=30)
    a = ap.parse_args()

    order = json.loads(ANCHOR.read_text(encoding="utf-8"))["order"]
    todo = [a.only] if a.only else order
    missing = [v for v in todo if not ok(v)]
    print(f"всего кусков {len(todo)}, нужно перекодировать {len(missing)}")
    print(f"  готовые: {[v for v in todo if v not in missing]}")
    print(f"  к работе: {missing}")
    if not missing:
        return 0

    t0 = time.time()
    done = 0
    for v in missing:
        print(f"[{done + 1}/{len(missing)}] {v}")
        if make(v, a.height, a.crf):
            done += 1
        left = len(missing) - done
        if done:
            per = (time.time() - t0) / done
            print(f"    ещё {left}, осталось примерно {per * left / 60:.0f} мин", flush=True)
    print(f"\nготово {done}/{len(missing)} за {(time.time() - t0) / 60:.0f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
