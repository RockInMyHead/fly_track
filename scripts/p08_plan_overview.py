#!/usr/bin/env python3
"""A whole-plan picture whose walls survive the shrink.

The page used to show only a small crop around the junction, because the plan drawn at the
size of its panel came out blank. The reason is the shrink itself: a plan of 5298 px drawn
into 565 px loses nine pixels in ten, and a wall is a thin dark line on a white sheet, so each
output pixel averages eight white pixels with one dark one and lands near white. The walls
were in the data and gone from the picture.

Shrinking by taking the darkest pixel of each block instead of the average fixes it. Any
block that contains a wall stays dark, so thin walls survive; at 565 px wide, 14% of blocks
contain a wall, which leaves a plan that reads as a plan rather than as a grey field.

The output keeps the source's convention — dark lines on a light sheet — because the page
inverts it for its dark theme, and matching that convention means the page needs no special
case for this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_overview(src: Path, dest: Path, width: int = 1400,
                  threshold: int = 150) -> tuple[int, int]:
    """Write a line-preserving thumbnail of `src` to `dest`. Returns its size.

    `threshold` is the grey level below which a pixel counts as a wall. The plan is white
    paper with black ink, so the two are far apart and the exact cut hardly matters; 150 sits
    between them and ignores the light grey of a scan.
    """
    import numpy as np
    from PIL import Image

    im = Image.open(src).convert("L")
    a = np.asarray(im)
    h, w = a.shape
    out_w = max(1, int(width))
    out_h = max(1, round(h * out_w / w))

    walls = a < threshold

    # Pad so the array divides into whole blocks, then ask of each block whether it holds any
    # wall at all. This is the "darkest pixel wins" rule: it is what keeps a one-pixel wall
    # alive when its block is mostly paper.
    pad_h = (-h) % out_h
    pad_w = (-w) % out_w
    if pad_h or pad_w:
        walls = np.pad(walls, ((0, pad_h), (0, pad_w)), constant_values=False)

    bh = walls.shape[0] // out_h
    bw = walls.shape[1] // out_w
    blocks = walls.reshape(out_h, bh, out_w, bw).any(axis=(1, 3))

    # dark walls on a light sheet, as in the source
    out = np.where(blocks, 0, 255).astype(np.uint8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "L").save(dest, optimize=True)
    return out_w, out_h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(ROOT / "data/p08/plan.png"))
    ap.add_argument("--dest", default=str(ROOT / "output/p08/plan_overview.png"))
    # Cut well above the width of the panel it is shown in: the panel is about 565 px and the
    # reader may magnify four or five times, and a picture cut at panel width would be upscaled
    # into mush at the first zoom step. 2400 keeps walls crisp to roughly four times.
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--threshold", type=int, default=150)
    args = ap.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    if not src.exists():
        print(f"нет плана: {src}")
        return 1
    w, h = make_overview(src, dest, args.width, args.threshold)
    print(f"{dest}  {w}x{h}  ({dest.stat().st_size / 1e6:.2f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
