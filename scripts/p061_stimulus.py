#!/usr/bin/env python3
"""
Shared stimulus for the clean direction test.

Both P06.1 (does direction reach these cells at all) and P06.2 (through which cells does
it arrive) must use the identical stimulus, otherwise a difference after ablation could
come from the stimulus rather than the ablation. The pieces live here and both import
them.

The stimulus, in one place
--------------------------
    frame     a still image, taken from the two clips at fixed positions
    LEFT      the frame slides left by SHIFT_PX every step
    RIGHT     the same, sliding right by the same amount
    STATIC    the frame does not move

Everything but the sign of motion is identical by construction: same source frame, same
duration, same step, same brightness and texture, same absolute displacement.

`np.roll` makes the shift cyclic, so no dark edge ever enters the frame. Truncating
instead would add a growing black band on one side, which is a large asymmetric change
and would confound the thing being measured.

The frame is also mirrored vertically before rolling. The frontend reads rows 30 to 70
percent of its 192x108 working image, and rolling the whole frame would wrap content
from the top and bottom edges into exactly those rows, where it would look stationary.
Mirroring gives a seamless field in which a horizontal roll is a pure translation
everywhere, centre band included.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO1 = ROOT / "data/p01r/VID00001.AVI"
VIDEO2 = ROOT / "data/p01r/VID00002.AVI"

N_FRAMES = 25
STEPS_PER_CLIP = 24
SHIFT_PX = 3
SETTLE_STEPS = 24
WARM_FRAMES = 8

CONDITIONS = (("LEFT", -1), ("RIGHT", 1), ("STATIC", 0))


def pick_frames(n_frames: int | None = None) -> list[np.ndarray]:
    """Stills spread across both clips, at fixed positions, so runs are comparable.

    Cached on disk, keyed by how many frames are wanted. Pulling frames from fixed
    positions in two hour-long files means decoding tens of thousands of frames
    sequentially, since seeking is unreliable on both containers. That costs several
    minutes but is the same cost for any number of samples, because the decoder advances
    through the file once either way, so it is paid once and reused.
    """
    from fly_vo.video_reader import VideoReader

    want = N_FRAMES if n_frames is None else n_frames
    cache_npz = ROOT / f"output/p061_synthetic/stimulus_frames_{want}.npz"
    if cache_npz.exists():
        try:
            arr = np.load(cache_npz)["frames"]
            if arr.shape[0] == want:
                return [arr[i] for i in range(arr.shape[0])]
        except Exception:
            pass

    out = []
    per = want // 2 + 1
    for path in (VIDEO1, VIDEO2):
        with VideoReader(path) as vr:
            n = vr.frame_count
            idx = [int(round(x)) for x in np.linspace(200, n - 200, per)]
            for i in idx:
                f = vr.frames(i, i + 1)
                if f:
                    out.append(f[0])
    out = out[:want]
    try:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        # compressed: these are full-resolution frames and the raw array is 6 MB each,
        # so the uncompressed cache reached 742 MB for 125 frames against 290 MB here
        np.savez_compressed(cache_npz, frames=np.stack(out))
    except Exception as e:
        print(f"    (кэш кадров не сохранён: {e})")
    return out


def mirror_field(frame: np.ndarray) -> np.ndarray:
    """Stack the frame with its vertical mirror, giving a seamless rolling field."""
    return np.vstack([frame, frame[::-1]])


def make_clip(frame: np.ndarray, direction: int, n: int = STEPS_PER_CLIP,
              shift: int = SHIFT_PX) -> list[np.ndarray]:
    """direction +1 slides right, -1 left, 0 does not move."""
    g = mirror_field(frame)
    out = []
    off = 0
    for _ in range(n):
        out.append(np.roll(g, off * direction, axis=1))
        off += shift
    return out


def bright_still(frame: np.ndarray) -> np.ndarray:
    """A uniform field of the frame's mean brightness, to settle activity between clips."""
    return np.full_like(frame, int(round(float(frame.mean()))))
