#!/usr/bin/env python3
"""P15 — turn the floor plan and the graph into a picture the fly can see.

FINAL TRACKER V1 is not used and not modified. Nothing here adds a bonus to a
graph edge. A local map is drawn, passed through the same visual encoder as a
camera frame, and only then may it change MaleCNS.

The map is egocentric: up is the way the hypothesis is facing, left and right
are the fly's left and right. Each live hypothesis gets its own picture.

A still drawing has no optic flow, and the encoder only injects motion. So the
map is a two-frame clip: the open passages brighten and shift outward. That is
the visual hint, not a label.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def forward_exits(g, node: str, in_edge: str) -> list[dict]:
    """Passages that continue from `node`, with the turn angle the graph already uses.

    `deg > 0` is RIGHT in the graph. Back is kept out: a reversal is not drawn as a
    way forward.
    """
    out = []
    for c in g.classify_candidates(node, in_edge, allow_back=True):
        if abs(float(c["deg"])) >= 135.0:
            continue
        out.append({"edge": c["edge"], "to": c["to"], "deg": float(c["deg"]),
                    "side": c["side"]})
    return out


def mirror_exits(exits: list[dict]) -> list[dict]:
    """Swap left and right. The control that should flip a real map effect."""
    mirrored = []
    for e in exits:
        d = -float(e["deg"])
        side = e["side"]
        if side == "LEFT":
            side = "RIGHT"
        elif side == "RIGHT":
            side = "LEFT"
        mirrored.append({**e, "deg": d, "side": side})
    return mirrored


def _ray(cx: float, cy: float, deg: float, length: float) -> tuple[int, int]:
    """Image point: 0° is up, positive degrees go to the right."""
    rad = math.radians(deg)
    return (int(round(cx + math.sin(rad) * length)),
            int(round(cy - math.cos(rad) * length)))


def render_synthetic(exits: list[dict], width: int = 96, height: int = 72,
                     frame: int = 0, n_frames: int = 2) -> np.ndarray:
    """Egocentric corridor sketch. `frame` 0 is the start, later frames flow outward."""
    img = np.full((height, width, 3), 18, np.uint8)
    cx, cy = width / 2.0, height / 2.0
    reach = 0.42 * min(width, height)
    grow = 1.0 + 0.35 * (frame / max(n_frames - 1, 1))
    thick = 2 if min(width, height) < 120 else 4
    for e in exits:
        x, y = _ray(cx, cy, float(e["deg"]), reach * grow)
        cv2.line(img, (int(cx), int(cy)), (x, y), (235, 235, 235), thick, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), max(2, thick), (255, 220, 80), -1)
    return img


def render_empty(width: int = 96, height: int = 72) -> np.ndarray:
    return np.full((height, width, 3), 18, np.uint8)


def sector_clip(exits: list[dict] | None, width: int = 96, height: int = 72,
                 n: int = 8, shift: int = 2, kind: str = "correct",
                 seed: int = 1) -> list[np.ndarray]:
    """One small marker per passage, slid a few pixels along that passage only.

    The marker stays in its own azimuth. The rest of the picture does not move.
    Image left is azimuth −1, image right is +1, and a passage at angle θ sits at
    sin(θ): −90° on the left, 0° in the middle, +90° on the right.
    """
    if kind == "empty" or not exits:
        still = render_empty(width, height)
        return [still.copy() for _ in range(n)]
    use = mirror_exits(exits) if kind == "mirror" else exits
    rng = np.random.default_rng(seed)
    markers = []
    pw, ph = max(8, width // 12), max(8, height // 9)
    for e in use:
        deg = float(e["deg"])
        patch = rng.integers(0, 256, (ph, pw), dtype=np.uint8)
        patch[::2, :] = 255
        markers.append((deg, patch))
    frames = []
    cx, cy = width / 2.0, height / 2.0
    radius = 0.22 * min(width, height)
    for i in range(n):
        img = np.full((height, width), 24, np.uint8)
        for deg, patch in markers:
            rad = math.radians(deg)
            along = shift * i
            x0 = int(round(cx + math.sin(rad) * (radius + along) - patch.shape[1] / 2))
            y0 = int(round(cy - math.cos(rad) * (radius + along) - patch.shape[0] / 2))
            ph_, pw_ = patch.shape
            y1, x1 = max(0, y0), max(0, x0)
            y2, x2 = min(height, y0 + ph_), min(width, x0 + pw_)
            if y2 <= y1 or x2 <= x1:
                continue
            img[y1:y2, x1:x2] = patch[y1 - y0:y1 - y0 + (y2 - y1),
                                      x1 - x0:x1 - x0 + (x2 - x1)]
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return frames


def stable_sector_clip(degrees: list[float], width: int = 96, height: int = 72,
                       n: int = 8, shift: int = 2, grating: bool = True) -> list[np.ndarray]:
    """Same drawing as `sector_clip`, but each angle keeps its own texture.

    A passage at −90° looks the same alone and inside a fork. That is what a
    sum of single-angle templates is allowed to assume. The texture is not a
    fitted weight.

    ``grating=True`` paints every other row white. That grating repeats every
    2 px, and the marker also steps 2 px, so a pure vertical slide reproduces
    the same picture. ``grating=False`` keeps the random patch, which has
    contrast both across and down, so a vertical slide is a real change.
    """
    exits = [{"edge": f"deg{int(d)}", "deg": float(d), "to": "", "side": ""} for d in degrees]
    if not exits:
        still = render_empty(width, height)
        return [still.copy() for _ in range(n)]
    markers = []
    pw, ph = max(8, width // 12), max(8, height // 9)
    for e in exits:
        deg = float(e["deg"])
        rng = np.random.default_rng(10_000 + int(round(deg * 10)))
        patch = rng.integers(0, 256, (ph, pw), dtype=np.uint8)
        if grating:
            patch[::2, :] = 255
        markers.append((deg, patch))
    frames = []
    cx, cy = width / 2.0, height / 2.0
    radius = 0.22 * min(width, height)
    for i in range(n):
        img = np.full((height, width), 24, np.uint8)
        for deg, patch in markers:
            rad = math.radians(deg)
            along = shift * i
            x0 = int(round(cx + math.sin(rad) * (radius + along) - patch.shape[1] / 2))
            y0 = int(round(cy - math.cos(rad) * (radius + along) - patch.shape[0] / 2))
            ph_, pw_ = patch.shape
            y1, x1 = max(0, y0), max(0, x0)
            y2, x2 = min(height, y0 + ph_), min(width, x0 + pw_)
            if y2 <= y1 or x2 <= x1:
                continue
            img[y1:y2, x1:x2] = patch[y1 - y0:y1 - y0 + (y2 - y1),
                                      x1 - x0:x1 - x0 + (x2 - x1)]
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return frames


# Fixed before P15.4 was run. Not a search over 5°, 10°, 20°, 30°.
CORRIDOR_DELTA_DEG = 15.0


def corridor_edges(heading_deg: float, delta: float = CORRIDOR_DELTA_DEG) -> list[float]:
    """Left and right walls of one passage. A heading of 0° becomes −δ and +δ."""
    return [float(heading_deg) - float(delta), float(heading_deg) + float(delta)]


def corridor_clip(headings: list[float], width: int = 96, height: int = 72,
                  n: int = 8, shift: int = 2, delta: float = CORRIDOR_DELTA_DEG) -> list[np.ndarray]:
    """One passage is two moving edges, not a marker on the midline.

    Each edge sits at heading ± δ and slides outward along its own ray, the same
    motion a single passage marker already used. δ stays 15°. The texture is the
    random patch: a 2 px grating would not change under a 2 px vertical step.
    """
    edges: list[float] = []
    for heading in headings:
        edges.extend(corridor_edges(heading, delta))
    return stable_sector_clip(edges, width, height, n=n, shift=shift, grating=False)


def directional_clip(exits: list[dict] | None, width: int = 96, height: int = 72,
                     n: int = 8, shift: int = 3, kind: str = "correct",
                     seed: int = 1) -> list[np.ndarray]:
    """Texture slid along each open passage. Left moves left, right moves right.

    Not an outward zoom. A mirror flips the slide. An empty clip does not move.
    """
    if kind == "empty" or not exits:
        still = render_empty(width, height)
        return [still.copy() for _ in range(n)]
    use = mirror_exits(exits) if kind == "mirror" else exits
    rng = np.random.default_rng(seed)
    # One texture per passage, built once and slid. A new texture every frame is flicker.
    patches = []
    for e in use:
        patch = rng.integers(0, 256, (height // 3, width // 5), dtype=np.uint8)
        patch[::3, :] = 255
        patches.append((float(e["deg"]), patch))
    frames = []
    for i in range(n):
        img = np.full((height, width), 28, np.uint8)
        for deg, patch in patches:
            dx = int(round(math.sin(math.radians(deg)) * shift * i))
            dy = int(round(-math.cos(math.radians(deg)) * shift * i))
            ph, pw = patch.shape
            y0 = height // 2 - ph // 2 + dy
            x0 = width // 2 - pw // 2 + dx
            y1, x1 = max(0, y0), max(0, x0)
            y2, x2 = min(height, y0 + ph), min(width, x0 + pw)
            if y2 <= y1 or x2 <= x1:
                continue
            img[y1:y2, x1:x2] = patch[y1 - y0:y1 - y0 + (y2 - y1),
                                      x1 - x0:x1 - x0 + (x2 - x1)]
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return frames


def map_clip(exits: list[dict] | None, width: int, height: int,
             kind: str) -> list[np.ndarray]:
    """Directional clip. `kind` is correct, mirror, empty, or wrong."""
    return directional_clip(exits, width, height, kind=kind)


def encode_clip(encoder, frames: list[np.ndarray]):
    """Last eye drive and inject after the clip. The first frame only warms the flow."""
    encoder.reset()
    eye, inject, metrics = None, [], {}
    for fr in frames:
        eye, inject, metrics = encoder.encode_frame(fr)
    return eye, inject, metrics


def mix_drives(camera, map_drive, alpha: float):
    """camera + alpha * map. alpha 0 returns the camera drive unchanged."""
    eye_c, inj_c = camera
    if alpha == 0.0 or map_drive is None:
        return eye_c, list(inj_c)
    eye_m, inj_m = map_drive
    eye = eye_c
    if eye_c is not None and eye_m is not None and eye_c.shape == eye_m.shape:
        eye = np.clip(eye_c + float(alpha) * eye_m, 0.0, 1.0).astype(np.float32)
    elif eye_c is None:
        eye = None if eye_m is None else np.clip(float(alpha) * eye_m, 0.0, 1.0).astype(np.float32)
    scaled = [(cells, float(alpha) * float(amt)) for cells, amt in inj_m if abs(amt) > 1e-8]
    return eye, list(inj_c) + scaled


def population_names(encoder) -> list[str]:
    names = list(encoder.t4_t5_groups.keys())
    names += list(encoder.loom_groups.keys())
    return names


def spikes_by_group(fired: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, int]:
    fired_set = set(np.asarray(fired, dtype=np.int64).tolist())
    return {name: sum(1 for i in idx if int(i) in fired_set)
            for name, idx in groups.items()}
