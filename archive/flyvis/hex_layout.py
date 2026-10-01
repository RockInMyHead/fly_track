"""Hexagonal retina layout for FlyVis (721 photoreceptors at extent=15)."""

from __future__ import annotations

import numpy as np


def hex_positions(n_hexals: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized (x, y) pixel coords for each hex receptor."""
    import flyvis.utils.hex_utils as hex_utils

    if n_hexals is None:
        n_hexals = 721
    radius = hex_utils.get_hextent(n_hexals)
    u, v = hex_utils.get_hex_coords(radius)
    x, y = hex_utils.hex_to_pixel(u, v)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-6)
    return x / scale, y / scale


def hex_layout_dict(extent: int = 15) -> dict:
    """JSON-serializable hex grid for the web UI."""
    n_hexals = 3 * extent * (extent + 1) + 1
    x, y = hex_positions(n_hexals)
    return {
        "n_hexals": int(len(x)),
        "hex_x": [round(float(v), 5) for v in x],
        "hex_y": [round(float(v), 5) for v in y],
        "extent": extent,
    }
