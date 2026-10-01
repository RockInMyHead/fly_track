"""PFN/hΔB-inspired body-to-world velocity transform."""

from __future__ import annotations

import numpy as np


def body_to_world(
    vx_body: np.ndarray,
    vy_body: np.ndarray,
    heading: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform body-centric velocity to allocentric world velocity.

    v_world = R(heading) @ v_body

    Body frame: x = forward, y = right.
    World frame: x = east, y = north.
    Heading: radians, 0 = east, increasing counter-clockwise.
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    vx_world = vx_body * cos_h - vy_body * sin_h
    vy_world = vx_body * sin_h + vy_body * cos_h
    return vx_world.astype(np.float32), vy_world.astype(np.float32)
