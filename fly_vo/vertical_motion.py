"""Signed vertical motion at each azimuth.

The frozen yaw frontend collapses every column to one number, so an upward
slide never reaches T4c/T4d. This is the same Reichardt idea turned on its
side: at each image column, compare the column with itself shifted up or down.

Sign (stated, not fitted):
    positive = image content moving downward (+row)
    negative = image content moving upward

The three detector offsets are the frozen yaw detectors, swapped onto the
vertical axis: (6 px, lag 2), (12 px, lag 3), (6 px, lag 3).
"""

from __future__ import annotations

import numpy as np

# Same three frozen detectors as optic_flow.FROZEN_YAW_DETECTORS_PX, on y.
VERTICAL_DETECTORS: tuple[tuple[int, int], ...] = ((6, 2), (12, 3), (6, 3))


def column_reichardt(prev: np.ndarray, curr: np.ndarray, dy: int) -> np.ndarray:
    """Signed vertical Reichardt, one value per column.

    ``prev`` and ``curr`` are HxW. Positive means content moved toward +row.
    """
    prev = np.asarray(prev, dtype=np.float32)
    curr = np.asarray(curr, dtype=np.float32)
    if dy < 1 or prev.shape[0] <= dy or prev.shape != curr.shape:
        return np.zeros(prev.shape[1] if prev.ndim == 2 else 0, dtype=np.float32)
    right = prev[:-dy] * curr[dy:]
    left = prev[dy:] * curr[:-dy]
    return (right - left).mean(axis=0).astype(np.float32)


def _prepare(img: np.ndarray) -> np.ndarray:
    """Per-column z-score, so a flat column cannot invent motion.

    Matches the range of the yaw panorama normalizer (clipped to ±3) without
    touching that normalizer.
    """
    img = np.asarray(img, dtype=np.float32)
    out = np.zeros_like(img)
    for x in range(img.shape[1]):
        col = img[:, x]
        scale = float(col.std())
        if scale < 0.02:
            continue
        out[:, x] = np.clip((col - float(col.mean())) / scale, -3.0, 3.0)
    return out


class VerticalMotionFrontend:
    """Running vertical flow, one signed value per azimuth bin."""

    def __init__(
        self,
        n_bins: int = 128,
        detectors: tuple[tuple[int, int], ...] = VERTICAL_DETECTORS,
    ):
        self.n_bins = n_bins
        self.detectors = detectors
        self._history: list[np.ndarray] = []

    def reset(self) -> None:
        self._history.clear()

    def process(self, img: np.ndarray) -> np.ndarray:
        """``img`` is float grayscale in 0..1, full encode resolution. Returns length ``n_bins``."""
        self._history.append(_prepare(img))
        need = max(lag for _dy, lag in self.detectors)
        if len(self._history) > need + 1:
            self._history = self._history[-(need + 1):]

        acc = np.zeros(img.shape[1], dtype=np.float64)
        used = 0
        curr = self._history[-1]
        for dy, lag in self.detectors:
            if len(self._history) <= lag:
                continue
            prev = self._history[-1 - lag]
            acc += column_reichardt(prev, curr, dy)
            used += 1
        if used == 0 or acc.size == 0:
            return np.zeros(self.n_bins, dtype=np.float64)
        acc /= used
        x_col = np.linspace(-1.0, 1.0, acc.size, dtype=np.float64)
        x_bin = np.linspace(-1.0, 1.0, self.n_bins, dtype=np.float64)
        return np.interp(x_bin, x_col, acc).astype(np.float64)
