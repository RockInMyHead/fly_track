"""P0.1Q — local 2-D motion field on an image grid.

Why this exists
---------------
The frozen P0.1R+ frontend collapsed every image column to a single luminance
value and ran the Reichardt correlator on that 1-D profile. That throws away all
vertical structure in the scene before any motion is measured: a wall, the
floor, a door edge and a person all smear into one number per azimuth. P0.1P then
showed the resulting per-frame signal changes sign several times inside a real
turn, on every event, including the "good" ones.

This module measures motion *first* and reduces afterwards:

    frame
      ↓
    per-cell local light adaptation (temporal EMA mean/dev, one scalar per cell)
      ↓
    Reichardt correlator at full resolution, per image row
      ↓
    average within each grid cell  ->  signed 2-D motion field (grid_h × grid_w)

Only the local signed motion is produced. There is no pooled LEFT/RIGHT decision
and no `yaw_frozen` anywhere in this file.

Sign convention
---------------
Positive = image content moved toward +x, i.e. toward +azimuth, exactly the same
convention as `optic_flow.reichardt_at_dx` on the azimuth panorama. This is
guaranteed by construction: `reichardt_2d` applies that same expression to every
image row instead of to a 1-D profile.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-8
DEV_FLOOR = 0.05
CLIP = 3.0


@dataclass
class GridFieldDiagnostics:
    """Local-motion diagnostics. No pooled direction is present."""

    grid_shape: tuple[int, int] = (0, 0)
    n_pairs: int = 0
    per_detector_yaw: dict[str, float] = field(default_factory=dict)
    per_detector_energy: dict[str, float] = field(default_factory=dict)
    row_agreement: float = 0.0
    active_cells: int = 0
    field_abs_mean: float = 0.0


def reichardt_2d(prev: np.ndarray, curr: np.ndarray, dx: int) -> np.ndarray:
    """Signed horizontal Reichardt correlator at pixel offset `dx`, per row.

    Identical expression to `optic_flow.reichardt_at_dx`, applied to every image
    row rather than to a collapsed column profile:

        R[y, i] = prev[y, i] * curr[y, i+dx] - prev[y, i+dx] * curr[y, i]

    Output shape is (H, W - dx); column i refers to image column i.
    """
    prev = prev.astype(np.float32)
    curr = curr.astype(np.float32)
    if dx < 1 or prev.shape[1] <= dx:
        return np.zeros((prev.shape[0], 0), dtype=np.float32)
    right = prev[:, :-dx] * curr[:, dx:]
    left = prev[:, dx:] * curr[:, :-dx]
    return (right - left).astype(np.float32)


def azimuth_profile(field2d: np.ndarray, n_bins: int) -> np.ndarray:
    """Collapse a 2-D field over elevation into the azimuth axis used by injection.

    A yaw moves every elevation the same way, so collapsing rows loses no
    directional information — it only averages noise. The collapse happens
    *after* motion has been measured on real 2-D structure, which is the whole
    difference from the P0.1R+ panorama path.
    """
    field2d = np.asarray(field2d, dtype=np.float64)
    if field2d.size == 0 or n_bins <= 0:
        return np.zeros(max(n_bins, 0), dtype=np.float64)
    col = field2d.mean(axis=0)
    if col.size == 1:
        return np.full(n_bins, float(col[0]), dtype=np.float64)
    x_col = np.linspace(-1.0, 1.0, col.size)
    x_bin = np.linspace(-1.0, 1.0, n_bins)
    return np.interp(x_bin, x_col, col)


class GridMotionField:
    """Local signed motion on a `grid_h × grid_w` image grid."""

    def __init__(
        self,
        grid_w: int = 16,
        grid_h: int = 8,
        detectors: tuple[tuple[int, int, int, int], ...] = ((6, 2, 4, 2), (12, 3, 8, 3), (6, 3, 4, 3)),
        alpha: float = 0.04,
    ):
        """`detectors` entries are (dx_px, dt_steps, dx_bins, dt_bins).

        The bin-space dx/dt are the frozen P0.1R+ detectors and are used only to
        name the reported keys, so grid2d numbers stay comparable with the
        panorama path.
        """
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.detectors = tuple(tuple(int(v) for v in d) for d in detectors)
        self.alpha = float(alpha)
        self._hist: deque[np.ndarray] = deque()
        self._cell_id: np.ndarray | None = None
        self._cell_shape = (0, 0)
        self._mu: np.ndarray | None = None
        self._dev: np.ndarray | None = None

    # ---------------------------------------------------------------- geometry
    @property
    def n_cells(self) -> int:
        return self.grid_h * self.grid_w

    def reset(self) -> None:
        self._hist.clear()
        self._cell_id = None
        self._cell_shape = (0, 0)
        self._mu = None
        self._dev = None

    def _ensure_geometry(self, h: int, w: int) -> None:
        if self._cell_id is not None and self._cell_shape == (h, w):
            return
        rows = np.clip((np.arange(h) * self.grid_h) // max(h, 1), 0, self.grid_h - 1)
        cols = np.clip((np.arange(w) * self.grid_w) // max(w, 1), 0, self.grid_w - 1)
        self._cell_id = (rows[:, None] * self.grid_w + cols[None, :]).astype(np.int64)
        self._cell_shape = (h, w)
        self._mu = None
        self._dev = None

    def _cell_mean(self, img: np.ndarray) -> np.ndarray:
        n = self.n_cells
        sums = np.bincount(self._cell_id.ravel(), weights=img.ravel(), minlength=n)
        cnts = np.bincount(self._cell_id.ravel(), minlength=n).astype(np.float64)
        return sums / np.maximum(cnts, 1.0)

    # ----------------------------------------------------------------- process
    def process(self, img: np.ndarray) -> tuple[np.ndarray, GridFieldDiagnostics]:
        """One frame of grayscale [0,1] → (field2d, diagnostics)."""
        img = np.asarray(img, dtype=np.float32)
        h, w = img.shape
        self._ensure_geometry(h, w)

        # ---- per-cell local light adaptation (temporal EMA mean/deviation) ----
        cell_mean = self._cell_mean(img)
        a = self.alpha
        if self._mu is None:
            self._mu = cell_mean.copy()
            self._dev = np.full_like(cell_mean, DEV_FLOOR)
        else:
            self._mu = (1.0 - a) * self._mu + a * cell_mean
            self._dev = (1.0 - a) * self._dev + a * np.abs(cell_mean - self._mu)

        norm = (img - self._mu[self._cell_id]) / (self._dev[self._cell_id] + EPS)
        norm = np.clip(norm, -CLIP, CLIP).astype(np.float32)

        self._hist.append(norm)
        max_hist = max(d[1] for d in self.detectors) + 1
        while len(self._hist) > max_hist:
            self._hist.popleft()

        # ---- full-resolution correlator, reduced onto the grid ----
        n = self.n_cells
        acc = np.zeros(n, dtype=np.float64)
        n_pairs = 0
        per_yaw: dict[str, float] = {}
        per_energy: dict[str, float] = {}

        for dx_px, dt, dx_bins, dt_bins in self.detectors:
            if len(self._hist) <= dt:
                continue
            prev = self._hist[-1 - dt]
            curr = self._hist[-1]
            R = reichardt_2d(prev, curr, dx_px)
            if R.shape[1] == 0:
                continue
            cid = self._cell_id[:, : R.shape[1]]
            sums = np.bincount(cid.ravel(), weights=R.ravel().astype(np.float64), minlength=n)
            cnts = np.bincount(cid.ravel(), minlength=n).astype(np.float64)
            cell = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), 0.0)
            acc += cell
            n_pairs += 1
            key = f"R_dx{dx_bins}_dt{dt_bins}"
            per_yaw[f"{key}_yaw"] = float(cell.mean())
            per_energy[key] = float(np.mean(np.abs(cell)))

        if n_pairs:
            flat = acc / n_pairs
        else:
            flat = np.zeros(n, dtype=np.float64)
        field2d = flat.reshape(self.grid_h, self.grid_w)

        # ---- diagnostics (no pooled decision; these are local statistics) ----
        diag = GridFieldDiagnostics(
            grid_shape=(self.grid_h, self.grid_w),
            n_pairs=n_pairs,
            per_detector_yaw=per_yaw,
            per_detector_energy=per_energy,
            field_abs_mean=float(np.mean(np.abs(field2d))),
            active_cells=int(np.count_nonzero(np.abs(field2d) > 1e-4)),
        )
        # How much the elevation rows inside one azimuth column agree in sign.
        sgn = np.sign(field2d)
        col_consensus = np.abs(sgn.mean(axis=0))
        diag.row_agreement = float(col_consensus.mean()) if col_consensus.size else 0.0
        return field2d, diag
