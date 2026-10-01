"""P0.1S — local motion by Lucas–Kanade tracking on a patch grid.

Why this replaces the Reichardt correlator
------------------------------------------
P0.1Q measured motion on a real 2-D grid instead of a collapsed column profile
and the signal improved per frame, but a cell-level audit showed the measurement
was still at chance: on the six real turns only 1% of grid cells held the correct
sign for 70% of the turn, and the pooled estimate was right 53% of the time. The
correlator itself — a product of two luminance samples at a fixed offset — is too
weak on this footage.

This module measures the same thing a different way: it tracks real image points
with pyramidal Lucas–Kanade and reports the actual displacement of each patch.

    frame t, frame t+1
        ↓
    fixed grid of image points (3 x 3 per cell)
        ↓
    calcOpticalFlowPyrLK  prev -> curr, then curr -> prev
        ↓
    keep points whose forward-backward error is small
        ↓
    per-cell median displacement  ->  local motion (dx, dy) in pixels

Only local motion is produced. There is no pooled LEFT/RIGHT decision and no
`yaw_frozen` anywhere in this file: the sign, polarity and direction conventions
live entirely in the fly's circuitry downstream.

Sign convention
---------------
`dx > 0` means the image content moved toward +x, i.e. rightward. This is the
same convention as `optic_flow.reichardt_at_dx` (verified by
`tests/test_p01s_lk_field.py::test_sign_matches_reichardt`), so the injection
polarities frozen in P0.1L stay valid and the two estimators are comparable.

Frame clock
-----------
`iter_video_at_brain_hz` samples 30 fps video at 50 Hz, so 40% of frames handed
to the estimator are exact copies of the previous frame. Lucas–Kanade is
*undefined* on such a pair: there is no displacement between two identical
images. With `hold_on_duplicate=True` the previous field is carried forward and
the sample is marked held, instead of reporting a spurious zero field. The number
of held samples is reported so the effect stays visible.

Units
-----
Displacement is reported in pixels divided by `ref_px` (default 1.0), so a
one-pixel frame-to-frame displacement is a field value of 1.0. This keeps the
frozen injection gain (`flow_inject_gain = 8.0`) meaningful: the frozen Reichardt
field saturated it around |field| ~ 0.25, and so does this one.

8-bit input
-----------
`cv2.calcOpticalFlowPyrLK` in this OpenCV build asserts `img.depth() == CV_8U`,
so the float drive is converted with a *fixed* scale (value * 255, no per-frame
percentile stretch). An adaptive stretch would break the brightness-constancy
assumption that Lucas–Kanade is built on, so it is deliberately avoided.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

EPS = 1e-8


@dataclass
class LKFieldDiagnostics:
    """Local tracking diagnostics. No pooled direction is present."""

    grid_shape: tuple[int, int] = (0, 0)
    n_seeds: int = 0
    n_valid: int = 0
    coverage: float = 0.0
    mean_fb_error: float = 0.0
    mean_abs_dx_px: float = 0.0
    mean_abs_dy_px: float = 0.0
    held: bool = False
    n_empty_cells: int = 0

    def as_dict(self) -> dict:
        return {
            "lk_coverage": self.coverage,
            "lk_n_valid": self.n_valid,
            "lk_mean_fb_error": self.mean_fb_error,
            "lk_mean_abs_dx_px": self.mean_abs_dx_px,
            "lk_mean_abs_dy_px": self.mean_abs_dy_px,
            "lk_held": self.held,
            "lk_empty_cells": self.n_empty_cells,
        }


class LKMotionField:
    """Per-cell local displacement from pyramidal Lucas–Kanade tracking."""

    def __init__(
        self,
        grid_w: int = 16,
        grid_h: int = 8,
        seed_layout: int = 3,
        win: int = 15,
        max_level: int = 3,
        fb_tol_px: float = 1.0,
        ref_px: float = 1.0,
        hold_on_duplicate: bool = True,
        fill_empty_from_column: bool = True,
    ):
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.seed_layout = max(int(seed_layout), 1)
        self.win = (int(win), int(win))
        self.max_level = int(max_level)
        self.fb_tol_px = float(fb_tol_px)
        self.ref_px = float(ref_px)
        self.hold_on_duplicate = bool(hold_on_duplicate)
        self.fill_empty_from_column = bool(fill_empty_from_column)

        self._prev: np.ndarray | None = None      # float drive, for duplicate detection
        self._prev_u8: np.ndarray | None = None   # 8-bit, for the tracker
        self._prev_field: np.ndarray | None = None
        self._prev_dy: np.ndarray | None = None
        self._seeds: np.ndarray | None = None
        self._seed_cell: np.ndarray | None = None
        self._shape: tuple[int, int] = (0, 0)

        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            0.03,
        )

    # ---------------------------------------------------------------- geometry
    @property
    def n_cells(self) -> int:
        return self.grid_h * self.grid_w

    def reset(self) -> None:
        self._prev = None
        self._prev_u8 = None
        self._prev_field = None
        self._prev_dy = None
        self._seeds = None
        self._seed_cell = None
        self._shape = (0, 0)

    def _ensure_seeds(self, h: int, w: int) -> None:
        """Fixed image-space seed grid, one `seed_layout^2` sub-grid per cell."""
        if self._seeds is not None and self._shape == (h, w):
            return
        margin = self.win[0] // 2 + 1
        cell_w, cell_h = w / self.grid_w, h / self.grid_h
        pts, cells = [], []
        L = self.seed_layout
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                for sy in range(L):
                    for sx in range(L):
                        x = (gx + (sx + 0.5) / L) * cell_w
                        y = (gy + (sy + 0.5) / L) * cell_h
                        if margin <= x <= w - 1 - margin and margin <= y <= h - 1 - margin:
                            pts.append((x, y))
                            cells.append(gy * self.grid_w + gx)
        self._seeds = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        self._seed_cell = np.asarray(cells, dtype=np.int64)
        self._shape = (h, w)
        self._prev = None
        self._prev_u8 = None
        self._prev_field = None
        self._prev_dy = None

    @staticmethod
    def _to_u8(img: np.ndarray) -> np.ndarray:
        """Fixed-scale conversion to 8-bit; no per-frame adaptation."""
        return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------- process
    def process(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, LKFieldDiagnostics]:
        """One frame → (dx field, dy field, diagnostics); fields in px / ref_px."""
        img = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
        h, w = img.shape
        self._ensure_seeds(h, w)
        zeros = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

        if self._prev is None:
            self._prev = img
            self._prev_u8 = self._to_u8(img)
            self._prev_field, self._prev_dy = zeros, zeros
            return zeros, zeros, LKFieldDiagnostics(
                grid_shape=(self.grid_h, self.grid_w), n_seeds=len(self._seed_cell)
            )

        diag = LKFieldDiagnostics(
            grid_shape=(self.grid_h, self.grid_w), n_seeds=int(len(self._seed_cell))
        )

        # A pair of identical frames carries no displacement: hold the last field.
        if self.hold_on_duplicate and np.array_equal(img, self._prev):
            diag.held = True
            diag.coverage = 1.0
            prev_field = self._prev_field if self._prev_field is not None else zeros
            prev_dy = self._prev_dy if self._prev_dy is not None else zeros
            self._prev = img
            return prev_field.copy(), prev_dy.copy(), diag

        curr_u8 = self._to_u8(img)
        prev_u8 = self._prev_u8 if self._prev_u8 is not None else curr_u8
        p0 = self._seeds
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            prev_u8, curr_u8, p0, None,
            winSize=self.win, maxLevel=self.max_level, criteria=self._criteria,
        )
        p0r, st0, _ = cv2.calcOpticalFlowPyrLK(
            curr_u8, prev_u8, p1, None,
            winSize=self.win, maxLevel=self.max_level, criteria=self._criteria,
        )
        good = (
            st1.reshape(-1).astype(bool)
            & st0.reshape(-1).astype(bool)
            & np.isfinite(p1.reshape(-1, 2)).all(axis=1)
            & np.isfinite(p0r.reshape(-1, 2)).all(axis=1)
        )
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good &= fb <= self.fb_tol_px

        n_valid = int(good.sum())
        diag.n_valid = n_valid
        diag.coverage = n_valid / max(len(self._seed_cell), 1)
        diag.mean_fb_error = float(fb[good].mean()) if n_valid else 0.0

        d = (p1.reshape(-1, 2) - p0.reshape(-1, 2))[good]
        cells = self._seed_cell[good]

        dx = np.full(self.n_cells, np.nan, dtype=np.float64)
        dy = np.full(self.n_cells, np.nan, dtype=np.float64)
        for ci in range(self.n_cells):
            m = cells == ci
            if not m.any():
                continue
            dx[ci] = float(np.median(d[m, 0]))
            dy[ci] = float(np.median(d[m, 1]))

        diag.n_empty_cells = int(np.isnan(dx).sum())
        if self.fill_empty_from_column:
            dx = self._fill_from_column(dx)
            dy = self._fill_from_column(dy)

        dx = np.nan_to_num(dx, nan=0.0)
        dy = np.nan_to_num(dy, nan=0.0)
        diag.mean_abs_dx_px = float(np.mean(np.abs(dx)))
        diag.mean_abs_dy_px = float(np.mean(np.abs(dy)))

        field_dx = (dx / self.ref_px).reshape(self.grid_h, self.grid_w)
        field_dy = (dy / self.ref_px).reshape(self.grid_h, self.grid_w)

        self._prev = img
        self._prev_u8 = curr_u8
        self._prev_field, self._prev_dy = field_dx, field_dy
        return field_dx, field_dy, diag

    def _fill_from_column(self, vals: np.ndarray) -> np.ndarray:
        """Fill empty cells from the same azimuth column (yaw is elevation-invariant)."""
        g = vals.reshape(self.grid_h, self.grid_w).copy()
        for x in range(self.grid_w):
            col = g[:, x]
            ok = ~np.isnan(col)
            if ok.all() or not ok.any():
                continue
            g[~ok, x] = float(np.median(col[ok]))
        return g.reshape(-1)
