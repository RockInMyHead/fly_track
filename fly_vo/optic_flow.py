"""Reichardt detector bank + temporal normalization + diagnostics.

Two motion-field geometries are supported:

* ``field_mode="panorama"`` (default, frozen P0.1R+): every image column is
  collapsed to one luminance value and the correlator runs on that 1-D profile.
* ``field_mode="grid2d"`` (P0.1Q): the correlator runs at full resolution on
  every image row and the result is read out on a ``grid_h x grid_w`` grid, so
  the vertical structure of the scene survives into the motion estimate.

Nothing in the grid2d path reads ``yaw_frozen`` or any pooled direction; it
produces a field of *local* signed motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .local_motion_field import GridMotionField, azimuth_profile
from .lk_motion_field import LKMotionField

SPATIAL_DX = (1, 2, 4, 8)
TEMPORAL_LAGS = (1, 2, 3, 4, 5)
ORACLE_LAGS = (1, 2, 3, 4, 5)
EPS = 1e-8

# Frozen P0.1R+ readout — equal weights, do not tune per video.
FROZEN_YAW_DETECTORS: tuple[tuple[int, int], ...] = ((4, 2), (8, 3), (4, 3))

# The same three frozen detectors expressed in *pixels* instead of azimuth bins.
# One azimuth bin = encode_width / n_azimuth_bins = 192/128 = 1.5 px, so
# dx 4 -> 6 px and dx 8 -> 12 px. Frozen, not tuned on results.
FROZEN_YAW_DETECTORS_PX: tuple[tuple[int, int], ...] = ((6, 2), (12, 3), (6, 3))


@dataclass
class FlowConfig:
    n_azimuth_bins: int = 128
    ema_tau_s: float = 0.5
    brain_dt: float = 0.02
    crop_mode: str = "center_band"  # full_frame | center_band
    crop_y0: float = 0.30
    crop_y1: float = 0.70
    spatial_dx: tuple[int, ...] = SPATIAL_DX
    temporal_lags: tuple[int, ...] = TEMPORAL_LAGS
    flow_inject_gain: float = 8.0
    # P0.1Q motion-field geometry.
    field_mode: str = "panorama"  # panorama | grid2d | lk
    grid_w: int = 16
    grid_h: int = 8
    grid_dx: tuple[int, ...] = tuple(p[0] for p in FROZEN_YAW_DETECTORS_PX)
    grid_dt: tuple[int, ...] = tuple(p[1] for p in FROZEN_YAW_DETECTORS_PX)
    # P0.1S Lucas-Kanade tracking (field_mode="lk").
    lk_seed_layout: int = 3
    lk_win: int = 15
    lk_max_level: int = 3
    lk_fb_tol_px: float = 1.0
    lk_ref_px: float = 1.0
    lk_hold_on_duplicate: bool = True


@dataclass
class FlowDiagnostics:
    """All pre-brain directional motion signals."""

    mean_luminance: float = 0.0
    contrast: float = 0.0
    frame_diff_mean: float = 0.0
    flow_L: float = 0.0
    flow_R: float = 0.0
    yaw_signal: float = 0.0
    yaw_frozen: float = 0.0
    expansion_signal: float = 0.0
    abs_flow_L: float = 0.0
    abs_flow_R: float = 0.0
    coherence_L: float = 0.0
    coherence_R: float = 0.0
    motion_energy: float = 0.0
    oracle_shift: float = 0.0
    oracle_bank: dict[str, float] = field(default_factory=dict)
    crop_mode: str = "full_frame"
    detector_bank: dict[str, float] = field(default_factory=dict)
    detector_signed: dict[str, float] = field(default_factory=dict)
    signed_flow_field: list[float] = field(default_factory=list)
    panorama: list[float] = field(default_factory=list)
    # P0.1Q: local 2-D motion field (grid_h rows x grid_w cols) and its geometry.
    motion_field_2d: list[list[float]] = field(default_factory=list)
    grid_shape: tuple[int, int] = (0, 0)
    grid_row_agreement: float = 0.0
    field_mode: str = "panorama"
    comb_empty_bins: int = 0

    def as_dict(self) -> dict:
        d = {
            "mean_luminance": self.mean_luminance,
            "contrast": self.contrast,
            "frame_diff_mean": self.frame_diff_mean,
            "flow_L": self.flow_L,
            "flow_R": self.flow_R,
            "yaw_signal": self.yaw_signal,
            "yaw_frozen": self.yaw_frozen,
            "expansion_signal": self.expansion_signal,
            "abs_flow_L": self.abs_flow_L,
            "abs_flow_R": self.abs_flow_R,
            "coherence_L": self.coherence_L,
            "coherence_R": self.coherence_R,
            "motion_energy": self.motion_energy,
            "oracle_shift": self.oracle_shift,
            "crop_mode": self.crop_mode,
            "field_mode": self.field_mode,
            "comb_empty_bins": self.comb_empty_bins,
            "grid_row_agreement": self.grid_row_agreement,
        }
        d.update(self.detector_bank)
        d.update(self.detector_signed)
        d.update(self.oracle_bank)
        return d


def apply_crop(img: np.ndarray, cfg: FlowConfig) -> np.ndarray:
    if cfg.crop_mode != "center_band":
        return img
    h = img.shape[0]
    y0 = int(cfg.crop_y0 * h)
    y1 = max(y0 + 1, int(cfg.crop_y1 * h))
    return img[y0:y1, :]


def legacy_azimuth_bin_counts(w: int, n_bins: int) -> np.ndarray:
    """Column counts produced by the old floor-binning (kept for auditing only).

    This is the mapping that created the 96 -> 128 comb: with `n_bins > w` some
    bins received no column at all. Only used to report how many bins *used* to
    be empty.
    """
    counts = np.zeros(n_bins, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    for col in range(w):
        b = int(np.floor((xs[col] + 1.0) * 0.5 * (n_bins - 1)))
        counts[int(np.clip(b, 0, n_bins - 1))] += 1.0
    return counts


def build_azimuth_panorama(img: np.ndarray, n_bins: int = 128) -> np.ndarray:
    """Map grayscale image to azimuth bins (left=-1 → right=+1). No per-frame norm.

    P0.1Q comb fix. The columns of `img` are *resampled* onto the bin centres by
    linear interpolation instead of being floor-binned. The previous version
    assigned each column to a single bin, so whenever `n_bins > encode_width`
    (128 > 96 here) a quarter of the bins never received a column and stayed
    permanently zero — a fixed comb superimposed on every panorama. Interpolation
    makes empty bins structurally impossible at any ratio of width to bins, and
    also stops columns from being silently dropped when `n_bins < encode_width`.
    """
    h, w = img.shape
    if n_bins <= 0:
        return np.zeros(0, dtype=np.float32)
    col = img.mean(axis=0).astype(np.float32)
    if w <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    if w == 1:
        return np.full(n_bins, float(col[0]), dtype=np.float32)
    x_col = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    x_bin = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
    return np.interp(x_bin, x_col, col).astype(np.float32)


def reichardt_at_dx(prev: np.ndarray, curr: np.ndarray, dx: int) -> np.ndarray:
    """Signed Reichardt correlator at spatial offset dx (not abs)."""
    prev = prev.astype(np.float32)
    curr = curr.astype(np.float32)
    if dx < 1 or len(prev) <= dx:
        return np.zeros(0, dtype=np.float32)
    right = prev[:-dx] * curr[dx:]
    left = prev[dx:] * curr[:-dx]
    n = min(len(right), len(left))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return (right[:n] - left[:n]).astype(np.float32)


def hemifield_stats(flow_field: np.ndarray) -> dict[str, float]:
    if len(flow_field) < 4:
        return {
            "flow_L": 0.0,
            "flow_R": 0.0,
            "abs_flow_L": 0.0,
            "abs_flow_R": 0.0,
            "coherence_L": 0.0,
            "coherence_R": 0.0,
            "yaw_signal": 0.0,
            "expansion_signal": 0.0,
            "motion_energy": 0.0,
        }
    mid = len(flow_field) // 2
    left = flow_field[:mid]
    right = flow_field[mid:]
    flow_l = float(np.mean(left))
    flow_r = float(np.mean(right))
    abs_l = float(np.mean(np.abs(left)))
    abs_r = float(np.mean(np.abs(right)))
    coh_l = abs(flow_l) / (abs_l + EPS)
    coh_r = abs(flow_r) / (abs_r + EPS)
    return {
        "flow_L": flow_l,
        "flow_R": flow_r,
        "abs_flow_L": abs_l,
        "abs_flow_R": abs_r,
        "coherence_L": coh_l,
        "coherence_R": coh_r,
        "yaw_signal": (flow_l + flow_r) / 2.0,
        "expansion_signal": (flow_r - flow_l) / 2.0,
        "motion_energy": float(np.mean(np.abs(flow_field))),
    }


def oracle_phase_shift(prev: np.ndarray, curr: np.ndarray) -> float:
    """Diagnostic-only 1D phase correlation shift (fraction of panorama width)."""
    if len(prev) != len(curr) or len(prev) < 8:
        return 0.0
    p = prev.astype(np.float32).reshape(1, -1)
    c = curr.astype(np.float32).reshape(1, -1)
    try:
        (dx, _), _ = cv2.phaseCorrelate(p, c)
        return float(np.clip(dx / len(curr), -0.5, 0.5))
    except cv2.error:
        return 0.0


class TemporalNormalizer:
    """Per-azimuth EMA mean + deviation normalization (not per-frame)."""

    def __init__(self, n_bins: int, alpha: float):
        self.n_bins = n_bins
        self.alpha = alpha
        self._mu: np.ndarray | None = None
        self._dev: np.ndarray | None = None

    def reset(self) -> None:
        self._mu = None
        self._dev = None

    def normalize(self, pan: np.ndarray) -> np.ndarray:
        pan = pan.astype(np.float32)
        a = self.alpha
        if self._mu is None:
            self._mu = pan.copy()
            self._dev = np.full_like(pan, 0.05, dtype=np.float32)
        else:
            self._mu = (1.0 - a) * self._mu + a * pan
            self._dev = (1.0 - a) * self._dev + a * np.abs(pan - self._mu)
        out = (pan - self._mu) / (self._dev + EPS)
        return np.clip(out, -3.0, 3.0).astype(np.float32)


class DirectionalMotionFrontend:
    """Multi-scale Reichardt bank + temporal normalization + oracle."""

    def __init__(self, cfg: FlowConfig | None = None):
        self.cfg = cfg or FlowConfig()
        alpha = self.cfg.brain_dt / max(self.cfg.ema_tau_s, 1e-3)
        alpha = float(np.clip(alpha, 0.01, 0.5))
        self._normalizer = TemporalNormalizer(self.cfg.n_azimuth_bins, alpha)
        self._grid = GridMotionField(
            grid_w=self.cfg.grid_w,
            grid_h=self.cfg.grid_h,
            detectors=tuple(
                (dx_px, dt, dx_bins, dt_bins)
                for (dx_px, dt), (dx_bins, dt_bins) in zip(
                    zip(self.cfg.grid_dx, self.cfg.grid_dt), FROZEN_YAW_DETECTORS
                )
            ),
            alpha=alpha,
        )
        self._lk = LKMotionField(
            grid_w=self.cfg.grid_w,
            grid_h=self.cfg.grid_h,
            seed_layout=self.cfg.lk_seed_layout,
            win=self.cfg.lk_win,
            max_level=self.cfg.lk_max_level,
            fb_tol_px=self.cfg.lk_fb_tol_px,
            ref_px=self.cfg.lk_ref_px,
            hold_on_duplicate=self.cfg.lk_hold_on_duplicate,
        )
        self._pan_history: list[np.ndarray] = []
        self._pan_raw_history: list[np.ndarray] = []
        self._prev_img: np.ndarray | None = None
        self._step_i = 0

    def reset(self) -> None:
        self._normalizer.reset()
        self._grid.reset()
        self._lk.reset()
        self._pan_history.clear()
        self._pan_raw_history.clear()
        self._prev_img = None
        self._step_i = 0

    def process(self, img: np.ndarray) -> FlowDiagnostics:
        """img: float32 grayscale [0,1], full encode resolution."""
        cfg = self.cfg
        cropped = apply_crop(img, cfg)
        mean_lum = float(img.mean())

        h, w = img.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        yg, xg = np.ogrid[:h, :w]
        mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= r**2
        centre = float(img[mask].mean()) if mask.any() else mean_lum
        surround = float(img[~mask].mean()) if (~mask).any() else mean_lum
        contrast = abs(centre - surround)

        diff_mean = 0.0
        if self._prev_img is not None:
            diff_mean = float(np.abs(img - self._prev_img).mean())
        self._prev_img = img.copy()

        pan_raw = build_azimuth_panorama(cropped, cfg.n_azimuth_bins)
        comb_empty = int(
            np.count_nonzero(
                legacy_azimuth_bin_counts(cropped.shape[1], cfg.n_azimuth_bins) == 0
            )
        )
        pan_norm = self._normalizer.normalize(pan_raw)
        self._pan_history.append(pan_norm)
        self._pan_raw_history.append(pan_raw)
        max_hist = max(max(cfg.temporal_lags), max(ORACLE_LAGS)) + 1
        if len(self._pan_history) > max_hist:
            self._pan_history = self._pan_history[-max_hist:]
            self._pan_raw_history = self._pan_raw_history[-max_hist:]

        oracle_bank: dict[str, float] = {}
        for lag in ORACLE_LAGS:
            if len(self._pan_raw_history) > lag:
                oracle_bank[f"oracle_dt{lag}"] = oracle_phase_shift(
                    self._pan_raw_history[-1 - lag], self._pan_raw_history[-1]
                )
        oracle = oracle_bank.get("oracle_dt1", 0.0)
        self._step_i += 1

        bank: dict[str, float] = {}
        signed_bank: dict[str, float] = {}
        field_by_key: dict[str, np.ndarray] = {}
        motion_field_2d: list[list[float]] = []
        grid_shape = (0, 0)
        grid_row_agreement = 0.0

        if cfg.field_mode == "lk":
            # P0.1S: pyramidal Lucas-Kanade on a patch grid. Real displacements,
            # not a luminance-product correlator. Still purely local.
            field2d_px, field_dy, ldiag = self._lk.process(cropped)
            bank.update(ldiag.as_dict())
            bank.update({
                "LK_abs_mean": ldiag.mean_abs_dx_px,
                "LK_abs_dy_mean": ldiag.mean_abs_dy_px,
            })
            signed_bank.update({
                "LK_dx_mean": float(field2d_px.mean()),
                "LK_dy_mean": float(field_dy.mean()),
            })
            signed_field = azimuth_profile(field2d_px, cfg.n_azimuth_bins).astype(np.float32)
            motion_field_2d = field2d_px.tolist()
            grid_shape = (int(field2d_px.shape[0]), int(field2d_px.shape[1]))
            # For the tracking path this slot reports seed coverage.
            grid_row_agreement = float(ldiag.coverage)
            frozen_yaw_vals = []
        elif cfg.field_mode == "grid2d":
            # P0.1Q: full-resolution correlator, read out on a 2-D grid. No
            # pooled direction is used here; the azimuth profile below is only
            # the injection interface (yaw is elevation-invariant).
            field2d, gdiag = self._grid.process(cropped)
            bank.update(gdiag.per_detector_energy)
            signed_bank.update(gdiag.per_detector_yaw)
            signed_field = azimuth_profile(field2d, cfg.n_azimuth_bins).astype(np.float32)
            motion_field_2d = field2d.tolist()
            grid_shape = (int(field2d.shape[0]), int(field2d.shape[1]))
            grid_row_agreement = float(gdiag.row_agreement)
            frozen_yaw_vals = [
                gdiag.per_detector_yaw.get(f"R_dx{dx}_dt{lag}_yaw", 0.0)
                for dx, lag in FROZEN_YAW_DETECTORS
            ]
        else:
            for lag in cfg.temporal_lags:
                if len(self._pan_history) <= lag:
                    continue
                prev_t = self._pan_history[-1 - lag]
                curr_t = self._pan_history[-1]
                for dx in cfg.spatial_dx:
                    field = reichardt_at_dx(prev_t, curr_t, dx)
                    if len(field) == 0:
                        continue
                    key = f"R_dx{dx}_dt{lag}"
                    st = hemifield_stats(field)
                    bank[key] = float(np.mean(np.abs(field)))
                    signed_bank[f"{key}_yaw"] = st["yaw_signal"]
                    field_by_key[key] = field

            frozen_fields = [
                field_by_key[f"R_dx{dx}_dt{lag}"]
                for dx, lag in FROZEN_YAW_DETECTORS
                if f"R_dx{dx}_dt{lag}" in field_by_key
            ]
            if frozen_fields:
                min_len = min(len(f) for f in frozen_fields)
                combined = np.zeros(min_len, dtype=np.float64)
                for f in frozen_fields:
                    combined += f[:min_len]
                combined /= len(frozen_fields)
                signed_field = combined.astype(np.float32)
            else:
                signed_field = np.zeros(max(cfg.n_azimuth_bins - 1, 0), dtype=np.float32)

            frozen_yaw_vals = [
                signed_bank.get(f"R_dx{dx}_dt{lag}_yaw", 0.0)
                for dx, lag in FROZEN_YAW_DETECTORS
            ]

        stats = hemifield_stats(signed_field)
        if frozen_yaw_vals:
            yaw_frozen = float(np.mean(frozen_yaw_vals))
        else:
            # This mode has no frozen dx/dt detector bank (the tracking path
            # measures displacement directly). Report the pooled profile mean as
            # the comparable logging quantity — still never injected anywhere.
            yaw_frozen = float(stats["yaw_signal"])
        stats["yaw_signal"] = yaw_frozen

        return FlowDiagnostics(
            mean_luminance=mean_lum,
            contrast=contrast,
            frame_diff_mean=diff_mean,
            flow_L=stats["flow_L"],
            flow_R=stats["flow_R"],
            yaw_signal=yaw_frozen,
            yaw_frozen=yaw_frozen,
            expansion_signal=stats["expansion_signal"],
            abs_flow_L=stats["abs_flow_L"],
            abs_flow_R=stats["abs_flow_R"],
            coherence_L=stats["coherence_L"],
            coherence_R=stats["coherence_R"],
            motion_energy=stats["motion_energy"],
            oracle_shift=oracle,
            oracle_bank=oracle_bank,
            crop_mode=cfg.crop_mode,
            detector_bank=bank,
            detector_signed=signed_bank,
            signed_flow_field=signed_field.tolist(),
            panorama=pan_raw[: min(64, len(pan_raw))].tolist(),
            motion_field_2d=motion_field_2d,
            grid_shape=grid_shape,
            grid_row_agreement=grid_row_agreement,
            field_mode=cfg.field_mode,
            comb_empty_bins=comb_empty,
        )


def tanh_drive(gain: float, value: float) -> float:
    return float(np.tanh(gain * value))
