"""Video → MaleCNS visual input (photoreceptors + local opponent motion inject).

The injected drive is a *local* signed motion field. No pooled LEFT/RIGHT decision
is computed anywhere in this module — that is the whole point of P0.1J. See
`fly_vo/local_motion_inject.py` for the mapping and its conventions.
"""

from __future__ import annotations

import cv2
import numpy as np

from .local_motion_inject import VERTICAL_FAMILIES, LocalOpponentMotionInjector
from .vertical_motion import VerticalMotionFrontend
from .malecns_engine import MaleCNSEngine
from .optic_flow import DirectionalMotionFrontend, FlowConfig, FlowDiagnostics
T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")
T4_T5_GROUP_NAMES = tuple(f"{t}_{s}" for t in T4_TYPES + T5_TYPES for s in ("L", "R"))

# Loom channel weights (LC4 / LPLC2). Loom is a magnitude pathway: it is driven by
# local image statistics only, never by a signed yaw decision.
LOOM_WEIGHT = 0.5
FRAME_DIFF_WEIGHT = 0.2
LPLC2_SCALE = 0.85


class VideoVisualEncoder:
    """Reichardt bank + temporal norm → local opponent T4a/b, T5a/b inject."""

    def __init__(
        self,
        engine: MaleCNSEngine,
        gain: float = 0.9,
        n_azimuth_bins: int = 128,
        flow_config: FlowConfig | None = None,
        azimuth_map: str | None = None,
    ):
        self.engine = engine
        self.gain = gain
        self.n_azimuth_bins = n_azimuth_bins
        fc = flow_config or FlowConfig(n_azimuth_bins=n_azimuth_bins)
        fc.n_azimuth_bins = n_azimuth_bins
        self.flow_config = fc
        self._frontend = DirectionalMotionFrontend(fc)
        self._last_flow: FlowDiagnostics | None = None

        engine._ensure_loaded()
        brain = engine.brain
        self.n_visual = len(brain.visual)
        self._azimuth = np.asarray(brain.azimuth, dtype=np.float32)

        self.loom_groups = {
            "LC4_L": self._cells(["LC4"], side="L"),
            "LC4_R": self._cells(["LC4"], side="R"),
            "LPLC2_L": self._cells(["LPLC2"], side="L"),
            "LPLC2_R": self._cells(["LPLC2"], side="R"),
        }

        self.motion_injector = LocalOpponentMotionInjector(
            brain, azimuth_npz=azimuth_map, field_bins=n_azimuth_bins
        )
        # Pitch pathway. encode_frame leaves it idle unless vertical=True, so the
        # yaw path that FINAL V1 already recorded does not change.
        self.vertical_injector = LocalOpponentMotionInjector(
            brain,
            azimuth_npz=azimuth_map,
            field_bins=n_azimuth_bins,
            families=VERTICAL_FAMILIES,
        )
        self._vertical = VerticalMotionFrontend(n_bins=n_azimuth_bins)
        # Kept for callers that enumerate T4/T5 populations (probes, viz).
        self.t4_t5_groups: dict[str, np.ndarray] = {
            key: self._cells([key.rsplit("_", 1)[0]], side=key[-1])
            for key in T4_T5_GROUP_NAMES
        }

    def _cells(self, types: list[str], side: str | None = None) -> np.ndarray:
        try:
            kwargs = {"side": side} if side else {}
            idx = self.engine.brain.cells(types, **kwargs)
            return idx if len(idx) else np.array([], dtype=int)
        except Exception:
            return np.array([], dtype=int)

    def encode_frame(
        self, bgr: np.ndarray, vertical: bool = False
    ) -> tuple[np.ndarray | None, list, dict]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = self._resize(gray)
        img = gray.astype(np.float32) / 255.0

        flow = self._frontend.process(img)
        self._last_flow = flow

        eye_drive = None
        if self.n_visual > 0:
            pan_full = np.asarray(flow.panorama, dtype=np.float32)
            if len(pan_full) < self.n_azimuth_bins:
                pan_full = np.pad(
                    pan_full, (0, self.n_azimuth_bins - len(pan_full)), mode="edge"
                )
            flat = self._pan_to_photoreceptors(pan_full[: self.n_azimuth_bins])
            eye_drive = np.clip(flat * self.gain, 0.0, 1.0).astype(np.float32)

        h, w = img.shape
        w2 = w // 2
        left_loom = float(np.abs(img[:, :w2].mean() - img[:, :w2].min()))
        right_loom = float(np.abs(img[:, w2:].mean() - img[:, w2:].min()))

        inject, diag = self._build_inject(flow, left_loom, right_loom)
        v_index = 0.0
        if vertical:
            vfield = self._vertical.process(img)
            v_entries, v_diag = self.vertical_injector.build(
                vfield,
                gain=self.gain,
                field_gain=self.flow_config.flow_inject_gain,
            )
            inject.extend(
                (cells, amt) for cells, amt in v_entries if len(cells) and amt > 1e-5
            )
            v_index = float(v_diag.opponent_index)

        metrics = flow.as_dict()
        metrics.update(
            {
                "motion": flow.frame_diff_mean,
                "mean_lum": flow.mean_luminance,
                "yaw_frozen": flow.yaw_frozen,  # logged for measurement only, never injected
                "eye_preview": flow.panorama,
                "pan_shift": flow.oracle_shift,
                "rotation_deg": 0.0,
                "step_rotation_deg": 0.0,
                "inject_mode": "local_opponent_2d" if vertical else "local_opponent",
                "vertical_opponent_index": v_index,
            }
        )
        metrics.update(diag.as_dict())
        return eye_drive, inject, metrics

    def _build_inject(
        self,
        flow: FlowDiagnostics,
        left_loom: float,
        right_loom: float,
    ) -> tuple[list, object]:
        """Local opponent motion for T4a/b + T5a/b, loom for LC4/LPLC2."""
        field = np.asarray(flow.signed_flow_field, dtype=np.float64)
        inject, diag = self.motion_injector.build(
            field,
            gain=self.gain,
            field_gain=self.flow_config.flow_inject_gain,
        )

        inject.append(
            (self.loom_groups["LC4_L"], self.gain * LOOM_WEIGHT * left_loom)
        )
        inject.append(
            (self.loom_groups["LPLC2_L"], self.gain * LOOM_WEIGHT * LPLC2_SCALE * left_loom)
        )
        inject.append(
            (self.loom_groups["LC4_R"], self.gain * LOOM_WEIGHT * right_loom)
        )
        inject.append(
            (self.loom_groups["LPLC2_R"], self.gain * LOOM_WEIGHT * LPLC2_SCALE * right_loom)
        )
        inject = [(cells, amt) for cells, amt in inject if len(cells) and amt > 1e-5]
        return inject, diag

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.engine.config
        return cv2.resize(
            gray,
            (cfg.encode_width, cfg.encode_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _pan_to_photoreceptors(self, pan: np.ndarray) -> np.ndarray:
        az = self._azimuth
        x_img = (az + 1.0) * 0.5
        x_pan = np.linspace(0.0, 1.0, len(pan), dtype=np.float32)
        return np.interp(x_img, x_pan, pan.astype(np.float32)).astype(np.float32)

    def reset(self) -> None:
        self._frontend.reset()
        self._vertical.reset()
        self._last_flow = None
