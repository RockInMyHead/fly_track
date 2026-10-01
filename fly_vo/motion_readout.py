"""Descending-neuron + pathway readout → ego-motion (forward / lateral / yaw)."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import FlyVOConfig
from .dopamine import AdaptiveParams
from .malecns_engine import BrainStepResult


class DescendingNeuronDecoder:
    """
    Map MaleCNS activity to body-frame velocity.

    Primary: descending neurons (DNa02 / DNg100 / MDN) when they fire.
    Fallback: visual_projection + looming + pan_shift when DN stay quiet
    (common with video inject — signal reaches VP but rarely DN at rest).
    """

    GROUPS = (
        "steer_L",
        "steer_R",
        "forward_L",
        "forward_R",
        "backward_L",
        "backward_R",
    )

    def __init__(
        self,
        config: FlyVOConfig | None = None,
        adaptive: AdaptiveParams | None = None,
    ):
        self.config = config or FlyVOConfig()
        self.adaptive = adaptive or AdaptiveParams()
        self._dn_history: deque[dict[str, float]] = deque(
            maxlen=self.config.steer_window
        )
        self._pathway_history: deque[dict[str, int]] = deque(
            maxlen=self.config.steer_window
        )
        self._last_source: str = "none"

    def reset(self) -> None:
        self._dn_history.clear()
        self._pathway_history.clear()
        self._last_source = "none"

    @property
    def last_source(self) -> str:
        return self._last_source

    def observe(self, step: BrainStepResult) -> None:
        self._dn_history.append(step.descending)
        self._pathway_history.append(step.pathway_spikes)

    def _dn_count(self, *groups: str, window: int | None = None) -> float:
        window = window or self.config.steer_window
        recent = list(self._dn_history)[-window:]
        if not recent:
            return 0.0
        return float(sum(d.get(g, 0.0) for d in recent for g in groups))

    def _pathway_mean(self, name: str, window: int = 5) -> float:
        recent = list(self._pathway_history)[-window:]
        if not recent:
            return 0.0
        return float(np.mean([d.get(name, 0) for d in recent]))

    def _descending_activity(self, window: int = 5) -> float:
        """Mean descending spikes per brain step (not sum over window)."""
        recent = list(self._dn_history)[-window:]
        if not recent:
            return 0.0
        per_step = [sum(d.get(g, 0.0) for g in self.GROUPS) for d in recent]
        return float(np.mean(per_step))

    def _decode_descending(self) -> tuple[float, float, float, float]:
        cfg = self.config
        ap = self.adaptive

        steer_l = self._dn_count("steer_L")
        steer_r = self._dn_count("steer_R")
        fwd = self._dn_count("forward_L", "forward_R")
        back = self._dn_count("backward_L", "backward_R")

        turn_spikes = steer_r - steer_l
        yaw_rate = 0.0
        vy = ap.lateral_bias

        if abs(turn_spikes) >= cfg.steer_margin:
            sign = 1.0 if turn_spikes > 0 else -1.0
            yaw_rate = sign * cfg.turn_gain * ap.yaw_gain_scale * abs(turn_spikes)
            vy += sign * cfg.lateral_gain * ap.yaw_gain_scale

        vx = cfg.forward_gain * ap.speed_scale * max(0.0, fwd - back)
        vx = min(vx, cfg.speed_cap)

        if abs(yaw_rate) < cfg.yaw_deadzone:
            yaw_rate = 0.0

        activity = fwd + back + abs(turn_spikes)
        confidence = float(np.clip(activity / (cfg.steer_margin * 4 + 1e-6), 0.05, 1.0))
        return vx, vy, yaw_rate, confidence

    def _decode_pathway(self, metrics: dict) -> tuple[float, float, float, float]:
        """Fallback when descending neurons are silent — uses optic-lobe drive + pan shift."""
        cfg = self.config
        ap = self.adaptive

        motion = float(metrics.get("motion", 0.0))
        pan = float(metrics.get("pan_shift", 0.0))
        rot_deg = float(metrics.get("rotation_deg", 0.0))
        loom = self._pathway_mean("looming", window=3)
        vp = self._pathway_mean("visual_projection", window=3)

        motion_norm = motion / max(cfg.pathway_motion_ref, 1e-6)
        vx = cfg.pathway_forward_gain * ap.speed_scale * (
            0.65 * motion_norm + 0.35 * loom * cfg.pathway_loom_gain
        )
        vx = float(np.clip(vx, 0.0, cfg.speed_cap))

        dt = max(cfg.brain_dt, 1e-6)
        # Sharp turns: ORB rotation (deg/step). Gentle drift: panorama pan shift.
        if abs(rot_deg) >= cfg.rotation_snap_deg:
            # Heading updated via compass.nudge in streaming — avoid double integration
            yaw_rate = 0.0
        elif abs(rot_deg) >= 0.08:
            yaw_rate = (
                -cfg.pathway_rot_gain
                * ap.yaw_gain_scale
                * float(np.radians(rot_deg))
                / dt
            )
        else:
            # Pan shift: positive = scene moves right → camera yaws left
            yaw_rate = -cfg.pathway_yaw_gain * ap.yaw_gain_scale * pan
        # Lateral only from learned bias — pan-derived vy caused left drift on straights.
        vy = ap.lateral_bias

        if abs(yaw_rate) < cfg.yaw_deadzone and abs(rot_deg) < 0.08:
            yaw_rate = 0.0
        # On straight segments ignore pan rotation (keep heading at 90°).
        if abs(rot_deg) < 2.2:
            yaw_rate = 0.0

        confidence = float(
            np.clip(
                0.3 * min(motion_norm, 2.0)
                + 0.3 * min(vp / 1500.0, 1.0)
                + 0.2 * min(loom / 100.0, 1.0)
                + 0.2 * min(abs(pan) * 50, 1.0),
                0.1,
                1.0,
            )
        )
        return vx, vy, yaw_rate, confidence

    def decode_step(
        self, metrics: dict | None = None
    ) -> tuple[float, float, float, float]:
        """
        Returns (vx_body, vy_body, yaw_rate, confidence).
        vx forward (+), vy right (+), yaw_rate rad/s (+ = turn right).
        """
        dn_act = self._descending_activity(window=5)
        if dn_act >= self.config.descending_min_activity:
            self._last_source = "descending"
            return self._decode_descending()

        if metrics and self.config.use_pathway_fallback:
            self._last_source = "pathway"
            return self._decode_pathway(metrics)

        self._last_source = "descending"
        return self._decode_descending()
