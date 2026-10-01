"""Estimate camera ego-motion from FlyVis neural activity only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dopamine import AdaptiveParams
from .flyvis_engine import FlyVisOutput


@dataclass
class EgoMotionFrame:
    vx_body: float  # forward (+)
    vy_body: float  # right (+)
    yaw_rate: float  # rad/s, positive = turn right
    confidence: float
    turning: bool = False


@dataclass
class EgoMotionSequence:
    vx: np.ndarray
    vy: np.ndarray
    yaw_rate: np.ndarray
    confidence: np.ndarray
    timestamps: np.ndarray
    turning: np.ndarray | None = None


class FlyEgoMotionDecoder:
    """
    Ego-motion from T4/T5 cardinal motion detectors.

    Critical: T4/T5 encode *optic flow on retina*, which is opposite
    to camera self-motion. When camera moves forward, scene moves
    backward -> T4a/T5a (180 deg) fire more than T4b/T5b (0 deg).

    Self-motion is therefore:
      forward  ~ T4a+T5a - T4b-T5b  (backward flow - forward flow)
      right    ~ T4d+T5d - T4c-T5c  (left flow - right flow)
    """

    CARDINAL = {
        "forward": ("T4b", "T5b"),
        "backward": ("T4a", "T5a"),
        "right": ("T4c", "T5c"),
        "left": ("T4d", "T5d"),
    }

    def __init__(
        self,
        smooth_alpha: float = 0.2,
        turn_threshold: float = 0.25,
        speed_gain: float = 2.5,
        turn_gain: float = 0.08,
        turn_lateral_ratio: float = 0.45,
        yaw_deadzone: float = 0.012,
        adaptive: AdaptiveParams | None = None,
    ):
        self.smooth_alpha = smooth_alpha
        self.turn_threshold = turn_threshold
        self.speed_gain = speed_gain
        self.turn_gain = turn_gain
        self.turn_lateral_ratio = turn_lateral_ratio
        self.yaw_deadzone = yaw_deadzone
        self.adaptive = adaptive or AdaptiveParams()
        self._prev: EgoMotionFrame | None = None

    def reset(self) -> None:
        """Clear smoothing state (call at chunk boundaries)."""
        self._prev = None

    def decode(self, output: FlyVisOutput) -> EgoMotionSequence:
        n = output.flow.shape[0]
        vx = np.zeros(n, dtype=np.float32)
        vy = np.zeros(n, dtype=np.float32)
        yaw = np.zeros(n, dtype=np.float32)
        conf = np.zeros(n, dtype=np.float32)
        turning = np.zeros(n, dtype=bool)

        for t in range(n):
            frame = self._decode_frame(
                {k: v[t] for k, v in output.t4.items()},
                {k: v[t] for k, v in output.t5.items()},
            )
            if self._prev is not None:
                a = self.smooth_alpha
                smoothed_yaw = a * frame.yaw_rate + (1 - a) * self._prev.yaw_rate
                frame = EgoMotionFrame(
                    vx_body=a * frame.vx_body + (1 - a) * self._prev.vx_body,
                    vy_body=a * frame.vy_body + (1 - a) * self._prev.vy_body,
                    yaw_rate=0.0 if not frame.turning else smoothed_yaw,
                    confidence=frame.confidence,
                    turning=frame.turning,
                )
            self._prev = frame
            vx[t] = frame.vx_body
            vy[t] = frame.vy_body
            yaw[t] = frame.yaw_rate
            conf[t] = frame.confidence
            turning[t] = frame.turning

        return EgoMotionSequence(
            vx=vx,
            vy=vy,
            yaw_rate=yaw,
            confidence=conf,
            timestamps=output.timestamps,
            turning=turning,
        )

    def _decode_frame(self, t4: dict[str, float], t5: dict[str, float]) -> EgoMotionFrame:
        self_fwd = self._cardinal_diff(t4, t5, "backward", "forward")
        self_lat = self._cardinal_diff(t4, t5, "left", "right") - self.adaptive.lateral_bias

        vx_body = max(0.0, self_fwd) * self.speed_gain * self.adaptive.speed_scale

        lat_mag = abs(self_lat)
        fwd_mag = abs(self_fwd)
        ratio = self.turn_lateral_ratio + self.adaptive.turn_ratio_boost
        turning = lat_mag > self.turn_threshold and lat_mag > fwd_mag * ratio
        yaw_rate = 0.0
        if turning:
            yaw_rate = float(
                np.clip(self_lat * self.turn_gain * self.adaptive.yaw_gain_scale, -0.08, 0.08)
            )
            if abs(yaw_rate) < self.yaw_deadzone:
                yaw_rate = 0.0
                turning = False
        vy_body = float(self_lat * 0.05) if turning else 0.0

        magnitude = np.sqrt(vx_body**2 + vy_body**2)
        confidence = float(np.clip(magnitude / (magnitude + 0.02), 0.0, 1.0))

        return EgoMotionFrame(vx_body, vy_body, yaw_rate, confidence, turning=turning)

    def _cardinal_diff(
        self,
        t4: dict[str, float],
        t5: dict[str, float],
        positive_key: str,
        negative_key: str,
    ) -> float:
        pos_cells = self.CARDINAL[positive_key]
        neg_cells = self.CARDINAL[negative_key]
        pos = sum(t4.get(c, 0.0) + t5.get(c, 0.0) for c in pos_cells)
        neg = sum(t4.get(c, 0.0) + t5.get(c, 0.0) for c in neg_cells)
        return float(pos - neg)
