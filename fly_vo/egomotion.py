"""Ego-motion from MaleCNS descending neuron activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dopamine import AdaptiveParams
from .malecns_engine import BrainStepResult
from .motion_readout import DescendingNeuronDecoder


@dataclass
class EgoMotionFrame:
    vx_body: float
    vy_body: float
    yaw_rate: float
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


class MaleCNSEgoMotionDecoder(DescendingNeuronDecoder):
    """Smooth per-step descending readout into ego-motion."""

    def __init__(
        self,
        smooth_alpha: float = 0.25,
        config=None,
        adaptive: AdaptiveParams | None = None,
    ):
        super().__init__(config=config, adaptive=adaptive)
        self.smooth_alpha = smooth_alpha
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0

    def reset(self) -> None:
        super().reset()
        self._vx = self._vy = self._yaw = 0.0

    def decode_frame(
        self, step: BrainStepResult, metrics: dict | None = None
    ) -> EgoMotionFrame:
        self.observe(step)
        vx, vy, yaw, conf = self.decode_step(metrics)
        rot = abs(float(metrics.get("rotation_deg", 0.0))) if metrics else 0.0
        a = self.smooth_alpha
        self._vx = (1 - a) * self._vx + a * vx
        self._vy = (1 - a) * self._vy + a * vy
        # Sharp turns: no yaw EMA (avoids smearing 90° corners into arcs)
        if rot >= 0.25:
            self._yaw = yaw
        else:
            self._yaw = (1 - a) * self._yaw + a * yaw
        turning = abs(self._yaw) > self.config.yaw_deadzone * 2
        return EgoMotionFrame(
            vx_body=self._vx,
            vy_body=self._vy,
            yaw_rate=self._yaw,
            confidence=conf,
            turning=turning,
        )

    def decode_cached(
        self,
        steps: list[BrainStepResult],
        metrics: list[dict] | None = None,
    ) -> EgoMotionSequence:
        self.reset()
        n = len(steps)
        vx = np.zeros(n, dtype=np.float32)
        vy = np.zeros(n, dtype=np.float32)
        yaw = np.zeros(n, dtype=np.float32)
        conf = np.zeros(n, dtype=np.float32)
        turning = np.zeros(n, dtype=bool)
        ts = np.array([s.timestamp for s in steps], dtype=np.float32)

        for i, step in enumerate(steps):
            m = metrics[i] if metrics and i < len(metrics) else None
            frame = self.decode_frame(step, m)
            vx[i] = frame.vx_body
            vy[i] = frame.vy_body
            yaw[i] = frame.yaw_rate
            conf[i] = frame.confidence
            turning[i] = frame.turning

        return EgoMotionSequence(vx=vx, vy=vy, yaw_rate=yaw, confidence=conf, timestamps=ts, turning=turning)
