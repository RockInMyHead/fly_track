"""Integrate world velocity into (x, y) trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FlyVOConfig
from .egomotion import EgoMotionSequence  # MaleCNS descending readout
from .heading import RingAttractorCompass
from .pfn import body_to_world


@dataclass
class Trajectory:
    timestamps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray  # radians
    vx_body: np.ndarray
    vy_body: np.ndarray
    yaw_rate: np.ndarray
    speed: np.ndarray
    confidence: np.ndarray


class PathIntegrator:
    """Central-complex-inspired path integration."""

    def __init__(self, config: FlyVOConfig | None = None):
        self.config = config or FlyVOConfig()
        self.compass = RingAttractorCompass(
            n_neurons=self.config.n_epg_neurons,
            gain=self.config.heading_gain,
        )

    def integrate(
        self,
        motion: EgoMotionSequence,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> Trajectory:
        n = len(motion.timestamps)
        default_dt = 1.0 / self.config.sim_hz
        if n > 1:
            dts = np.diff(motion.timestamps, prepend=motion.timestamps[0])
            dts[0] = float(np.median(np.diff(motion.timestamps)))
        else:
            dts = np.array([default_dt], dtype=np.float32)

        x = np.zeros(n, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        heading = np.zeros(n, dtype=np.float32)
        x[0] = x0
        y[0] = y0

        gain = self.config.velocity_gain
        for i in range(n):
            dt = float(dts[i])
            if i > 0:
                self.compass.step(float(motion.yaw_rate[i]), dt)
            heading[i] = self.compass.heading
            vx_w, vy_w = body_to_world(
                np.array([motion.vx[i] * gain]),
                np.array([motion.vy[i] * gain]),
                np.array([heading[i]]),
            )
            if i > 0:
                x[i] = x[i - 1] + vx_w[0] * dt
                y[i] = y[i - 1] + vy_w[0] * dt

        speed = np.sqrt(motion.vx**2 + motion.vy**2)

        return Trajectory(
            timestamps=motion.timestamps,
            x=x,
            y=y,
            heading=heading,
            vx_body=motion.vx,
            vy_body=motion.vy,
            yaw_rate=motion.yaw_rate,
            speed=speed,
            confidence=motion.confidence,
        )
