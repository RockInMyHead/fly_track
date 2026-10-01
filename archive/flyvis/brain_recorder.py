"""Record FlyVis neural activity for visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import FlyVOConfig
from .egomotion import FlyEgoMotionDecoder
from .fly_eye import FlyEyeRenderer
from .flyvis_engine import FlyVisEngine
from .heading import RingAttractorCompass
from .video_io import load_video


@dataclass
class BrainSnapshot:
    timestamp: float
    t4: dict[str, float]
    t5: dict[str, float]
    flow_fx: list[float]  # hex field (721)
    flow_fy: list[float]
    epg_bump: list[float]
    heading_deg: float
    vx: float
    vy: float
    yaw_rate: float


@dataclass
class BrainRecording:
    snapshots: list[BrainSnapshot]
    sim_hz: float
    start_seconds: float


class BrainRecorder:
    """Run FlyVis on a segment and record per-frame brain state."""

    T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
    T5_TYPES = ("T5a", "T5b", "T5c", "T5d")

    def __init__(self, config: FlyVOConfig | None = None):
        self.config = config or FlyVOConfig()
        self.eye = FlyEyeRenderer(
            extent=self.config.eye_extent,
            kernel_size=self.config.eye_kernel_size,
        )
        self.engine = FlyVisEngine(self.config)
        self.egomotion = FlyEgoMotionDecoder(self.config.motion_smooth_alpha)
        self.compass = RingAttractorCompass(
            n_neurons=self.config.n_epg_neurons,
            gain=self.config.heading_gain,
        )

    def record(
        self,
        video_path: str,
        start_seconds: float = 300.0,
        duration_seconds: float = 30.0,
        stride: int = 5,
    ) -> BrainRecording:
        """Record brain activity every `stride` simulation frames."""
        video = load_video(
            video_path,
            sim_hz=self.config.sim_hz,
            max_seconds=duration_seconds,
            max_size=self.config.max_frame_size,
            start_seconds=start_seconds,
        )
        lum = self.eye.render(video.frames)
        self.engine._ensure_loaded()
        dt = 1.0 / self.config.sim_hz

        first_frame = lum[:, 0:1].squeeze(2)
        stationary = self.engine._network.fade_in_state(  # type: ignore[union-attr]
            self.config.fade_in_seconds, dt, first_frame
        )

        with torch.no_grad():
            responses = self.engine._network.simulate(  # type: ignore[union-attr]
                lum, dt, initial_state=stationary, as_layer_activity=True
            )
            flow = self.engine._decoder(responses.activity).squeeze(0).cpu().numpy()

        central = responses.central
        snapshots: list[BrainSnapshot] = []

        for i in range(0, flow.shape[0], stride):
            t4 = {k: float(central[k][0, i]) for k in self.T4_TYPES}
            t5 = {k: float(central[k][0, i]) for k in self.T5_TYPES}

            frame_flow = flow[i]
            omega = float(np.sum(
                -self._hex_y(flow.shape[-1]) * frame_flow[0]
                + self._hex_x(flow.shape[-1]) * frame_flow[1]
            ) / flow.shape[-1])

            self.compass.step(omega, dt * stride)
            vx = -float(np.mean(frame_flow[1]))
            vy = float(np.mean(frame_flow[0]))

            snapshots.append(
                BrainSnapshot(
                    timestamp=float(video.timestamps[i]),
                    t4=t4,
                    t5=t5,
                    flow_fx=frame_flow[0].tolist(),
                    flow_fy=frame_flow[1].tolist(),
                    epg_bump=self.compass.bump.copy().tolist(),
                    heading_deg=self.compass.heading_deg,
                    vx=vx,
                    vy=vy,
                    yaw_rate=omega,
                )
            )

        return BrainRecording(
            snapshots=snapshots,
            sim_hz=self.config.sim_hz,
            start_seconds=start_seconds,
        )

    @staticmethod
    def _hex_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
        import flyvis.utils.hex_utils as hex_utils

        radius = hex_utils.get_hextent(n)
        u, v = hex_utils.get_hex_coords(radius)
        x, y = hex_utils.hex_to_pixel(u, v)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def _hex_x(self, n: int) -> np.ndarray:
        x, _ = self._hex_positions(n)
        return x / (np.max(np.abs(x)) + 1e-6)

    def _hex_y(self, n: int) -> np.ndarray:
        _, y = self._hex_positions(n)
        return y / (np.max(np.abs(y)) + 1e-6)
