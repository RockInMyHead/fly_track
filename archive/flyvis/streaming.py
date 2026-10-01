"""Streaming fly visual odometry — process video in batches, emit trajectory points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from .config import FlyVOConfig
from .dopamine import AdaptiveParams, load_adaptive
from .egomotion import FlyEgoMotionDecoder
from .fly_eye import FlyEyeRenderer
from .flyvis_engine import FlyVisEngine, FlyVisOutput
from .path_integration import PathIntegrator
from .pfn import body_to_world
from .video_preview import encode_preview_jpeg


@dataclass
class TrajectoryPoint:
    timestamp: float
    x: float
    y: float
    heading_deg: float
    vx_body: float
    vy_body: float
    speed: float
    confidence: float
    frame_jpeg: bytes | None = None
    t4: dict[str, float] | None = None
    t5: dict[str, float] | None = None
    epg: list[float] | None = None
    lum_hex: list[float] | None = None
    flow_mag: list[float] | None = None
    yaw_rate: float | None = None
    turning: bool | None = None


@dataclass
class SegmentNeuralCache:
    """Cached FlyVis outputs for fast dopamine re-integration."""

    outputs: list[FlyVisOutput]
    start_seconds: float
    end_seconds: float


class StreamingFlyVO:
    """Stateful streaming processor for live trajectory generation."""

    def __init__(
        self,
        config: FlyVOConfig | None = None,
        batch_frames: int = 25,
        adaptive: AdaptiveParams | None = None,
    ):
        self.config = config or FlyVOConfig()
        self.batch_frames = batch_frames
        self.adaptive = adaptive if adaptive is not None else load_adaptive()
        self.engine = FlyVisEngine(self.config)
        self.eye = FlyEyeRenderer(
            extent=self.config.eye_extent,
            kernel_size=self.config.eye_kernel_size,
        )
        self.egomotion = FlyEgoMotionDecoder(
            self.config.motion_smooth_alpha,
            turn_threshold=self.config.turn_threshold,
            speed_gain=self.config.speed_gain,
            turn_gain=self.config.turn_gain,
            turn_lateral_ratio=self.config.turn_lateral_ratio,
            yaw_deadzone=self.config.yaw_deadzone,
            adaptive=self.adaptive,
        )
        self.integrator = PathIntegrator(self.config)
        self._first_chunk = True
        self._x = 0.0
        self._y = 0.0

    @property
    def device(self) -> str:
        self.engine._ensure_loaded()
        return self.engine.device

    def reload_adaptive(self) -> AdaptiveParams:
        """Reload learned params from disk (after dopamine pulse)."""
        self.adaptive = load_adaptive()
        self.egomotion.adaptive = self.adaptive
        return self.adaptive

    def _yaw_step(self, yaw_rate: float, dt: float) -> None:
        self.integrator.compass.step(yaw_rate + self.adaptive.yaw_rate_bias, dt)

    def reset(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading_deg: float = 90.0,
    ) -> None:
        self._first_chunk = True
        self._x = x0
        self._y = y0
        self.egomotion.reset()
        self.integrator = PathIntegrator(self.config)
        self.integrator.compass.set_heading_deg(
            heading_deg + self.adaptive.heading_offset_deg
        )
        self.engine._ensure_loaded()

    def stream(
        self,
        video_path: str,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        emit_stride: int = 5,
    ) -> Iterator[TrajectoryPoint]:
        """
        Yield trajectory points while reading video sequentially.

        Args:
            emit_stride: emit every N simulation frames (lower = smoother, more WS traffic)
        """
        self.reload_adaptive()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0.0
        if end_seconds is None:
            end_seconds = duration
        end_seconds = min(end_seconds, duration)

        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
        sim_hz = self.config.sim_hz
        dt = 1.0 / sim_hz
        t_video = start_seconds
        batch_raw: list[np.ndarray] = []
        batch_t: list[float] = []
        batch_bgr: list[np.ndarray] = []
        point_idx = 0

        def flush_batch() -> Iterator[TrajectoryPoint]:
            nonlocal point_idx, batch_raw, batch_t, batch_bgr
            if not batch_raw:
                return iter(())

            frames = self._resize_batch(batch_raw)
            timestamps = np.array(batch_t, dtype=np.float32)
            lum = self.eye.render(frames)
            out, _ = self.engine.simulate_batch(
                lum,
                timestamps,
                initial_state=None if self._first_chunk else object(),
            )
            self._first_chunk = False
            motion = self.egomotion.decode(out)
            points: list[TrajectoryPoint] = []

            for i in range(len(motion.timestamps)):
                if i > 0:
                    self._yaw_step(float(motion.yaw_rate[i]), dt)
                heading = self.integrator.compass.heading
                gain = self.config.velocity_gain
                vx_w, vy_w = body_to_world(
                    np.array([motion.vx[i] * gain]),
                    np.array([motion.vy[i] * gain]),
                    np.array([heading]),
                )
                if i > 0:
                    self._x += float(vx_w[0] * dt)
                    self._y += float(vy_w[0] * dt)

                if point_idx % emit_stride == 0:
                    speed = float(np.hypot(motion.vx[i], motion.vy[i]))
                    frame_idx = min(i, len(batch_bgr) - 1)
                    frame_jpeg = None
                    if batch_bgr:
                        try:
                            frame_jpeg = encode_preview_jpeg(batch_bgr[frame_idx])
                        except Exception:
                            frame_jpeg = None
                    turning = bool(motion.turning[i]) if motion.turning is not None else None
                    brain = self._brain_snapshot(out, lum, i, float(motion.yaw_rate[i]))
                    points.append(
                        TrajectoryPoint(
                            timestamp=float(motion.timestamps[i]),
                            x=self._x,
                            y=self._y,
                            heading_deg=self.integrator.compass.heading_deg,
                            vx_body=float(motion.vx[i]),
                            vy_body=float(motion.vy[i]),
                            speed=speed,
                            confidence=float(motion.confidence[i]),
                            frame_jpeg=frame_jpeg,
                            turning=turning,
                            **brain,
                        )
                    )
                point_idx += 1

            batch_raw = []
            batch_t = []
            batch_bgr = []
            return iter(points)

        while t_video < end_seconds - 1e-6:
            ok, frame = cap.read()
            if not ok:
                break

            batch_bgr.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            batch_raw.append(gray)
            batch_t.append(t_video)
            t_video += 1.0 / fps

            if len(batch_raw) >= self.batch_frames:
                yield from flush_batch()

        cap.release()
        yield from flush_batch()

    def run_segment(
        self,
        video_path: str,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        heading_deg: float = 90.0,
        emit_stride: int = 4,
    ) -> list[TrajectoryPoint]:
        """Run a full segment and return all trajectory points."""
        self.reload_adaptive()
        self.reset(x0=x0, y0=y0, heading_deg=heading_deg)
        return list(
            self.stream(
                video_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                emit_stride=emit_stride,
            )
        )

    def collect_neural_cache(
        self,
        video_path: str,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        progress=None,
    ) -> SegmentNeuralCache:
        """Run FlyVis once and cache neural outputs for fast dopamine search."""
        self.engine._ensure_loaded()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0.0
        if end_seconds is None:
            end_seconds = duration
        end_seconds = min(end_seconds, duration)

        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
        t_video = start_seconds
        batch_raw: list[np.ndarray] = []
        batch_t: list[float] = []
        outputs: list[FlyVisOutput] = []
        first_chunk = True
        span = max(end_seconds - start_seconds, 1e-6)

        def flush_cache_batch() -> None:
            nonlocal first_chunk, batch_raw, batch_t
            if not batch_raw:
                return
            frames = self._resize_batch(batch_raw)
            timestamps = np.array(batch_t, dtype=np.float32)
            lum = self.eye.render(frames)
            out, _ = self.engine.simulate_batch(
                lum,
                timestamps,
                initial_state=None if first_chunk else object(),
            )
            first_chunk = False
            outputs.append(out)
            batch_raw = []
            batch_t = []
            if progress:
                progress("cache", t_video, end_seconds, (t_video - start_seconds) / span)

        while t_video < end_seconds - 1e-6:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            batch_raw.append(gray)
            batch_t.append(t_video)
            t_video += 1.0 / fps
            if len(batch_raw) >= self.batch_frames:
                flush_cache_batch()

        flush_cache_batch()
        cap.release()
        return SegmentNeuralCache(
            outputs=outputs,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    def integrate_from_cache(
        self,
        cache: SegmentNeuralCache,
        adaptive: AdaptiveParams,
        x0: float = 0.0,
        y0: float = 0.0,
        heading_deg: float = 90.0,
        emit_stride: int = 4,
    ) -> list[TrajectoryPoint]:
        """Re-decode ego-motion + path integration from cached FlyVis outputs."""
        self.adaptive = adaptive.clamp()
        self.egomotion.adaptive = self.adaptive
        self.egomotion.reset()
        self.integrator = PathIntegrator(self.config)
        self.integrator.compass.set_heading_deg(
            heading_deg + self.adaptive.heading_offset_deg
        )
        dt = 1.0 / self.config.sim_hz
        x, y = x0, y0
        points: list[TrajectoryPoint] = []
        point_idx = 0

        for out in cache.outputs:
            motion = self.egomotion.decode(out)
            for i in range(len(motion.timestamps)):
                if i > 0:
                    self._yaw_step(float(motion.yaw_rate[i]), dt)
                gain = self.config.velocity_gain
                vx_w, vy_w = body_to_world(
                    np.array([motion.vx[i] * gain]),
                    np.array([motion.vy[i] * gain]),
                    np.array([self.integrator.compass.heading]),
                )
                if i > 0:
                    x += float(vx_w[0] * dt)
                    y += float(vy_w[0] * dt)
                if point_idx % emit_stride == 0:
                    points.append(
                        TrajectoryPoint(
                            timestamp=float(motion.timestamps[i]),
                            x=x,
                            y=y,
                            heading_deg=self.integrator.compass.heading_deg,
                            vx_body=float(motion.vx[i]),
                            vy_body=float(motion.vy[i]),
                            speed=float(np.hypot(motion.vx[i], motion.vy[i])),
                            confidence=float(motion.confidence[i]),
                        )
                    )
                point_idx += 1
        return points

    def _brain_snapshot(
        self, out, lum, frame_idx: int, yaw_rate: float
    ) -> dict:
        """Extract per-frame retina + central-column brain activity."""
        t4 = {
            k: round(float(out.t4[k][frame_idx]), 4)
            for k in self.config.t4_types
            if k in out.t4
        }
        t5 = {
            k: round(float(out.t5[k][frame_idx]), 4)
            for k in self.config.t5_types
            if k in out.t5
        }
        epg = [round(float(v), 4) for v in self.integrator.compass.bump]
        lum_row = lum[0, frame_idx, 0, :].detach().cpu().numpy()
        lum_hex = [round(float(v), 4) for v in lum_row]
        flow = out.flow[frame_idx]
        flow_mag = [round(float(v), 4) for v in np.hypot(flow[0], flow[1])]
        return {
            "t4": t4,
            "t5": t5,
            "epg": epg,
            "lum_hex": lum_hex,
            "flow_mag": flow_mag,
            "yaw_rate": round(yaw_rate, 4),
        }

    def _resize_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        max_size = self.config.max_frame_size
        out = []
        for gray in frames:
            if max_size and max(gray.shape) > max_size:
                scale = max_size / max(gray.shape)
                gray = cv2.resize(
                    gray,
                    (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            out.append(gray)
        return np.stack(out, axis=0)


def split_neural_cache(
    cache: SegmentNeuralCache, frac: float = 0.7
) -> tuple[SegmentNeuralCache, SegmentNeuralCache, bool]:
    """Split cached FlyVis outputs by timestamp. Returns (train, test, held_out)."""
    t0, t1 = cache.start_seconds, cache.end_seconds
    if t1 - t0 < 15.0 or not cache.outputs:
        return cache, cache, False
    split_t = t0 + (t1 - t0) * float(np.clip(frac, 0.55, 0.85))
    train: list[FlyVisOutput] = []
    test: list[FlyVisOutput] = []
    for out in cache.outputs:
        ts = np.asarray(out.timestamps)
        mid = float(ts.mean()) if len(ts) else split_t
        (train if mid < split_t else test).append(out)
    if len(train) < 2 or len(test) < 2:
        return cache, cache, False
    return (
        SegmentNeuralCache(train, t0, split_t),
        SegmentNeuralCache(test, split_t, t1),
        True,
    )
