"""Streaming MaleCNS visual odometry — video → whole brain → trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

from .config import FlyVOConfig
from .dopamine import AdaptiveParams, load_adaptive
from .egomotion import MaleCNSEgoMotionDecoder
from .heading import RingAttractorCompass
from .malecns_engine import BrainStepResult, MaleCNSEngine
from .pfn import body_to_world
from .brain_clock import iter_video_at_brain_hz
from .video_preview import encode_preview_jpeg
from .visual_encoder import VideoVisualEncoder


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
    # MaleCNS viz payloads
    pathways: dict[str, int] | None = None
    descending: dict[str, float] | None = None
    n_spikes: int | None = None
    eye_preview: list[float] | None = None
    yaw_rate: float | None = None
    turning: bool | None = None


@dataclass
class SegmentNeuralCache:
    """Cached MaleCNS brain steps for fast dopamine re-integration."""

    steps: list[BrainStepResult]
    start_seconds: float
    end_seconds: float
    metrics: list[dict] = field(default_factory=list)


class StreamingMaleCNS:
    """Stateful streaming processor: MP4 → MaleCNS v1.0 → x, y, θ."""

    def __init__(
        self,
        config: FlyVOConfig | None = None,
        emit_every_n_steps: int = 1,
        adaptive: AdaptiveParams | None = None,
    ):
        self.config = config or FlyVOConfig()
        self.emit_every_n_steps = emit_every_n_steps
        self.adaptive = adaptive if adaptive is not None else load_adaptive()
        self.engine = MaleCNSEngine(self.config)
        self.encoder = VideoVisualEncoder(self.engine, gain=self.config.visual_gain)
        self.egomotion = MaleCNSEgoMotionDecoder(
            smooth_alpha=self.config.motion_smooth_alpha,
            config=self.config,
            adaptive=self.adaptive,
        )
        self.compass = RingAttractorCompass(
            n_neurons=self.config.n_epg_neurons,
            gain=self.config.heading_gain,
        )
        self._x = 0.0
        self._y = 0.0
        self._pending_rot_deg = 0.0
        self._turn_streak = 0
        self._turn_mode_steps = 0
        self._turn_sign = 0.0
        self._corner_snapped = False
        self._stream_time = 0.0
        self._arc_sign = 0.0

    @property
    def device(self) -> str:
        self.engine._ensure_loaded()
        return self.engine.device

    def reload_adaptive(self) -> AdaptiveParams:
        self.adaptive = load_adaptive()
        self.egomotion.adaptive = self.adaptive
        return self.adaptive

    def _integrate_heading(
        self,
        motion,
        metrics: dict | None,
        dt_step: float,
    ) -> None:
        """Hold heading on straights; sharp L-snap or gradual arc rotation."""
        rot = float(metrics.get("rotation_deg", 0.0)) if metrics else 0.0
        step = float(metrics.get("step_rotation_deg", 0.0)) if metrics else 0.0
        pan = float(metrics.get("pan_shift", 0.0)) if metrics else 0.0
        t = float(metrics.get("timestamp", 0.0)) if metrics else 0.0
        if t <= 0.0:
            t = self._stream_time
        mov = float(metrics.get("motion", 0.0)) if metrics else 0.0
        ap = self.adaptive
        cfg = self.config

        if mov < 0.018:
            rot = 0.0

        # ORB clip ceiling (±20°) on straights = glitch, not an L-turn (~10°).
        ref_ok = cfg.corner_ref_deg <= abs(rot) < 15.0
        confirmed = (
            t >= cfg.corner_min_time_s
            and ref_ok
            and abs(step) >= cfg.corner_step_min
            and step * rot > 0
            and abs(pan) >= cfg.corner_pan_min
        )

        # Sharp L-turn (51-70 style): confirmed ref 10–15° + pan + step → ±90° snap.
        if (
            not self._corner_snapped
            and ref_ok
            and mov >= 0.018
            and confirmed
        ):
            snap = cfg.corner_snap_deg
            self.compass.nudge_heading_deg(-snap if rot > 0 else snap)
            self._corner_snapped = True
            self._turn_mode_steps = 90
            self._turn_sign = 1.0 if rot > 0 else -1.0
            return

        # Gentle arc (130-160 end): moderate ref + low pan → partial turn, not L-snap.
        is_arc = (
            t >= cfg.arc_min_time_s
            and 5.0 <= abs(rot) < 15.0
            and abs(pan) < cfg.corner_pan_min
            and mov >= 0.018
        )
        if is_arc:
            if self._arc_sign == 0.0:
                self._arc_sign = 1.0 if rot > 0 else -1.0
            if rot * self._arc_sign > 0:
                arc_gain = cfg.pathway_rot_gain * ap.yaw_gain_scale * cfg.arc_gain_scale
                self.compass.nudge_heading_deg(-rot * arc_gain)
            self._turn_mode_steps = max(self._turn_mode_steps, 20)
        elif abs(rot) < 2.0:
            self._arc_sign = 0.0

        if self._turn_mode_steps > 0:
            self._turn_mode_steps -= 1
        # else: lock heading (no compass.step — avoids ORB noise on straights)

    def reset(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading_deg: float = 90.0,
        seed: int = 64,
    ) -> None:
        self._x = x0
        self._y = y0
        self._pending_rot_deg = 0.0
        self._turn_streak = 0
        self._turn_mode_steps = 0
        self._turn_sign = 0.0
        self._corner_snapped = False
        self._stream_time = 0.0
        self._arc_sign = 0.0
        self.engine.reset(seed=seed)
        self.encoder.reset()
        self.egomotion.reset()
        self.compass = RingAttractorCompass(
            n_neurons=self.config.n_epg_neurons,
            gain=self.config.heading_gain,
        )
        self.compass.set_heading_deg(heading_deg + self.adaptive.heading_offset_deg)

    def stream(
        self,
        video_path: str,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        include_viz: bool = True,
    ) -> Iterator[TrajectoryPoint]:
        brain_dt = self.config.brain_dt
        step_i = 0
        prev_t_video: float | None = None

        for t_video, frame in iter_video_at_brain_hz(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            brain_dt=brain_dt,
        ):
            bgr = self._maybe_resize(frame)
            eye_drive, inject, metrics = self.encoder.encode_frame(bgr)
            self._stream_time = t_video
            metrics["timestamp"] = t_video
            result = self.engine.step(t_video, eye_drive=eye_drive, inject=inject)
            motion = self.egomotion.decode_frame(result, metrics)

            dt_step = brain_dt if prev_t_video is None else float(t_video - prev_t_video)
            dt_step = max(dt_step, 1e-6)

            if step_i > 0:
                self._integrate_heading(motion, metrics, dt_step)
                gain = self.config.velocity_gain
                vx_w, vy_w = body_to_world(
                    np.array([motion.vx_body * gain]),
                    np.array([motion.vy_body * gain]),
                    np.array([self.compass.heading]),
                )
                self._x += float(vx_w[0] * dt_step)
                self._y += float(vy_w[0] * dt_step)

            if step_i % self.emit_every_n_steps == 0:
                jpeg = None
                if include_viz:
                    jpeg = encode_preview_jpeg(frame, max_width=480)
                yield TrajectoryPoint(
                    timestamp=t_video,
                    x=self._x,
                    y=self._y,
                    heading_deg=self.compass.heading_deg,
                    vx_body=motion.vx_body,
                    vy_body=motion.vy_body,
                    speed=float(np.hypot(motion.vx_body, motion.vy_body)),
                    confidence=motion.confidence,
                    frame_jpeg=jpeg,
                    pathways=result.pathway_spikes if include_viz else None,
                    descending=result.descending if include_viz else None,
                    n_spikes=result.n_active if include_viz else None,
                    eye_preview=metrics.get("eye_preview") if include_viz else None,
                    yaw_rate=motion.yaw_rate,
                    turning=motion.turning,
                )
            prev_t_video = t_video
            step_i += 1

    def collect_neural_cache(
        self,
        video_path: str,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        progress=None,
    ) -> SegmentNeuralCache:
        """Run MaleCNS once and cache brain steps for dopamine search."""
        self.engine.reset()
        self.encoder.reset()

        brain_dt = self.config.brain_dt
        step_i = 0
        steps: list[BrainStepResult] = []
        metrics_list: list[dict] = []
        span = max((end_seconds or 0) - start_seconds, 1e-6)

        for t_video, frame in iter_video_at_brain_hz(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            brain_dt=brain_dt,
        ):
            bgr = self._maybe_resize(frame)
            eye_drive, inject, metrics = self.encoder.encode_frame(bgr)
            result = self.engine.step(t_video, eye_drive=eye_drive, inject=inject)
            steps.append(result)
            metrics_list.append(metrics)
            step_i += 1
            if progress and step_i % 10 == 0:
                progress("cache", t_video, start_seconds + span, (t_video - start_seconds) / span)

        if progress:
            progress("cache", start_seconds + span, start_seconds + span, 1.0)
        return SegmentNeuralCache(
            steps=steps,
            start_seconds=start_seconds,
            end_seconds=start_seconds + span if end_seconds is None else end_seconds,
            metrics=metrics_list,
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
        """Re-decode trajectory from cached brain steps (no re-simulation)."""
        self.adaptive = adaptive.clamp()
        self.egomotion.adaptive = self.adaptive
        self.egomotion.reset()
        self.compass = RingAttractorCompass(
            n_neurons=self.config.n_epg_neurons,
            gain=self.config.heading_gain,
        )
        self.compass.set_heading_deg(heading_deg + self.adaptive.heading_offset_deg)
        self._pending_rot_deg = 0.0
        self._turn_streak = 0
        self._turn_mode_steps = 0
        self._turn_sign = 0.0
        self._corner_snapped = False
        self._stream_time = 0.0
        self._arc_sign = 0.0

        x, y = x0, y0
        points: list[TrajectoryPoint] = []

        for i, step in enumerate(cache.steps):
            m = cache.metrics[i] if i < len(cache.metrics) else None
            if m is not None:
                m = dict(m)
                m["timestamp"] = step.timestamp
            motion = self.egomotion.decode_frame(step, m)
            if i > 0:
                dt_step = float(step.timestamp - cache.steps[i - 1].timestamp)
                dt_step = max(dt_step, 1e-6)
                self._integrate_heading(motion, m, dt_step)
                gain = self.config.velocity_gain
                vx_w, vy_w = body_to_world(
                    np.array([motion.vx_body * gain]),
                    np.array([motion.vy_body * gain]),
                    np.array([self.compass.heading]),
                )
                x += float(vx_w[0] * dt_step)
                y += float(vy_w[0] * dt_step)

            if i % emit_stride == 0:
                points.append(
                    TrajectoryPoint(
                        timestamp=step.timestamp,
                        x=x,
                        y=y,
                        heading_deg=self.compass.heading_deg,
                        vx_body=motion.vx_body,
                        vy_body=motion.vy_body,
                        speed=float(np.hypot(motion.vx_body, motion.vy_body)),
                        confidence=motion.confidence,
                        pathways=step.pathway_spikes,
                        descending=step.descending,
                        n_spikes=step.n_active,
                        yaw_rate=motion.yaw_rate,
                        turning=motion.turning,
                    )
                )
        return points

    def _maybe_resize(self, bgr: np.ndarray) -> np.ndarray:
        max_size = self.config.max_frame_size
        h, w = bgr.shape[:2]
        if max_size and max(h, w) > max_size:
            scale = max_size / max(h, w)
            # INTER_LINEAR keeps ORB corner features stable (INTER_AREA smears rotation).
            bgr = cv2.resize(
                bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        return bgr


# Alias for web server compatibility
StreamingFlyVO = StreamingMaleCNS


def split_neural_cache(
    cache: SegmentNeuralCache, frac: float = 0.7
) -> tuple[SegmentNeuralCache, SegmentNeuralCache, bool]:
    """Split cached brain steps by timestamp."""
    t0, t1 = cache.start_seconds, cache.end_seconds
    if t1 - t0 < 15.0 or len(cache.steps) < 20:
        return cache, cache, False
    split_t = t0 + (t1 - t0) * float(np.clip(frac, 0.55, 0.85))
    train_idx = [i for i, s in enumerate(cache.steps) if s.timestamp < split_t]
    test_idx = [i for i, s in enumerate(cache.steps) if s.timestamp >= split_t]
    if len(train_idx) < 10 or len(test_idx) < 10:
        return cache, cache, False
    train_steps = [cache.steps[i] for i in train_idx]
    test_steps = [cache.steps[i] for i in test_idx]
    train_metrics = [cache.metrics[i] for i in train_idx if i < len(cache.metrics)]
    test_metrics = [cache.metrics[i] for i in test_idx if i < len(cache.metrics)]
    return (
        SegmentNeuralCache(train_steps, t0, split_t, metrics=train_metrics),
        SegmentNeuralCache(test_steps, split_t, t1, metrics=test_metrics),
        True,
    )
