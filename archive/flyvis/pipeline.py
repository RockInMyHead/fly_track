"""End-to-end fly visual odometry pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FlyVOConfig
from .egomotion import FlyEgoMotionDecoder
from .fly_eye import FlyEyeRenderer
from .flyvis_engine import FlyVisEngine
from .path_integration import PathIntegrator, Trajectory
from .video_io import get_video_info, load_video
from .visualize import plot_trajectory


@dataclass
class PipelineResult:
    trajectory: Trajectory
    csv_path: Path
    png_path: Path


class FlyVisualOdometryPipeline:
    """
    P0: video.mp4 -> FlyVis -> ego-motion -> heading -> path integration -> trajectory
    """

    def __init__(self, config: FlyVOConfig | None = None):
        self.config = config or FlyVOConfig()
        self.eye = FlyEyeRenderer(
            extent=self.config.eye_extent,
            kernel_size=self.config.eye_kernel_size,
        )
        self.engine = FlyVisEngine(self.config)
        self.egomotion = FlyEgoMotionDecoder(
            self.config.motion_smooth_alpha,
            turn_threshold=self.config.turn_threshold,
            speed_gain=self.config.speed_gain,
            turn_gain=self.config.turn_gain,
            turn_lateral_ratio=self.config.turn_lateral_ratio,
        )
        self.integrator = PathIntegrator(self.config)

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        max_seconds: float | None = None,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
    ) -> PipelineResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        duration, fps, w, h = get_video_info(str(video_path))
        if end_seconds is not None:
            process_seconds = max(0.0, end_seconds - start_seconds)
        elif max_seconds is not None:
            process_seconds = min(max_seconds, duration - start_seconds)
        else:
            process_seconds = duration - start_seconds

        chunk = self.config.chunk_seconds
        t_end = start_seconds + process_seconds

        print(f"Video: {duration:.1f}s @ {fps:.1f} fps ({w}x{h})")
        print(f"Segment: {start_seconds:.1f}s – {t_end:.1f}s ({process_seconds:.1f}s)")

        trajectories: list[Trajectory] = []
        x0, y0 = 0.0, 0.0
        t = start_seconds
        chunk_idx = 0

        while t < t_end - 1e-6:
            chunk_len = min(chunk, t_end - t)
            chunk_idx += 1
            print(f"\n=== Chunk {chunk_idx}: {t:.0f}s – {t + chunk_len:.0f}s ===")

            video = load_video(
                str(video_path),
                sim_hz=self.config.sim_hz,
                max_seconds=chunk_len,
                max_size=self.config.max_frame_size,
                start_seconds=t,
            )
            print(f"  Loaded {video.frames.shape[0]} frames ({video.frames.shape[1]}x{video.frames.shape[2]})")

            lum = self.eye.render(video.frames)
            flyvis_out = self.engine.simulate(lum, video.timestamps)
            motion = self.egomotion.decode(flyvis_out)
            traj = self.integrator.integrate(motion, x0=x0, y0=y0)

            trajectories.append(traj)
            x0, y0 = float(traj.x[-1]), float(traj.y[-1])
            t += chunk_len

        traj = self._merge_trajectories(trajectories)

        print(f"\nTotal: {len(traj.timestamps)} points, path length {self._path_length(traj):.1f}")
        csv_path = output_dir / "trajectory.csv"
        png_path = output_dir / "trajectory.png"
        self._save_csv(traj, csv_path)
        plot_trajectory(
            traj,
            png_path,
            title=f"Fly VO · {start_seconds:.0f}s–{t_end:.0f}s",
        )

        print(f"Done. CSV: {csv_path}")
        print(f"      PNG: {png_path}")
        return PipelineResult(trajectory=traj, csv_path=csv_path, png_path=png_path)

    @staticmethod
    def _merge_trajectories(parts: list[Trajectory]) -> Trajectory:
        if len(parts) == 1:
            return parts[0]
        return Trajectory(
            timestamps=np.concatenate([p.timestamps for p in parts]),
            x=np.concatenate([p.x for p in parts]),
            y=np.concatenate([p.y for p in parts]),
            heading=np.concatenate([p.heading for p in parts]),
            vx_body=np.concatenate([p.vx_body for p in parts]),
            vy_body=np.concatenate([p.vy_body for p in parts]),
            yaw_rate=np.concatenate([p.yaw_rate for p in parts]),
            speed=np.concatenate([p.speed for p in parts]),
            confidence=np.concatenate([p.confidence for p in parts]),
        )

    @staticmethod
    def _path_length(traj: Trajectory) -> float:
        return float(np.sum(np.hypot(np.diff(traj.x), np.diff(traj.y))))

    def _save_csv(self, traj: Trajectory, path: Path) -> None:
        df = pd.DataFrame(
            {
                "timestamp": traj.timestamps,
                "x": traj.x,
                "y": traj.y,
                "heading_deg": np.degrees(traj.heading),
                "vx_body": traj.vx_body,
                "vy_body": traj.vy_body,
                "yaw_rate": traj.yaw_rate,
                "speed": traj.speed,
                "confidence": traj.confidence,
            }
        )
        df.to_csv(path, index=False)
