#!/usr/bin/env python3
"""Batch MaleCNS visual odometry: video → trajectory CSV + plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from fly_vo.config import FlyVOConfig
from fly_vo.streaming import StreamingMaleCNS
from fly_vo.visualize import plot_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="MaleCNS v1.0 visual odometry")
    parser.add_argument("video", type=str, help="Input video path")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (s)")
    parser.add_argument("--end", type=float, default=None, help="End time (s)")
    parser.add_argument("--heading", type=float, default=90.0, help="Initial heading (deg)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FlyVOConfig(device=args.device)
    proc = StreamingMaleCNS(cfg)
    proc.reset(heading_deg=args.heading)
    print(f"Loading MaleCNS brain on {proc.device}…")
    points = list(
        proc.stream(
            args.video,
            start_seconds=args.start,
            end_seconds=args.end,
            include_viz=False,
        )
    )
    if not points:
        print("No trajectory points generated.")
        return

    csv_path = out_dir / "trajectory.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("timestamp,x,y,heading_deg,vx_body,vy_body,speed,confidence\n")
        for p in points:
            f.write(
                f"{p.timestamp:.4f},{p.x:.4f},{p.y:.4f},{p.heading_deg:.2f},"
                f"{p.vx_body:.4f},{p.vy_body:.4f},{p.speed:.4f},{p.confidence:.3f}\n"
            )

    plot_path = out_dir / "trajectory.png"
    plot_trajectory(points, plot_path)
    print(f"Wrote {len(points)} points → {csv_path}")
    print(f"Plot → {plot_path}")


if __name__ == "__main__":
    main()
