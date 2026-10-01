"""Plot trajectory outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .path_integration import Trajectory


def plot_trajectory(
    traj: Trajectory | Sequence,
    output_path: str | Path,
    title: str = "MaleCNS Visual Odometry",
) -> None:
    if hasattr(traj, "x"):
        x = np.asarray(traj.x)
        y = np.asarray(traj.y)
        heading = np.asarray(traj.heading)
        timestamps = np.asarray(traj.timestamps)
        speed = np.asarray(traj.speed)
    else:
        points = list(traj)
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        heading = np.radians([p.heading_deg for p in points])
        timestamps = np.array([p.timestamp for p in points])
        speed = np.array([p.speed for p in points])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x, y, "b-", linewidth=1.5, alpha=0.8)
    ax.plot(x[0], y[0], "go", markersize=10, label="start")
    ax.plot(x[-1], y[-1], "ro", markersize=10, label="end")

    step = max(1, len(x) // 15)
    for i in range(0, len(x), step):
        dx = 0.05 * np.cos(heading[i])
        dy = 0.05 * np.sin(heading[i])
        ax.arrow(x[i], y[i], dx, dy, head_width=0.02, color="gray", alpha=0.5)

    ax.set_xlabel("x (arbitrary units)")
    ax.set_ylabel("y (arbitrary units)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    t = timestamps - timestamps[0] if len(timestamps) else timestamps
    ax2.plot(t, speed, "g-", label="speed")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
