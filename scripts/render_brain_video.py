#!/usr/bin/env python3
"""Render composite MP4: video + trajectory + brain activity + fly schematic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("output/vid00001_full/brain_data.json"))
    p.add_argument("--preview", type=Path, default=Path("output/vid00001_full/preview.mp4"))
    p.add_argument("-o", type=Path, default=Path("output/vid00001_full/brain_tour.mp4"))
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    data = json.loads(args.data.read_text())
    brain = data["brain"]["frames"]
    traj = data["trajectory"]
    t0 = data["brain"]["start_seconds"]

    cap = cv2.VideoCapture(str(args.preview))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(vid_fps / args.fps))

    fig = plt.figure(figsize=(12, 7), facecolor="#0f1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    ax_video = fig.add_subplot(gs[0, 0:2])
    ax_fly = fig.add_subplot(gs[0, 2])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_t4t5 = fig.add_subplot(gs[1, 1])
    ax_epg = fig.add_subplot(gs[1, 2], polar=True)

    ax_traj.plot(traj["x"], traj["y"], color="#1e3a5f", linewidth=0.8)
    ax_traj.plot(traj["x"][0], traj["y"][0], "go", markersize=5)
    dot, = ax_traj.plot([], [], "ro", markersize=6)
    ax_traj.set_aspect("equal")
    ax_traj.set_facecolor("#0f1117")

    types = ["a", "b", "c", "d"]
    x = np.arange(4)
    bars_t4 = ax_t4t5.bar(x - 0.15, [0] * 4, 0.3, color="#22c55e", label="T4")
    bars_t5 = ax_t4t5.bar(x + 0.15, [0] * 4, 0.3, color="#f97316", label="T5")
    ax_t4t5.set_xticks(x)
    ax_t4t5.set_xticklabels(["ant", "post", "dors", "vent"], fontsize=7)
    ax_t4t5.set_facecolor("#0f1117")
    ax_t4t5.legend(fontsize=7)

    n_epg = len(brain[0]["epg"])
    angles = np.linspace(0, 2 * np.pi, n_epg, endpoint=False)
    epg_bars = ax_epg.bar(angles, brain[0]["epg"], width=2 * np.pi / n_epg * 0.8, color="#a855f7")
    ax_epg.set_facecolor("#0f1117")

    out = None
    frame_i = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step != 0:
            frame_i += 1
            continue

        t_local = frame_i / vid_fps
        t_global = t0 + t_local
        bi = min(int(t_local / 30 * len(brain)), len(brain) - 1)
        bf = brain[bi]

        ti = max(i for i, ts in enumerate(traj["timestamps"]) if ts <= t_global)

        ax_video.clear()
        ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_video.axis("off")
        ax_video.set_title(f"Camera · t={t_global:.1f}s", color="white", fontsize=10)

        ax_fly.clear()
        ax_fly.set_xlim(-1.5, 1.5)
        ax_fly.set_ylim(-1, 1)
        ax_fly.set_aspect("equal")
        ax_fly.axis("off")
        ax_fly.set_facecolor("#0f1117")
        h = np.radians(bf["heading"])
        wp = np.sin(t_local * 20) * 0.3
        ax_fly.add_patch(plt.Circle((0, 0), 0.25, color="#5c4033"))
        hx, hy = 0.35 * np.cos(h), 0.35 * np.sin(h)
        ax_fly.add_patch(plt.Circle((hx, hy), 0.15, color="#3d2b1f"))
        for sign in (-1, 1):
            ax_fly.plot([0, sign * 0.5 * np.cos(h + np.pi / 2 + wp)],
                        [0, sign * 0.5 * np.sin(h + np.pi / 2 + wp)], color="#94a3b8", lw=2)
        ax_fly.set_title(f"Fly θ={bf['heading']:.0f}°", color="#94a3b8", fontsize=9)

        dot.set_data([traj["x"][ti]], [traj["y"][ti]])

        for bar, v in zip(bars_t4, [bf["t4"][f"T4{t}"] for t in types]):
            bar.set_height(v)
        for bar, v in zip(bars_t5, [bf["t5"][f"T5{t}"] for t in types]):
            bar.set_height(v)

        for bar, v in zip(epg_bars, bf["epg"]):
            bar.set_height(v)

        fig.suptitle("Fly Visual Odometry — Brain Tour", color="white", fontsize=12)
        img = fig_to_bgr(fig)

        if out is None:
            h, w = img.shape[:2]
            out = cv2.VideoWriter(str(args.o), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        out.write(img)
        written += 1
        frame_i += 1

    cap.release()
    if out:
        out.release()
    plt.close(fig)
    print(f"Saved {written} frames -> {args.o}")


if __name__ == "__main__":
    main()
