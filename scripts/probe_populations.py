#!/usr/bin/env python3
"""
Probe which MaleCNS populations respond to camera motion in a video.

Usage:
    python scripts/probe_populations.py /path/to/video.mp4 --duration 30
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import VideoVisualEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe MaleCNS visual responses")
    parser.add_argument("video", type=str)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gain", type=float, default=0.9)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cfg = FlyVOConfig(device=args.device, visual_gain=args.gain)
    engine = MaleCNSEngine(cfg)
    print(f"Loading MaleCNS v1.0 ({engine.n_neurons if engine._brain else '…'} neurons)…")
    engine._ensure_loaded()
    print(f"  neurons: {engine.n_neurons:,}")
    print(f"  photoreceptors: {engine.n_visual:,}")
    print(f"  device: {engine.device}")

    encoder = VideoVisualEncoder(engine, gain=args.gain)
    pathways = engine.pathway_indices()
    for name, idx in pathways.items():
        print(f"  pathway '{name}': {len(idx)} neurons")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(args.duration * fps)
    frames_per_step = max(1, int(round(fps * cfg.brain_dt)))

    spike_hist: dict[str, list[int]] = defaultdict(list)
    motion_hist: list[float] = []
    pan_hist: list[float] = []

    engine.reset()
    encoder.reset()
    frame_i = 0
    step = 0

    pbar = tqdm(total=total_frames, unit="frame")
    while frame_i < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % frames_per_step == 0:
            eye_drive, inject, metrics = encoder.encode_frame(frame)
            result = engine.step(step * cfg.brain_dt, eye_drive=eye_drive, inject=inject)
            for name, count in result.pathway_spikes.items():
                spike_hist[name].append(count)
            motion_hist.append(metrics["motion"])
            pan_hist.append(metrics["pan_shift"])
            step += 1
        frame_i += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    motion = np.array(motion_hist)
    pan = np.array(pan_hist)
    print("\n=== Visual metrics ===")
    print(f"  motion energy: mean={motion.mean():.4f}  std={motion.std():.4f}")
    print(f"  pan shift:     mean={pan.mean():.4f}  std={pan.std():.4f}")

    print("\n=== Pathway spike counts (mean ± std per step) ===")
    ranked = []
    for name, counts in spike_hist.items():
        arr = np.array(counts, dtype=float)
        ranked.append((arr.mean(), arr.std(), name, len(counts)))
    ranked.sort(reverse=True)
    for mean, std, name, n in ranked:
        print(f"  {name:22s}  {mean:6.2f} ± {std:5.2f}  ({n} steps)")

    print("\n=== Descending neurons (last step) ===")
    engine.reset()
    encoder.reset()
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(min(50, total_frames)):
        ok, frame = cap.read()
        if not ok:
            break
    if ok:
        eye_drive, inject, _ = encoder.encode_frame(frame)
        result = engine.step(0.0, eye_drive=eye_drive, inject=inject)
        for k, v in sorted(result.descending.items()):
            print(f"  {k}: {v:.0f} spikes")

    cap.release()
    print("\nDone. Use these pathways to tune motion_readout.py gains.")


if __name__ == "__main__":
    main()
