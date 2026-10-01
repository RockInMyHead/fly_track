#!/usr/bin/env python3
"""Batch-train MaleCNS params per video clip (user teacher or OpenCV proxy)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fly_vo.config import FlyVOConfig
from fly_vo.dopamine import (
    AdaptiveParams,
    load_adaptive,
    path_loss,
    save_adaptive,
    video_key_from_path,
    dopamine_optimize,
)
from fly_vo.streaming import StreamingMaleCNS
from fly_vo.teacher_from_video import estimate_teacher_from_video
from fly_vo.teacher_io import load_teacher_record, save_teacher


def _fly_xy(points) -> list[tuple[float, float]]:
    return [(p.x, p.y) for p in points]


def train_one(
    video: Path,
    *,
    rounds: int = 3,
    heading_deg: float = 90.0,
    out_dir: Path,
) -> dict:
    vk = video_key_from_path(video)
    adaptive_path = Path(__file__).resolve().parents[1] / "data" / "dopamine_state.json"

    rec = load_teacher_record(video)
    if rec is not None:
        teacher, teacher_source = rec
    else:
        teacher = estimate_teacher_from_video(str(video), heading_deg=heading_deg)
        save_teacher(video, teacher, source="opencv", meta={"note": "auto proxy"})
        teacher_source = "opencv"

    params = AdaptiveParams()
    save_adaptive(params, adaptive_path, video_key=vk)

    cfg = FlyVOConfig()
    proc = StreamingMaleCNS(cfg, adaptive=params)
    cache = proc.collect_neural_cache(str(video))

    history: list[dict] = []
    best_loss = float("inf")
    best_params = params

    for rnd in range(1, rounds + 1):
        params = load_adaptive(adaptive_path, video_key=vk)
        proc.adaptive = params
        proc.egomotion.adaptive = params

        fly_before = proc.integrate_from_cache(
            cache, params, heading_deg=heading_deg, emit_stride=4
        )
        loss_before = path_loss(_fly_xy(fly_before), teacher)

        new_params, lb, la, fly_after, trials = dopamine_optimize(
            params,
            cache,
            proc,
            teacher,
            x0=0.0,
            y0=0.0,
            heading_deg=heading_deg,
        )
        save_adaptive(new_params, adaptive_path, video_key=vk)

        if la < best_loss:
            best_loss = la
            best_params = new_params

        p0, pn = fly_after[0], fly_after[-1]
        rec = {
            "round": rnd,
            "trials": trials,
            "loss_before": lb,
            "loss_after": la,
            "yaw_gain": new_params.yaw_gain_scale,
            "speed_scale": new_params.speed_scale,
            "end": {"x": pn.x, "y": pn.y, "heading": pn.heading_deg},
            "delta_heading": pn.heading_deg - p0.heading_deg,
        }
        history.append(rec)
        print(
            f"  r{rnd}: loss {lb:.3f}→{la:.3f}  "
            f"yaw×{new_params.yaw_gain_scale:.2f} spd×{new_params.speed_scale:.2f}  "
            f"Δh={rec['delta_heading']:.1f}°"
        )
        if la < 0.35:
            break

    result = {
        "video": str(video),
        "video_key": vk,
        "teacher_source": teacher_source,
        "teacher_points": len(teacher),
        "rounds": history,
        "best_loss": best_loss,
        "params": asdict(best_params),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{vk}_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    save_teacher(
        video,
        teacher,
        source=teacher_source,
        meta={"best_loss": best_loss},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch train per-video MaleCNS params")
    parser.add_argument(
        "videos",
        nargs="*",
        help="Video paths (default: Downloads/VID00001_*.mp4)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("output/batch_train"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--heading", type=float, default=90.0)
    args = parser.parse_args()

    if args.videos:
        videos = [Path(v).expanduser() for v in args.videos]
    else:
        dl = Path.home() / "Downloads"
        videos = sorted(dl.glob("VID00001_*.mp4"))

    if not videos:
        raise SystemExit("No videos found")

    summary = []
    print(f"Training {len(videos)} clip(s), {args.rounds} round(s) each\n")
    for video in videos:
        if not video.exists():
            print(f"SKIP missing: {video}")
            continue
        print(f"=== {video.name} ===")
        try:
            result = train_one(
                video,
                rounds=args.rounds,
                heading_deg=args.heading,
                out_dir=args.output,
            )
            summary.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summary.append({"video": str(video), "error": str(exc)})

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nDone → {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
