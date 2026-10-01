#!/usr/bin/env python3
"""
Forensic diagnostic pack for MaleCNS VO — one video → full outputs folder.

Usage:
    PYTHONPATH=. python scripts/forensic_pack.py /path/to/video.mp4 -o output/forensic_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.dopamine import load_adaptive, path_loss, video_key_from_path
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.motion_readout import DescendingNeuronDecoder
from fly_vo.streaming import StreamingMaleCNS
from fly_vo.teacher_io import load_teacher
from fly_vo.visualize import plot_trajectory


def _log(lines: list[str], msg: str) -> None:
    print(msg)
    lines.append(msg)


def _save_eye_panorama(path: Path, pan: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(pan, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("panorama bin")
    ax.set_ylabel("luminance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_frame(path: Path, bgr: np.ndarray) -> None:
    cv2.imwrite(str(path), bgr)


def run_forensic(video: Path, out_dir: Path, duration: float | None) -> None:
    log_lines: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visual_samples"
    vis_dir.mkdir(exist_ok=True)

    cfg = FlyVOConfig(max_frame_size=480, brain_dt=0.020)
    vk = video_key_from_path(str(video))
    adaptive = load_adaptive(video_key=vk)
    teacher = load_teacher(str(video))

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_dur = n_frames / fps if n_frames > 0 else 0.0
    cap.release()

    end_s = duration if duration else vid_dur
    _log(log_lines, f"=== Forensic run {datetime.now(timezone.utc).isoformat()} ===")
    _log(log_lines, f"Video: {video}")
    _log(log_lines, f"FPS={fps:.2f} frames={n_frames} duration={vid_dur:.2f}s run_end={end_s:.2f}s")
    _log(log_lines, f"Video key: {vk}")
    _log(log_lines, f"Adaptive params: {adaptive}")

    proc = StreamingMaleCNS(cfg, adaptive=adaptive, emit_every_n_steps=1)
    proc.reset(heading_deg=90.0)
    _log(log_lines, f"Brain device: {proc.device}")
    proc.engine._ensure_loaded()
    _log(log_lines, f"Neurons: {proc.engine.n_neurons:,}  photoreceptors: {proc.engine.n_visual:,}")

    for name, idx in proc.engine.pathway_indices().items():
        _log(log_lines, f"  pathway {name}: {len(idx)} cells")

    # --- Full stream with per-step debug ---
    steps: list[dict] = []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fpp = max(1, int(round(fps * cfg.brain_dt)))
    fi = 0
    step_i = 0
    prev_t = None
    spike_hist: dict[str, list[int]] = defaultdict(list)
    source_hist: list[str] = []
    sample_times = {0.0, end_s * 0.25, end_s * 0.5, end_s * 0.75, max(0, end_s - 0.5)}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if t >= end_s - 1e-6:
            break
        if fi % fpp == 0:
            bgr = proc._maybe_resize(frame)
            eye, inj, metrics = proc.encoder.encode_frame(bgr)
            inject_info = []
            for cells, strength in inj:
                inject_info.append({"n_cells": len(cells), "strength": float(strength)})

            result = proc.engine.step(t, eye_drive=eye, inject=inj)
            motion = proc.egomotion.decode_frame(result, metrics)
            source_hist.append(proc.egomotion.last_source)

            dt = t - prev_t if prev_t is not None else fpp / fps
            h_before = proc.compass.heading_deg
            if step_i > 0:
                proc._stream_time = t
                metrics["timestamp"] = t
                proc._integrate_heading(motion, metrics, dt)
                gain = cfg.velocity_gain
                from fly_vo.pfn import body_to_world

                vx_w, vy_w = body_to_world(
                    np.array([motion.vx_body * gain]),
                    np.array([motion.vy_body * gain]),
                    np.array([proc.compass.heading]),
                )
                proc._x += float(vx_w[0] * dt)
                proc._y += float(vy_w[0] * dt)

            for name, count in result.pathway_spikes.items():
                spike_hist[name].append(count)

            steps.append(
                {
                    "t": round(t, 4),
                    "x": round(proc._x, 4),
                    "y": round(proc._y, 4),
                    "heading_deg": round(proc.compass.heading_deg, 2),
                    "vx_body": round(motion.vx_body, 4),
                    "vy_body": round(motion.vy_body, 4),
                    "yaw_rate": round(motion.yaw_rate, 4),
                    "readout_source": proc.egomotion.last_source,
                    "motion": round(metrics.get("motion", 0), 5),
                    "pan_shift": round(metrics.get("pan_shift", 0), 5),
                    "rotation_deg": round(metrics.get("rotation_deg", 0), 3),
                    "step_rotation_deg": round(metrics.get("step_rotation_deg", 0), 3),
                    "n_spikes": result.n_active,
                    "pathway_spikes": result.pathway_spikes,
                    "descending": {k: round(v, 2) for k, v in result.descending.items()},
                    "inject": inject_info,
                    "heading_delta": round(proc.compass.heading_deg - h_before, 3)
                    if step_i > 0
                    else 0.0,
                    "corner_snapped": proc._corner_snapped,
                }
            )

            if any(abs(t - st) < 0.15 for st in sample_times):
                tag = f"t{t:05.1f}s"
                _save_frame(vis_dir / f"frame_{tag}.jpg", bgr)
                pan = np.array(metrics.get("eye_preview", []), dtype=np.float32)
                if len(pan):
                    _save_eye_panorama(vis_dir / f"panorama_{tag}.png", pan, f"eye panorama {tag}")
                with open(vis_dir / f"metrics_{tag}.json", "w") as f:
                    json.dump(
                        {
                            "metrics": metrics,
                            "inject": inject_info,
                            "pathway_spikes": result.pathway_spikes,
                            "descending": result.descending,
                            "eye_drive_stats": {
                                "min": float(eye.min()) if eye is not None else None,
                                "max": float(eye.max()) if eye is not None else None,
                                "mean": float(eye.mean()) if eye is not None else None,
                                "len": len(eye) if eye is not None else 0,
                            },
                        },
                        f,
                        indent=2,
                    )

            prev_t = t
            step_i += 1
        fi += 1
    cap.release()

    # --- Trajectory outputs ---
    traj_csv = out_dir / "trajectory.csv"
    with traj_csv.open("w") as f:
        f.write(
            "timestamp,x,y,heading_deg,vx_body,vy_body,yaw_rate,readout_source,"
            "motion,pan_shift,rotation_deg,n_spikes\n"
        )
        for s in steps:
            f.write(
                f"{s['t']},{s['x']},{s['y']},{s['heading_deg']},"
                f"{s['vx_body']},{s['vy_body']},{s['yaw_rate']},{s['readout_source']},"
                f"{s['motion']},{s['pan_shift']},{s['rotation_deg']},{s['n_spikes']}\n"
            )

    steps_json = out_dir / "steps_debug.json"
    steps_json.write_text(json.dumps(steps, indent=2), encoding="utf-8")

    traj_meta = {
        "video": str(video),
        "video_key": vk,
        "duration_s": end_s,
        "n_steps": len(steps),
        "brain_hz": cfg.sim_hz,
        "adaptive_params": adaptive.__dict__,
        "corner_snapped": proc._corner_snapped,
        "final": steps[-1] if steps else {},
        "teacher_loss": path_loss([(s["x"], s["y"]) for s in steps], teacher)
        if teacher is not None and len(teacher) >= 2
        else None,
    }
    (out_dir / "trajectory_meta.json").write_text(json.dumps(traj_meta, indent=2), encoding="utf-8")

    # Trajectory plot
    from fly_vo.streaming import TrajectoryPoint

    points = [
        TrajectoryPoint(
            timestamp=s["t"],
            x=s["x"],
            y=s["y"],
            heading_deg=s["heading_deg"],
            vx_body=s["vx_body"],
            vy_body=s["vy_body"],
            speed=float(np.hypot(s["vx_body"], s["vy_body"])),
            confidence=1.0,
        )
        for s in steps
    ]
    plot_trajectory(points, out_dir / "trajectory.png")

    # Heading / rotation timeline
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ts = [s["t"] for s in steps]
    axes[0].plot(ts, [s["heading_deg"] for s in steps], "b-", linewidth=0.8)
    axes[0].set_ylabel("heading °")
    axes[0].set_title("Heading")
    axes[1].plot(ts, [s["rotation_deg"] for s in steps], "r-", linewidth=0.5)
    axes[1].axhline(10, color="gray", ls="--", alpha=0.5, label="corner ref")
    axes[1].axhline(-10, color="gray", ls="--", alpha=0.5)
    axes[1].set_ylabel("ORB rot °")
    axes[1].legend()
    axes[2].plot(ts, [s["x"] for s in steps], label="x")
    axes[2].plot(ts, [s["y"] for s in steps], label="y")
    axes[2].set_ylabel("position")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "timeline.png", dpi=120)
    plt.close(fig)

    # Pathway summary
    pathway_summary = {}
    for name, counts in spike_hist.items():
        arr = np.asarray(counts, dtype=float)
        pathway_summary[name] = {
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "max": int(arr.max()),
            "n_steps": len(counts),
        }
    (out_dir / "pathway_summary.json").write_text(
        json.dumps(pathway_summary, indent=2), encoding="utf-8"
    )

    from collections import Counter

    src_counts = Counter(source_hist)
    _log(log_lines, "\n=== Readout source ===")
    for k, v in src_counts.most_common():
        _log(log_lines, f"  {k}: {v}/{len(source_hist)} steps ({100*v/len(source_hist):.1f}%)")

    _log(log_lines, "\n=== Pathway spikes/step (mean) ===")
    for name, stats in sorted(pathway_summary.items(), key=lambda x: -x[1]["mean"]):
        _log(log_lines, f"  {name:22s} {stats['mean']:6.1f} ± {stats['std']:.1f}")

    _log(log_lines, "\n=== Checkpoints ===")
    motion_arr = np.array([s["motion"] for s in steps])
    _log(log_lines, f"  [1] motion energy: mean={motion_arr.mean():.4f} std={motion_arr.std():.4f}")
    _log(log_lines, f"  [4] descending forward mean: {pathway_summary.get('forward', {}).get('mean', 0):.2f} spikes/step")
    _log(log_lines, f"  [4] descending steer mean:   {pathway_summary.get('steering', {}).get('mean', 0):.2f} spikes/step")
    _log(log_lines, f"  [7] readout: {src_counts.most_common(1)[0][0]}")
    _log(log_lines, f"  [8] corner_snapped: {proc._corner_snapped}")
    if traj_meta["teacher_loss"] is not None:
        _log(log_lines, f"  [9] teacher loss: {traj_meta['teacher_loss']:.4f}")

    # Copy pipeline doc
    pipeline_src = ROOT / "docs" / "PIPELINE_FORENSIC.md"
    if pipeline_src.exists():
        shutil.copy(pipeline_src, out_dir / "PIPELINE_FORENSIC.md")

    readme = out_dir / "README.txt"
    readme.write_text(
        textwrap.dedent(
            f"""
            Forensic pack for {video.name}
            Generated: {datetime.now(timezone.utc).isoformat()}

            Files:
              trajectory.csv       — x,y,heading per brain step
              steps_debug.json     — full per-step debug (spikes, inject, ORB)
              trajectory_meta.json — run metadata + adaptive params
              pathway_summary.json — mean pathway spike counts
              trajectory.png       — 2D path plot
              timeline.png         — heading, ORB rot, x/y vs time
              visual_samples/      — sample frames + eye panoramas + inject metrics
              run.log              — console log
              PIPELINE_FORENSIC.md — pipeline map

            Key finding: descending neurons usually silent; trajectory from ORB pathway fallback.
            """
        ).strip(),
        encoding="utf-8",
    )

    log_path = out_dir / "run.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nWrote forensic pack → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic diagnostic pack")
    parser.add_argument("video", type=str)
    parser.add_argument("-o", "--output", type=str, default="output/forensic_run")
    parser.add_argument("--duration", type=float, default=None, help="Max seconds to process")
    args = parser.parse_args()
    run_forensic(Path(args.video), Path(args.output), args.duration)


if __name__ == "__main__":
    main()
