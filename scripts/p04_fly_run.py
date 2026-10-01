#!/usr/bin/env python3
"""
P0.4 — drive the whole clip through MaleCNS and record every signal a trajectory
could be built from.

The question this answers: **can the fly's brain draw the trajectory itself?**

The chain is

    video -> frontend -> T4/T5 injection -> connectome -> descending neurons

and a trajectory needs a yaw rate. So every stage is recorded at every brain step:

    yaw_frozen        the frontend's own scalar estimate (the validated part)
    inject_lr         the T4/T5 lateral differential the encoder actually applied
    t4_lr, t5_lr      the T4 and T5 populations, fired per step
    steer_lr          DNa02 left minus right
    fwd_lr            DNg100 left minus right
    hedge_lr          MDN left minus right
    escape_lr         DNp01 left minus right
    all_dn_lr         every descending neuron summed as left minus right

If the trajectory from `yaw_frozen` and the trajectory from `steer_lr` agree, the
brain carries the signal. If only the frontend one is usable, the loss is in the
brain, and that is the finding.

Polarity is never taken from the data being tested: the sign convention is fixed on
the first `--calib` seconds and applied unchanged afterwards, so the evaluation
segment cannot influence its own orientation.

Runtime is roughly 1.5 s of compute per second of video, so the full clip is about
half an hour. Progress is written in chunks and merged at the end, so an interruption
costs at most one chunk.

Usage:
    PYTHONPATH=. python scripts/p04_fly_run.py --duration 30          # smoke test
    PYTHONPATH=. python scripts/p04_fly_run.py                        # full clip
    PYTHONPATH=. python scripts/p04_fly_run.py --merge-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.neural_probe import group_spike_counts
from fly_vo.visual_encoder import VideoVisualEncoder
from fly_vo.video_reader import VideoReader

VIDEO = ROOT / "data/p01r/VID00001.AVI"
OUT = ROOT / "output/p04_fly"
CHUNKS = OUT / "chunks"
CHUNK_S = 60.0

SIGNALS = [
    "yaw_frozen", "inject_lr", "t4_lr", "t5_lr",
    "steer_lr", "fwd_lr", "hedge_lr", "escape_lr", "all_dn_lr",
    "motion", "n_active",
]


def merge_only() -> None:
    files = sorted(CHUNKS.glob("chunk_*.npz"))
    if not files:
        print("нет чанков")
        return
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("t", *SIGNALS)}
    for f in files:
        d = np.load(f)
        for k in acc:
            acc[k].append(d[k])
    out = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(OUT / "fly_signals.npz", **out)
    print(f"объединено {len(files)} чанков -> {len(out['t'])} шагов, "
          f"{out['t'][-1] - out['t'][0]:.0f} с")
    for k in SIGNALS:
        v = out[k]
        print(f"  {k:>11s}  средн {v.mean():+8.4f}  стд {v.std():8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds of video; default is the whole clip")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_only()
        return

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine, flow_config=None)
    dt = float(engine.config.brain_dt)

    with VideoReader(VIDEO) as vr:
        total = vr.duration_s
    t_end = total if args.duration is None else min(args.start + args.duration, total)

    print("P0.4 — прогон мозга по ролику")
    print(f"  видео {VIDEO.name}: {total:.0f} с, шаг мозга {dt * 1000:.0f} мс "
          f"({1 / dt:.0f} Гц)")
    print(f"  окно {args.start:.0f} .. {t_end:.0f} с, "
          f"ожидается {int((t_end - args.start) / dt):,} шагов")
    print(f"  оценка времени: {(t_end - args.start) * 1.5 / 60:.0f} мин\n")

    engine.reset(seed=64)
    encoder.reset()

    t0 = time.perf_counter()
    done = 0
    seg_start = args.start
    while seg_start < t_end:
        seg_end = min(seg_start + CHUNK_S, t_end)
        tag = f"{int(seg_start):05d}_{int(seg_end):05d}"
        path = CHUNKS / f"chunk_{tag}.npz"
        if path.exists():
            seg_start = seg_end
            done += int((seg_end - seg_start) / dt)
            continue

        rows = {k: [] for k in ("t", *SIGNALS)}
        first_of_chunk = True
        for t, frame in iter_video_at_brain_hz(str(VIDEO), seg_start, seg_end, dt):
            # brain_timestamps is inclusive at both ends, so every chunk after the
            # first would repeat the step its predecessor ended on
            if first_of_chunk:
                first_of_chunk = False
                if seg_start > args.start:
                    continue
            eye_drive, inject, metrics = encoder.encode_frame(frame)
            result = engine.step(t, eye_drive=eye_drive, inject=inject)

            g = group_spike_counts(result, encoder)
            d = result.descending

            # every descending neuron, left minus right, as one blunt aggregate
            all_lr = 0.0
            for key, val in d.items():
                if key.endswith("_L"):
                    all_lr += float(val)
                elif key.endswith("_R"):
                    all_lr -= float(val)

            rows["t"].append(t)
            rows["yaw_frozen"].append(float(metrics.get("yaw_frozen",
                                                        metrics.get("yaw_signal", 0.0))))
            rows["inject_lr"].append(float(metrics.get("local_lr_asymmetry", 0.0)))
            rows["t4_lr"].append(float(g.get("T4a_L", 0) + g.get("T4b_L", 0)
                                       - g.get("T4a_R", 0) - g.get("T4b_R", 0)))
            rows["t5_lr"].append(float(g.get("T5a_L", 0) + g.get("T5b_L", 0)
                                       - g.get("T5a_R", 0) - g.get("T5b_R", 0)))
            rows["steer_lr"].append(float(d.get("steer_L", 0.0) - d.get("steer_R", 0.0)))
            rows["fwd_lr"].append(float(d.get("forward_L", 0.0) - d.get("forward_R", 0.0)))
            rows["hedge_lr"].append(float(d.get("backward_L", 0.0) - d.get("backward_R", 0.0)))
            rows["escape_lr"].append(float(d.get("escape_L", 0.0) - d.get("escape_R", 0.0)))
            rows["all_dn_lr"].append(all_lr)
            rows["motion"].append(float(metrics.get("motion", 0.0)))
            rows["n_active"].append(float(metrics.get("local_n_active_cells", 0)))

        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32)
                                     for k, v in rows.items()})
        done += len(rows["t"])
        el = time.perf_counter() - t0
        rate = el / max(seg_end - args.start, 1e-9)
        left = (t_end - seg_end) * rate / 60
        print(f"  {seg_end:7.0f} / {t_end:.0f} с   шагов {done:7,}   "
              f"прошло {el / 60:5.1f} мин   осталось ~{left:5.1f} мин", flush=True)
        seg_start = seg_end

    merge_only()


if __name__ == "__main__":
    main()
