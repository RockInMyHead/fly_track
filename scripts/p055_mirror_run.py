#!/usr/bin/env python3
"""
P05.5 — mirror control on the real video.

Why this run exists
-------------------
P05.4 came out invalid for a structural reason: the controls chosen to isolate the path
do not isolate the *stimulus*. DNg100 and MDN receive exactly zero input from the 48
depth-1 carriers, yet they still separated the turns at AUC 0.806 and 0.589, and
DNp17 at 0.944 sat no further above chance than DNa02 at 0.919. Something other than the
path was separating left from right turns.

Mirroring the video removes that possibility entirely. The same frames, the same scene,
the same walking, only flipped horizontally. So:

    a neuron that reads turn direction   must FLIP   - a leftward turn now looks
                                                     rightward to it
    a neuron that reads the scene        must NOT    - a busy corridor mirrored is
                                                     just as busy

The labels never change; they describe the camera, not the image. That is what makes the
test decisive: scene differences are held fixed by construction, and only the direction
of image motion is reversed.

Two passes, because one is not interpretable alone
--------------------------------------------------
    control   unmirrored, my runner. Must reproduce P05.3. If it does not, the runner
              differs from the one that produced the frozen test and nothing below counts.
    mirror    same frames, flipped. The only thing that changed is the direction of
              image motion.

Everything else is held at the P05.3 settings: seed 64, the same encoder, the same 46
recorded cells, the same dynamics, no weight touched.

Usage:
    PYTHONPATH=. python scripts/p055_mirror_run.py --tag control
    PYTHONPATH=. python scripts/p055_mirror_run.py --tag mirror --mirror
    PYTHONPATH=. python scripts/p055_mirror_run.py --tag mirror --merge-only
"""

from __future__ import annotations

import argparse
import importlib.util
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
from fly_vo.visual_encoder import VideoVisualEncoder
from fly_vo.video_reader import VideoReader

VIDEO = ROOT / "data/p01r/VID00001.AVI"
P053_RUN = ROOT / "scripts/p053_threshold_run.py"
OUT = ROOT / "output/p055_mirror"
CHUNK_S = 60.0
SEED = 64


def load_target_builder():
    spec = importlib.util.spec_from_file_location("p053_run", P053_RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_targets


def merge(tag: str) -> None:
    chunks = OUT / f"chunks_{tag}"
    files = sorted(chunks.glob("chunk_*.npz"))
    if not files:
        print(f"нет чанков для {tag}")
        return
    keys = list(np.load(files[0]).keys())
    acc = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys:
            acc[k].append(d[k])
    merged = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(OUT / f"trace_{tag}.npz", **merged)
    print(f"  {tag}: объединено {len(files)} чанков -> {len(merged['t'])} шагов, "
          f"{merged['t'][-1] - merged['t'][0]:.0f} с")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["control", "mirror"])
    ap.add_argument("--mirror", action="store_true",
                    help="flip every frame horizontally")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    chunks = OUT / f"chunks_{args.tag}"
    chunks.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(args.tag)
        return

    build_targets = load_target_builder()
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    targets, carriers = build_targets(engine.brain)
    brain = engine.brain
    idx = np.array([t["cell"] for t in targets], dtype=np.int64)
    n_t = len(targets)

    with VideoReader(VIDEO) as vr:
        total = vr.duration_s
    t_end = total if args.duration is None else min(args.duration, total)

    print(f"P05.5 — прогон '{args.tag}'"
          f"{' (ЗЕРКАЛЬНЫЙ)' if args.mirror else ' (контрольный, без зеркала)'}")
    print(f"  seed {SEED}, нейронов {n_t}, носителей из P05.1 {len(carriers)}")
    print(f"  видео {total:.0f} с, окно 0..{t_end:.0f} с")
    print(f"  зеркало: {'кадр переворачивается по горизонтали' if args.mirror else 'нет'}")
    print(f"  оценка: {t_end * 1.5 / 60:.0f} мин\n")

    engine.reset(seed=SEED)
    encoder = VideoVisualEncoder(engine, flow_config=None)
    dt = float(engine.config.brain_dt)

    t0 = time.perf_counter()
    seg_start = 0.0
    while seg_start < t_end:
        seg_end = min(seg_start + CHUNK_S, t_end)
        path = chunks / f"chunk_{int(seg_start):05d}_{int(seg_end):05d}.npz"
        if path.exists():
            seg_start = seg_end
            continue
        rows: dict[str, list] = {k: [] for k in
                                 ("t", "v_pre", "fired", "total_in",
                                  "drive_carrier", "drive_L", "drive_R", "n_active")}
        first = True
        for t, frame in iter_video_at_brain_hz(str(VIDEO), seg_start, seg_end, dt):
            if first:
                first = False
                if seg_start > 0:
                    continue
            if args.mirror:
                # horizontal flip reverses left and right; this is the whole experiment
                frame = np.ascontiguousarray(frame[:, ::-1])
            eye_drive, inject, _ = encoder.encode_frame(frame)

            v_pre = brain.v.ravel()[idx].copy()
            act = np.zeros(brain.n, dtype=np.float64)
            if len(brain.fired):
                act[brain.fired] = 1.0

            tot = np.empty(n_t)
            drv = np.empty(n_t)
            dl = np.empty(n_t)
            dr = np.empty(n_t)
            for i, tg in enumerate(targets):
                a = act[tg["pre"]]
                tot[i] = np.dot(tg["w"], a)
                drv[i] = np.dot(tg["w"] * tg["carrier"], a)
                dl[i] = np.dot(tg["w"] * tg["pref_L"], a)
                dr[i] = np.dot(tg["w"] * tg["pref_R"], a)

            result = engine.step(t, eye_drive=eye_drive, inject=inject)
            rows["t"].append(t)
            rows["v_pre"].append(v_pre)
            rows["fired"].append(np.isin(idx, result.fired))
            rows["total_in"].append(tot)
            rows["drive_carrier"].append(drv)
            rows["drive_L"].append(dl)
            rows["drive_R"].append(dr)
            rows["n_active"].append(len(result.fired))

        np.savez_compressed(path, **{
            "t": np.asarray(rows["t"], np.float32),
            "v_pre": np.asarray(rows["v_pre"], np.float32),
            "fired": np.asarray(rows["fired"], np.uint8),
            "total_in": np.asarray(rows["total_in"], np.float32),
            "drive_carrier": np.asarray(rows["drive_carrier"], np.float32),
            "drive_L": np.asarray(rows["drive_L"], np.float32),
            "drive_R": np.asarray(rows["drive_R"], np.float32),
            "n_active": np.asarray(rows["n_active"], np.int32)})
        el = time.perf_counter() - t0
        left = (t_end - seg_end) * (el / max(seg_end, 1e-9)) / 60
        print(f"  {seg_end:7.0f} / {t_end:.0f} с  шагов {len(rows['t']):6d}  "
              f"прошло {el / 60:5.1f} мин  осталось ~{left:5.1f} мин", flush=True)
        seg_start = seg_end

    merge(args.tag)


if __name__ == "__main__":
    main()
