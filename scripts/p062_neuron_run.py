#!/usr/bin/env python3
"""
P06.2 — drive the second video through the brain and record the frozen cells.

The targets, the seed and the dynamics come from `p053_threshold_run.py` unchanged; only
the video differs. That script is left untouched so the first clip's recording stays
reproducible byte for byte.

Why this is worth running before any labels exist
-------------------------------------------------
The recording does not depend on the labels. It is 20 minutes of compute that can happen
while the windows are being labelled by eye, so that the moment the labels land the test
is a single command.

What makes the second clip a real test
--------------------------------------
Different route, different corridors, different objects. The cells were chosen on the
first clip, so if they still prefer the same direction here, they are reading something
about motion rather than a particular feature of a particular scene. That is the control
the mirror pass on one clip could not give.

Usage:
    PYTHONPATH=. python scripts/p062_neuron_run.py --duration 60
    PYTHONPATH=. python scripts/p062_neuron_run.py
    PYTHONPATH=. python scripts/p062_neuron_run.py --merge-only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
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

VIDEO = Path(os.environ.get("FLY_VIDEO", ROOT / "data/p01r/VID00002.AVI"))
P053_RUN = ROOT / "scripts/p053_threshold_run.py"
# The output directory is overridable so the same runner can record a new clip without
# touching the VID00002 results. Defaults are unchanged.
OUT = Path(os.environ.get("FLY_OUT", ROOT / "output/p06_neurons"))
CHUNKS = OUT / "chunks"
CHUNK_S = 60.0
SEED = 64
DT_FALLBACK = 0.02


def load_p053():
    spec = importlib.util.spec_from_file_location("p053_run", P053_RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge() -> None:
    files = sorted(CHUNKS.glob("chunk_*.npz"))
    if not files:
        print("нет чанков")
        return
    keys = list(np.load(files[0]).keys())
    acc = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys:
            acc[k].append(d[k])
    merged = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(OUT / "spike_trace.npz", **merged)
    print(f"объединено {len(files)} чанков -> {len(merged['t'])} шагов, "
          f"{merged['t'][-1] - merged['t'][0]:.0f} с")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--video", default=None,
                    help="clip to drive; default keeps the previous choice of VID00002")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output folder, so no earlier recording is lost")
    args = ap.parse_args()

    global VIDEO, OUT, CHUNKS
    if args.video:
        VIDEO = (Path(args.video) if str(args.video).startswith("/")
                 else ROOT / "data/p01r" / args.video)
    if args.tag:
        OUT = ROOT / f"output/p06_neurons_{args.tag}"
        CHUNKS = OUT / "chunks"
    if not VIDEO.exists():
        print(f"нет видео: {VIDEO}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge()
        return

    mod = load_p053()
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    # the target set and the wiring come from p053 and are not re-derived
    targets, carriers = mod.build_targets(engine.brain)
    brain = engine.brain
    idx = np.array([t["cell"] for t in targets], dtype=np.int64)
    n_t = len(targets)
    dt = float(engine.config.brain_dt)

    with VideoReader(VIDEO) as vr:
        total = vr.duration_s
        n_frames = vr.frame_count
        declared = vr.declared_count
    t_end = total if args.duration is None else min(args.duration, total)

    print("P06.2 — видео через мозг, замороженные цели")
    print(f"  цели и seed взяты из {P053_RUN.name} без изменений: {n_t} клеток, "
          f"seed {SEED}")
    print(f"  видео {VIDEO.name}: {total:.1f} с, {n_frames} кадров "
          f"(контейнер заявляет {declared}), читается последовательно")
    if declared > 0 and abs(n_frames - declared) > declared * 0.02:
        print(f"    расхождение с заголовком: {n_frames / declared:.4f} — "
              f"заголовку верить нельзя, длительность берётся по факту")
    print(f"  окно 0..{t_end:.0f} с, шаг {dt * 1000:.0f} мс")
    print(f"  оценка: {t_end * 1.5 / 60:.0f} мин\n")

    engine.reset(seed=SEED)
    encoder = VideoVisualEncoder(engine, flow_config=None)

    t0 = time.perf_counter()
    seg = 0.0
    while seg < t_end:
        end = min(seg + CHUNK_S, t_end)
        path = CHUNKS / f"chunk_{int(seg):05d}_{int(end):05d}.npz"
        if path.exists():
            seg = end
            continue
        rows: dict[str, list] = {k: [] for k in
                                 ("t", "v_pre", "fired", "total_in",
                                  "drive_carrier", "drive_L", "drive_R", "n_active")}
        first = True
        for t, frame in iter_video_at_brain_hz(str(VIDEO), seg, end, dt):
            if first:
                first = False
                if seg > 0:
                    continue
            eye, inject, _ = encoder.encode_frame(frame)
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
            result = engine.step(t, eye_drive=eye, inject=inject)
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
        left = (t_end - end) * (el / max(end, 1e-9)) / 60
        print(f"  {end:7.0f} / {t_end:.0f} с  шагов {len(rows['t']):6d}  "
              f"прошло {el / 60:5.1f} мин  осталось ~{left:5.1f} мин", flush=True)
        seg = end

    merge()

    keys = ["cell", "cell_type", "side", "set", "in_degree", "in_weight",
            "carrier_weight", "w_from_left_carriers", "w_from_right_carriers",
            "w_exc_from_left", "w_exc_from_right", "w_inh_from_left",
            "w_inh_from_right", "signed_from_left", "signed_from_right"]
    with (OUT / "targets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for tg in targets:
            w.writerow({k: tg.get(k) for k in keys})
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
