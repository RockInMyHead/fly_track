#!/usr/bin/env python3
"""
P05.3 — is the spike threshold what kills the turn signal?

The dynamics in the engine, exactly as `FlyBrain.step` implements them:

    current = synaptic_input(fired) * gain       gain  = 3.0
    v      *= decay                              decay = 0.8187 per 20 ms step
    v      += current + tonic                    tonic = 0.14
    v      += noise                              amplitude 0.22, rate 1.2 Hz
    fired   = flatnonzero(v >= 1.0)              THRESHOLD = 1.0
    v[fired] = 0                                 reset

Two numbers follow from that and they frame the whole question:

    baseline v with no input      tonic / (1 - decay) = 0.772
    distance to threshold         1.0 - 0.772        = 0.228
    noise step size                                  = 0.220

So the neuron lives 0.228 below threshold and the noise kicks it by 0.220, which is 96
percent of the entire usable range. Any directional input has to move v by a meaningful
fraction of 0.228 to change the firing rate, and it does so against a perturbation of
comparable size. That is the hypothesis this measures.

What is recorded, per step, for each chosen neuron
-------------------------------------------------
    v_pre            membrane potential as it is evaluated against the threshold
    fired            whether it crossed
    total_input      sum of w over every presynaptic neuron that fired on the previous
                     step, i.e. the whole synaptic current before gain
    drive_carrier    the same sum restricted to the carriers of P05.1
    drive_L, drive_R the carriers split by their own direction preference

The split by preference is the point of the exercise. A neuron can receive a large
carrier input and still carry nothing, if the carriers that prefer LEFT and the ones
that prefer RIGHT deliver equal and opposite weight. That is cancellation at the single
neuron, which the type-level projection of P05.2 could not see.

The comparison set
------------------
    keeps direction   DNp17, DNa07, DNp26, DNp20, DNbe001, DNp11
    same input, loses DNpe056 (5.3%), DNge030 (2.3%), DNg82 (2.0%), DNpe037 (1.7%)
    little input      DNpe022, DNp31, DNa02, MDN, DNg100, DNp01

DNpe056 against DNbe001 is the clean test: 5.3 against 5.4 percent of input from the
same path, one carrying the direction at AUC 0.96 and the other at 0.50.

No threshold is tuned, no weight is changed, and the video, seed, encoder and dynamics
are those of the P05 run.

Usage:
    PYTHONPATH=. python scripts/p053_threshold_run.py --duration 60
    PYTHONPATH=. python scripts/p053_threshold_run.py
    PYTHONPATH=. python scripts/p053_threshold_run.py --merge-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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
P051_GROUPS = ROOT / "output/p051_signal_break/group_metrics.csv"
GROUPS = ROOT / "output/p05_layer_trace/groups.json"
OUT = ROOT / "output/p053_threshold"
CHUNKS = OUT / "chunks"
CHUNK_S = 60.0
THRESHOLD = 1.0
SEED = 64

KEEPS = ["DNp17", "DNa07", "DNp26", "DNp20", "DNbe001", "DNp11"]
SAME_INPUT_LOSES = ["DNpe056", "DNge030", "DNg82", "DNpe037"]
LITTLE_INPUT = ["DNpe022", "DNp31", "DNa02", "MDN", "DNg100", "DNp01"]
TARGETS = KEEPS + SAME_INPUT_LOSES + LITTLE_INPUT


def box_mat(M: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return M
    c = np.cumsum(M, axis=0, dtype=np.float64)
    c = np.vstack([np.zeros((1, M.shape[1])), c])
    o = c[n:] - c[:-n]
    left = (n - 1) // 2
    right = n - 1 - left
    o = np.vstack([np.repeat(o[:1], left, axis=0), o, np.repeat(o[-1:], right, axis=0)])
    return o[: M.shape[0]] / n


def build_targets(brain) -> tuple[list[dict], np.ndarray]:
    """Cells to watch, with their true presynaptic partners.

    The matrix is CSR with row = presynaptic source and `indices` = its postsynaptic
    targets; `_propagate` scatters the outgoing weights of each spiking neuron into
    `indices`. A neuron's INPUT is therefore its COLUMN, not its row, and the rows of
    the cells we care about are useless here. Incoming edges are recovered by scanning
    every edge once and keeping the ones that land on a target cell.
    """
    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    by_group = {int(r["group"]): r for r in csv.DictReader(P051_GROUPS.open())}
    carriers = {int(r["group"]) for r in by_group.values()
                if r["carries"] == "True" and r["in_analysis"] == "True"
                and int(r["depth_bin"]) == 1
                and r["superclass"] != "descending_neuron"}
    pref = {g: ("RIGHT" if float(by_group[g]["pol_64"]) > 0 else "LEFT")
            for g in carriers}

    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)
    wt = np.asarray(brain.weights)
    n = brain.n

    # cell -> its carrier group, for the presynaptic mask
    cell_carrier: dict[int, int] = {}
    for g in carriers:
        m = meta[g]
        for c in np.flatnonzero((ct == m["cell_type"]) & (sd == m["side"])).tolist():
            cell_carrier[c] = g

    want: list[tuple[str, str, int]] = []          # (type, side, cell)
    for nm in TARGETS:
        for side in ("L", "R"):
            for c in np.flatnonzero((ct == nm) & (sd == side)).tolist():
                want.append((nm, side, int(c)))
    pos = np.full(n, -1, np.int32)
    for k, (_, _, c) in enumerate(want):
        pos[c] = k

    # ---- one pass over every edge, keeping those aimed at a watched cell --------
    counts = np.diff(ip).astype(np.int64)
    edge_src = np.repeat(np.arange(n, dtype=np.int32), counts)
    hit = np.flatnonzero(pos[ix] >= 0)
    srcs = edge_src[hit]
    tpos = pos[ix[hit]]
    ws = wt[hit].astype(np.float64)
    order = np.argsort(tpos, kind="stable")
    srcs, tpos, ws = srcs[order], tpos[order], ws[order]
    bounds = np.searchsorted(tpos, np.arange(len(want) + 1))
    del edge_src, hit, tpos, order

    rows = []
    for k, (nm, side, c) in enumerate(want):
        pre = srcs[bounds[k]:bounds[k + 1]].astype(np.int64)
        w = ws[bounds[k]:bounds[k + 1]]
        if len(pre) == 0:
            continue
        car_idx = np.array([cell_carrier.get(int(p), -1) for p in pre.tolist()])
        is_car = car_idx >= 0
        pref_L = np.array([g >= 0 and pref[g] == "LEFT" for g in car_idx])
        pref_R = np.array([g >= 0 and pref[g] == "RIGHT" for g in car_idx])
        rows.append({
            "cell": c, "cell_type": nm, "side": side,
            "set": ("keeps" if nm in KEEPS else
                    "same_input_loses" if nm in SAME_INPUT_LOSES else
                    "little_input"),
            "pre": pre, "w": w,
            "carrier": is_car, "pref_L": pref_L, "pref_R": pref_R,
            "in_degree": int(len(pre)),
            "in_weight": float(np.abs(w).sum()),
            "carrier_weight": float(np.abs(w[is_car]).sum()),
            "w_from_left_carriers": float(np.abs(w[pref_L]).sum()),
            "w_from_right_carriers": float(np.abs(w[pref_R]).sum()),
            "w_exc_from_left": float(w[pref_L & (w >= 0)].sum()),
            "w_exc_from_right": float(w[pref_R & (w >= 0)].sum()),
            "w_inh_from_left": float(w[pref_L & (w < 0)].sum()),
            "w_inh_from_right": float(w[pref_R & (w < 0)].sum()),
            "signed_from_left": float(w[pref_L].sum()),
            "signed_from_right": float(w[pref_R].sum()),
        })
    return rows, carriers


def merge_only() -> None:
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
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_only()
        return

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    targets, carriers = build_targets(engine.brain)
    brain = engine.brain
    n_t = len(targets)
    idx = np.array([t["cell"] for t in targets], dtype=np.int64)

    print("P05.3 — проверка спайкового порога")
    print(f"  динамика: decay {float(brain.decay):.4f}, gain {float(brain.gain):.1f}, "
          f"porог {THRESHOLD}, tonic {float(brain.tonic):.2f}, "
          f"шум {float(brain.noise_amp):.2f} @ {float(brain.noise_hz):.1f} Гц")
    base = float(brain.tonic) / (1.0 - float(brain.decay))
    print(f"  базовое v без входа = tonic/(1-decay) = {base:.4f}")
    print(f"  расстояние до порога              = {THRESHOLD - base:.4f}")
    print(f"  шум за шаг                        = {float(brain.noise_amp):.4f}  "
          f"({float(brain.noise_amp) / (THRESHOLD - base) * 100:.0f}% всего запаса)")
    print(f"\n  нейронов под наблюдением: {n_t}")
    for s in ("keeps", "same_input_loses", "little_input"):
        sub = [t for t in targets if t["set"] == s]
        print(f"    {s:>18s}: {len(sub)} клеток, "
              f"вход от пути {np.mean([t['carrier_weight'] for t in sub]):6.2f}")
    print(f"  носителей из P05.1: {len(carriers)} групп "
          f"({sum(1 for t in targets if t['w_from_left_carriers'] > 0)} целевых клеток "
          f"получают хоть что-то от них)")
    for t in targets:
        if t["cell_type"] in ("DNbe001", "DNpe056") and t["side"] == "L":
            print(f"\n  {t['cell_type']} {t['side']}: вход. связей {t['in_degree']}, "
                  f"вес от пути {t['carrier_weight']:.2f} "
                  f"(от LEFT-предпочитающих {t['w_from_left_carriers']:.2f}, "
                  f"от RIGHT {t['w_from_right_carriers']:.2f})")
            print(f"      возбуждение от LEFT {t['w_exc_from_left']:+.2f}, "
                  f"торможение от LEFT {t['w_inh_from_left']:+.2f}")
            print(f"      возбуждение от RIGHT {t['w_exc_from_right']:+.2f}, "
                  f"торможение от RIGHT {t['w_inh_from_right']:+.2f}")

    encoder = VideoVisualEncoder(engine, flow_config=None)
    dt = float(engine.config.brain_dt)
    with VideoReader(VIDEO) as vr:
        total = vr.duration_s
    t_end = total if args.duration is None else min(args.duration, total)
    print(f"\n  видео {total:.0f} с, окно 0..{t_end:.0f} с, шаг {dt * 1000:.0f} мс")
    print(f"  оценка: {t_end * 1.5 / 60:.0f} мин\n")

    engine.reset(seed=SEED)
    encoder.reset()
    t0 = time.perf_counter()
    seg_start = 0.0
    while seg_start < t_end:
        seg_end = min(seg_start + CHUNK_S, t_end)
        tag = f"{int(seg_start):05d}_{int(seg_end):05d}"
        path = CHUNKS / f"chunk_{tag}.npz"
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
            eye_drive, inject, _ = encoder.encode_frame(frame)

            # the value that will be compared against the threshold
            v_pre = brain.v.ravel()[idx].copy()
            prev_fired = brain.fired
            act = np.zeros(brain.n, dtype=np.float64)
            if len(prev_fired):
                act[prev_fired] = 1.0

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
            fired_mask = np.isin(idx, result.fired)

            rows["t"].append(t)
            rows["v_pre"].append(v_pre)
            rows["fired"].append(fired_mask)
            rows["total_in"].append(tot)
            rows["drive_carrier"].append(drv)
            rows["drive_L"].append(dl)
            rows["drive_R"].append(dr)
            rows["n_active"].append(len(result.fired))

        payload = {"t": np.asarray(rows["t"], np.float32),
                   "v_pre": np.asarray(rows["v_pre"], np.float32),
                   "fired": np.asarray(rows["fired"], np.uint8),
                   "total_in": np.asarray(rows["total_in"], np.float32),
                   "drive_carrier": np.asarray(rows["drive_carrier"], np.float32),
                   "drive_L": np.asarray(rows["drive_L"], np.float32),
                   "drive_R": np.asarray(rows["drive_R"], np.float32),
                   "n_active": np.asarray(rows["n_active"], np.int32)}
        np.savez_compressed(path, **payload)
        el = time.perf_counter() - t0
        left = (t_end - seg_end) * (el / max(seg_end, 1e-9)) / 60
        print(f"  {seg_end:7.0f} / {t_end:.0f} с  шагов {len(rows['t']):6d}  "
              f"прошло {el / 60:5.1f} мин  осталось ~{left:5.1f} мин", flush=True)
        seg_start = seg_end

    merge_only()

    with (OUT / "targets.csv").open("w", newline="") as f:
        keys = ["cell", "cell_type", "side", "set", "in_degree", "carrier_weight",
                "w_from_left_carriers", "w_from_right_carriers",
                "w_exc_from_left", "w_exc_from_right",
                "w_inh_from_left", "w_inh_from_right"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for tg in targets:
            w.writerow({k: tg[k] for k in keys})
    (OUT / "dynamics.json").write_text(json.dumps({
        "decay": float(brain.decay), "gain": float(brain.gain),
        "threshold": THRESHOLD, "tonic": float(brain.tonic),
        "noise_amp": float(brain.noise_amp), "noise_hz": float(brain.noise_hz),
        "baseline_v": base, "distance_to_threshold": THRESHOLD - base,
        "noise_fraction_of_range": float(brain.noise_amp) / (THRESHOLD - base),
        "seed": SEED,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
