#!/usr/bin/env python3
"""
P05 — where inside MaleCNS does the turn signal disappear?

The chain was believed to run through four to six synaptic layers. The connectome says
otherwise, and that changes the question:

    T4a/b, T5a/b          the injected layer                        depth 0
    LPC1, LLPC1, Y11 ...  their direct recipients, visual projection depth 1
    DNa02, DNg100, MDN..  descending neurons                        depth 2

Breadth-first from the T4/T5 population reaches 912 of the 1314 descending neurons in
two hops. So there is no deep chain to walk: whatever is lost, is lost across one or
two synapses.

What is recorded
----------------
Every step, the spike count of each chosen (cell type, side) pair, so that a layer's
left/right differential can be formed later:

    L0  T4a/b and T5a/b, per side
    L1  the 27 types that earlier work confirmed as direction selective under a
        mirrored synthetic stimulus, per side, plus aggregate depth-1
    L2  every descending neuron type, per side

Nothing is trained, no weight is touched, the dynamics and the visual frontend are the
same as in the P0.4 run. Only additional populations are being watched.

Reading is sequential through the fixed reader, and each chunk skips the step its
predecessor ended on, so no timestamp repeats.

Usage:
    PYTHONPATH=. python scripts/p05_layer_trace.py --groups-only
    PYTHONPATH=. python scripts/p05_layer_trace.py --duration 60
    PYTHONPATH=. python scripts/p05_layer_trace.py
    PYTHONPATH=. python scripts/p05_layer_trace.py --merge-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
OUT = ROOT / "output/p05_layer_trace"
CHUNKS = OUT / "chunks"
GROUPS_JSON = OUT / "groups.json"
CHUNK_S = 60.0

SOURCE_TYPES = ("T4a", "T4b", "T5a", "T5b")

# The 27 types confirmed in P0.1K as reversing their left/forward/right ordering under a
# mirrored stimulus, with 5 of 5 seeds agreeing. Taken from that report, not re-derived.
CONFIRMED = [
    "C3", "TmY20", "LPC1", "LLPC1", "Y1", "TmY9q__perp", "TmY9q", "Y12", "Tm27",
    "Y11", "TmY5a", "LPi05", "LPi10", "Tlp5", "TmY16", "Y4", "LPi01", "T2",
    "TmY15", "cLP01", "LPC2", "LPT23", "Li05", "LPT31", "LPTe02", "VST2", "LLPt",
]

REFERENCE = ["yaw_frozen", "inject_lr"]


def build_groups(brain) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Map every neuron to a group index, and describe each group.

    Returns (group_of, meta, depths) where `group_of` is -1 for neurons not watched.
    """
    ct = np.asarray(brain.cell_type)
    sc = np.asarray(brain.superclass)
    sd = np.asarray(brain.side)
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)

    # ---- depth layers by BFS from T4/T5 -----------------------------------
    depths = np.full(brain.n, -1, np.int32)
    src = np.flatnonzero(np.isin(ct, SOURCE_TYPES))
    depths[src] = 0
    frontier = src
    for d in range(1, 6):
        if len(frontier) == 0:
            break
        starts = ip[frontier]
        cnt = ip[frontier + 1] - starts
        if cnt.sum() == 0:
            break
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tg = ix[np.repeat(starts, cnt) + off]
        new = np.unique(tg[depths[tg] < 0])
        depths[new] = d
        frontier = new

    want: list[tuple[int, str, str]] = []   # (layer, cell_type, side)
    for t in SOURCE_TYPES:
        for s in ("L", "R"):
            want.append((0, t, s))

    # every type that the confirmed list names, both sides, so the left and right
    # populations can be compared rather than just the aggregate
    for t in CONFIRMED:
        for s in ("L", "R"):
            want.append((1, t, s))

    # every descending neuron type, both sides
    dn_types = sorted(set(str(t) for t in np.unique(ct[np.isin(sc, ("descending_neuron",
                                                                   "descending_neuron_tbc"))])))
    for t in dn_types:
        for s in ("L", "R"):
            want.append((2, t, s))

    group_of = np.full(brain.n, -1, np.int32)
    meta: list[dict] = []
    for layer, t, s in want:
        idx = np.flatnonzero((ct == t) & (sd == s))
        if len(idx) == 0:
            continue
        if group_of[idx[0]] >= 0:      # already assigned by an earlier entry
            continue
        # keep the type together: assign all of its neurons on this side
        idx = np.flatnonzero((ct == t) & (sd == s) & (group_of < 0))
        if len(idx) == 0:
            continue
        g = len(meta)
        group_of[idx] = g
        meta.append({"layer": layer, "cell_type": t, "side": s,
                     "n_cells": int(len(idx)),
                     "mean_depth": float(depths[idx].mean()),
                     "superclass": str(sc[idx[0]])})

    # aggregate groups: all of depth 1, all of depth 2, all DNs
    for layer, mask, name in [
        (1, (depths == 1), "ALL_depth1"),
        (2, (depths >= 2) & np.isin(sc, ("descending_neuron", "descending_neuron_tbc")), "ALL_DN"),
    ]:
        for s in ("L", "R"):
            # only neurons no individual type has already claimed, otherwise the
            # aggregate would erase the confirmed types assigned just above
            idx = np.flatnonzero(mask & (sd == s) & (group_of < 0))
            if len(idx) == 0:
                continue
            g = len(meta)
            group_of[idx] = g
            meta.append({"layer": layer, "cell_type": name, "side": s,
                         "n_cells": int(len(idx)),
                         "mean_depth": float(depths[idx].mean()),
                         "superclass": "aggregate"})
    return group_of, meta, depths


def merge_only() -> None:
    files = sorted(CHUNKS.glob("chunk_*.npz"))
    if not files:
        print("нет чанков")
        return
    first = np.load(files[0])
    keys = list(first.keys())
    acc: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for f in files:
        dd = np.load(f)
        for k in keys:
            acc[k].append(dd[k])
    merged = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(OUT / "layer_signals.npz", **merged)
    G = int(merged["G"].shape[1])
    print(f"объединено {len(files)} чанков -> {len(merged['t'])} шагов, "
          f"{merged['t'][-1] - merged['t'][0]:.0f} с, {G} групп")


def main() -> None:
    global OUT, CHUNKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--groups-only", action="store_true")
    ap.add_argument("--seed", type=int, default=64)
    args = ap.parse_args()

    # a second seed must not overwrite the first run's recording
    if args.seed != 64:
        OUT = OUT.parent / f"{OUT.name}_seed{args.seed}"
        CHUNKS = OUT / "chunks"

    OUT.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_only()
        return

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    group_of, meta, depths = build_groups(engine.brain)
    G = len(meta)

    by_layer = Counter(m["layer"] for m in meta)
    print("P05 — послойная трассировка сигнала поворота")
    print(f"  нейронов под наблюдением: {int(np.sum(group_of >= 0)):,} в {G} группах")
    print(f"  групп по слоям: L0={by_layer[0]}, L1={by_layer[1]}, L2={by_layer[2]}")
    from collections import Counter as C2
    dl = C2(int(depths[np.flatnonzero(group_of >= 0)[i]]) for i in range(0))
    print(f"  нисходящих типов: {sum(1 for m in meta if m['superclass'] == 'descending_neuron')}")
    print()
    for layer in (0, 1, 2):
        ms = [m for m in meta if m["layer"] == layer]
        tot = sum(m["n_cells"] for m in ms)
        print(f"  слой {layer}: {len(ms):3d} групп, {tot:6,} нейронов")
    if args.groups_only:
        GROUPS_JSON.write_text(json.dumps(
            {"groups": meta, "n_neurons_watched": int(np.sum(group_of >= 0))},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {GROUPS_JSON}")
        return

    GROUPS_JSON.write_text(json.dumps(
        {"groups": meta, "n_neurons_watched": int(np.sum(group_of >= 0))},
        indent=2, ensure_ascii=False), encoding="utf-8")

    encoder = VideoVisualEncoder(engine, flow_config=None)
    dt = float(engine.config.brain_dt)
    with VideoReader(VIDEO) as vr:
        total = vr.duration_s
    t_end = total if args.duration is None else min(args.start + args.duration, total)

    print(f"\n  видео {total:.0f} с, окно {args.start:.0f}..{t_end:.0f} с, "
          f"шаг {dt * 1000:.0f} мс")
    print(f"  групп к записи: {G}  (матрица {G} x {int((t_end - args.start) / dt):,})\n")

    engine.reset(seed=args.seed)
    encoder.reset()
    t0 = time.perf_counter()
    seg_start = args.start
    while seg_start < t_end:
        seg_end = min(seg_start + CHUNK_S, t_end)
        tag = f"{int(seg_start):05d}_{int(seg_end):05d}"
        path = CHUNKS / f"chunk_{tag}.npz"
        if path.exists():
            seg_start = seg_end
            continue

        counts: list[np.ndarray] = []
        # numpy output for the watched groups, plus the two scalars
        keep = np.zeros(G, np.float32)
        rows_extra = {k: [] for k in REFERENCE}
        ts: list[float] = []
        first = True
        for t, frame in iter_video_at_brain_hz(str(VIDEO), seg_start, seg_end, dt):
            if first:
                first = False
                if seg_start > args.start:
                    continue
            eye_drive, inject, metrics = encoder.encode_frame(frame)
            result = engine.step(t, eye_drive=eye_drive, inject=inject)
            g = group_of[result.fired]
            g = g[g >= 0]
            vec = np.bincount(g, minlength=G).astype(np.float32) if g.size \
                else np.zeros(G, np.float32)
            counts.append(vec)
            ts.append(t)
            rows_extra["yaw_frozen"].append(float(metrics.get("yaw_frozen", 0.0)))
            rows_extra["inject_lr"].append(float(metrics.get("local_lr_asymmetry", 0.0)))

        arr = np.stack(counts).astype(np.float32)
        payload = {"t": np.asarray(ts, np.float32), "G": arr}
        for k in REFERENCE:
            payload[k] = np.asarray(rows_extra[k], np.float32)
        np.savez_compressed(path, **payload)
        el = time.perf_counter() - t0
        left = (t_end - seg_end) * (el / max(seg_end - args.start, 1e-9)) / 60
        print(f"  {seg_end:7.0f} / {t_end:.0f} с   групп {G}   "
              f"прошло {el / 60:5.1f} мин   осталось ~{left:5.1f} мин", flush=True)
        seg_start = seg_end

    merge_only()


if __name__ == "__main__":
    main()
