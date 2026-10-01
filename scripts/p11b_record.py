#!/usr/bin/env python3
"""P11.B step 2 — record the candidates while the fly walks the same five recordings.

Same videos, same seed, same clock, same one-minute chunking as the P09 recording of the
descending cells, so that this can be compared against the levels already measured.

Recorded per cell, not per cell type. That choice is the whole lesson of P11.A: a pooled number
over a population was what destroyed the direction information in yaw, and pooling the cells of a
type before looking at them would invite exactly the same mistake here.

Counts are accumulated into 200 ms bins rather than kept per 20 ms step. A cell fires at most once
per step, so a bin holds at most ten spikes and fits in one byte, and a hundred-millisecond
question about temporal form is answered comfortably by five points per second.

Usage:
    PYTHONPATH=. python scripts/p11b_record.py [--video VID00001] [--top 80]
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

from fly_vo.brain_clock import iter_video_at_brain_hz  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

SEED = 64          # the seed P09 used for the descending cells
CHUNK_S = 60.0
BIN_S = 0.2


def video_of(name: str) -> Path:
    return ROOT / "data/p01r" / f"{name}.AVI"


def span_of(name: str, dt: float) -> float:
    """Length to walk, taken from the P09 trace so both recordings cover the same interval."""
    p = ROOT / f"output/p09/trace_{name}.npz"
    if not p.exists():
        from fly_vo.video_reader import VideoReader
        with VideoReader(video_of(name)) as vr:
            return vr.duration_s
    return float(np.load(p)["t"][-1]) + dt


def chosen_cells(top: int) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """The candidate cells: every cell of the selected types, with its type and side."""
    cand = json.loads((ROOT / "output/p11b/candidates.json").read_text(encoding="utf-8"))
    types = [r["cell_type"] for r in cand["chosen"][:top]]
    engine = MaleCNSEngine(FlyVOConfig())
    brain = engine.brain
    idx, ct, sd = [], [], []
    for t in types:
        cells = brain.cells([t])
        if len(cells) == 0:
            continue
        idx.append(cells)
        ct.extend([t] * len(cells))
        sd.extend(np.asarray(brain.side)[cells].tolist())
    idx = np.concatenate(idx).astype(np.int64)
    order = np.argsort(idx)
    idx = idx[order]
    ct = [ct[i] for i in order.tolist()]
    sd = [sd[i] for i in order.tolist()]
    return idx, ct, sd, np.asarray(types)


def record(name: str, out: Path, idx: np.ndarray) -> None:
    chunks = out / f"chunks_{name}"
    chunks.mkdir(parents=True, exist_ok=True)
    dest = out / f"cells_{name}.npz"
    if dest.exists():
        print(f"  {name}: уже записано, пропуск")
        return

    engine = MaleCNSEngine(FlyVOConfig())
    encoder = VideoVisualEncoder(engine, flow_config=None)
    dt = float(engine.config.brain_dt)
    steps_per_bin = max(int(round(BIN_S / dt)), 1)
    total = span_of(name, dt)

    # lookup: brain cell -> position in the recorded set, -1 elsewhere
    sel = np.full(engine.brain.n + 1, -1, dtype=np.int32)
    sel[idx] = np.arange(len(idx), dtype=np.int32)
    n_sel = len(idx)

    print(f"  {name}: {total:.0f} с, клеток под наблюдением {n_sel}, "
          f"бин {BIN_S*1000:.0f} мс ({steps_per_bin} шагов)", flush=True)
    engine.reset(seed=SEED)
    t0 = time.perf_counter()
    seg = 0.0
    while seg < total:
        end = min(seg + CHUNK_S, total)
        path = chunks / f"chunk_{int(seg):05d}_{int(end):05d}.npz"
        if path.exists():
            seg = end
            continue
        counts, bin_t = [], []
        acc = np.zeros(n_sel, dtype=np.int32)
        n_in_bin = 0
        bin_start = seg
        for t, frame in iter_video_at_brain_hz(str(video_of(name)), seg, end, dt):
            eye, inject, _ = encoder.encode_frame(frame)
            res = engine.step(t, eye_drive=eye, inject=inject)
            if len(res.fired):
                m = sel[res.fired]
                m = m[m >= 0]
                if len(m):
                    acc += np.bincount(m, minlength=n_sel).astype(np.int32)
            n_in_bin += 1
            if n_in_bin >= steps_per_bin:
                counts.append(acc.astype(np.uint8))
                bin_t.append(bin_start + n_in_bin * dt * 0.5)
                acc = np.zeros(n_sel, dtype=np.int32)
                bin_start += steps_per_bin * dt
                n_in_bin = 0
        if n_in_bin:
            counts.append(np.minimum(acc, 255).astype(np.uint8))
            bin_t.append(bin_start + n_in_bin * dt * 0.5)
        np.savez_compressed(path, t=np.asarray(bin_t, np.float32),
                            counts=np.asarray(counts, np.uint8))
        el = time.perf_counter() - t0
        left = (total - end) * (el / max(end, 1e-9)) / 60
        print(f"    {end:7.0f}/{total:.0f} с  прошло {el/60:5.1f} мин  "
              f"осталось ~{left:4.1f} мин", flush=True)
        seg = end

    files = sorted(chunks.glob("chunk_*.npz"))
    t = np.concatenate([np.load(f)["t"] for f in files])
    counts = np.concatenate([np.load(f)["counts"] for f in files])
    np.savez_compressed(dest, t=t, counts=counts, cells=idx)
    print(f"  записано {dest}: {counts.shape[0]} бинов x {counts.shape[1]} клеток")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "output/p11b"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--top", type=int, default=80)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    idx, ct, sd, types = chosen_cells(args.top)
    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    names = sorted({l["clip"] for l in frozen["labels"]})
    todo = [args.video] if args.video else names

    print("=" * 100)
    print("P11.B шаг 2 — запись кандидатов на тех же пяти роликах")
    print("=" * 100)
    print(f"  типов: {len(types)}, клеток: {len(idx)}, seed {SEED}, бины {BIN_S*1000:.0f} мс")
    print(f"  из них получают вход от encoder напрямую: T4a/b, T5a/b, LC4, LPLC2")
    print()
    for n in todo:
        if not video_of(n).exists():
            print(f"  {n}: НЕТ ФАЙЛА")
            continue
        record(n, out, idx)

    meta = {"cells": idx.tolist(), "cell_type": ct, "side": sd,
            "types": [str(t) for t in types], "n_sources": int(len(idx)),
            "seed": SEED, "bin_s": BIN_S, "n_types": int(len(types))}
    (out / "record_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"  метаданные: {out/'record_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
