"""P0.1I probe helpers — shared by the smoke test and the real-event gate.

Kept free of any tuning knobs: everything here is measurement only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .brain_clock import iter_video_at_brain_hz
from .config import FlyVOConfig
from .malecns_engine import MaleCNSEngine
from .visual_encoder import VideoVisualEncoder

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
EMA_TAU = 0.5


def dn_layout(brain) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dn_indices, slot_map, cell_type_per_dn, side_per_dn)."""
    sc = np.asarray(brain.superclass)
    idx = np.where(sc == "descending_neuron")[0]
    ct = np.asarray(brain.cell_type)[idx]
    sd = np.asarray(brain.side)[idx]
    slot = np.full(int(sc.shape[0]), -1, dtype=np.int64)
    slot[idx] = np.arange(len(idx), dtype=np.int64)
    return idx, slot, ct, sd


def fam_indices(ct_slot: np.ndarray, sd_slot: np.ndarray, ctype: str) -> tuple[list[int], list[int]]:
    il = [i for i in range(len(ct_slot)) if str(ct_slot[i]) == ctype and str(sd_slot[i]) == "L"]
    ir = [i for i in range(len(ct_slot)) if str(ct_slot[i]) == ctype and str(sd_slot[i]) == "R"]
    return il, ir


def load_canonical_polarity(families_csv: Path) -> dict[str, int]:
    """+1 if synthetic LEFT raised L-R, else -1. Frozen source, never real events."""
    out: dict[str, int] = {}
    if not families_csv.exists():
        return out
    for r in csv.DictReader(families_csv.open()):
        out[r["cell_type"]] = 1 if float(r["defl_LEFT"]) > 0 else -1
    return out


def oriented_sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def capture_event(
    video: Path,
    start: float,
    cfg: FlyVOConfig,
    pre_roll_s: float = PRE_ROLL_S,
    dur_s: float = EVENT_DUR_S,
) -> dict:
    """Run the frozen frontend over one event window (seed-independent)."""
    t0 = max(0.0, start - pre_roll_s)
    t1 = start + dur_s
    encoder = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain)
    frames: list[dict] = []
    for t, frame in iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt):
        eye, inject, metrics = encoder.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject, "metrics": metrics})

    n_pre = min(int(round(pre_roll_s / cfg.brain_dt)), max(0, len(frames) - 1))
    tail = frames[n_pre:] or frames

    def med(key: str) -> float:
        return float(np.median([float(f["metrics"].get(key, 0.0)) for f in tail]))

    return {
        "frames": frames,
        "n_pre": n_pre,
        "yaw_frozen": med("yaw_frozen"),
        "t4_lr": med("inject_t4_lr"),
        "t5_lr": med("inject_t5_lr"),
        "combined_lr": med("inject_combined_lr"),
        "scene_common": med("inject_scene_common"),
    }


def replay_dn(
    engine: MaleCNSEngine,
    frames: list[dict],
    slot: np.ndarray,
    n_dn: int,
    seed: int,
) -> np.ndarray:
    """Replay cached frontend into MaleCNS; return per-step DN counts [n_dn, n_steps]."""
    engine.reset(seed=seed)
    counts = np.zeros((n_dn, len(frames)), dtype=np.int32)
    for i, fr in enumerate(frames):
        res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
        fired = res.fired
        if len(fired):
            hits = slot[fired]
            hits = hits[hits >= 0]
            if len(hits):
                np.add.at(counts[:, i], hits, 1)
    return counts


def dn_lr_rate(counts: np.ndarray, il: list[int], ir: list[int], lo: int, hi: int, dt: float) -> float:
    """L-R firing rate (Hz) of one DN family over steps [lo, hi)."""
    span = max(hi - lo, 1) * dt
    l = float(counts[il, lo:hi].sum()) / span if il else 0.0
    r = float(counts[ir, lo:hi].sum()) / span if ir else 0.0
    return l - r
