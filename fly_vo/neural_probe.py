"""Utilities for P0 neural observability probes."""

from __future__ import annotations

import numpy as np

from .malecns_engine import BrainStepResult
from .visual_encoder import T4_T5_GROUP_NAMES, VideoVisualEncoder

LOOM_GROUP_NAMES = ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R")
ALL_PROBE_GROUPS = LOOM_GROUP_NAMES + T4_T5_GROUP_NAMES


def group_spike_counts(
    result: BrainStepResult,
    encoder: VideoVisualEncoder,
) -> dict[str, int]:
    """Spike count per T4/T5/LC4/LPLC2 group for one brain step."""
    fired = set(result.fired.tolist())
    counts: dict[str, int] = {}
    for key in LOOM_GROUP_NAMES:
        idx = encoder.loom_groups.get(key, np.array([], dtype=int))
        counts[key] = sum(1 for i in idx if i in fired)
    for key in T4_T5_GROUP_NAMES:
        idx = encoder.t4_t5_groups.get(key, np.array([], dtype=int))
        counts[key] = sum(1 for i in idx if i in fired)
    return counts


def rolling_rates(
    history: list[dict[str, int]],
    window: int = 8,
) -> dict[str, float]:
    """Mean spikes/step over last `window` steps (100–200 ms @ 50 Hz)."""
    if not history:
        return {k: 0.0 for k in ALL_PROBE_GROUPS}
    tail = history[-window:]
    out: dict[str, float] = {}
    for key in ALL_PROBE_GROUPS:
        out[key] = float(np.mean([h.get(key, 0) for h in tail]))
    return out
