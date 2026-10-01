"""MaleCNS v1.0 whole-CNS simulation via flybrain."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import FlyVOConfig


@dataclass
class BrainStepResult:
    """One MaleCNS simulation step."""

    timestamp: float
    fired: np.ndarray
    pathway_spikes: dict[str, int] = field(default_factory=dict)
    descending: dict[str, float] = field(default_factory=dict)
    n_active: int = 0


class MaleCNSEngine:
    """Wrap flybrain.FlyBrain — 166,700 neurons, MaleCNS v1.0."""

    def __init__(self, config: FlyVOConfig | None = None):
        self.config = config or FlyVOConfig()
        self._brain = None
        self._pathway_idx: dict[str, np.ndarray] = {}
        self._descending_groups: dict[str, np.ndarray] = {}

    @property
    def brain(self):
        self._ensure_loaded()
        return self._brain

    @property
    def device(self) -> str:
        return self.brain.device

    @property
    def n_neurons(self) -> int:
        return self.brain.n

    @property
    def n_visual(self) -> int:
        return len(self.brain.visual)

    def _data_dir(self) -> Path | None:
        if self.config.data_dir:
            return Path(self.config.data_dir).expanduser()
        root = Path(__file__).resolve().parents[1] / "data" / "malecns"
        os.environ.setdefault("FLY_DATA", str(root))
        return root

    def _ensure_loaded(self) -> None:
        if self._brain is not None:
            return

        from flybrain import FlyBrain

        data = self._data_dir()
        self._brain = FlyBrain(
            data=data,
            device=self.config.device,
            dt=self.config.brain_dt,
            sensory_input=False,
        )
        self._init_pathways()
        self._init_descending_groups()

    def _init_pathways(self) -> None:
        brain = self._brain
        for name, types in self.config.pathway_types.items():
            try:
                idx = brain.cells(list(types))
                if len(idx):
                    self._pathway_idx[name] = idx
            except Exception:
                pass

        # fly.ai pre-built readout groups (DNa02 L/R, DNg100 L/R, …)
        for key in (
            "steer_L",
            "steer_R",
            "forward_L",
            "forward_R",
            "backward_L",
            "backward_R",
            "escape_L",
            "escape_R",
        ):
            if key in brain.groups:
                self._descending_groups[key] = brain.groups[key]

    def _init_descending_groups(self) -> None:
        """Fallback if pre-built groups missing."""
        brain = self._brain
        mapping = {
            "steer_L": (["DNa02"], "L"),
            "steer_R": (["DNa02"], "R"),
            "forward_L": (["DNg100"], "L"),
            "forward_R": (["DNg100"], "R"),
            "backward_L": (["MDN"], "L"),
            "backward_R": (["MDN"], "R"),
            "escape_L": (["DNp01"], "L"),
            "escape_R": (["DNp01"], "R"),
        }
        for key, (types, side) in mapping.items():
            if key not in self._descending_groups:
                idx = brain.cells(types, side=side)
                if len(idx):
                    self._descending_groups[key] = idx

    def reset(self, seed: int = 64) -> None:
        self._ensure_loaded()
        self._brain.reset(seed)

    def step(
        self,
        timestamp: float,
        eye_drive: np.ndarray | None = None,
        inject: list | None = None,
    ) -> BrainStepResult:
        self._ensure_loaded()
        fired = self._brain.step(eye_drive=eye_drive, inject=inject or ())
        fired_arr = np.asarray(fired, dtype=np.int64)
        fired_set = set(fired_arr.tolist())

        pathway_spikes = {
            name: sum(1 for i in idx if i in fired_set)
            for name, idx in self._pathway_idx.items()
        }
        descending = {
            key: float(sum(1 for i in idx if i in fired_set))
            for key, idx in self._descending_groups.items()
        }

        return BrainStepResult(
            timestamp=timestamp,
            fired=fired_arr,
            pathway_spikes=pathway_spikes,
            descending=descending,
            n_active=len(fired_arr),
        )

    def pathway_indices(self) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        return dict(self._pathway_idx)
