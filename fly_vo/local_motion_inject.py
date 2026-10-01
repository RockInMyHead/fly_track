"""P0.1J — local opponent motion injection.

Replaces the P0.1I pooled-signal injection. The old path computed one scalar
`yaw_frozen`, decided LEFT/RIGHT in Python, and handed that answer to T4/T5.
This module injects only the *local* signed motion field, leaving the decision
to the connectome.

What is injected
----------------
For a horizontal-motion cell at retinotopic azimuth u:

    a-family (T4a, T5a):  drive = gain * max(0, +m(u))
    b-family (T4b, T5b):  drive = gain * max(0, -m(u))

where m(u) is the signed local motion at that azimuth (positive = image content
moving in +panorama direction). The a/b split makes the pair opponent, which is
the only way direction survives into the circuit: if both families received the
same value the population would carry no direction and the connectome could not
recover it.

Not injected
------------
* `yaw_frozen`, or any pooled left/right scalar — never read in this module.
* T4c / T4d / T5c / T5d from a yaw field. They are the vertical detectors. A
  separate injector (``families=VERTICAL_FAMILIES``) may drive them, and only
  from a vertical motion field. The default ``build`` path never does.
* Any left/right asymmetry: a yaw rotates the image the same way in both eyes,
  so both eyes receive the same opponent drive. Whatever left/right structure
  appears in the descending neurons is produced by the connectome, not by us.

Retinotopy
----------
T4/T5 have no azimuth in `brain.npz`, but each cell is one optic column and the
value is recoverable from weighted photoreceptor input two hops upstream
(`scripts/p01j_t4_azimuth_map.py`, cached in `data/p01r/t4_azimuth.npz`).
Cells whose azimuth could not be resolved receive no drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Azimuth bins used to group cells. One scalar per (cell group, bin) is the
# granularity the inject interface allows: `brain.step(inject=[(idx, amount)])`
# adds one amount to every index in `idx`.
N_AZ_BINS = 32

# Horizontal-motion families. `prefers` is the sign of the local motion each
# family is tuned to. The a/b opposition is the biological fact; which family
# gets +1 is a stated convention, fixed here and not tuned on results.
HORIZONTAL_FAMILIES: dict[str, dict] = {
    "a": {"types": ("T4a", "T5a"), "prefers": +1.0},
    "b": {"types": ("T4b", "T5b"), "prefers": -1.0},
}

# Vertical detectors. The default injector does not drive them: a yaw field has
# no vertical component. A second injector, built with VERTICAL_FAMILIES and fed
# a vertical motion field, does. Positive field = image content moving down.
# c prefers that direction, d prefers upward. Same opponent split as a/b, not a
# weight fitted to a result.
VERTICAL_FAMILIES: dict[str, dict] = {
    "c": {"types": ("T4c", "T5c"), "prefers": +1.0},
    "d": {"types": ("T4d", "T5d"), "prefers": -1.0},
}
VERTICAL_TYPES = ("T4c", "T4d", "T5c", "T5d")

DEFAULT_AZIMUTH_NPZ = Path(__file__).resolve().parents[1] / "data/p01r/t4_azimuth.npz"


@dataclass
class LocalMotionDiagnostics:
    """Everything the injector reports. No pooled decision is present."""

    bins: int = N_AZ_BINS
    drive_by_group_bin: dict[str, list[float]] = field(default_factory=dict)
    family_totals: dict[str, float] = field(default_factory=dict)
    side_totals: dict[str, float] = field(default_factory=dict)
    opponent_index: float = 0.0
    left_right_asymmetry: float = 0.0
    n_active_cells: int = 0

    def as_dict(self) -> dict:
        d = {
            "local_opponent_index": self.opponent_index,
            "local_lr_asymmetry": self.left_right_asymmetry,
            "local_n_active_cells": self.n_active_cells,
        }
        for k, v in self.family_totals.items():
            d[f"local_{k}_total"] = v
        for k, v in self.side_totals.items():
            d[f"local_{k}_total"] = v
        return d


class LocalOpponentMotionInjector:
    """Builds per-azimuth opponent drive for T4a/b and T5a/b from a local motion field."""

    def __init__(
        self,
        brain,
        azimuth: np.ndarray | None = None,
        azimuth_npz: Path | str | None = None,
        n_az_bins: int = N_AZ_BINS,
        field_bins: int = 128,
        families: dict[str, dict] | None = None,
    ):
        self.brain = brain
        self.n_az_bins = n_az_bins
        self.field_bins = field_bins
        self.families = families if families is not None else HORIZONTAL_FAMILIES

        if azimuth is None:
            path = Path(azimuth_npz) if azimuth_npz else DEFAULT_AZIMUTH_NPZ
            if not path.exists():
                raise FileNotFoundError(
                    f"missing T4/T5 azimuth map: {path}\n"
                    "generate it with: PYTHONPATH=. python scripts/p01j_t4_azimuth_map.py"
                )
            azimuth = np.asarray(np.load(path, allow_pickle=False)["azimuth"], dtype=np.float32)
        azimuth = np.asarray(azimuth, dtype=np.float32)
        if len(azimuth) != brain.n:
            raise ValueError(f"azimuth map has {len(azimuth)} entries, brain has {brain.n}")

        # azimuth of each cell-group bin centre, in the same -1..+1 axis as the field
        self.bin_azimuth = np.linspace(-1.0, 1.0, n_az_bins, dtype=np.float64)

        # group -> (family, prefers, list of per-bin index arrays)
        self.groups: dict[str, tuple[str, float, list[np.ndarray]]] = {}
        self.n_unresolved = 0
        for fam, spec in self.families.items():
            for t in spec["types"]:
                for side in ("L", "R"):
                    idx = np.asarray(brain.cells([t], side=side), dtype=np.int64)
                    if len(idx) == 0:
                        continue
                    az = azimuth[idx]
                    ok = np.isfinite(az)
                    self.n_unresolved += int((~ok).sum())
                    idx, az = idx[ok], az[ok]
                    if len(idx) == 0:
                        continue
                    b = np.clip(
                        np.round((az + 1.0) * 0.5 * (n_az_bins - 1)).astype(np.int64),
                        0,
                        n_az_bins - 1,
                    )
                    bins = [idx[b == bi] for bi in range(n_az_bins)]
                    self.groups[f"{t}_{side}"] = (fam, float(spec["prefers"]), bins)

        self.n_cells = sum(
            int(len(c)) for _f, _p, bins in self.groups.values() for c in bins
        )

    # ------------------------------------------------------------------ field
    def _field_at_bins(self, field: np.ndarray) -> np.ndarray:
        """Sample the signed motion field at each azimuth bin centre."""
        field = np.asarray(field, dtype=np.float64)
        if field.size == 0:
            return np.zeros(self.n_az_bins, dtype=np.float64)
        az = -1.0 + 2.0 * np.arange(field.size) / max(self.field_bins - 1, 1)
        return np.interp(self.bin_azimuth, az, field)

    # ------------------------------------------------------------------ build
    def build(
        self,
        field: np.ndarray,
        gain: float,
        field_gain: float,
    ) -> tuple[list, LocalMotionDiagnostics]:
        """Return (inject entries, diagnostics) for one brain step."""
        raw = self._field_at_bins(field)
        m = np.tanh(field_gain * raw)  # bounded local signed motion

        inject: list = []
        diag = LocalMotionDiagnostics(bins=self.n_az_bins)
        fam_totals = {k: 0.0 for k in self.families}
        side_totals = {"L": 0.0, "R": 0.0}
        pos_key = next(k for k, spec in self.families.items() if spec["prefers"] > 0)
        neg_key = next(k for k, spec in self.families.items() if spec["prefers"] < 0)

        for key, (fam, prefers, bins) in self.groups.items():
            side = key[-1]
            drives = []
            for bi, cells in enumerate(bins):
                if len(cells) == 0:
                    drives.append(0.0)
                    continue
                d = gain * max(0.0, prefers * float(m[bi]))
                drives.append(d)
                if d > 1e-5:
                    inject.append((cells, d))
                    fam_totals[fam] += d * len(cells)
                    side_totals[side] += d * len(cells)
                    diag.n_active_cells += int(len(cells))
            diag.drive_by_group_bin[key] = drives

        total = fam_totals[pos_key] + fam_totals[neg_key]
        diag.family_totals = fam_totals
        diag.side_totals = side_totals
        # +1 = the family that prefers positive field, -1 = the opposite family.
        # For the default injector that is a versus b. For the vertical injector, c versus d.
        diag.opponent_index = (
            (fam_totals[pos_key] - fam_totals[neg_key]) / total if total > 0 else 0.0
        )
        lr = side_totals["L"] + side_totals["R"]
        diag.left_right_asymmetry = (side_totals["L"] - side_totals["R"]) / lr if lr > 0 else 0.0
        return inject, diag
