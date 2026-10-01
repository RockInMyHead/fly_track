"""P0.1J invariants for local opponent motion injection.

Runs standalone (no pytest needed):

    PYTHONPATH=. python tests/test_local_opponent_invariants.py

Uses a stub brain + synthetic azimuths so the algebra is checked without loading
the connectome.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.local_motion_inject import N_AZ_BINS, LocalOpponentMotionInjector

FIELD_BINS = 128
N_PER_GROUP = 20


class _StubBrain:
    """Minimal stand-in: `n` plus a `cells(types, side)` lookup."""

    def __init__(self) -> None:
        self._types = [
            (t, side) for t in ("T4a", "T5a", "T4b", "T5b", "T4c", "T4d") for side in ("L", "R")
        ]
        self._index: dict[tuple[str, str], list[int]] = {}
        self.azimuth: list[float] = []
        for t, side in self._types:
            start = len(self.azimuth)
            if t in ("T4c", "T4d"):
                vals = [float("nan")] * N_PER_GROUP  # vertical, no azimuth
            else:
                lo, hi = (-1.0, -0.05) if side == "L" else (0.05, 1.0)
                vals = list(np.linspace(lo, hi, N_PER_GROUP))
            self.azimuth.extend(vals)
            self._index[(t, side)] = list(range(start, len(self.azimuth)))
        self.n = len(self.azimuth)

    def cells(self, types, side=None):
        out: list[int] = []
        for t in types:
            out.extend(self._index.get((t, side), []))
        return np.asarray(out, dtype=np.int64)


def _injector() -> LocalOpponentMotionInjector:
    brain = _StubBrain()
    return LocalOpponentMotionInjector(
        brain, azimuth=np.asarray(brain.azimuth, dtype=np.float32), field_bins=FIELD_BINS
    )


def _uniform_field(value: float) -> np.ndarray:
    return np.full(FIELD_BINS, value, dtype=np.float64)


def _left_only_field(value: float) -> np.ndarray:
    f = np.zeros(FIELD_BINS, dtype=np.float64)
    f[: FIELD_BINS // 2] = value
    return f


def _build(inj, field, gain=1.0, fg=8.0):
    return inj.build(np.asarray(field, dtype=np.float64), gain=gain, field_gain=fg)


# ---------------------------------------------------------------- invariants
def test_zero_field_gives_no_drive() -> None:
    """A still / balanced field must not move the opponent code at all."""
    _inj, d = _build(_injector(), _uniform_field(0.0))
    assert d.opponent_index == 0.0, d.opponent_index
    assert d.n_active_cells == 0, d.n_active_cells
    assert d.family_totals["a"] == 0.0 and d.family_totals["b"] == 0.0


def test_uniform_field_has_no_lateral_bias() -> None:
    """A spatially flat motion field must drive both eyes equally."""
    _inj, d = _build(_injector(), _uniform_field(0.5))
    assert abs(d.left_right_asymmetry) < 1e-9, d.left_right_asymmetry
    assert abs(d.side_totals["L"] - d.side_totals["R"]) < 1e-9


def test_opponent_index_tracks_field_sign() -> None:
    """Field sign selects the family; the code is opponent, not one-sided."""
    for value, expect in ((0.5, +1), (-0.5, -1)):
        _inj, d = _build(_injector(), _uniform_field(value))
        got = math.copysign(1, d.opponent_index)
        assert got == expect, f"field {value}: opponent_index {d.opponent_index}"
        # only one family should carry drive
        if expect > 0:
            assert d.family_totals["b"] == 0.0
            assert d.family_totals["a"] > 0.0
        else:
            assert d.family_totals["a"] == 0.0
            assert d.family_totals["b"] > 0.0


def test_opponent_index_is_antisymmetric() -> None:
    """Flipping the field must flip the opponent index exactly."""
    _i1, pos = _build(_injector(), _uniform_field(0.4))
    _i2, neg = _build(_injector(), _uniform_field(-0.4))
    assert abs(pos.opponent_index + neg.opponent_index) < 1e-9, (
        f"{pos.opponent_index} vs {neg.opponent_index}"
    )


def test_spatial_resolution() -> None:
    """A half-field stimulus must light up the corresponding eye only."""
    _inj, d = _build(_injector(), _left_only_field(0.5))
    assert d.side_totals["L"] > 0.0
    assert d.side_totals["R"] == 0.0, d.side_totals
    assert d.left_right_asymmetry > 0.99


def test_vertical_detectors_never_driven() -> None:
    """T4c/T4d (pitch) must not appear in any inject entry."""
    inj = _injector()
    brain = inj.brain
    vertical = set(brain.cells(["T4c", "T4d"]).tolist())
    entries, _d = _build(inj, _uniform_field(0.5))
    for cells, _amt in entries:
        assert not (set(np.asarray(cells).tolist()) & vertical), "vertical detector received yaw drive"


def test_no_pooled_signal_in_module() -> None:
    """The injector must not read any pooled yaw quantity — code, not prose."""
    import ast

    tree = ast.parse((ROOT / "fly_vo/local_motion_inject.py").read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.arg):
            used.add(node.arg)
    for banned in ("yaw_frozen", "yaw_signal", "left_yaw", "right_yaw"):
        assert banned not in used, f"pooled quantity {banned!r} used in injection code"


def test_bins_cover_the_field() -> None:
    """Every azimuth bin must be addressable, and bin centres span -1..+1."""
    inj = _injector()
    assert len(inj.bin_azimuth) == N_AZ_BINS
    assert abs(inj.bin_azimuth[0] + 1.0) < 1e-9
    assert abs(inj.bin_azimuth[-1] - 1.0) < 1e-9


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} invariants passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
