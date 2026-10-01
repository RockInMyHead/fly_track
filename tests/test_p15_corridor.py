"""Corridor edges are θ±15°. No brain, and δ is not a search.

    PYTHONPATH=. .venv/bin/python tests/test_p15_corridor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_map_vision as M


def test_delta_is_fifteen() -> None:
    assert M.CORRIDOR_DELTA_DEG == 15.0


def test_edges() -> None:
    assert M.corridor_edges(0) == [-15.0, 15.0]
    assert M.corridor_edges(-90) == [-105.0, -75.0]
    assert M.corridor_edges(90) == [75.0, 105.0]


def test_straight_walls_sit_on_opposite_sides() -> None:
    frames = M.corridor_clip([0.0], width=96, height=72)
    frame = frames[0][:, :, 0]
    mid = frame.shape[1] // 2
    assert int(frame[:, :mid].max()) > 24
    assert int(frame[:, mid:].max()) > 24
    assert float(np.abs(frames[0].astype(float) - frames[-1].astype(float)).mean()) > 0.0


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
