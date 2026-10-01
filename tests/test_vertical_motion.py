"""Sign of the vertical Reichardt. No brain, no fitted threshold.

    PYTHONPATH=. .venv/bin/python tests/test_vertical_motion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.vertical_motion import column_reichardt


def _bar(row: int, h: int = 48, w: int = 12, thick: int = 4) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float32)
    img[row:row + thick, :] = 1.0
    return img


def test_upward_is_negative() -> None:
    field = column_reichardt(_bar(30), _bar(24), 6)
    assert float(field.mean()) < 0.0, field.mean()


def test_downward_is_positive() -> None:
    field = column_reichardt(_bar(24), _bar(30), 6)
    assert float(field.mean()) > 0.0, field.mean()


def test_vertical_sign_flips() -> None:
    up = column_reichardt(_bar(30), _bar(24), 6)
    down = column_reichardt(_bar(24), _bar(30), 6)
    assert abs(float(up.mean() + down.mean())) < 1e-5


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
