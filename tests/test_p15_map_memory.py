"""Pulse length and the mirror of a pair. No brain.

    PYTHONPATH=. .venv/bin/python tests/test_p15_map_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_6_map_memory as S


def test_times_are_the_ones_written_down() -> None:
    assert S.DELAYS_MS == (0, 200, 500, 1000)
    assert S.MAP_MS == 200
    assert S.MAP_STEPS == 10
    assert S.CAM_STEPS == 50


def test_mirror_and_wrong_pairs() -> None:
    assert S.mirror_pair(("LEFT", "STRAIGHT")) == ("STRAIGHT", "RIGHT")
    assert S.mirror_pair(("LEFT", "RIGHT")) == ("LEFT", "RIGHT")
    assert S.wrong_pair(("LEFT", "STRAIGHT")) == ("STRAIGHT", "RIGHT")
    assert S.wrong_pair(("LEFT", "RIGHT")) == ("LEFT", "STRAIGHT")


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
