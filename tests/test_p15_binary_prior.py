"""Map prior is zero on triples and on a weak margin. No brain.

    PYTHONPATH=. .venv/bin/python tests/test_p15_binary_prior.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_5_binary_prior as B


def test_triple_is_ignored() -> None:
    assert B.map_prior(3, ["LEFT", "STRAIGHT", "RIGHT"], 0.4, ["LEFT", "STRAIGHT"]) == 0.0


def test_weak_margin_is_zero() -> None:
    assert B.map_prior(2, ["LEFT", "STRAIGHT"], 0.049, ["LEFT", "STRAIGHT"]) == 0.0
    assert B.map_prior(2, ["LEFT", "STRAIGHT"], 0.002, ["LEFT", "RIGHT"]) == 0.0


def test_binary_agree_and_contradict() -> None:
    assert B.map_prior(2, ["LEFT", "STRAIGHT"], 0.08, ["STRAIGHT", "LEFT"]) == B.BETA
    assert B.map_prior(2, ["RIGHT", "STRAIGHT"], 0.08, ["LEFT", "STRAIGHT"]) == -B.BETA


def test_single_and_same_class_are_not_a_pair() -> None:
    assert B.map_prior(2, ["LEFT"], 0.2, ["LEFT", "STRAIGHT"]) == 0.0
    assert B.map_prior(2, ["LEFT", "RIGHT"], 0.2, ["LEFT", "LEFT"]) == 0.0


def test_bands() -> None:
    assert B.passage_class(-90) == "LEFT"
    assert B.passage_class(-60) == "LEFT"
    assert B.passage_class(-45) is None
    assert B.passage_class(0) == "STRAIGHT"
    assert B.passage_class(15) == "STRAIGHT"
    assert B.passage_class(30) is None
    assert B.passage_class(60) == "RIGHT"


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
