"""P0.1I algebraic invariants for the signed-yaw injection mapping.

Runs standalone (no pytest required):

    PYTHONPATH=. python tests/test_inject_invariants.py

Also collectable by pytest if it is ever installed.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.legacy_p01i_inject import (
    T4_TYPES,
    T5_TYPES,
    compute_inject_strengths,
)
from fly_vo.optic_flow import FlowDiagnostics

GAIN = 0.9
FG = 8.0


def _flow(
    yaw: float,
    expansion: float = 0.0,
    flow_l: float = 0.0,
    flow_r: float = 0.0,
    frame_diff: float = 0.0,
) -> FlowDiagnostics:
    d = FlowDiagnostics()
    d.yaw_signal = yaw
    d.yaw_frozen = yaw
    d.expansion_signal = expansion
    d.flow_L = flow_l
    d.flow_R = flow_r
    d.frame_diff_mean = frame_diff
    return d


def _strengths(yaw: float, **scene) -> dict[str, float]:
    scene = dict(scene)
    left_loom = float(scene.pop("left_loom", 0.0))
    right_loom = float(scene.pop("right_loom", 0.0))
    s, _diag = compute_inject_strengths(
        _flow(yaw, **scene),
        left_loom=left_loom,
        right_loom=right_loom,
        gain=GAIN,
        flow_inject_gain=FG,
    )
    return s


def _lr(strengths: dict[str, float]) -> tuple[float, float, float]:
    t4_l = sum(strengths[f"{t}_L"] for t in T4_TYPES)
    t4_r = sum(strengths[f"{t}_R"] for t in T4_TYPES)
    t5_l = sum(strengths[f"{t}_L"] for t in T5_TYPES)
    t5_r = sum(strengths[f"{t}_R"] for t in T5_TYPES)
    return t4_l - t4_r, t5_l - t5_r, (t4_l + t5_l) - (t4_r + t5_r)


def _sign(x: float) -> int:
    return 1 if x > 1e-12 else (-1 if x < -1e-12 else 0)


# ---------------------------------------------------------------- invariants
def test_left_yaw_all_positive() -> None:
    """yaw < 0 (LEFT) => T4_LR > 0, T5_LR > 0, combined > 0."""
    t4, t5, comb = _lr(_strengths(-0.2))
    assert t4 > 0, f"T4_LR should be > 0 for LEFT, got {t4}"
    assert t5 > 0, f"T5_LR should be > 0 for LEFT, got {t5}"
    assert comb > 0, f"combined_LR should be > 0 for LEFT, got {comb}"


def test_right_yaw_all_negative() -> None:
    """yaw > 0 (RIGHT) => T4_LR < 0, T5_LR < 0, combined < 0."""
    t4, t5, comb = _lr(_strengths(+0.2))
    assert t4 < 0, f"T4_LR should be < 0 for RIGHT, got {t4}"
    assert t5 < 0, f"T5_LR should be < 0 for RIGHT, got {t5}"
    assert comb < 0, f"combined_LR should be < 0 for RIGHT, got {comb}"


def test_t4_t5_share_one_convention() -> None:
    """T4 and T5 must never disagree in sign (pre-P0.1I they cancelled)."""
    for yaw in (-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0):
        t4, t5, comb = _lr(_strengths(yaw))
        assert _sign(t4) == _sign(t5) == _sign(comb), (
            f"convention mismatch at yaw={yaw}: T4={t4}, T5={t5}, combined={comb}"
        )


def test_amplitude_gain_is_1_05x() -> None:
    """Signed lateralisation must be (0.55 + 0.50) * sum over subtypes, not 0.05x.

    combined_LR = len(T4_TYPES+4) * 0.55 + len(T5_TYPES) * 0.50, scaled by gain.
    """
    yaw = -0.2
    _t4, _t5, comb = _lr(_strengths(yaw))
    n_sub = len(T4_TYPES) + len(T5_TYPES)
    expected = (
        len(T4_TYPES) * 0.55 + len(T5_TYPES) * 0.50
    ) * GAIN * abs(math.tanh(FG * yaw))
    assert abs(abs(comb) - expected) < 1e-9, f"expected {expected}, got {abs(comb)}"

    # The pre-P0.1I T5 convention would have produced (0.55 - 0.50) instead of (0.55 + 0.50).
    old_expected = (
        len(T4_TYPES) * 0.55 - len(T5_TYPES) * 0.50
    ) * GAIN * abs(math.tanh(FG * yaw))
    assert abs(comb) > 10 * abs(old_expected), (
        f"lateralisation not restored: |{abs(comb)}| vs old |{abs(old_expected)}|"
    )
    assert n_sub == 8


def test_zero_yaw_arbitrary_scene_is_zero() -> None:
    """yaw == 0 => combined_LR ~ 0 for arbitrary scene terms."""
    rng = random.Random(0)
    for _ in range(200):
        kwargs = {
            "expansion": rng.uniform(-3, 3),
            "flow_l": rng.uniform(-3, 3),
            "flow_r": rng.uniform(-3, 3),
            "frame_diff": rng.uniform(0, 1),
        }
        t4, t5, comb = _lr(_strengths(0.0, **kwargs))
        assert abs(t4) < 1e-9 and abs(t5) < 1e-9 and abs(comb) < 1e-9, (
            f"scene terms leaked into L-R at yaw=0: T4={t4}, T5={t5}, combined={comb}"
        )


def test_scene_cannot_flip_left_sign() -> None:
    """With yaw pinned LEFT, 300 random scenes must never flip combined_LR."""
    rng = random.Random(1)
    for _ in range(300):
        kwargs = {
            "expansion": rng.uniform(-5, 5),
            "flow_l": rng.uniform(-5, 5),
            "flow_r": rng.uniform(-5, 5),
            "frame_diff": rng.uniform(0, 2),
            "left_loom": rng.uniform(0, 1),
            "right_loom": rng.uniform(0, 1),
        }
        _t4, _t5, comb = _lr(_strengths(-0.2, **kwargs))
        assert comb > 1e-9, f"LEFT sign flipped by scene terms: combined={comb}, scene={kwargs}"


def test_scene_cannot_flip_right_sign() -> None:
    """With yaw pinned RIGHT, 300 random scenes must never flip combined_LR."""
    rng = random.Random(2)
    for _ in range(300):
        kwargs = {
            "expansion": rng.uniform(-5, 5),
            "flow_l": rng.uniform(-5, 5),
            "flow_r": rng.uniform(-5, 5),
            "frame_diff": rng.uniform(0, 2),
            "left_loom": rng.uniform(0, 1),
            "right_loom": rng.uniform(0, 1),
        }
        _t4, _t5, comb = _lr(_strengths(+0.2, **kwargs))
        assert comb < -1e-9, f"RIGHT sign flipped by scene terms: combined={comb}, scene={kwargs}"


def test_scene_terms_still_modulate_activity() -> None:
    """Common-mode terms must still raise overall optic-lobe drive (not be dropped)."""
    base = _strengths(0.0)
    loud = _strengths(0.0, expansion=3.0, flow_l=3.0, flow_r=3.0)
    assert loud["T4a_L"] > base["T4a_L"]
    assert abs(loud["T4a_L"] - loud["T4a_R"]) < 1e-9


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
