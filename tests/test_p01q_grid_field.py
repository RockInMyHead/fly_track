"""P0.1Q invariants: no empty azimuth bins, and a 2-D field that carries direction.

Runs standalone (no pytest required):

    PYTHONPATH=. python tests/test_p01q_grid_field.py

Also collectable by pytest if it is ever installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.local_motion_field import GridMotionField, azimuth_profile, reichardt_2d
from fly_vo.optic_flow import (
    FlowConfig,
    build_azimuth_panorama,
    legacy_azimuth_bin_counts,
)


# --------------------------------------------------------------------- comb
def _striped(h: int, w: int) -> np.ndarray:
    x = np.arange(w, dtype=np.float32)
    return np.tile(0.5 + 0.4 * np.sin(x * 0.7), (h, 1)).astype(np.float32)


def _texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.random((h, w)).astype(np.float32)
    for _ in range(2):
        img = 0.25 * (np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1) + 2.0 * img)
    return img


def test_legacy_binning_really_has_holes():
    """The old 96 -> 128 mapping is the defect we fixed: it must show holes."""
    counts = legacy_azimuth_bin_counts(96, 128)
    holes = int(np.count_nonzero(counts == 0))
    assert holes > 0, "legacy binning unexpectedly had no holes"
    print(f"    legacy 96 -> 128 empty bins: {holes}/128")


def test_panorama_has_no_empty_bins_at_any_ratio():
    """New interpolation mapping: every bin is populated, at 96 and at 192 cols."""
    for w in (96, 192, 320):
        flat = _striped(40, w)
        pan = build_azimuth_panorama(flat, 128)
        assert pan.shape == (128,)
        assert np.all(np.isfinite(pan))
        expect = np.interp(
            np.linspace(-1.0, 1.0, 128),
            np.linspace(-1.0, 1.0, w),
            flat.mean(axis=0),
        )
        assert np.allclose(pan, expect, atol=1e-6)
        assert not np.any(pan == 0.0), f"zero bin at w={w}"
        print(f"    w={w:4d} -> 128 bins, empty = 0")


def test_panorama_preserves_column_order():
    """Azimuth stays monotone: a right-brighter image gives a rising panorama."""
    img = np.tile(np.linspace(0.0, 1.0, 200, dtype=np.float32), (30, 1))
    pan = build_azimuth_panorama(img, 128)
    assert np.all(np.diff(pan) >= -1e-6)


# ------------------------------------------------------------------ 2-D sign
def test_reichardt_2d_sign_follows_horizontal_shift():
    """Content shifted toward +x must give a positive 2-D correlator."""
    base = _texture(24, 64, seed=1)
    for dx in (2, 4, 6):
        assert np.mean(reichardt_2d(base, np.roll(base, dx, axis=1), dx)) > 0
        assert np.mean(reichardt_2d(base, np.roll(base, -dx, axis=1), dx)) < 0


def test_grid_field_carries_shift_direction():
    """The grid field keeps the sign, and every cell agrees."""
    base = _texture(48, 128, seed=2)

    field = GridMotionField(grid_w=16, grid_h=8, detectors=((4, 1, 4, 1),), alpha=0.5)
    for step in range(2):
        f2d, diag = field.process(np.roll(base, 4, axis=1) if step else base)
        if step:
            assert f2d.shape == (8, 16)
            assert np.all(f2d > 0), "rightward shift must be positive in every cell"
            assert diag.row_agreement > 0.9

    field.reset()
    for step in range(2):
        f2d, _ = field.process(np.roll(base, -4, axis=1) if step else base)
        if step:
            assert np.all(f2d < 0), "leftward shift must be negative in every cell"


def test_grid_field_ignores_vertical_shift():
    """Vertical motion must not masquerade as a horizontal (yaw) signal.

    A single texture can still produce a sizeable random value, so the claim
    tested here is that the *systematic* response to a vertical shift is
    negligible next to the response to a horizontal shift of the same size.
    """
    h_vals, v_vals = [], []
    for seed in range(24):
        base = _texture(48, 128, seed=seed)

        hf = GridMotionField(grid_w=16, grid_h=8, detectors=((4, 1, 4, 1),), alpha=0.5)
        for step in range(2):
            f2d, _ = hf.process(np.roll(base, 4, axis=1) if step else base)
            if step:
                h_vals.append(float(np.mean(f2d)))

        vf = GridMotionField(grid_w=16, grid_h=8, detectors=((4, 1, 4, 1),), alpha=0.5)
        for step in range(2):
            f2d, _ = vf.process(np.roll(base, 5, axis=0) if step else base)
            if step:
                v_vals.append(float(np.mean(f2d)))

    h_mean = float(np.mean(h_vals))
    v_bias = abs(float(np.mean(v_vals)))
    assert h_mean > 0, "horizontal shift did not produce a positive response"
    assert v_bias < 0.05 * h_mean, (
        f"vertical bias {v_bias:.2e} is not negligible vs horizontal {h_mean:.2e}"
    )
    print(f"    horizontal {h_mean:.3e}  vs  vertical bias {v_bias:.3e}"
          f"  ({v_bias / h_mean:.1%})")


def test_no_pooled_direction_in_module():
    """The grid path must not use a pooled yaw scalar in code."""
    import ast

    import fly_vo.local_motion_field as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    banned = {"yaw_frozen", "left_yaw", "right_yaw"}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.add(node.value)
    assert not (used & banned), f"pooled term leaked into the 2-D module: {used & banned}"


def test_azimuth_profile_matches_column_means():
    field = np.tile(np.arange(16, dtype=np.float64), (8, 1))
    prof = azimuth_profile(field, 128)
    assert prof.shape == (128,)
    assert np.all(np.diff(prof) >= -1e-9)
    assert abs(prof.mean() - 7.5) < 1e-6


def test_default_field_mode_is_legacy_panorama():
    """Existing scripts keep the frozen P0.1R+ behaviour unless they opt in."""
    assert FlowConfig().field_mode == "panorama"
    cfg = FlowConfig(field_mode="grid2d")
    assert cfg.grid_w == 16 and cfg.grid_h == 8
    assert cfg.grid_dx == (6, 12, 6) and cfg.grid_dt == (2, 3, 3)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
