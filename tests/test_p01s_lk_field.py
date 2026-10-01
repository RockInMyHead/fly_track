"""P0.1S invariants for the Lucas–Kanade local motion field.

Runs standalone (no pytest required):

    PYTHONPATH=. python tests/test_p01s_lk_field.py

Also collectable by pytest if it is ever installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.lk_motion_field import LKMotionField
from fly_vo.optic_flow import FlowConfig, reichardt_at_dx

H, W = 144, 256


def texture(seed: int = 0) -> np.ndarray:
    """High-contrast blobs — LK needs gradient structure to lock onto."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.float32)
    for _ in range(240):
        y, x = rng.integers(6, H - 6), rng.integers(6, W - 6)
        r = int(rng.integers(2, 5))
        img[max(0, y - r):y + r, max(0, x - r):x + r] += float(rng.uniform(0.3, 1.0))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img -= img.min()
    return (img / max(img.max(), 1e-6) * 0.85 + 0.05).astype(np.float32)


def shift(img: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Translate content by (+dx, +dy); positive dx moves content rightward."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def field_of(seq: list[np.ndarray], **kw) -> tuple[np.ndarray, np.ndarray, object]:
    lk = LKMotionField(grid_w=16, grid_h=8, **kw)
    out = None
    for img in seq:
        out = lk.process(img)
    return out


def lowfreq_texture(seed: int = 0) -> np.ndarray:
    """Large soft blobs — enough low-frequency energy for a 96-wide profile."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.float32)
    for _ in range(18):
        y, x = rng.integers(30, H - 30), rng.integers(40, W - 40)
        r = int(rng.integers(18, 34))
        yy, xx = np.ogrid[:H, :W]
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * r * r))
    for _ in range(18):
        y, x = rng.integers(30, H - 30), rng.integers(40, W - 40)
        r = int(rng.integers(18, 34))
        yy, xx = np.ogrid[:H, :W]
        img -= 0.7 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * r * r))
    img -= img.min()
    return (img / max(img.max(), 1e-6) * 0.85 + 0.05).astype(np.float32)


# ------------------------------------------------------------------ accuracy
def test_dx_matches_known_shift():
    """A known rightward shift must be recovered in pixels, per cell."""
    base = texture(1)
    for s in (1, 2, 3, 5):
        dx, dy, diag = field_of([base, shift(base, dx=s)])
        med = float(np.median(dx))
        assert abs(med - s) < 0.6, f"shift {s}: reported {med:.2f} px"
        assert diag.coverage > 0.8, f"shift {s}: coverage {diag.coverage:.2f}"
        print(f"    shift {s} px -> median dx {med:+.2f} px (coverage {diag.coverage:.0%})")


def test_sign_follows_shift_direction():
    base = texture(2)
    dxr, _, _ = field_of([base, shift(base, dx=3)])
    dxl, _, _ = field_of([base, shift(base, dx=-3)])
    assert np.all(dxr > 0), "rightward shift must be positive in every cell"
    assert np.all(dxl < 0), "leftward shift must be negative in every cell"


def test_sign_matches_reichardt():
    """LK and the Reichardt correlator must agree in sign (no polarity inversion)."""
    base = lowfreq_texture(3)
    for s in (2, 4):
        dx, _, _ = field_of([base, shift(base, dx=s)])
        lk_mean = float(np.mean(dx))

        col_prev = base.mean(axis=0)
        col_curr = shift(base, dx=s).mean(axis=0)
        r_mean = float(np.mean(reichardt_at_dx(col_prev, col_curr, 4)))

        assert np.sign(lk_mean) == np.sign(r_mean) == +1, (
            f"shift {s}: LK {lk_mean:+.3f} vs Reichardt {r_mean:+.4f}"
        )
        print(f"    shift {s}: LK {lk_mean:+.2f} px, Reichardt {r_mean:+.4f} — same sign")


def test_vertical_only_shift_gives_no_yaw():
    """Vertical image motion is not yaw: dx must stay near zero while dy tracks it."""
    base = texture(4)
    dx, dy, diag = field_of([base, shift(base, dy=3)])
    assert abs(float(np.mean(dx))) < 0.3, f"vertical shift leaked into dx: {np.mean(dx):+.3f}"
    assert abs(float(np.mean(dy)) - 3.0) < 0.6, f"dy did not track the shift: {np.mean(dy):+.2f}"


def test_field_is_local_not_pooled():
    """A radial (scaling) stimulus must split into a left/right pattern."""
    base = texture(5)
    zoom = cv2.resize(base, (int(W * 1.06), int(H * 1.06)), interpolation=cv2.INTER_LINEAR)
    y0 = (zoom.shape[0] - H) // 2
    x0 = (zoom.shape[1] - W) // 2
    zoom = zoom[y0:y0 + H, x0:x0 + W]
    dx, _, _ = field_of([base, zoom])
    left, right = float(dx[:, :8].mean()), float(dx[:, 8:].mean())
    assert left < 0 < right, f"scaling did not produce opposed halves: L {left:+.2f} R {right:+.2f}"
    assert abs(left - right) > 1.0, "halves are not separated enough to be a local measurement"
    print(f"    scaling halved the field: left {left:+.2f} px  right {right:+.2f} px")


# ----------------------------------------------------------- duplicate frames
def test_duplicate_frame_is_held_not_zeroed():
    """LK is undefined on identical frames; the previous field must be carried."""
    base = texture(6)
    moved = shift(base, dx=3)
    lk = LKMotionField(grid_w=16, grid_h=8, hold_on_duplicate=True)
    lk.process(base)
    dx1, _, _ = lk.process(moved)
    dx2, _, d2 = lk.process(moved.copy())
    assert d2.held, "duplicate frame was not detected"
    assert np.allclose(dx1, dx2), "held field differs from the previous one"

    lk2 = LKMotionField(grid_w=16, grid_h=8, hold_on_duplicate=False)
    lk2.process(base)
    lk2.process(moved)
    dx3, _, d3 = lk2.process(moved.copy())
    assert not d3.held
    assert float(np.max(np.abs(dx3))) < 0.2, "without holding, a duplicate must read ~zero motion"


def test_empty_cells_are_filled_from_column():
    """A seed-free patch must not leave a hole in the injected field."""
    blank = np.zeros((H, W), np.float32)
    base = texture(7)
    # a featureless band in the middle rows: LK cannot lock there
    img = base.copy()
    img[H // 3: 2 * H // 3, :] = 0.5
    lk = LKMotionField(grid_w=16, grid_h=8, fill_empty_from_column=True)
    lk.process(img)
    dx, _, diag = lk.process(shift(img, dx=3))
    assert diag.n_empty_cells >= 0
    # with column filling, no cell is left as a hard zero unless the column is
    # entirely empty, so the field should be uniformly signed
    assert np.all(dx > 0), "column filling left holes in the field"
    print(f"    coverage {diag.coverage:.0%}, empty cells before filling {diag.n_empty_cells}")


# --------------------------------------------------------------------- purity
def test_no_pooled_direction_in_module():
    """The tracking path must not use a pooled yaw scalar in code."""
    import ast

    import fly_vo.lk_motion_field as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    banned = {"yaw_frozen", "left_yaw", "right_yaw"}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name,)):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.add(node.value)
    assert not (used & banned), f"pooled term leaked into the tracking module: {used & banned}"


def test_legacy_mode_still_default():
    assert FlowConfig().field_mode == "panorama"
    cfg = FlowConfig(field_mode="lk")
    assert cfg.lk_win == 15 and cfg.lk_seed_layout == 3
    assert cfg.lk_hold_on_duplicate is True


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
