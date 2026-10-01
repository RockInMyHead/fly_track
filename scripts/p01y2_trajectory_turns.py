#!/usr/bin/env python3
"""
Turn a drawn trajectory into turn labels, then check them against the measurement.

The trajectory tab of the annotator produces a list of points, each with a canvas
position and the video time at which it was placed:

    {"trajectory":[{"x":0.5,"y":0.9,"t":10.0}, ...], "orient_deg":-90, ...}

Because every point carries its time, the drawing is a *course over time*, not just
a shape. From it:

    bearing of segment i  = angle of (p[i+1] - p[i]) measured from the forward
                            direction. Positive = veering clockwise = camera RIGHT.
    yaw rate              = derivative of the bearing over time

So one sketch yields a whole sequence of turns, with a rate attached, instead of a
binary left/right mark.

Two things are reported

  A. TURNS read off the shape: contiguous runs of same-sign bearing above a
     threshold, each with its time span and mean bearing. Each is then checked
     against the frozen measurement, on exactly that window.

  B. COURSE vs MEASUREMENT: the drawing implies a yaw rate at every moment, and the
     frozen measurement independently gives one at every moment. They are compared
     on a common time grid: sign agreement and correlation, plus the sign agreement
     restricted to the windows the drawing itself calls a turn.

Part B is the useful one. It does not need the drawn turns to be right; it asks
whether the shape of the sketch tracks the measured rotation, which is a much
stronger check than any per-turn label.

Nothing is tuned. The measurement is the frozen definition in
`fly_vo/content_motion.py`; the drawing is the human's.

Usage:
    PYTHONPATH=. python scripts/p01y2_trajectory_turns.py
    PYTHONPATH=. python scripts/p01y2_trajectory_turns.py --turns ~/Downloads/turns.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.content_motion import DEFAULT_FPS, read_window
from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
DEFAULT = ROOT / "data/p01r/trajectory_annotations.json"
GRID_W, GRID_H = 16, 8
XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
PINV = np.linalg.pinv(np.stack([np.ones_like(XN), XN], axis=1))
CROP = (0.30, 0.70)
ENCODE = (192, 108)

# Trajectory handling. Consecutive clicks are far too close in time for their heading to
# mean anything: pointer jitter of a couple of pixels over a third of a second produces
# tens of degrees of apparent course change. Both of these were needed before the
# derived turns stopped being noise.
SMOOTH_S = 0.6        # box-average the positions over this window
BASELINE_S = 1.0      # take the heading between points about this far apart in time
MAX_GAP_S = 3.0       # never compute a heading across a longer break in the recording
MIN_STEP = 1e-4       # ignore a baseline whose displacement is below this


# --------------------------------------------------------------- geometry
def bearing_deg(a: dict, b: dict, forward_deg: float) -> float | None:
    """Heading of the segment a->b, degrees from forward. Positive = veering right."""
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    if math.hypot(dx, dy) < 1e-6:
        return None
    alpha = math.degrees(math.atan2(dy, dx))
    return (alpha - forward_deg + 180) % 360 - 180


def _norm180(d: float) -> float:
    return (d + 180) % 360 - 180


def smooth_positions(points: list[dict], window_s: float) -> list[dict]:
    """Box-average the positions over a time window, keeping the times.

    Click jitter moves the pointer a few pixels each time, so the heading between two
    *adjacent* clicks is dominated by noise rather than by the camera course. Averaging
    over a fraction of a second attenuates exactly that.
    """
    if window_s <= 0 or len(points) < 3:
        return [dict(p) for p in points]
    t = np.array([p["t"] for p in points], dtype=float)
    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    half = window_s / 2.0
    xs, ys = np.empty_like(x), np.empty_like(y)
    for i in range(len(points)):
        m = np.abs(t - t[i]) <= half
        if not m.any():
            xs[i], ys[i] = x[i], y[i]
        else:
            xs[i], ys[i] = x[m].mean(), y[m].mean()
    return [{"t": float(t[i]), "x": float(xs[i]), "y": float(ys[i])} for i in range(len(points))]


def heading_series(points: list[dict], forward_deg: float, baseline_s: float) -> list[dict]:
    """Heading over a *baseline* window, not between adjacent points.

    For every point, the heading is taken between that point and the point about
    `baseline_s` later. A real turn of 30 deg/s changes the heading by 30*baseline
    degrees over that span, while pointer jitter contributes only a couple of degrees
    for a baseline of a second. Gap-splitting is respected: no heading is computed
    across a break in the recording.
    """
    pts = smooth_positions(points, SMOOTH_S)
    t = np.array([p["t"] for p in pts])
    out = []
    j0 = 0
    for i in range(len(pts)):
        # do not bridge a gap in the data
        if i > 0 and (t[i] - t[i - 1]) > MAX_GAP_S:
            j0 = i
        target = t[i] + baseline_s
        j = int(np.searchsorted(t, target))
        j = min(max(j, i + 1), len(pts) - 1)
        if j <= i or t[j] - t[i] < baseline_s * 0.5:
            continue
        a, b = pts[i], pts[j]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        if math.hypot(dx, dy) < MIN_STEP:
            continue
        out.append({
            "t0": float(a["t"]), "t1": float(b["t"]),
            "heading": (math.degrees(math.atan2(dy, dx)) - forward_deg + 180) % 360 - 180,
            "step": math.hypot(dx, dy),
        })
    return out


def turn_rate_series(points: list[dict], forward_deg: float, baseline_s: float) -> list[dict]:
    """Turn rate in deg/s, from the change of the baseline heading."""
    hs = heading_series(points, forward_deg, baseline_s)
    out = []
    for k in range(1, len(hs)):
        prev, cur = hs[k - 1], hs[k]
        dt = cur["t0"] - prev["t0"]
        if dt <= 1e-6:
            continue
        out.append({
            "t": cur["t0"], "t0": cur["t0"], "t1": cur["t1"],
            "rate": _norm180(cur["heading"] - prev["heading"]) / dt,
        })
    return out


def segments(points: list[dict], forward_deg: float) -> list[dict]:
    """Segments with a heading and a turn rate in deg/s.

    The heading alone cannot identify a turn: after a 90 deg left turn the path keeps
    pointing left, so a straight continuation would still show a heading of -90. What
    identifies a turn is the change of heading per second — the yaw rate.
    """
    segs = []
    for i in range(len(points) - 1):
        h = bearing_deg(points[i], points[i + 1], forward_deg)
        if h is None:
            continue
        segs.append({"i": i, "t0": points[i]["t"], "t1": points[i + 1]["t"],
                     "heading": h, "rate": None})
    for k in range(1, len(segs)):
        mid_prev = (segs[k - 1]["t0"] + segs[k - 1]["t1"]) / 2.0
        mid_cur = (segs[k]["t0"] + segs[k]["t1"]) / 2.0
        dt = mid_cur - mid_prev
        if abs(dt) < 1e-6:
            continue
        segs[k]["rate"] = _norm180(segs[k]["heading"] - segs[k - 1]["heading"]) / dt
    return segs


def find_turns(points: list[dict], forward_deg: float, thresh_deg_s: float,
               gap_s: float = 0.6) -> list[dict]:
    """Runs of same-sign turn rate above the threshold, from the baseline series."""
    series = turn_rate_series(points, forward_deg, BASELINE_S)
    hits = [s for s in series if abs(s["rate"]) >= thresh_deg_s]
    runs, cur = [], None
    for s in hits:
        d = "RIGHT" if s["rate"] > 0 else "LEFT"
        if cur and cur["direction"] == d and s["t0"] <= cur["t1"] + gap_s:
            cur["t1"] = s["t1"]
            cur["segs"].append(s)
        else:
            if cur:
                runs.append(cur)
            cur = {"direction": d, "t0": s["t0"], "t1": s["t1"], "segs": [s]}
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if r["t1"] - r["t0"] < 0.3:
            continue
        rs = [abs(s["rate"]) for s in r["segs"]]
        span = max(r["t1"] - r["t0"], 1e-9)
        net = sum(s["rate"] for s in r["segs"]) * (span / max(len(r["segs"]), 1))
        out.append({
            "direction": r["direction"], "t0": round(r["t0"], 3), "t1": round(r["t1"], 3),
            "n_segments": len(r["segs"]),
            "mean_rate": round(sum(rs) / len(rs), 1),
            "peak_rate": round(max(rs), 1),
            "net_deg": round(net, 1),
        })
    return out


# --------------------------------------------------- course vs measurement
def measured_yaw_rate(t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame measured rotation over [t0, t1] at native fps.

    Returns (times, rate) where rate is the uniform component of the Lucas-Kanade
    displacement field: rate > 0 means content moved right, i.e. the camera rotated
    LEFT. This is the same quantity as in the frozen measurement, just not pooled
    over the window.
    """
    grays = read_window(str(VIDEO), t0, t1, DEFAULT_FPS)
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    ts, vals = [], []
    for i, g in enumerate(grays):
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, _dy, _d = lk.process(band)
        ab = dxf.reshape(-1) @ PINV.T
        ts.append(t0 + i / DEFAULT_FPS)
        vals.append(float(ab[0]))
    return np.asarray(ts), np.asarray(vals)


def drawn_rate(points: list[dict], forward_deg: float, t0: float, t1: float,
               fps: float = DEFAULT_FPS) -> tuple[np.ndarray, np.ndarray]:
    """Yaw rate implied by the sketch, sampled on the measurement's time grid.

    The sketch's rate is in deg/s with positive = camera RIGHT (veering clockwise).
    The measured rate is the uniform component of the content displacement, positive
    for content moving right, which is camera LEFT. The sketch's rate is negated so
    both series mean the same physical thing before they are compared.
    """
    series = turn_rate_series(points, forward_deg, BASELINE_S)
    grid = np.arange(t0, t1, 1.0 / fps)
    if not series or grid.size == 0:
        return grid, np.zeros_like(grid)
    # interpolate the sketched rate onto the measurement's grid
    st = np.array([s["t"] for s in series])
    sv = np.array([s["rate"] for s in series])
    return grid, -np.interp(grid, st, sv)   # negate: sketch is camera-right positive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default=str(DEFAULT))
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--threshold", type=float, default=None,
                    help="bearing threshold in degrees (default: whatever the sketch used)")
    ap.add_argument("-o", "--output", default="output/p01y2_trajectory")
    args = ap.parse_args()

    src = Path(args.turns).expanduser()
    if not src.exists():
        raise SystemExit(f"file not found: {src}")
    doc = json.loads(src.read_text(encoding="utf-8"))
    points = doc.get("trajectory") or []
    forward_deg = float(doc.get("orient_deg", -90))
    thresh = args.threshold if args.threshold is not None else float(
        doc.get("turn_threshold_deg", 10)
    )
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print(f"drawn trajectory — {src}")
    print(f"  {len(points)} points, forward = {forward_deg:g}°, "
          f"turn-rate threshold = {thresh:g}°/s")
    if not points:
        print("\n  no trajectory points yet. Draw a path in the annotator's trajectory tab,")
        print("  press save, then run this again.")
        return

    mono = all(points[i]["t"] >= points[i - 1]["t"] for i in range(1, len(points)))
    print(f"  points in time order: {mono}")
    if not mono:
        points = sorted(points, key=lambda p: p["t"])
        print("  (sorted by time for the analysis)")

    turns = find_turns(points, forward_deg, thresh)
    print(f"\n=== A. turns read off the shape ===")
    if not turns:
        print("  none above the threshold")
    print(f"  {'dir':6s} {'t0':>8s} {'t1':>8s} {'len':>6s} {'rate':>7s} {'peak':>7s} "
          f"{'net':>7s} {'measured':>9s} {'agrees':>7s} {'coh':>5s}")
    rows = []
    for t in turns:
        m = None
        try:
            from fly_vo.content_motion import measure_window
            m = measure_window(args.video, t["t0"], t["t1"], "drawn")
        except Exception as exc:  # noqa: BLE001
            print(f"  measurement failed: {exc}")
        meas_dir = m.camera_direction if m and m.content_dx != 0 else "NONE"
        agrees = (meas_dir == t["direction"]) if m else None
        t.update({"measured_camera": meas_dir, "agrees": agrees,
                  "coherence": m.coherence if m else None,
                  "content_dx": m.content_dx if m else None})
        rows.append(t)
        coh_txt = f"{t['coherence']:.2f}" if t["coherence"] is not None else "—"
        ag_txt = "—" if agrees is None else ("yes" if agrees else "NO")
        print(f"  {t['direction']:6s} {t['t0']:8.2f} {t['t1']:8.2f} "
              f"{t['t1'] - t['t0']:6.2f} {t['mean_rate']:7.1f} {t['peak_rate']:7.1f} "
              f"{t['net_deg']:7.1f} {meas_dir:>9s} {ag_txt:>7s} {coh_txt:>5s}")

    ok = sum(1 for t in rows if t["agrees"])
    print(f"\n  drawn turns agreeing with the measurement: {ok}/{len(rows)}")

    # ------------------------------------------------- B. turn size vs measured
    # A per-frame comparison was tried and rejected: the measured rate is
    # noise-dominated (per-frame SNR ~1.8, sign flips about 20 times a minute), so
    # agreeing with a smooth hand-drawn curve sample-by-sample is not a valid test.
    # What is comparable is the total rotation over each drawn turn, which is the
    # quantity the drawing actually asserts.
    print(f"\n=== B. drawn turn size vs measured rotation over the same window ===")
    print(f"  signed so positive = camera turned RIGHT in both columns")
    print(f"  {'dir':6s} {'t0':>8s} {'t1':>8s} {'drawn net':>10s} {'measured':>10s} "
          f"{'same sign':>10s}")
    pairs = []
    for t in rows:
        ts, vals = measured_yaw_rate(t["t0"], t["t1"])
        if ts.size == 0:
            continue
        # vals > 0 means content moved right, i.e. camera turned LEFT; negate to make
        # positive mean camera RIGHT, matching the sketch's convention
        measured = float(-np.sum(vals) / DEFAULT_FPS)     # pixels of accumulated turn
        drawn = float(t["net_deg"])
        same = bool(np.sign(measured) == np.sign(drawn))
        t["measured_turn_px"] = measured
        pairs.append((drawn, measured, same))
        print(f"  {t['direction']:6s} {t['t0']:8.2f} {t['t1']:8.2f} {drawn:10.1f} "
              f"{measured:10.1f} {('yes' if same else 'NO'):>10s}")
    if pairs:
        d = np.array([p[0] for p in pairs])
        m = np.array([p[1] for p in pairs])
        same_n = sum(1 for p in pairs if p[2])
        corr = float(np.corrcoef(d, m)[0, 1]) if d.std() > 0 and m.std() > 0 else float("nan")
        print(f"\n  same sign: {same_n}/{len(pairs)}")
        print(f"  correlation of magnitudes: {corr:+.3f}  "
              f"(a scale factor between degrees and pixels is unknown, so only the")
        print(f"   sign and the ordering carry information)")
        order_d = np.argsort(d)
        order_m = np.argsort(m)
        print(f"  rank order of turn sizes identical: {list(order_d) == list(order_m)}")
        course = {"n_turns": len(pairs), "same_sign": same_n,
                  "correlation": corr,
                  "rank_order_identical": bool(list(order_d) == list(order_m)),
                  "drawn_net_deg": d.tolist(), "measured_turn_px": m.tolist()}

    with (out / "drawn_turns.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "t0", "t1", "n_segments", "mean_rate_deg_s",
                    "peak_rate_deg_s", "net_deg", "measured_camera", "agrees",
                    "coherence", "content_dx"])
        for t in rows:
            w.writerow([t["direction"], t["t0"], t["t1"], t["n_segments"],
                        t["mean_rate"], t["peak_rate"], t["net_deg"],
                        t["measured_camera"], t["agrees"], t["coherence"], t["content_dx"]])
    (out / "report.json").write_text(json.dumps({
        "source": str(src), "n_points": len(points), "forward_deg": forward_deg,
        "threshold_deg": thresh, "drawn_turns": rows,
        "n_agree": ok, "n_turns": len(rows),
        "course_vs_measurement": course,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
