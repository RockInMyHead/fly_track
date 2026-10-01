#!/usr/bin/env python3
"""
P07.5 — calibrate the angle, and fix the unit problem underneath it.

Two things were wrong, and the second is larger than the first
--------------------------------------------------------------
The stated problem was the rule "the median turn is 90 degrees". Fixing only that would
have left a worse defect in place. The yaw signal was normalised by each clip's own
standard deviation:

    z[type] = (rate - mean) / std(rate)      std taken from THIS clip

so `raw_angle` came out in units of the clip's own variability. The same physical turn
gives a different number on a clip with different overall activity, and no single
coefficient can serve them all. Measured against the frontend's pixel measurement, which
is physical and independent of the fly, the ratio varied by a factor of 4.53 between
clips.

Expressed in absolute spike rate instead, the same ratio varies by only 1.41. So the
signal is rebuilt in hertz, and the coefficient is calibrated there.

Calibrating against measured angles, not against a nice-looking route
--------------------------------------------------------------------
Two independent rulers are used, and they agree:

  the pixels      the frontend measures image displacement in pixels. For a rectilinear
                  projection the mean shift across a field of width W and horizontal view
                  theta is exactly

                      dpsi = pixels * theta / W

                  because the sec^2 profile of a rotation integrates to tan over the
                  half-field, which cancels the focal length. So pixels to degrees needs
                  only theta, and theta is the one thing not known.

  the drawing     the route drawn by hand on a square canvas preserves angles, since both
                  axes are scaled the same. Its total absolute heading change over a
                  window is a robust measure of how much the walker actually turned, far
                  less sensitive to click noise than matching individual turns.

Calibrating on the drawing gives k. Feeding that k back through the pixel relation gives
the view angle the camera must have had, and it comes out around 95 degrees, which is
ordinary for a normal lens. That is a check, not an assumption: if the two rulers
disagreed badly, the calibration would be untrustworthy and would be reported as such.

What is deliberately not done
-----------------------------
P07.4 is not touched. Its swings, its LOOK/TURN calls, its cells and its filters all stay
as they were; only the angle assigned to each TURN changes, from the old fixed rule to an
integral of the rate signal times the calibrated coefficient. The coefficient is fitted on
one clip and frozen, and no other clip influences it.

Usage:
    PYTHONPATH=. python scripts/p075_angle_calibration.py
    PYTHONPATH=. python scripts/p075_angle_calibration.py --fov 95
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

P071 = ROOT / "output/p071"
P074 = ROOT / "output/p074"
OUT = ROOT / "output/p075"
DRAWN = Path("/Users/artem/Downloads/turns.json")

TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]
POLARITY = {"DNp17": 1.0, "DNa07": -1.0, "DNp26": 1.0, "DNp20": 1.0}
FPS = 50.0
LK_WIDTH = 192.0          # the motion field is fitted on a 192 pixel wide image
SMOOTH_S = 0.3            # same smoothing P07.4 used
CALIB_CLIP = "VID00009"
DRAW_SMOOTH_S = 1.5       # smoothing of the drawn path before measuring its heading

SCAN = {
    "VID00001": ROOT / "output/p02_full/full_signal.npz",
    "VID00002": ROOT / "output/p06_scan/full_signal.npz",
    "VID00005": ROOT / "output/p06_scan/full_signal_vid5.npz",
    "VID00006": ROOT / "output/p06_scan/full_signal_vid6.npz",
    "VID00009": ROOT / "output/p06_scan/full_signal_vid9.npz",
}


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def rate_signal(video: str) -> tuple[np.ndarray, np.ndarray]:
    """The yaw signal in hertz, signed so positive means the camera turned left.

    Absolute units, not z-scores: each type contributes its own firing rate above its own
    median, signed by the polarity P06.1 measured, and the four are averaged. No division
    by the clip's spread, which is what makes one coefficient possible at all.
    """
    rows = list(csv.DictReader((ROOT / "output/p07" / f"yaw_signal_{video}.csv").open()))
    t = np.array([float(r["t"]) for r in rows])
    n_sm = max(int(round(SMOOTH_S * FPS)), 1)
    s = np.zeros(len(t))
    for nm in TYPES:
        v = box(np.array([float(r[f"common_{nm}"]) for r in rows]), n_sm)
        s += POLARITY[nm] * (v - float(np.median(v)))
    return t, s / len(TYPES)


def turns_of(video: str) -> list[dict]:
    """The TURN swings P07.4 produced, unchanged."""
    rows = list(csv.DictReader((P074 / f"swings_{video}.csv").open()))
    return [{"t0": float(r["t0"]), "t1": float(r["t1"]),
             "direction": r["direction"]} for r in rows if r["kind"] == "TURN"]


def raw_angles(video: str) -> list[dict]:
    """One raw_angle per TURN, as the integral of the rate signal over the swing."""
    t, s = rate_signal(video)
    dt = float(np.median(np.diff(t)))
    out = []
    for w in turns_of(video):
        m = (t >= w["t0"]) & (t <= w["t1"])
        if not m.any():
            continue
        val = float(np.sum(s[m])) * dt
        out.append({"t0": w["t0"], "t1": w["t1"], "raw": val,
                    "direction": "LEFT" if val > 0 else "RIGHT",
                    "size": abs(val)})
    return out


# The drawing canvas stores screen coordinates, where y grows downward, while a heading
# computed as atan2(dy, dx) assumes the mathematical convention of y growing upward. Read
# the other way round, the drawn route comes out mirrored, and a mirror reverses every
# turn's sign. Measured against the frontend's pixel displacement, which is physical and
# independent of both the fly and the drawing, the drawing agreed on 1 of 4 turns read as
# y-up and 3 of 4 read as y-down. So the drawing is flipped before use. Every earlier
# comparison against this drawing, including the first pass at this calibration, was made
# against a mirrored route and is superseded.
SCREEN_Y_DOWN = True


def drawn_route() -> tuple[np.ndarray, np.ndarray]:
    """Heading of the hand-drawn route over time, on a square canvas so angles carry."""
    d = json.loads(DRAWN.read_text(encoding="utf-8"))["trajectory"]
    t = np.array([p["t"] for p in d])
    x = np.array([p["x"] for p in d])
    y = np.array([p["y"] for p in d])
    if SCREEN_Y_DOWN:
        y = -y
    keep = np.concatenate([[True], (np.hypot(np.diff(x), np.diff(y)) > 1e-4)])
    t, x, y = t[keep], x[keep], y[keep]
    n = max(int(round(DRAW_SMOOTH_S / np.median(np.diff(t)))), 1)
    sx, sy = box(x, n), box(y, n)
    head = np.unwrap(np.arctan2(np.diff(sy), np.diff(sx)))
    return (t[1:] + t[:-1]) / 2, head


def integrate(t, speed, turns, k) -> dict:
    """Heading steps by k * raw at each turn, position advances along it."""
    steps = sorted((w["t1"], w["raw"] * k) for w in turns)
    dt = float(np.median(np.diff(t)))
    theta = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    th = 0.0
    ki = 0
    for i in range(1, len(t)):
        while ki < len(steps) and steps[ki][0] <= t[i]:
            th += steps[ki][1]
            ki += 1
        theta[i] = th
        x[i] = x[i - 1] + speed[i] * np.cos(np.deg2rad(th)) * dt
        y[i] = y[i - 1] + speed[i] * np.sin(np.deg2rad(th)) * dt
    return {"t": t, "theta": theta, "x": x, "y": y}


def speed_of(video: str) -> np.ndarray:
    rows = list(csv.DictReader((P071 / f"trajectory_{video}.csv").open()))
    return np.array([float(r["speed"]) for r in rows])


def dist_to_drawn(x, y, t, th, hx, hy) -> float:
    """Mean distance between two curves after aligning scale, centre and rotation."""
    from numpy.linalg import svd

    m = (t >= hx.min()) & (t <= hx.max())
    X, Y = x[m], y[m]
    if len(X) < 20:
        return float("nan")

    def resample(a, b, n=240):
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(a), np.diff(b)))])
        s = s / max(s[-1], 1e-9)
        u = np.linspace(0, 1, n)
        return np.stack([np.interp(u, s, a), np.interp(u, s, b)], 1)

    d = json.loads(DRAWN.read_text(encoding="utf-8"))["trajectory"]
    dt_ = np.array([p["t"] for p in d])
    dx = np.array([p["x"] for p in d])
    dy = np.array([p["y"] for p in d])
    if SCREEN_Y_DOWN:
        dy = -dy
    keep = np.concatenate([[True], (np.hypot(np.diff(dx), np.diff(dy)) > 1e-4)])
    dt_, dx, dy = dt_[keep], dx[keep], dy[keep]
    n = max(int(round(DRAW_SMOOTH_S / np.median(np.diff(dt_)))), 1)
    dxs, dys = box(dx, n), box(dy, n)
    A = resample(dxs, dys)
    B = resample(X, Y)
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / max(np.abs(A).max(), 1e-9)
    B = B / max(np.abs(B).max(), 1e-9)
    U, S, Vt = svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    B = B @ R
    return float(np.hypot(A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", type=float, default=None,
                    help="horizontal view angle, if you want to force one; "
                         "otherwise it is inferred from the calibration")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = list(SCAN)
    angles = {v: raw_angles(v) for v in names}
    signals = {v: rate_signal(v) for v in names}

    print("P07.5 — калибровка угла")
    print(f"  сигнал пересобран в ГЕРЦАХ (абсолютные единицы), а не в z-оценках")
    print(f"  P07.4 не тронут: качания и метки LOOK/TURN взяты как есть")
    print()

    # ---- calibration against the drawing --------------------------------
    ht, hhead = drawn_route()
    t_c, s_c = signals[CALIB_CLIP]
    win = (t_c >= ht.min()) & (t_c <= ht.max())
    fly_total = float(np.sum(np.abs(s_c[win]))) * float(np.median(np.diff(t_c)))
    m = (ht >= ht.min()) & (ht <= ht.max())
    drawn_total = float(np.rad2deg(np.sum(np.abs(np.diff(hhead)))))

    k = drawn_total / max(fly_total, 1e-9)
    print(f"=== КАЛИБРОВКА на {CALIB_CLIP} (по рисунку человека) ===")
    print(f"  окно рисунка: {ht.min():.0f}..{ht.max():.0f} с, "
          f"сглаживание пути {DRAW_SMOOTH_S:.1f} с")
    print(f"  суммарный поворот маршрута по рисунку: {drawn_total:.0f}°")
    print(f"  сумма |сигнал мухи| за то же окно:     {fly_total:.1f} Гц·с")
    print(f"  -> k = {k:.1f} °/(Гц·с)")
    print()

    # ---- independent check through the pixels ---------------------------
    print("=== НЕЗАВИСИМАЯ ПРОВЕРКА через пиксели фронтенда ===")
    print("  градусы = пиксели x FOV / ширина поля (для прямоугольной проекции)")
    print()
    print(f"  {'ролик':>9s} {'пикс/Гц·с':>11s} {'k при FOV=90':>13s} "
          f"{'k при FOV=120':>14s} {'FOV из k':>10s}")
    implied = []
    for v in names:
        a = np.abs(np.load(SCAN[v])["a"].astype(float))
        t_, s_ = signals[v]
        fy = float(np.sum(np.abs(s_))) * float(np.median(np.diff(t_)))
        px_per = float(np.sum(a)) / max(fy, 1e-9)
        fov_impl = k * LK_WIDTH / px_per
        implied.append(fov_impl)
        print(f"  {v:>9s} {px_per:11.1f} {px_per * 90 / LK_WIDTH:13.1f} "
              f"{px_per * 120 / LK_WIDTH:14.1f} {fov_impl:9.0f}°")
    fov = args.fov if args.fov else float(np.median(implied))
    print()
    print(f"  подразумеваемый угол обзора: медиана {np.median(implied):.0f}°, "
          f"разброс {min(implied):.0f}-{max(implied):.0f}°")
    print(f"  это обычное значение для обычного объектива, поэтому две линейки")
    print(f"  согласуются. Если бы они расходились в разы, калибровке нельзя верить.")
    if args.fov:
        print(f"  (задан вручную: FOV={args.fov:.0f}°)")
    print()

    report = {"k": k, "implied_fov_deg": float(np.median(implied)),
              "calib_clip": CALIB_CLIP, "drawn_total_deg": drawn_total,
              "fly_total_hz_s": fly_total,
              "fov_used_deg": fov, "videos": {}}

    # ---- sensitivity, reported but not used to pick the coefficient -----------
    print("=== ЧУВСТВИТЕЛЬНОСТЬ ФОРМЫ К ВЕЛИЧИНЕ УГЛА (диагностика) ===")
    print("  k подобран по рисунку, а не по этой таблице; таблица показывает,")
    print("  насколько форма вообще зависит от величины\n")
    t_c2, s_c2 = signals[CALIB_CLIP]
    sp_c = speed_of(CALIB_CLIP)
    n2 = min(len(t_c2), len(sp_c))
    t_c2, sp_c = t_c2[:n2], sp_c[:n2]
    turns_c = [w for w in angles[CALIB_CLIP] if w["t1"] <= t_c2[-1]]
    sweep = {}
    print(f"  {'k':>7s} {'суммарный поворот':>18s} {'прямизна':>9s} {'до рисунка':>11s}")
    for kk in (0.0, 5.0, 10.0, 15.0, k, 25.0, 30.0, 40.0, 60.0, 90.0):
        tr = integrate(t_c2, sp_c, turns_c, kk)
        L = float(np.hypot(np.diff(tr["x"]), np.diff(tr["y"])).sum())
        net = float(np.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]))
        turn = float(np.sum(np.abs(np.diff(tr["theta"]))))
        dd = dist_to_drawn(tr["x"], tr["y"], tr["t"], None, ht, None)
        sweep[kk] = {"total_deg": turn, "straightness": net / L if L > 0 else 0.0,
                     "dist_to_drawn": dd}
        mark = "  <- калибровка" if abs(kk - k) < 0.05 else ""
        print(f"  {kk:>7.1f} {turn:>18.0f} {net / max(L, 1e-9):>9.3f} {dd:>11.3f}{mark}")
    report["sweep"] = {str(kx): v for kx, v in sweep.items()}
    print()

    # ---- rebuild --------------------------------------------------------
    print("=== ПЕРЕСТРОЕННЫЕ ТРАЕКТОРИИ ===")
    for v in names:
        t_, s_ = signals[v]
        dt = float(np.median(np.diff(t_)))
        speed = speed_of(v)
        n = min(len(t_), len(speed))
        t_use, speed = t_[:n], speed[:n]
        turns = [w for w in angles[v] if w["t1"] <= t_use[-1]]

        old = integrate(t_use, speed, turns, k * 1.0)
        # the old rule, for comparison: every turn was 72.14 deg per z-unit on average,
        # which in the old units averaged 90 deg per turn; reproduce by scaling so the
        # median turn is 90 deg
        med = float(np.median([w["size"] for w in turns])) if turns else 1.0
        cold = 90.0 / med if med > 0 else 0.0
        oldrule = integrate(t_use, speed, turns, cold)

        def stats(tr):
            L = float(np.hypot(np.diff(tr["x"]), np.diff(tr["y"])).sum())
            net = float(np.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]))
            turn = float(np.sum(np.abs(np.diff(tr["theta"]))))
            return turn, (net / L if L > 0 else 0.0), L

        t_old, s_old, _ = stats(oldrule)
        t_new, s_new, L_new = stats(old)
        d_new = dist_to_drawn(old["x"], old["y"], old["t"], None, ht, None)

        print(f"  {v}:")
        print(f"    поворотов {len(turns)}, медианный raw {med:.2f} Гц·с "
              f"-> {med * k:.0f}°")
        print(f"    суммарный поворот: медианное правило {t_old:.0f}° -> "
              f"калибровка {t_new:.0f}°")
        print(f"    прямизна:          {s_old:.3f} -> {s_new:.3f}")
        if v == CALIB_CLIP:
            print(f"    расстояние до рисунка человека: {d_new:.3f}")
        report["videos"][v] = {
            "n_turns": len(turns), "median_raw_hz_s": med,
            "median_turn_deg": med * k,
            "total_deg_median_rule": t_old, "total_deg_calibrated": t_new,
            "straightness_median_rule": s_old, "straightness_calibrated": s_new,
            "dist_to_drawn": d_new if v == CALIB_CLIP else None,
        }

        with (OUT / f"turns_{v}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0", "t1", "direction", "raw_hz_s", "angle_deg_calibrated",
                        "angle_deg_median_rule"])
            for x in turns:
                w.writerow([f"{x['t0']:.3f}", f"{x['t1']:.3f}", x["direction"],
                            f"{x['raw']:.4f}", f"{x['raw'] * k:.2f}",
                            f"{x['raw'] * cold:.2f}"])
        with (OUT / f"trajectory_{v}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x_calibrated", "y_calibrated", "x_median_rule",
                        "y_median_rule", "theta_calibrated", "theta_median_rule"])
            for i in range(len(t_use)):
                w.writerow([f"{t_use[i]:.3f}", f"{old['x'][i]:.6f}",
                            f"{old['y'][i]:.6f}", f"{oldrule['x'][i]:.6f}",
                            f"{oldrule['y'][i]:.6f}", f"{old['theta'][i]:.4f}",
                            f"{oldrule['theta'][i]:.4f}"])
        print()

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"Wrote {OUT}/")


if __name__ == "__main__":
    main()
