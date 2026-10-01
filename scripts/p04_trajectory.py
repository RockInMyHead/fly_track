#!/usr/bin/env python3
"""
P0.4 — dead-reckoning trajectory, read out at each stage of the fly's visual system.

The question: at which stage can the direction of travel still be read out?

The chain we have built is

    video -> frontend (retina + optic lobe) -> T4/T5 injection -> connectome -> DNs

and each stage can be asked for a yaw rate. Integrating a yaw rate together with a
forward speed gives a trajectory. So the same video is integrated three times:

    frontend   yaw from the uniform component of the measured flow field
    T4/T5      yaw from the injected population itself
    descending yaw from the DN groups the engine already tracks

If the trajectories agree, the connectome preserves the signal. If only the frontend
one is usable, the loss happens inside the brain, and that is the finding.

What is honest to claim
-----------------------
There is no ground-truth trajectory for this clip: it is a person walking through a
workshop with a chest camera, and nobody recorded where they went. So absolute accuracy
cannot be measured. What can be measured:

  * the SHAPE, which is what a trajectory is for. A walk through a building should go
    somewhere and turn, not wander as a blob and not run in a straight line.
  * the AGREEMENT between stages, which is internal consistency and does not need truth.
  * the total turning and path length, against what a walking person plausibly does.

The scale is not determined either. Yaw is calibrated so that the turns confirmed by eye
come out at a plausible angle, and forward speed is calibrated to a walking pace; both
are stated in the output as assumptions rather than measurements. The shape does not
depend on the forward scale at all, and only weakly on the yaw scale.

Usage:
    PYTHONPATH=. python scripts/p04_trajectory.py --stage frontend
    PYTHONPATH=. python scripts/p04_trajectory.py --stage frontend --t1 120
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

SIGNAL = ROOT / "output/p02_full/full_signal.npz"
OUT = ROOT / "output/p04_trajectory"
FPS = 30.0

# Integration parameters.
SMOOTH_S = 1.0            # box smoothing of the rate signals, in seconds
STOP_ACTIVITY = 0.60      # px/frame of local motion below which the walk is stationary
MIN_SPEED = 0.05          # fraction of median speed below which nothing is happening

# Calibration. Both are stated assumptions, not measurements.
DEG_PER_PX = 0.60         # yaw: from the hand-confirmed turns; a ~1.5 s turn accumulates
                          # 150 px of uniform displacement and is treated as about 90 deg
WALK_SPEED = 1.40         # m/s, a normal walking pace; sets the forward scale only


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def integrate(yaw_rate_dps: np.ndarray, speed_mps: np.ndarray,
              t: np.ndarray) -> dict:
    """Heading and position by dead reckoning, midpoint rule."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / FPS
    heading = np.empty(len(t))
    theta = np.pi / 2.0                     # start pointing "north" on the plot
    for i in range(len(t)):
        heading[i] = theta
        theta += np.deg2rad(yaw_rate_dps[i]) * dt
    # unwrap not needed: we keep theta continuous by construction
    vx = speed_mps * np.cos(heading)
    vy = speed_mps * np.sin(heading)
    x = np.concatenate([[0.0], np.cumsum(vx[:-1] * dt)])
    y = np.concatenate([[0.0], np.cumsum(vy[:-1] * dt)])
    path_len = float(np.sum(np.abs(speed_mps) * dt))
    net_disp = float(np.hypot(x[-1], y[-1]))
    # total turning, and the net change of heading
    dtheta = np.diff(np.concatenate([[heading[0]], heading]))
    return {
        "t": t, "x": x, "y": y, "heading": heading,
        "yaw_rate": yaw_rate_dps, "speed": speed_mps,
        "path_length_m": path_len, "net_displacement_m": net_disp,
        "closing_ratio": net_disp / max(path_len, 1e-9),
        "total_turning_deg": float(np.sum(np.abs(np.rad2deg(dtheta)))),
        "net_turning_deg": float(np.rad2deg(heading[-1] - heading[0])),
    }


def frontend_trajectory(sig: dict, t0: float, t1: float, smooth_s: float) -> dict:
    """Yaw from the uniform component, speed from the radial component.

    a > 0 means the content moved right, which is the camera turning LEFT, so the
    heading rate is +a. Speed comes from the divergence: for a camera moving forward
    the image expands, and the expansion coefficient scales with speed.
    """
    n = max(int(round(smooth_s * FPS)), 1)
    a = box(sig["a"], n)
    div = box(sig["b"] + sig["c"], n)
    act = box(sig["act_x"], n)
    t = sig["t"]

    m = (t >= t0) & (t <= t1)
    a, div, act, t = a[m], div[m], act[m], t[m]

    yaw = a * DEG_PER_PX                       # deg/s, already per frame at native fps
    yaw = yaw * FPS
    # forward speed proportional to expansion, scaled so the median moving speed is a
    # walking pace. This fixes the units only; the shape is unaffected.
    moving = act >= STOP_ACTIVITY
    raw_speed = np.maximum(div, 0.0)
    if moving.any() and np.median(raw_speed[moving]) > 1e-9:
        k = WALK_SPEED / np.median(raw_speed[moving])
    else:
        k = 0.0
    speed = raw_speed * k
    speed[~moving] = 0.0
    return integrate(yaw, speed, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="frontend")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--smooth", type=float, default=SMOOTH_S)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sig = {k: v for k, v in np.load(SIGNAL).items()}
    t_all = sig["t"]
    t1 = float(t_all[-1]) if args.t1 is None else args.t1

    print("P0.4 — траектория методом счисления пути")
    print(f"  окно видео: {args.t0:.0f} .. {t1:.0f} с  ({t1 - args.t0:.0f} с)")
    print(f"  сглаживание {args.smooth} с, порог остановки {STOP_ACTIVITY} px/кадр")
    print(f"  масштаб поворота {DEG_PER_PX} °/px — допущение, откалибровано по поворотам, "
          f"подтверждённым глазами")
    print(f"  масштаб скорости {WALK_SPEED} м/с — допущение; влияет только на масштаб, "
          f"не на форму\n")

    tr = frontend_trajectory(sig, args.t0, t1, args.smooth)

    print(f"=== что получилось ===")
    print(f"  длина пути          {tr['path_length_m']:8.0f} м")
    print(f"  смещение от старта  {tr['net_displacement_m']:8.0f} м")
    print(f"  отношение смещения к пути (замкнутость): {tr['closing_ratio']:.2f}")
    print(f"  суммарный поворот   {tr['total_turning_deg']:8.0f} °")
    print(f"  чистый поворот      {tr['net_turning_deg']:+8.0f} °")
    print(f"  движение           {np.mean(tr['speed'] > 0) * 100:8.0f}% времени")

    # sanity: a walking person at 1.4 m/s for this duration
    dur = t1 - args.t0
    print(f"\n  для сравнения: {dur:.0f} с ходьбы на 1.4 м/с даёт {dur * 1.4:.0f} м пути")

    # write the raw trajectory
    with (OUT / f"trajectory_{args.stage}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_m", "y_m", "heading_deg", "yaw_rate_deg_s", "speed_m_s"])
        for i in range(len(tr["t"])):
            w.writerow([f"{tr['t'][i]:.3f}", f"{tr['x'][i]:.3f}", f"{tr['y'][i]:.3f}",
                        f"{np.rad2deg(tr['heading'][i]) % 360:.2f}",
                        f"{tr['yaw_rate'][i]:.3f}", f"{tr['speed'][i]:.4f}"])

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ax = axes[0][0]
    ax.plot(tr["x"], tr["y"], lw=0.8, color="tab:blue")
    ax.plot(tr["x"][0], tr["y"][0], "o", color="tab:green", ms=9, label="start")
    ax.plot(tr["x"][-1], tr["y"][-1], "s", color="tab:red", ms=9, label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.set_title(f"траектория по счислению, {args.t0:.0f}–{t1:.0f} с\n"
                 f"путь {tr['path_length_m']:.0f} м, смещение "
                 f"{tr['net_displacement_m']:.0f} м, замкнутость "
                 f"{tr['closing_ratio']:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(tr["t"], np.rad2deg(tr["heading"]) % 360, lw=0.8, color="tab:purple")
    ax.set_xlabel("время, с"); ax.set_ylabel("курс, °")
    ax.set_title("курс (угол heading)"); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(tr["t"], tr["yaw_rate"], lw=0.6, color="tab:orange")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("время, с"); ax.set_ylabel("скорость поворота, °/с")
    ax.set_title("скорость поворота, знак: + = камера влево"); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(tr["t"], tr["speed"], lw=0.6, color="tab:green")
    ax.set_xlabel("время, с"); ax.set_ylabel("скорость, м/с")
    ax.set_title("поступательная скорость"); ax.grid(alpha=0.3)

    fig.suptitle("P0.4 — траектория мухи по счислению пути, шкала не измерена", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"trajectory_{args.stage}.png", dpi=130)
    plt.close(fig)

    (OUT / f"report_{args.stage}.json").write_text(json.dumps({
        "stage": args.stage, "t0": args.t0, "t1": t1,
        "assumptions": {"deg_per_px": DEG_PER_PX, "walk_speed_mps": WALK_SPEED,
                        "smooth_s": args.smooth, "stop_activity": STOP_ACTIVITY},
        "caveat": "истинной траектории нет, поэтому абсолютная точность не измерена; "
                  "форма не зависит от масштаба скорости и слабо зависит от масштаба поворота",
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else None)
                    for k, v in tr.items()
                    if k not in ("t", "x", "y", "heading", "yaw_rate", "speed")},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/trajectory_{args.stage}.png")


if __name__ == "__main__":
    main()
