#!/usr/bin/env python3
"""
P0.4 — the trajectory, bearing by bearing.

Why the naive integration failed
--------------------------------
Integrating the yaw rate every frame produced a scribble: 1328 m of path and 12865 deg
of total turning, which is 36 full rotations. A person walking through a workshop does
not spin on the spot. The diagnostics show why:

    accumulated |a| over detected turns   11738 px
    accumulated |a| over all frames       21442 px      <- nearly half is not turning

So the signal oscillates even when the course is steady, and position integrates the
heading, so the low-frequency part of that oscillation walks the path around. This is
the ordinary dead-reckoning failure: yaw-rate noise integrates into a random walk in
heading, and the heading error integrates again into position.

What is done instead
--------------------
Only the part of the chain that was actually validated is used. Turn detection scored
96% on direction across the reviewed windows, so the trajectory is built turn by turn:

    a detected turn rotates the heading by (accumulated a) x scale, with its sign
    between turns the heading is held constant and the position advances forward

Nothing is integrated during steady stretches, so their noise cannot accumulate. The
heading is therefore piecewise constant with steps at the turns, which is exactly what
walking through a corridor looks like.

The scale, and what is honest about it
--------------------------------------
There is no ground truth for this clip, so the absolute rotation cannot be measured.
Two things are reported instead:

  * the SHAPE, which needs no scale;
  * the distribution of turn sizes in px, from which a scale can be *argued*. If the
    corridor turns of a workshop are taken to be right angles, the scale follows; that
    assumption is printed, not hidden.

Usage:
    PYTHONPATH=. python scripts/p04_trajectory_turns.py
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
WINDOWS = ROOT / "data/p01r/windows_v3.json"
OUT = ROOT / "output/p04_trajectory"
FPS = 30.0

WALK_SPEED = 1.40        # m/s, sets the forward scale only; the shape is unaffected
DEFAULT_TURN_DEG = 90.0  # assumed angle of a corridor turn, used to fix the yaw scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-deg", type=float, default=DEFAULT_TURN_DEG,
                    help="assumed angle of a typical corridor turn")
    ap.add_argument("--percentile", type=float, default=60.0,
                    help="which percentile of turn sizes is taken to be that angle")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sig = {k: v for k, v in np.load(SIGNAL).items()}
    t = sig["t"]
    a_sig = sig["a"]
    act = sig["act_x"]

    win = json.loads(WINDOWS.read_text(encoding="utf-8"))["windows"]
    turns = [w for w in win if w["kind"] in ("LEFT", "RIGHT")]
    fwd = [w for w in win if w["kind"] == "FWD"]
    stat = [w for w in win if w["kind"] == "STATIC"]

    # accumulated turn size per detected turn, in px
    sizes = []
    for w in turns:
        m = (t >= w["t0"]) & (t <= w["t1"])
        sizes.append(float(np.sum(a_sig[m])))
    sizes = np.array(sizes)
    mag = np.abs(sizes)

    ref = float(np.percentile(mag, args.percentile))
    deg_per_px = args.turn_deg / max(ref, 1e-9)

    print("P0.4 — траектория по поворотам")
    print(f"  обнаружено поворотов: {len(turns)}  (LEFT "
          f"{sum(1 for w in turns if w['kind'] == 'LEFT')}, "
          f"RIGHT {sum(1 for w in turns if w['kind'] == 'RIGHT')})")
    print(f"  участков вперёд: {len(fwd)}, стоянок: {len(stat)}")
    print(f"\n  размеры поворотов, |накопленный a| в px:")
    for p in (10, 25, 50, 75, 90, 100):
        print(f"    p{p:<3d} {np.percentile(mag, p):7.0f}")
    print(f"\n  масштаб: {args.percentile:.0f}-й процентиль ({ref:.0f} px) принят за "
          f"{args.turn_deg:.0f}°")
    print(f"    -> {deg_per_px:.3f} °/px")
    print(f"    предположение: повороты в цеху прямые. Это допущение, не измерение.")

    # ---- build the trajectory -------------------------------------------------
    # Walk the timeline; at a turn, step the heading; otherwise advance straight ahead.
    events: list[dict] = []
    events += [{"t0": w["t0"], "t1": w["t1"], "dtheta": sizes[i] * deg_per_px,
                "kind": w["kind"]}
               for i, w in enumerate(turns)]
    events += [{"t0": w["t0"], "t1": w["t1"], "dtheta": 0.0, "kind": "FWD"}
               for w in fwd]
    events += [{"t0": w["t0"], "t1": w["t1"], "dtheta": 0.0, "kind": "STATIC"}
               for w in stat]
    events.sort(key=lambda e: e["t0"])

    total_turn = float(np.sum([e["dtheta"] for e in events]))
    dtm = float(np.median(np.diff(t)))
    theta = np.pi / 2
    x = y = 0.0
    heading_at = []
    traj = []
    prev_end = 0.0
    for e in events:
        # advance straight from the previous event to this one
        gap = max(0.0, e["t0"] - prev_end)
        if gap > 0 and e["kind"] != "STATIC":
            dist = WALK_SPEED * gap
            x += dist * np.cos(theta)
            y += dist * np.sin(theta)
            traj.append({"t": prev_end, "x": x, "y": y,
                         "heading_deg": np.rad2deg(theta) % 360, "event": "travel"})
        # inside the event
        dur = e["t1"] - e["t0"]
        if e["kind"] in ("LEFT", "RIGHT"):
            seg = int(round(e["dtheta"] / max(abs(e["dtheta"]), 1e-9) * 1))  # sign only
            steps = max(int(round(dur / dtm)), 1)
            for k in range(steps):
                frac = (k + 1) / steps
                th = theta + np.deg2rad(e["dtheta"] * frac)
                # rotate in place; the radius of the turn is not modelled
                x += WALK_SPEED * dtm * np.cos(th) if e["kind"] == "FWD" else 0.0
                y += WALK_SPEED * dtm * np.sin(th) if e["kind"] == "FWD" else 0.0
                traj.append({"t": e["t0"] + k * dtm, "x": x, "y": y,
                             "heading_deg": np.rad2deg(th) % 360, "event": e["kind"]})
            theta += np.deg2rad(e["dtheta"])
            heading_at.append((e["t1"], np.rad2deg(theta) % 360))
        prev_end = max(prev_end, e["t1"])

    arr_x = np.array([p["x"] for p in traj])
    arr_y = np.array([p["y"] for p in traj])

    print(f"\n=== траектория ===")
    print(f"  точек               {len(traj)}")
    print(f"  суммарный поворот   {abs(total_turn):7.0f} °")
    print(f"  чистый поворот      {total_turn:+7.0f} °")
    print(f"  конечная позиция    x={arr_x[-1]:7.0f} м  y={arr_y[-1]:7.0f} м")
    print(f"  смещение от старта  {np.hypot(arr_x[-1], arr_y[-1]):7.0f} м")
    print(f"  размах              x {arr_x.max() - arr_x.min():.0f} м, "
          f"y {arr_y.max() - arr_y.min():.0f} м")

    with (OUT / "trajectory_turns.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "x", "y", "heading_deg", "event"])
        w.writeheader()
        w.writerows(traj)

    # ---- plot ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(arr_x, arr_y, lw=1.0, color="tab:blue", zorder=1)
    for e in events:
        if e["kind"] in ("LEFT", "RIGHT"):
            m = [p for p in traj if e["t0"] - 0.05 <= p["t"] <= e["t1"] + 0.05]
            if m:
                ax.plot([p["x"] for p in m], [p["y"] for p in m],
                        color=("tab:green" if e["kind"] == "LEFT" else "tab:purple"),
                        lw=2.6, zorder=2)
    ax.plot(arr_x[0], arr_y[0], "o", color="tab:green", ms=11, zorder=3, label="старт")
    ax.plot(arr_x[-1], arr_y[-1], "s", color="tab:red", ms=11, zorder=3, label="конец")
    ax.set_aspect("equal")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.set_title(f"траектория по поворотам\n{len(turns)} поворотов, "
                 f"суммарно {abs(total_turn):.0f}°\n"
                 f"масштаб: {deg_per_px:.2f}°/px (допущение)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(mag, bins=24, color="tab:orange")
    ax.axvline(ref, color="tab:red", ls="--",
               label=f"{args.percentile:.0f}% = {ref:.0f} px = {args.turn_deg:.0f}°")
    ax.set_xlabel("размер поворота, |накопленный a|, px")
    ax.set_ylabel("число поворотов")
    ax.set_title("распределение размеров поворотов\nиз него выведен масштаб")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    hd = [p["heading_deg"] for p in traj]
    tt = [p["t"] for p in traj]
    ax.plot(tt, hd, lw=0.9, color="tab:purple")
    ax.set_xlabel("время, с"); ax.set_ylabel("курс, °")
    ax.set_title("курс во времени\nступени = повороты")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    kinds = ["LEFT", "RIGHT", "FWD", "STATIC"]
    cnt = [sum(1 for w in win if w["kind"] == k) for k in kinds]
    ax.bar(kinds, cnt, color=["tab:green", "tab:purple", "tab:blue", "0.6"])
    ax.set_ylabel("окон")
    ax.set_title("из чего состоит ролик")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(tt, arr_x - arr_x[0], lw=0.9, label="x")
    ax.plot(tt, arr_y - arr_y[0], lw=0.9, label="y")
    ax.set_xlabel("время, с"); ax.set_ylabel("смещение, м")
    ax.set_title("накопление смещения")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("P0.4 — траектория по обнаруженным поворотам. "
                 "Абсолютный масштаб не измерен, форма от него не зависит.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "trajectory_turns.png", dpi=130)
    plt.close(fig)

    (OUT / "report_turns.json").write_text(json.dumps({
        "method": "heading steps only at detected turns; nothing integrated during "
                  "steady travel, so its noise cannot accumulate",
        "why_not_naive": "integrating the yaw rate every frame gave 12865 deg of total "
                         "turning and a scribble; accumulated |a| over all frames is "
                         "21442 px against 11738 px over the detected turns, so about "
                         "half the signal is oscillation",
        "assumptions": {"walk_speed_mps": WALK_SPEED, "turn_deg": args.turn_deg,
                        "percentile": args.percentile, "deg_per_px": deg_per_px},
        "caveat": "нет истинной траектории, абсолютная точность не измерена; "
                  "масштаб выведен из допущения, что повороты в цеху прямые",
        "n_turns": len(turns), "n_fwd": len(fwd), "n_static": len(stat),
        "total_turning_deg": abs(total_turn), "net_turning_deg": total_turn,
        "final_x_m": float(arr_x[-1]), "final_y_m": float(arr_y[-1]),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/trajectory_turns.png")


if __name__ == "__main__":
    main()
