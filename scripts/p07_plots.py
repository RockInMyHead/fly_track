#!/usr/bin/env python3
"""
P07 step 7-8 — the figures: trajectory, signals, and the side-by-side comparison.

Three pictures, no numbers that are not in the csv files.

    trajectory.png   the route from MaleCNS alone, start and end marked
    signals.png      yaw, forward and heading against time
    compare.png      the MaleCNS route beside the old frontend route, never merged

The comparison is the point of the exercise. The frontend trajectory was built by P0.4
from the motion measurement; this one is built only from the recorded activity of four
descending neuron types. They are drawn on the same axes but in separate panels, because
overlaying them would suggest an agreement that does not exist.

Usage:
    PYTHONPATH=. python scripts/p07_plots.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p07"
FRONTEND = ROOT / "output/p04_trajectory/trajectory_turns.csv"
VIDEOS = ["VID00001", "VID00002"]


def read_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    out = {}
    for c in rows[0].keys():
        try:
            out[c] = np.array([float(r[c]) for r in rows])
        except (ValueError, TypeError):
            # the frontend file carries an 'event' column with names in it
            out[c] = np.array([r[c] for r in rows])
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {v: read_csv(OUT / f"trajectory_{v}.csv") for v in VIDEOS}
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))

    # ---- trajectory --------------------------------------------------------
    fig, axes = plt.subplots(1, len(VIDEOS), figsize=(7 * len(VIDEOS), 6.4))
    if len(VIDEOS) == 1:
        axes = [axes]
    for ax, v in zip(axes, VIDEOS):
        d = data[v]
        ax.plot(d["x"], d["y"], lw=0.8, color="tab:blue", zorder=1)
        ax.plot(d["x"][0], d["y"][0], "o", color="tab:green", ms=11, zorder=3,
                label="START")
        ax.plot(d["x"][-1], d["y"][-1], "s", color="tab:red", ms=11, zorder=3,
                label="END")
        r = report["videos"][v]
        ax.set_aspect("equal")
        ax.set_xlabel("x (условные единицы)")
        ax.set_ylabel("y (условные единицы)")
        ax.set_title(f"{v} — траектория только из MaleCNS\n"
                     f"путь {r['path_length']:.0f}, смещение "
                     f"{r['net_displacement']:.0f}, прямизна {r['straightness']:.2f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("P07 — траектория из выходов MaleCNS. "
                 "Масштаб условный: абсолютный из одного видео неизвестен.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "trajectory.png", dpi=120)
    plt.close(fig)

    # ---- signals -----------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for v, col in zip(VIDEOS, ("tab:blue", "tab:orange")):
        d = data[v]
        axes[0].plot(d["t"], d["yaw_signal"], lw=0.5, color=col, label=v)
        axes[1].plot(d["t"], d["speed"], lw=0.5, color=col, label=v)
        axes[2].plot(d["t"], np.rad2deg(d["theta"]), lw=0.7, color=col, label=v)
        axes[3].plot(d["t"], d["n_active"], lw=0.4, color=col, label=v)
    for ax, lbl in zip(axes, ["yaw_signal (>0 = камера влево)",
                              "speed = max(0, forward_signal)",
                              "курс, градусы (условные)",
                              "общая активность мозга, для контроля"]):
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("время, с")
    fig.suptitle("P07 — сигналы во времени")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "signals.png", dpi=120)
    plt.close(fig)

    # ---- comparison --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    d = data["VID00001"]
    axes[0].plot(d["x"], d["y"], lw=0.8, color="tab:blue")
    axes[0].plot(d["x"][0], d["y"][0], "o", color="tab:green", ms=11)
    axes[0].plot(d["x"][-1], d["y"][-1], "s", color="tab:red", ms=11)
    axes[0].set_aspect("equal")
    r = report["videos"]["VID00001"]
    axes[0].set_title(f"A. только MaleCNS\nпуть {r['path_length']:.0f}, "
                      f"прямизна {r['straightness']:.2f}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.3)

    if FRONTEND.exists():
        f = read_csv(FRONTEND)
        axes[1].plot(f["x"], f["y"], lw=1.4, color="0.35")
        axes[1].plot(f["x"][0], f["y"][0], "o", color="tab:green", ms=11)
        axes[1].plot(f["x"][-1], f["y"][-1], "s", color="tab:red", ms=11)
        axes[1].set_aspect("equal")
        axes[1].set_title("B. старый фронтенд, для сравнения\n"
                          "(из P0.4, только по измерению движения)")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "нет траектории фронтенда", ha="center")
        axes[1].axis("off")

    fig.suptitle("P07 — сравнение. Панели намеренно раздельные, "
                 "наложение подразумевало бы согласие, которого нет.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "compare.png", dpi=120)
    plt.close(fig)

    print(f"Wrote {OUT}/trajectory.png, signals.png, compare.png")


if __name__ == "__main__":
    main()
