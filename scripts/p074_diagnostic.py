#!/usr/bin/env python3
"""
P07.4 — the diagnostic figure the task asks for: camera against heading.

Four panels per clip, all on one time axis where time matters:

    camera_yaw      the accumulated yaw, which is where the camera points
    body_heading    the same curve with looks removed, which is where the walker goes
    events          each swing marked LOOK or TURN
    trajectories    the route before and after the separation, drawn separately

The point of putting camera and heading on the same axes is that the difference between
them is exactly the set of excursions that were classified as looks. Where the two curves
run together the walker was going the way the camera pointed; where the heading leaves the
camera out and comes back, a look has been removed.

Usage:
    PYTHONPATH=. python scripts/p074_diagnostic.py
    PYTHONPATH=. python scripts/p074_diagnostic.py --video VID00006
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

OUT = ROOT / "output/p074"
P072 = ROOT / "output/p072"


def read(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    out = {}
    for c in rows[0].keys():
        try:
            out[c] = np.array([float(r[c]) for r in rows])
        except (ValueError, TypeError):
            out[c] = np.array([r[c] for r in rows])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    clips = [args.video] if args.video else list(report["videos"].keys())

    for v in clips:
        tr = read(OUT / f"trajectory_{v}.csv")
        sw = read(OUT / f"swings_{v}.csv")
        rep = report["videos"][v]

        # the heading that would follow every swing, for reference
        looks = sw["kind"] == "LOOK"
        turns = sw["kind"] == "TURN"

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25],
                              hspace=0.30, wspace=0.24)

        # --- camera against heading --------------------------------------
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(tr["t"], tr["camera_yaw"], lw=0.8, color="tab:blue",
                label="camera_yaw — куда смотрит камера")
        # heading in the same units: convert degrees back to yaw units
        scale = report["scale"]
        ax.plot(tr["t"], tr["theta_after"] / scale, lw=1.2, color="tab:green",
                label="body_heading — куда идёт человек (взгляды убраны)")
        for k in np.flatnonzero(looks):
            ax.axvspan(sw["t0"][k], sw["t1"][k], color="tab:orange", alpha=0.35)
        for k in np.flatnonzero(turns):
            ax.axvspan(sw["t0"][k], sw["t1"][k], color="tab:purple", alpha=0.25)
        ax.set_ylabel("накопленный поворот, единицы")
        ax.set_xlabel("время, с")
        ax.set_title(f"{v} — камера против направления движения\n"
                     f"оранжевым взгляды ({rep['n_look']}), фиолетовым повороты "
                     f"({rep['n_turn']})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # --- heading in degrees ------------------------------------------
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(tr["t"], tr["theta_before"], lw=0.8, color="0.65",
                label="курс: было (все события)")
        ax.plot(tr["t"], tr["theta_after"], lw=1.2, color="tab:green",
                label="курс: стало (только повороты)")
        ax.set_xlabel("время, с")
        ax.set_ylabel("курс, градусы (условные)")
        ax.set_title(f"курс\n{rep['turn_deg_before']:.0f}° -> "
                     f"{rep['turn_deg_after']:.0f}°")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # --- trajectory before -------------------------------------------
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(tr["x_before"], tr["y_before"], lw=0.9, color="0.6")
        ax.plot(tr["x_before"][0], tr["y_before"][0], "o", color="tab:green", ms=8)
        ax.plot(tr["x_before"][-1], tr["y_before"][-1], "s", color="tab:red", ms=8)
        ax.set_aspect("equal")
        ax.set_title(f"траектория ДО\nпрямизна {rep['straightness_before']:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

        # --- trajectory after --------------------------------------------
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(tr["x_after"], tr["y_after"], lw=0.9, color="tab:green")
        ax.plot(tr["x_after"][0], tr["y_after"][0], "o", color="tab:green", ms=8)
        ax.plot(tr["x_after"][-1], tr["y_after"][-1], "s", color="tab:red", ms=8)
        ax.set_aspect("equal")
        ax.set_title(f"траектория ПОСЛЕ (взгляды убраны)\n"
                     f"прямизна {rep['straightness_after']:.3f}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

        # --- swing sizes --------------------------------------------------
        ax = fig.add_subplot(gs[1, 2])
        order = np.argsort(sw["t0"])
        col = ["tab:orange" if sw["kind"][i] == "LOOK" else
               ("tab:purple" if sw["kind"][i] == "TURN" else "0.7") for i in order]
        ax.bar(np.arange(len(order)), sw["size"][order], color=col)
        ax.set_xlabel("качание, по времени")
        ax.set_ylabel("величина, единицы")
        ax.set_title(f"качания: оранжевое — взгляд, фиолетовое — поворот\n"
                     f"взглядов {rep['n_look']}, поворотов {rep['n_turn']}, "
                     f"неясных {rep['n_inconclusive']}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle("P07.4 — взгляд камеры против поворота маршрута. "
                     "Взгляды не меняют курс.", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"diagnostic_{v}.png", dpi=120)
        plt.close(fig)
        print(f"  {v}: diagnostic_{v}.png")

    # ---- one summary figure across clips -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = list(report["videos"])
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [report["videos"][n]["n_look"] for n in names], 0.4,
                color="tab:orange", label="взгляды")
    axes[0].bar(x + 0.2, [report["videos"][n]["n_turn"] for n in names], 0.4,
                color="tab:purple", label="повороты")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, fontsize=8)
    axes[0].set_ylabel("событий")
    axes[0].set_title("сколько взглядов и поворотов найдено")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - 0.2, [report["videos"][n]["turn_deg_before"] for n in names], 0.4,
                color="0.65", label="было")
    axes[1].bar(x + 0.2, [report["videos"][n]["turn_deg_after"] for n in names], 0.4,
                color="tab:green", label="стало")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, fontsize=8)
    axes[1].set_ylabel("суммарный поворот, градусы")
    axes[1].set_title("сколько поворота убрано вместе со взглядами")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("P07.4 — итог по роликам")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "summary.png", dpi=120)
    plt.close(fig)
    print(f"  summary.png")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
