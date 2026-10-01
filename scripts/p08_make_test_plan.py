#!/usr/bin/env python3
"""
P08 — a placeholder floor plan, so that the editor and the tracker can be exercised
before a real plan of the room is drawn on.

This is not a plan of the actual room. It is a synthetic warehouse-shaped image with
corridors laid out to match the small test graph the stage description uses:

        C
        |
A ------B------ D
                |
                E

The image lands in `data/p08/plan.png` and a matching `data/p08/graph.json` is written
only when that file does not already exist, so a graph drawn by hand is never
overwritten by running this. `--force` overrides that.

Usage:
    PYTHONPATH=. python scripts/p08_make_test_plan.py
    PYTHONPATH=. python scripts/p08_make_test_plan.py --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/p08"

W, H = 1600, 1000

# normalized positions, chosen so the graph matches the drawing above
NODES = [("A", 0.10, 0.50), ("B", 0.40, 0.50), ("C", 0.40, 0.20),
         ("D", 0.75, 0.50), ("E", 0.75, 0.80)]
EDGES = [("A_B", "A", "B"), ("B_C", "B", "C"), ("B_D", "B", "D"), ("D_E", "D", "E")]


def corridor(ax, p, q, width_px=110):
    """A floor-coloured strip between two normalized points, i.e. a walkable lane."""
    x0, y0 = p[0] * W, (1 - p[1]) * H
    x1, y1 = q[0] * W, (1 - q[1]) * H
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) > abs(dy):
        ax.add_patch(mpatches.Rectangle((min(x0, x1), y0 - width_px / 2),
                                        abs(dx), width_px,
                                        fc="#e9edf2", ec="#cfd6e0", lw=1, zorder=2))
    else:
        ax.add_patch(mpatches.Rectangle((x0 - width_px / 2, min(y0, y1)),
                                        width_px, abs(dy),
                                        fc="#e9edf2", ec="#cfd6e0", lw=1, zorder=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                   help="overwrite data/p08/graph.json if it exists")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # image convention, y grows downward
    ax.set_facecolor("#b9c2cf")

    # outer walls
    ax.add_patch(mpatches.Rectangle((30, 30), W - 60, H - 60, fc="#dfe5ec",
                                    ec="#7b8494", lw=6, zorder=1))

    d = {n: (x, y) for n, x, y in NODES}
    for _, a, b in EDGES:
        corridor(ax, d[a], d[b])

    # racks and equipment, placed away from the corridors so they read as obstacles
    rng = np.random.default_rng(0)
    for _ in range(38):
        x = rng.uniform(60, W - 60)
        y = rng.uniform(60, H - 60)
        # keep clear of every corridor
        ok = True
        for _, a, b in EDGES:
            (x0, y0), (x1, y1) = d[a], d[b]
            px0, py0 = x0 * W, (1 - y0) * H
            px1, py1 = x1 * W, (1 - y1) * H
            dx, dy = px1 - px0, py1 - py0
            L2 = dx * dx + dy * dy or 1
            t = max(0.0, min(1.0, ((x - px0) * dx + (y - py0) * dy) / L2))
            if np.hypot(px0 + t * dx - x, py0 + t * dy - y) < 150:
                ok = False
                break
        if not ok:
            continue
        w = rng.uniform(70, 150)
        h = rng.uniform(45, 90)
        ax.add_patch(mpatches.Rectangle((x - w / 2, y - h / 2), w, h,
                                        fc="#9aa6b6", ec="#79838f", lw=1, zorder=3))

    ax.text(W / 2, 62, "ЗАГЛУШКА — это не план реального помещения",
            ha="center", va="center", fontsize=20, color="#5a6472", zorder=5)
    ax.text(W / 2, H - 48, "загрузите настоящий план через редактор: /graph",
            ha="center", va="center", fontsize=15, color="#5a6472", zorder=5)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    path = OUT_DIR / "plan.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"план-заглушка записан: {path} ({W}x{H})")

    graph_path = OUT_DIR / "graph.json"
    if graph_path.exists() and not args.force:
        print(f"{graph_path} уже существует — не трогаю (--force чтобы перезаписать)")
        return
    doc = {
        "image": "plan.png",
        "placeholder": True,
        "nodes": [{"id": n, "x": x, "y": y} for n, x, y in NODES],
        "edges": [{"id": i, "from": a, "to": b, "bidirectional": True}
                  for i, a, b in EDGES],
        "start": {"type": "edge", "edge": "A_B", "from": "A", "to": "B"},
        "meters_per_pixel": None,
        "known_distance": None,
    }
    graph_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"тестовый граф записан: {graph_path}")
    print("  A_B, B_C, B_D, D_E; старт A → B")


if __name__ == "__main__":
    main()
