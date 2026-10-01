#!/usr/bin/env python3
"""
P08 — figures for the graph route, and the comparison with the free P07 trajectory.

What is drawn, and what the comparison does not claim
-----------------------------------------------------
Stage 25 asks for P07 and P08 side by side. They cannot be drawn on the same axes: P07's
coordinates come from integrating an arbitrary speed, so they carry no metric meaning, and
P08's come from the plan of the room, so their shape is the plan's. Overlaying them would
put two unrelated coordinate systems on one pair of axes and invite a reading that does
not exist. So they are drawn one above the other, each in its own frame, and what the
comparison is worth is stated in the caption: the question is the sequence of turns, not
the shape of the line.

Usage:
    PYTHONPATH=. python scripts/p08_plot.py
    PYTHONPATH=. python scripts/p08_plot.py --video VID00002
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402

OUT = ROOT / "output/p08"
P071 = ROOT / "output/p071"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def draw_plan(ax, g: Graph, plan_path: Path) -> None:
    if plan_path.exists():
        img = plt.imread(plan_path)
        ax.imshow(img, extent=(0, g.img_w, g.img_h, 0), alpha=0.55, zorder=0)
    for eid, e in g.edges.items():
        a, b = g.pos(e["from"]), g.pos(e["to"])
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#3d6b8f", lw=1.2, zorder=1)
    for nid, n in g.nodes.items():
        x, y = g.pos(nid)
        ax.plot([x], [y], "o", color="#5cc8ff", ms=5, zorder=3)
        ax.text(x + 12, y - 10, nid, color="#1a1d26", fontsize=9, zorder=4)
    ax.set_xlim(0, g.img_w)
    ax.set_ylim(g.img_h, 0)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID00001")
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    g = Graph.load(args.graph)
    seq = read_csv(OUT / "edge_sequence.csv")
    dec = read_csv(OUT / "decisions.csv")
    path = read_csv(OUT / "graph_trajectory.csv")
    p07 = read_csv(P071 / f"trajectory_{args.video}.csv")
    plan = ROOT / "data/p08" / (g.doc.get("image") or "plan.png")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.0], hspace=0.32, wspace=0.18)

    # ---- the route over the plan ----------------------------------------
    ax = fig.add_subplot(gs[0, :])
    draw_plan(ax, g, plan)
    if path:
        x = np.array([float(r["x_px"]) for r in path])
        y = np.array([float(r["y_px"]) for r in path])
        ax.plot(x, y, color="#f97316", lw=2.0, alpha=0.85, zorder=5,
                label="путь по графу (P08)")
        ax.plot([x[0]], [y[0]], "o", color="#22c55e", ms=13, zorder=6,
                label="начало")
        ax.plot([x[-1]], [y[-1]], "o", color="#ef4444", ms=13, zorder=6,
                label="конец")
    for s in seq:
        a, b = g.pos(s["from"]), g.pos(s["to"])
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color="#f97316", lw=1.6,
                                    shrinkA=14, shrinkB=14), zorder=4)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"P08 — маршрут как проход по графу · {args.video} · "
                 f"рёбер в последовательности: {len(seq)}", fontsize=11)

    # ---- edge sequence over time ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    names = []
    for i, s in enumerate(seq):
        t0, t1 = float(s["time_start"]), float(s["time_end"])
        lab = f"{s['from']}→{s['to']}"
        ax.barh(i, t1 - t0, left=t0, height=0.7, color="#3d6b8f")
        if len(seq) <= 60:
            ax.text(t1 + 1, i, lab, va="center", fontsize=8)
        names.append(lab)
    ax.set_yticks(range(len(seq)))
    ax.set_yticklabels([f"{i + 1}" for i in range(len(seq))], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("время, с")
    ax.set_title("последовательность рёбер", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # ---- yaw at each decision -------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if dec:
        ys = [float(d["yaw_integral"]) for d in dec]
        cols = []
        for d in dec:
            r = d["reason"]
            cols.append({"ONLY_OPTION": "0.7", "STRAIGHT": "#94a3b8",
                         "MALE_LEFT": "#60a5fa", "MALE_RIGHT": "#fb923c",
                         "BACK": "#f87171", "DELAYED_DECISION": "#fbbf24",
                         "RECOVERY": "#a78bfa"}.get(r, "0.5"))
        ax.bar(range(len(dec)), ys, color=cols)
        ax.axhline(0.40, color="k", ls="--", lw=1, label="порог 0.40")
        ax.axhline(-0.40, color="k", ls="--", lw=1)
        ax.axhline(1.20, color="#b91c1c", ls=":", lw=1, label="порог разворота")
        ax.axhline(-1.20, color="#b91c1c", ls=":", lw=1)
        ax.set_xticks(range(len(dec)))
        ax.set_xticklabels([f"{float(d['time']):.0f}с" for d in dec], fontsize=7,
                           rotation=45)
        ax.set_ylabel("интеграл сигнала поворота")
        ax.legend(fontsize=8)
    ax.set_title("сигнал на каждой развилке", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ---- P07 free trajectory against P08 --------------------------------
    ax = fig.add_subplot(gs[2, 0])
    if p07:
        x = np.array([float(r["x"]) for r in p07])
        y = np.array([float(r["y"]) for r in p07])
        th = np.array([float(r["theta"]) for r in p07])
        sc = ax.scatter(x, y, c=th, s=1.5, cmap="twilight")
        ax.plot([x[0]], [y[0]], "o", color="#22c55e", ms=10)
        ax.plot([x[-1]], [y[-1]], "o", color="#ef4444", ms=10)
        plt.colorbar(sc, ax=ax, label="курс, рад", fraction=0.045)
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        L = float(np.hypot(np.diff(x), np.diff(y)).sum())
        ax.set_title(f"P07 — свободная траектория\nпрямизна {net / max(L, 1e-9):.3f}, "
                     f"единицы условные", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    draw_plan(ax, g, plan)
    if path:
        x = np.array([float(r["x_px"]) for r in path])
        y = np.array([float(r["y_px"]) for r in path])
        ax.plot(x, y, color="#f97316", lw=2.0, zorder=5)
    ax.set_title("P08 — траектория по графу\nкоординаты — пиксели плана, "
                 "другая система", fontsize=10)

    fig.suptitle("P08 — граф помещения против свободной траектории P07. "
                 "Оси разные: сравним маршрут, а не форму линии", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "trajectory.png", dpi=120)
    fig.savefig(OUT / "comparison.png", dpi=120)
    plt.close(fig)

    # ---- report ----------------------------------------------------------
    reasons: dict[str, int] = {}
    for d in dec:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    notes = [
        "P07 строит линию свободным интегрированием курса, P08 — проходом по графу.",
        "Оси на рисунке разные, потому что единицы разные: у P07 условные, у P08 "
        "пиксели плана.",
        "Прямизна P07 приведена как справка и не является показателем успеха P08.",
        "Главное для P08 — последовательность рёбер и отсутствие невозможных переходов.",
    ]
    problems = g.validate()
    impossible = 0
    for i in range(1, len(seq)):
        prev, cur = seq[i - 1], seq[i]
        if prev["to"] != cur["from"]:
            impossible += 1
    rep = {
        "video": args.video, "graph": args.graph,
        "n_edges_in_sequence": len(seq),
        "n_decisions": len(dec), "reasons": reasons,
        "impossible_transitions": impossible,
        "graph_problems": problems,
        "notes": notes,
    }
    (OUT / "report_plot.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"рисунки: {OUT}/trajectory.png, {OUT}/comparison.png")
    print(f"  рёбер в последовательности {len(seq)}, решений {len(dec)}, "
          f"невозможных переходов {impossible}")
    for n in notes:
        print(f"  · {n}")


if __name__ == "__main__":
    main()
