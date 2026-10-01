#!/usr/bin/env python3
"""P08.4A — the old route and the new one, drawn on the same plan.

Numbers say the sequence lost two edges and kept the same end; a picture says whether the
route through the building is actually the same. Old in one colour, new in another, both
over the plan, so a difference is a difference you can point at.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def polyline(traj: list[dict]) -> tuple[list[float], list[float]]:
    return [float(r["x_px"]) for r in traj], [float(r["y_px"]) for r in traj]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default=str(ROOT / "output/p084a/OLD_graph_trajectory.csv"))
    ap.add_argument("--new", default=str(ROOT / "output/p084a/new/graph_trajectory.csv"))
    ap.add_argument("--plan", default=str(ROOT / "data/p08/plan.png"))
    ap.add_argument("--graph-old", default=str(ROOT / "data/p08/graph_before_p084a.json"))
    ap.add_argument("--graph-new", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--out", default=str(ROOT / "output/p084a/route_before_after.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plan = Image.open(args.plan).convert("L")
    W, H = plan.size
    go = json.loads(Path(args.graph_old).read_text(encoding="utf-8"))
    gn = json.loads(Path(args.graph_new).read_text(encoding="utf-8"))

    ox, oy = polyline(rows(Path(args.old)))
    nx, ny = polyline(rows(Path(args.new)))

    fig, ax = plt.subplots(figsize=(16, 16 * H / W), dpi=110)
    ax.imshow(plan, cmap="gray", vmin=0, vmax=255)

    def pos(g, nid):
        n = [x for x in g["nodes"] if x["id"] == nid]
        return (n[0]["x"] * g["img_w"], n[0]["y"] * g["img_h"]) if n else None

    # the edges that exist in one graph and not the other, so the repair is visible
    eo = {tuple(sorted((e["from"], e["to"]))) for e in go["edges"]}
    en = {tuple(sorted((e["from"], e["to"]))) for e in gn["edges"]}
    for pair in sorted(eo - en):
        a, b = pos(go, pair[0]), pos(go, pair[1])
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#ff4d4d", lw=3, alpha=0.75, zorder=3)
    for pair in sorted(en - eo):
        a, b = pos(gn, pair[0]), pos(gn, pair[1])
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#3ddc84", lw=3, alpha=0.75, zorder=3)

    ax.plot(ox, oy, color="#ff9f1c", lw=5, alpha=0.9, zorder=4, label="маршрут ДО починки")
    ax.plot(nx, ny, color="#2ec4ff", lw=2.5, alpha=1.0, zorder=5, label="маршрут ПОСЛЕ починки")

    for nid, dx, dy in (("J37", 40, -40), ("T54", 40, 40), ("T50", 40, -40), ("T14", 40, 40)):
        for g, colour, tag in ((go, "#ff9f1c", "до"), (gn, "#2ec4ff", "после")):
            p = pos(g, nid)
            if p:
                ax.plot([p[0]], [p[1]], "o", color=colour, ms=9,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)
        p = pos(gn, nid) or pos(go, nid)
        if p:
            ax.annotate(nid, (p[0] + dx, p[1] + dy), color="#ffffff", fontsize=13,
                        weight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#000000", ec=colour, alpha=0.7))

    # frame the part of the plan the route actually uses
    pad = 120
    ax.set_xlim(max(0, min(min(ox), min(nx)) - pad), min(W, max(max(ox), max(nx)) + pad))
    ax.set_ylim(min(H, max(max(oy), max(ny)) + pad), max(0, min(min(oy), min(ny)) - pad))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("P08.4A — маршрут VID00006 до и после починки графа\n"
                 "красное — рёбра, которых больше нет; зелёное — рёбра, которых не было",
                 fontsize=15)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.85)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, facecolor="#101216")
    print(f"{args.out}  {fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
