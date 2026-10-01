#!/usr/bin/env python3
"""P08.4B — what the angles at this graph's junctions actually look like.

The claim being tested is that a 45 degree boundary was throwing away the numbers that
decide a junction. That is a claim about the graph, not about the fly, so it can be settled
by measuring the graph: at every junction, for every way in, what are the ways on.

What the measurement is for
---------------------------
A boundary at 45 degrees is only harmless if almost nothing falls just under it. If the
angles cluster near 0 or spread near 90 the boundary never mattered much and the phase is
housekeeping. If a good number sit between 15 and 45 degrees, then the old rule was calling
real turns "straight on", and two such ways on at the same junction were indistinguishable to
it by construction. The histogram answers that directly, and the audit lists the passages
that change caption.

Two numbers are counted separately and must not be confused:

  * how many passages the old boundary labelled STRAIGHT but which are 15 to 45 degrees off.
    These are real turns that were being treated as carrying straight on.
  * how many junctions offer two ways on closer than 15 degrees to each other. These are the
    GEOMETRY_AMBIGUOUS cases, where no signal can help because the map does not distinguish
    them either.

The second is expected to be small. If it is not, the graph needs drawing, not the rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p084b_geometry as geo  # noqa: E402
import p08_graph as G  # noqa: E402

OLD_STRAIGHT_DEG = 45.0     # the boundary this phase removes from the decision
OLD_BACK_DEG = 135.0


def old_side(deg: float) -> str:
    """The caption the P08 rule decided on, kept here only to be counted against."""
    ad = abs(deg)
    if ad < OLD_STRAIGHT_DEG:
        return "STRAIGHT"
    if ad > OLD_BACK_DEG:
        return "BACK"
    return "RIGHT" if deg > 0 else "LEFT"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--out", default=str(ROOT / "output/p084b"))
    ap.add_argument("--min-degree", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = G.Graph.load(Path(args.graph))

    rows: list[dict] = []
    for node in sorted(g.nodes):
        deg = g.degree(node)
        if deg < args.min_degree:
            continue
        for in_edge in sorted(g.edges_at(node)):
            if not g.can_travel(in_edge, node):
                continue
            for c in g.classify_candidates(node, in_edge, allow_back=True):
                if c["edge"] == in_edge:
                    continue
                a = c["deg"]
                new = geo.display_side(a)
                old = old_side(a)
                rows.append({
                    "node": node,
                    "node_degree": deg,
                    "in_edge": in_edge,
                    "out_edge": c["edge"],
                    "angle_deg": round(a, 3),
                    "abs_angle_deg": round(abs(a), 3),
                    "caption_new": new,
                    "caption_old": old,
                    "was_straight_now_turn": int(old == "STRAIGHT" and new != "STRAIGHT"),
                    "is_reversal": int(geo.is_reversal(a)),
                    "geometry_score": round(geo.geometry_score(a), 4),
                    "back_penalty": round(geo.back_penalty(a), 4),
                })

    fields = ["node", "node_degree", "in_edge", "out_edge", "angle_deg", "abs_angle_deg",
              "caption_new", "caption_old", "was_straight_now_turn", "is_reversal",
              "geometry_score", "back_penalty"]
    with (out / "graph_angle_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summaries
    nodes = sorted({r["node"] for r in rows})
    counts_new = Counter(r["caption_new"] for r in rows)
    counts_old = Counter(r["caption_old"] for r in rows)
    reclassified = [r for r in rows if r["was_straight_now_turn"]]

    # per (node, in_edge): the two straightest ways on, and how close they are
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["node"], r["in_edge"]), []).append(r)
    near_pairs = []
    for (node, in_edge), grp in groups.items():
        on = sorted([r for r in grp if not r["is_reversal"]],
                    key=lambda r: r["abs_angle_deg"])
        if len(on) < 2:
            continue
        gap = abs(on[0]["angle_deg"] - on[1]["angle_deg"])
        if gap < geo.AMBIGUOUS_DEG:
            near_pairs.append({"node": node, "in_edge": in_edge, "gap_deg": round(gap, 2),
                               "a": on[0]["out_edge"], "angle_a": on[0]["angle_deg"],
                               "b": on[1]["out_edge"], "angle_b": on[1]["angle_deg"]})

    print("=" * 88)
    print("P08.4B — АУДИТ УГЛОВ ВСЕГО ГРАФА")
    print("=" * 88)
    print(f"граф: {len(g.nodes)} узлов, {len(g.edges)} рёбер")
    print(f"узлов степени >= {args.min_degree}: {len(nodes)}")
    print(f"пар (узел, вход): {len(groups)}   записей (выход): {len(rows)}")
    print()

    print("─── подписи выходов: новый порог (15°/135°) против старого (45°/135°) ───")
    for cap in ("STRAIGHT", "LEFT", "RIGHT", "BACK"):
        print(f"  {cap:<10} было {counts_old.get(cap, 0):>6}   стало {counts_new.get(cap, 0):>6}")
    print()

    print(f"─── выходы, которые старый порог звал STRAIGHT, а они поворот: "
          f"{len(reclassified)} из {len(rows)} ({100*len(reclassified)/max(len(rows),1):.1f}%) ───")
    band35 = [r for r in reclassified if r["abs_angle_deg"] <= 35]
    print(f"    из них в полосе 15°..35° (самая спорная): {len(band35)}")
    worst = sorted(reclassified, key=lambda r: r["abs_angle_deg"])[:15]
    for r in worst:
        print(f"    {r['node']:<6} {r['in_edge']:<14} → {r['out_edge']:<14} "
              f"{r['angle_deg']:>+7.1f}°")
    print()

    print(f"─── развилки с двумя выходами ближе {geo.AMBIGUOUS_DEG}°: {len(near_pairs)} ───")
    for p in sorted(near_pairs, key=lambda x: x["gap_deg"])[:20]:
        print(f"    {p['node']:<6} вход {p['in_edge']:<14} "
              f"{p['a']} {p['angle_a']:+.1f}°  против  {p['b']} {p['angle_b']:+.1f}°"
              f"   разница {p['gap_deg']:.1f}°")
    print()

    # Where the old boundary could not help: a junction whose two straightest ways on both
    # fell under 45 degrees, so the old rule saw one direction and could not choose between
    # them by geometry at all.
    both_old_straight = []
    for (node, in_edge), grp in groups.items():
        on = sorted([r for r in grp if not r["is_reversal"]],
                    key=lambda r: r["abs_angle_deg"])
        if len(on) >= 2 and all(r["abs_angle_deg"] < OLD_STRAIGHT_DEG for r in on[:2]):
            both_old_straight.append({"node": node, "in_edge": in_edge,
                                      "a": on[0]["out_edge"], "angle_a": on[0]["angle_deg"],
                                      "b": on[1]["out_edge"], "angle_b": on[1]["angle_deg"],
                                      "gap": round(abs(on[0]["angle_deg"] -
                                                       on[1]["angle_deg"]), 2)})
    print(f"─── развилки, где ОБА лучших выхода были < 45° (старое правило их не различало): "
          f"{len(both_old_straight)} ───")
    for p in sorted(both_old_straight, key=lambda x: x["gap"])[:20]:
        print(f"    {p['node']:<6} вход {p['in_edge']:<14} {p['a']} {p['angle_a']:+.1f}°"
              f"  против  {p['b']} {p['angle_b']:+.1f}°   разница {p['gap']:.1f}°")
    print()

    # ---------------------------------------------------------------- picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), dpi=110)
    fig.patch.set_facecolor("#101216")
    for ax in axes:
        ax.set_facecolor("#101216")
        ax.tick_params(colors="#c8cede")
        for s in ax.spines.values():
            s.set_color("#39415a")

    ax = axes[0]
    abs_angles = [r["abs_angle_deg"] for r in rows]
    ax.hist(abs_angles, bins=36, range=(0, 180), color="#3d7dd8", edgecolor="#101216")
    ax.axvline(geo.STRAIGHT_DISPLAY_DEG, color="#3ddc84", lw=2.5,
               label=f"новый порог подписи {geo.STRAIGHT_DISPLAY_DEG:.0f}°")
    ax.axvline(OLD_STRAIGHT_DEG, color="#ff9f1c", lw=2.5, ls="--",
               label=f"старый порог решения {OLD_STRAIGHT_DEG:.0f}°")
    band = [a for a in abs_angles if geo.STRAIGHT_DISPLAY_DEG < a <= OLD_STRAIGHT_DEG]
    ax.axvspan(geo.STRAIGHT_DISPLAY_DEG, OLD_STRAIGHT_DEG, color="#ff4d4d", alpha=0.18)
    ax.set_title(f"углы выходов на развилках\n{len(band)} выходов в полосе 15°–45°: "
                 f"старое правило звало их «прямо»", color="#ffffff", fontsize=12)
    ax.set_xlabel("|отклонение от курса|, градусы", color="#c8cede")
    ax.set_ylabel("число выходов", color="#c8cede")
    ax.legend(facecolor="#1a1f2b", edgecolor="#39415a", labelcolor="#e8ecf5", fontsize=10)

    ax = axes[1]
    caps = ["STRAIGHT", "LEFT", "RIGHT", "BACK"]
    x = range(len(caps))
    oldv = [counts_old.get(c, 0) for c in caps]
    newv = [counts_new.get(c, 0) for c in caps]
    ax.bar([i - 0.2 for i in x], oldv, width=0.4, color="#ff9f1c", label="старый порог 45°")
    ax.bar([i + 0.2 for i in x], newv, width=0.4, color="#3ddc84", label="новый порог 15°")
    ax.set_xticks(list(x))
    ax.set_xticklabels(caps, color="#c8cede")
    ax.set_title("подпись выхода: было против стало\n(подпись ничего не решает)",
                 color="#ffffff", fontsize=12)
    ax.set_ylabel("число выходов", color="#c8cede")
    for i, (o, n) in enumerate(zip(oldv, newv)):
        ax.text(i - 0.2, o, str(o), ha="center", va="bottom", color="#ff9f1c", fontsize=10)
        ax.text(i + 0.2, n, str(n), ha="center", va="bottom", color="#3ddc84", fontsize=10)
    ax.legend(facecolor="#1a1f2b", edgecolor="#39415a", labelcolor="#e8ecf5", fontsize=10)

    ax = axes[2]
    gaps = []
    for grp in groups.values():
        on = sorted([r for r in grp if not r["is_reversal"]],
                    key=lambda r: r["abs_angle_deg"])
        if len(on) >= 2:
            gaps.append(abs(on[0]["angle_deg"] - on[1]["angle_deg"]))
    ax.hist(gaps, bins=36, range=(0, 180), color="#b07de0", edgecolor="#101216")
    ax.axvline(geo.AMBIGUOUS_DEG, color="#ff4d4d", lw=2.5,
               label=f"порог неразличимости {geo.AMBIGUOUS_DEG:.0f}°")
    ax.axvspan(0, geo.AMBIGUOUS_DEG, color="#ff4d4d", alpha=0.18)
    ax.set_title(f"разница между двумя самыми прямыми выходами\n"
                 f"{len(near_pairs)} развилок ниже порога: сигнал их не различит",
                 color="#ffffff", fontsize=12)
    ax.set_xlabel("разница углов, градусы", color="#c8cede")
    ax.set_ylabel("число пар (узел, вход)", color="#c8cede")
    ax.legend(facecolor="#1a1f2b", edgecolor="#39415a", labelcolor="#e8ecf5", fontsize=10)

    fig.tight_layout()
    fig.savefig(out / "angle_distribution.png", dpi=110, facecolor="#101216")

    Path(out / "graph_angle_audit_summary.json").write_text(json.dumps({
        "graph": args.graph,
        "nodes": len(g.nodes), "edges": len(g.edges),
        "choice_nodes": len(nodes), "pairs_node_in": len(groups), "exit_rows": len(rows),
        "captions_old": dict(counts_old), "captions_new": dict(counts_new),
        "was_straight_now_turn": len(reclassified),
        "was_straight_now_turn_15_35": len(band35),
        "ambiguous_pairs": len(near_pairs),
        "both_best_under_45": len(both_old_straight),
        "ambiguous_detail": near_pairs,
        "both_best_under_45_detail": both_old_straight,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"записано: {out/'graph_angle_audit.csv'}")
    print(f"записано: {out/'graph_angle_audit_summary.json'}")
    print(f"записано: {out/'angle_distribution.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
