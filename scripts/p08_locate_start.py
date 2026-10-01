#!/usr/bin/env python3
"""
P08 — where did the walk start? A hypothesis, and how it is scored.

There is no recorded start. The graph has 209 passages and the walk began on one of them
in one of two directions, and without that the tracker has nowhere to begin.

What is scored, and why it is not circular
------------------------------------------
The score uses only the agreement between two things that were produced independently:

    the graph, drawn by hand from the building
    the yaw signal, which comes from the frozen P07 pipeline and knows nothing about the
    building

If the walk is placed on the wrong passage, then minutes later it will arrive at junctions
in the wrong order, and at those junctions the fly will report a turn for which no passage
exists — it will say left inside a place the graph says has no left. Placed correctly, those
contradictions should be rare, because a person walking a building does not turn into
walls.

So the score is the rate of contradiction: the fraction of junctions where the signal names
a side and the graph has no passage on that side. The threshold used for naming a side here
is deliberately lower than the tracker's, 0.20 against 0.40, because the contradictions of
interest are exactly the occasions that fall between the two; the tracker's own behaviour
is unchanged.

What this is not
----------------
It is a hypothesis, not ground truth. Nothing here has been compared against where the
walker actually was, because nobody has recorded that. The result is a ranked list, and it
should be confirmed by eye against the video before a route built from it is believed. Two
independent readings are reported for each candidate: the contradiction rate, and how much
of the route the fly had to be asked about, since a start that sails through long corridors
asking nothing has a low rate for the wrong reason.

Usage:
    PYTHONPATH=. python scripts/p08_locate_start.py
    PYTHONPATH=. python scripts/p08_locate_start.py --video VID00002 --top 15
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
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402
from p08_graph_tracker import load_signals, relative_speed, track  # noqa: E402

OUT = ROOT / "output/p08"
DIAG_THRESHOLD = 0.20     # lower than the tracker's 0.40, to see the near misses too
MIN_SIGNAL_DECISIONS = 15  # below this a rate is too noisy to rank on


def score(decisions: list[dict], g: Graph, seq: list[dict]) -> dict:
    """Contradiction rate between the named direction and the passages that exist."""
    n_choice = n_signal = n_match = 0
    for d in decisions:
        sides = [s for s in d["candidate_sides"].split("|") if s]
        if len(sides) < 2:
            continue
        n_choice += 1
        I = d["yaw_integral"]
        if abs(I) < DIAG_THRESHOLD:
            continue
        n_signal += 1
        # the documented convention: positive integral means the camera turned left
        want = "LEFT" if I > 0 else "RIGHT"
        if want in sides:
            n_match += 1
    rate = 1.0 - n_match / n_signal if n_signal else float("nan")
    dist = g.route_meters([s["edge"] for s in seq])
    return {
        "n_choice": n_choice,
        "n_signal": n_signal,
        "n_match": n_match,
        "contradiction_rate": rate,
        "n_edges": len(seq),
        "n_back": sum(1 for d in decisions if d["reason"] == "BACK"),
        "distinct_edges": len({s["edge"] for s in seq}),
        "route_meters": dist,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID00001")
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--t1", type=float, default=None, help="restrict to the first N seconds")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    g = Graph.load(args.graph)
    t, yaw, tf, speed, alt = load_signals(args.video, "yaw_signal_deadband")
    if args.t1 is not None:
        m = t <= args.t1
        t, yaw, speed = t[m], yaw[m], speed[m]
    rel = relative_speed(speed)

    starts = []
    for eid, e in g.edges.items():
        starts.append({"type": "edge", "edge": eid, "from": e["from"], "to": e["to"]})
        if e["bidirectional"]:
            starts.append({"type": "edge", "edge": eid, "from": e["to"], "to": e["from"]})

    print("P08 — подбор старта по согласованности графа с сигналом поворота")
    print(f"  видео {args.video}, {t[-1]:.0f} с, {len(t)} кадров")
    print(f"  граф: {len(g.nodes)} узлов, {len(g.edges)} рёбер, "
          f"развилок {len(g.choice_nodes())}")
    print(f"  перебрано стартов: {len(starts)} (ребро × направление)")
    print(f"  противоречие = сигнал называет сторону, а прохода на этой стороне нет")
    print(f"  порог называния стороны здесь {DIAG_THRESHOLD:.2f}, "
          f"у трекера он 0.40")
    print()
    print("  это гипотеза, а не истина: сверять с видео глазами.")
    print()

    rows = []
    for k, s in enumerate(starts):
        res = track(g, t, yaw, rel, start=s)
        sc = score(res["decisions"], g, res["edge_sequence"])
        sc.update({"start_edge": s["edge"], "from": s["from"], "to": s["to"]})
        rows.append(sc)
        if (k + 1) % 100 == 0:
            print(f"  проверено {k + 1} из {len(starts)}")

    print(f"  проверено {len(starts)}")
    print()

    usable = [r for r in rows if r["n_signal"] >= MIN_SIGNAL_DECISIONS]
    print(f"  стартов с достаточным числом названных сторон "
          f"(>= {MIN_SIGNAL_DECISIONS}): {len(usable)} из {len(rows)}")
    if len(usable) < 10:
        print("  слишком мало данных для ранжирования: сигнал называет сторону редко.")
        print("  тогда подбор старта этим способом не работает, и старт надо указать вручную.")
    print()

    ranked = sorted(usable, key=lambda r: (r["contradiction_rate"], -r["n_signal"]))
    print(f"=== ЛУЧШИЕ {args.top} ПО ДОЛЕ ПРОТИВОРЕЧИЙ ===")
    print(f"  {'ребро':>18s} {'напр.':>9s} {'противореч.':>12s} {'названо':>8s} "
          f"{'совпало':>8s} {'рёбер':>6s} {'разных':>7s} {'метров':>8s} {'BACK':>5s}")
    for r in ranked[:args.top]:
        print(f"  {r['start_edge']:>18s} {r['from'] + '→' + r['to']:>9s} "
              f"{r['contradiction_rate'] * 100:11.0f}% {r['n_signal']:8d} "
              f"{r['n_match']:8d} {r['n_edges']:6d} {r['distinct_edges']:7d} "
              f"{r['route_meters'] if r['route_meters'] else 0:8.0f} {r['n_back']:5d}")
    print()

    print("=== ХУДШИЕ 5 ===")
    for r in ranked[-5:]:
        print(f"  {r['start_edge']:>18s} {r['from'] + '→' + r['to']:>9s} "
              f"{r['contradiction_rate'] * 100:11.0f}% "
              f"(названо {r['n_signal']})")
    print()

    rates = [r["contradiction_rate"] for r in usable]
    print(f"  доля противоречий по всем стартам: медиана {np.median(rates) * 100:.0f}%, "
          f"лучшая {min(rates) * 100:.0f}%, худшая {max(rates) * 100:.0f}%")
    if usable:
        best = ranked[0]
        spread = max(rates) - min(rates)
        print(f"  разброс между стартами: {spread * 100:.0f} процентных пунктов")
        if spread < 0.15:
            print("  разброс мал: сигнал почти не различает старты, "
                  "и это не надёжный способ локализации.")
        else:
            print("  разброс заметный: у стартов действительно разная согласованность.")
    print()

    with (OUT / f"start_candidates_{args.video}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (
            r["contradiction_rate"] if np.isfinite(r["contradiction_rate"]) else 9)))

    (OUT / "start_localization.json").write_text(json.dumps({
        "video": args.video,
        "diag_threshold": DIAG_THRESHOLD,
        "min_signal_decisions": MIN_SIGNAL_DECISIONS,
        "n_starts_tested": len(starts),
        "n_usable": len(usable),
        "best": ranked[:args.top],
        "note": ("Гипотеза, не истина. Считается только согласованность между графом, "
                 "нарисованным человеком, и сигналом поворота из P07. Никакого "
                 "известного положения на местности не использовано."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}/start_candidates_{args.video}.csv and start_localization.json")


if __name__ == "__main__":
    main()
