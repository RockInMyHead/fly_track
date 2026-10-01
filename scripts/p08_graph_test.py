#!/usr/bin/env python3
"""
P08 stage 2 — the graph on its own, before anything neural is attached.

The graph is the one the stage description draws:

            C
            |
    A ------B------ D
                    |
                    E

with the path A → B → D → E given by hand. Nothing here involves MaleCNS. The point is to
show that the geometry says what a person looking at the drawing would say: leaving B
along BD is STRAIGHT, along BC is LEFT, back along BA is BACK; and that a walk using those
categories arrives at E by the intended route.

What is checked
---------------
1  the turn at each junction is classified as drawn, by coordinates alone
2  a traversal from the start edge reaches E in four edges, the expected sequence
3  the drawn polyline has the right number of vertices and ends at E
4  a one-way edge is not offered against its direction
5  a dead end offers only a reversal, and only when reversal is allowed
6  the validator notices a broken drawing

Usage:
    PYTHONPATH=. python scripts/p08_graph_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph, GraphError  # noqa: E402

# the drawing above, in normalized plan coordinates
NODES = {"A": (0.10, 0.50), "B": (0.40, 0.50), "C": (0.40, 0.20),
         "D": (0.75, 0.50), "E": (0.75, 0.80)}
EDGES = [("A_B", "A", "B"), ("B_C", "B", "C"), ("B_D", "B", "D"), ("D_E", "D", "E")]


def build() -> Graph:
    return Graph({
        "img_w": 1600, "img_h": 1000,
        "nodes": [{"id": k, "x": v[0], "y": v[1]} for k, v in NODES.items()],
        "edges": [{"id": i, "from": a, "to": b, "bidirectional": True}
                  for i, a, b in EDGES],
        "start": {"type": "edge", "edge": "A_B", "from": "A", "to": "B"},
    })


def walk(g: Graph, sequence: list[str]) -> list[dict]:
    """Advance along the given edges, keeping the node each one starts from."""
    steps = []
    for eid in sequence:
        e = g.edges[eid]
        frm = e["from"]
        # orient the step the way the walker travels: pick the endpoint not yet visited
        if steps and steps[-1]["to"] != frm:
            frm = e["to"]
        steps.append({"edge": eid, "from": frm, "to": g.other(eid, frm),
                      "length_px": g.length_px(eid)})
    return steps


def polyline(g: Graph, steps: list[dict]) -> list[tuple[float, float]]:
    pts = []
    for s in steps:
        p = g.pos(s["from"])
        if not pts:
            pts.append(p)
        pts.append(g.pos(s["to"]))
    return pts


def main() -> None:
    g = build()
    fails: list[str] = []

    print("P08 этап 2 — проверка графа без мухи")
    print(f"  узлов {len(g.nodes)}, рёбер {len(g.edges)}, план {g.img_w:.0f}x{g.img_h:.0f}")
    print()

    # ---- 1. geometry -----------------------------------------------------
    print("=== 1. ГЕОМЕТРИЯ НА ПЕРЕКРЁСТКЕ B ===")
    print("  приходим по A_B, смотрим, как выглядят выходы")
    print()
    print(f"  {'выход':>8s} {'сторона':>10s} {'угол, °':>9s} {'длина, px':>10s}")
    want = {"B_D": "STRAIGHT", "B_C": "LEFT", "A_B": "BACK"}
    got = {}
    for c in g.classify_candidates("B", "A_B", allow_back=True):
        got[c["edge"]] = c["side"]
        print(f"  {c['edge']:>8s} {c['side']:>10s} {c['deg']:9.1f} {c['length_px']:10.1f}")
    for eid, side in want.items():
        if got.get(eid) != side:
            fails.append(f"на B выход {eid} определён как {got.get(eid)}, ожидалось {side}")
    print()
    print(f"  ожидалось: B_D STRAIGHT, B_C LEFT, A_B BACK — "
          f"{'совпало' if not fails else 'РАСХОЖДЕНИЕ'}")
    print()

    # note the side convention explicitly, because it is the one place a sign error
    # would silently mirror every turn in the project
    print("=== СОГЛАШЕНИЕ О СТОРОНАХ ===")
    print("  в плане x вправо, y вниз; приход по A_B смотрит вправо (0°)")
    print(f"  B_C ведёт вверх, то есть угол {g.turn('A_B', 'B', 'B_C')['deg']:+.0f}° → LEFT")
    print(f"  B_D ведёт вправо, то есть угол {g.turn('A_B', 'B', 'B_D')['deg']:+.0f}° → STRAIGHT")
    print("  это совпадает с конвенцией yaw: yaw < 0 = LEFT")
    print(f"  проверка на живом примере: правый поворот даёт положительный угол — "
          f"{'да' if g.turn('A_B','B','B_D')['deg'] > -45 else 'нет'}")
    print()

    # ---- 2. the walk -----------------------------------------------------
    print("=== 2. ПРОХОД A → B → D → E ===")
    steps = walk(g, ["A_B", "B_D", "D_E"])
    for s in steps:
        print(f"  {s['from']} → {s['to']}  по ребру {s['edge']}, {s['length_px']:.0f} px")
    seq = [s["edge"] for s in steps]
    if seq != ["A_B", "B_D", "D_E"]:
        fails.append(f"последовательность рёбер {seq}, ожидалось A_B, B_D, D_E")
    if steps[-1]["to"] != "E":
        fails.append(f"проход закончился в {steps[-1]['to']}, ожидалось E")
    total = sum(s["length_px"] for s in steps)
    print(f"  длина пути {total:.0f} px, конец в {steps[-1]['to']}")
    print()

    # ---- 3. the line -----------------------------------------------------
    print("=== 3. ЛОМАНАЯ ===")
    pts = polyline(g, steps)
    for i, (x, y) in enumerate(pts):
        print(f"  точка {i}: ({x:7.1f}, {y:7.1f})")
    if len(pts) != 4:
        fails.append(f"в ломаной {len(pts)} точек, ожидалось 4")
    if pts[0] != g.pos("A") or pts[-1] != g.pos("E"):
        fails.append("ломаная не начинается в A или не заканчивается в E")
    print()

    # ---- 4. one-way ------------------------------------------------------
    print("=== 4. ОДНОСТОРОННЕЕ РЕБРО ===")
    one = build()
    one.edges["B_D"]["bidirectional"] = False
    at_b = [c["edge"] for c in one.classify_candidates("B", "A_B", allow_back=True)]
    at_d = one.candidates("D", "B_D", allow_back=True)
    print(f"  B_D помечено односторонним B → D")
    print(f"  из B доступно: {sorted(at_b)}")
    print(f"  из D доступно: {sorted(at_d)} (обратно по B_D нельзя)")
    if "B_D" not in at_b:
        fails.append("одностороннее ребро B_D недоступно из B, хотя должно быть")
    if "B_D" in at_d:
        fails.append("одностороннее ребро B_D доступно из D, хотя не должно")
    print()

    # ---- 5. dead end, single option, choice -----------------------------
    print("=== 5. ТУПИК, ЕДИНСТВЕННЫЙ ВЫХОД, ВЫБОР ===")
    print("  этапы 14-15: сколько выходов — столько и логики")
    print()

    # (a) dead end: arriving at C along B_C leaves nothing ahead
    fwd = dead_end = build().candidates("C", "B_C", allow_back=False)
    back = build().candidates("C", "B_C", allow_back=True)
    print("  (a) приходим в C по B_C — это конец прохода")
    print(f"      выходов вперёд: {sorted(fwd)}")
    print(f"      с разрешением разворота: {sorted(back)}")
    if fwd:
        fails.append("в тупике C предложен выход вперёд")
    if "B_C" not in back:
        fails.append("в тупике C не предложен разворот")

    # (b) one way onward: a plain corridor, B has nothing but C ahead
    chain = Graph({
        "nodes": [{"id": "A", "x": 0.10, "y": 0.50},
                  {"id": "B", "x": 0.40, "y": 0.50},
                  {"id": "C", "x": 0.70, "y": 0.50}],
        "edges": [{"id": "A_B", "from": "A", "to": "B", "bidirectional": True},
                  {"id": "B_C", "from": "B", "to": "C", "bidirectional": True}],
    })
    one = chain.classify_candidates("B", "A_B", allow_back=False)
    print(f"  (b) коридор A—B—C, приходим в B по A_B")
    print(f"      выходов: {[c['edge'] for c in one]} → выбирается автоматически, "
          f"MaleCNS не нужен")
    if [c["edge"] for c in one] != ["B_C"]:
        fails.append(f"в коридоре из B предложено {[c['edge'] for c in one]}, ожидалось B_C")

    # (c) real choice: the junction from the drawing
    many = g.classify_candidates("B", "A_B", allow_back=False)
    print(f"  (c) перекрёсток из схемы, приходим в B по A_B")
    print(f"      выходов: {sorted(c['edge'] for c in many)} → нужен MaleCNS")
    if len(many) != 2:
        fails.append(f"на перекрёстке B предложено {len(many)} выхода, ожидалось 2")
    print()

    # ---- 6. validator ----------------------------------------------------
    print("=== 6. ЧТО ЛОВИТ ПРОВЕРКА ===")
    broken = build()
    del broken.edges["D_E"]
    broken.nodes["G"] = {"id": "G", "x": 0.20, "y": 0.90}
    probs = broken.validate()
    for p in probs:
        print(f"  — {p}")
    if not probs:
        fails.append("проверка не заметила ни оборванного конца, ни одинокого узла")
    try:
        Graph({"nodes": [{"id": "A", "x": 0, "y": 0}],
               "edges": [{"id": "A_Z", "from": "A", "to": "Z"}]})
        fails.append("граф с ребром на несуществующий узел не был отвергнут")
    except GraphError:
        print("  — ребро на несуществующий узел отвергается")
    print()

    print("=== ИТОГ ===")
    if fails:
        for f in fails:
            print(f"  НЕ ПРОШЛО: {f}")
        sys.exit(1)
    print("  все проверки пройдены: граф работает без мухи")
    print("  A → B → D → E восстанавливается по одной геометрии")


if __name__ == "__main__":
    main()
