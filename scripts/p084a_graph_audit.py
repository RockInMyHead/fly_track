#!/usr/bin/env python3
"""Find the defects in the P08 graph that are not about the fly or the rules.

P08.4A is a repair of the map, not of the tracker. Changing both at once would leave the
question of which one helped unanswerable, so this script only looks at the graph: whether a
passage exists at all, whether two nodes are really the same place, and whether an edge
actually runs where it claims to.

Five checks, each of which found something real on this graph:

  1. edges shorter than a threshold. A ten centimetre passage is not a passage; it is almost
     always one junction split across two nodes, and it hands the tracker a "choice" that is
     not a choice.
  2. node pairs closer together than a threshold, not already joined. The same defect seen
     from the other side.
  3. an edge that passes through a node which is not one of its ends. When a corridor runs
     straight past a junction, the edge should end at that junction and continue beyond it;
     drawn as one edge, the tracker skips the junction entirely and never considers the ways
     out that meet there.
  4. nodes joined to only one other node, and nodes that are their own little island. These
     are where a route dead-ends, and a wrong choice there costs a reversal.
  5. chains of nodes joined only by tiny edges, which is how a single wide place gets carved
     into a row of nodes that the tracker can wander along.

Nothing here reads the fly. The output is a list of places where the map disagrees with the
building, and every fix is a choice a person should look at.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Geo:
    """Positions in plan pixels, which is the frame the geometry is exact in."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.w = float(doc["img_w"])
        self.h = float(doc["img_h"])
        self.mpp = doc.get("meters_per_pixel") or 0.0
        self.pos = {n["id"]: (n["x"] * self.w, n["y"] * self.h) for n in doc["nodes"]}
        self.edges = {e["id"]: e for e in doc["edges"]}

    def at(self, node: str) -> tuple[float, float]:
        return self.pos[node]

    def ends(self, eid: str) -> tuple[str, str]:
        e = self.edges[eid]
        return e["from"], e["to"]

    def other(self, eid: str, node: str) -> str:
        a, b = self.ends(eid)
        return b if a == node else a

    def px_len(self, eid: str) -> float:
        a, b = self.ends(eid)
        (x1, y1), (x2, y2) = self.at(a), self.at(b)
        return math.hypot(x2 - x1, y2 - y1)

    def m(self, px: float) -> float:
        return px * self.mpp if self.mpp else float("nan")

    def edges_at(self, node: str) -> list[str]:
        return [eid for eid, e in self.edges.items()
                if node in (e["from"], e["to"])]

    def degree(self, node: str) -> int:
        return len(self.edges_at(node))


def point_seg_px(p, a, b) -> float:
    """Distance from a point to a segment, in pixels."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + dx * t), py - (ay + dy * t))


# --------------------------------------------------------------------------- checks
def check_tiny_edges(g: Geo, limit_m: float) -> list[dict]:
    out = []
    for eid in g.edges:
        m = g.m(g.px_len(eid))
        if m == m and m < limit_m:            # not NaN
            a, b = g.ends(eid)
            out.append({"edge": eid, "from": a, "to": b, "meters": round(m, 4),
                        "degrees": [g.degree(a), g.degree(b)]})
    return sorted(out, key=lambda d: d["meters"])


def check_close_nodes(g: Geo, limit_m: float) -> list[dict]:
    ids = list(g.pos)
    joined = {tuple(sorted(g.ends(eid))) for eid in g.edges}
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            m = g.m(math.hypot(g.pos[a][0] - g.pos[b][0], g.pos[a][1] - g.pos[b][1]))
            if m == m and m < limit_m:
                out.append({"a": a, "b": b, "meters": round(m, 4),
                            "already_joined": tuple(sorted((a, b))) in joined,
                            "degrees": [g.degree(a), g.degree(b)]})
    return sorted(out, key=lambda d: d["meters"])


def check_edge_through_node(g: Geo, limit_m: float) -> list[dict]:
    """An edge whose line runs through a node that is not one of its ends.

    This is the defect that hides a junction: the passage is real, the node is real, but no
    edge joins them, so the tracker goes past without the way out ever being offered. A small
    tolerance matters here — a corridor and the node marking it are never exactly collinear.
    """
    out = []
    for eid, e in g.edges.items():
        a, b = e["from"], e["to"]
        pa, pb = g.at(a), g.at(b)
        for nid in g.pos:
            if nid in (a, b):
                continue
            d = g.m(point_seg_px(g.at(nid), pa, pb))
            if d == d and d < limit_m:
                # only interesting when the node sits between the ends, not off to one side
                seg = g.m(g.px_len(eid))
                to_a = g.m(math.hypot(g.at(nid)[0] - pa[0], g.at(nid)[1] - pa[1]))
                to_b = g.m(math.hypot(g.at(nid)[0] - pb[0], g.at(nid)[1] - pb[1]))
                out.append({"edge": eid, "node": nid, "off_line_m": round(d, 4),
                            "edge_len_m": round(seg, 3),
                            "node_degree": g.degree(nid),
                            "between": d == d and to_a <= seg + 1e-6 and to_b <= seg + 1e-6})
    return sorted(out, key=lambda d: d["off_line_m"])


def check_dead_ends(g: Geo) -> dict:
    dead = [n for n in g.pos if g.degree(n) == 1]
    lonely = [n for n in g.pos if g.degree(n) == 0]
    return {"degree_1": sorted(dead), "degree_0": sorted(lonely),
            "n_degree_1": len(dead), "n_degree_0": len(lonely)}


def check_components(g: Geo) -> list[dict]:
    seen = set()
    comps = []
    for start in g.pos:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for eid in g.edges_at(n):
                o = g.other(eid, n)
                if o not in seen:
                    seen.add(o)
                    stack.append(o)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return [{"size": len(c), "nodes": sorted(c)[:12]} for c in comps]


def check_tiny_chains(g: Geo, limit_m: float) -> list[dict]:
    """Runs of nodes connected only by tiny edges — one wide place carved into a row."""
    tiny = {eid for eid in g.edges
            if (lambda m: m == m and m < limit_m)(g.m(g.px_len(eid)))}
    if not tiny:
        return []
    # group nodes touched by tiny edges, then split into connected groups
    touched = {n for eid in tiny for n in g.ends(eid)}
    seen, groups = set(), []
    for start in touched:
        if start in seen:
            continue
        stack, grp = [start], []
        seen.add(start)
        while stack:
            n = stack.pop()
            grp.append(n)
            for eid in tiny:
                if n in g.ends(eid):
                    o = g.other(eid, n)
                    if o not in seen:
                        seen.add(o)
                        stack.append(o)
        groups.append({"nodes": sorted(grp), "n": len(grp)})
    return sorted(groups, key=lambda d: -d["n"])


# --------------------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--tiny-m", type=float, default=0.5,
                    help="рёбра короче этого считаются дефектом")
    ap.add_argument("--near-m", type=float, default=0.5,
                    help="узлы ближе этого считаются одним местом")
    ap.add_argument("--through-m", type=float, default=0.6,
                    help="насколько узел может отойти от линии ребра и всё ещё считаться на нём")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    g = Geo(load(Path(args.graph)))
    print(f"граф: {len(g.doc['nodes'])} узлов, {len(g.doc['edges'])} рёбер, "
          f"масштаб {g.mpp:.6f} м/пиксель" if g.mpp else "граф без масштаба")
    print()

    tiny = check_tiny_edges(g, args.tiny_m)
    print(f"=== 1. рёбра короче {args.tiny_m} м: {len(tiny)} ===")
    for d in tiny:
        print(f"  {d['edge']:<14} {d['meters']:7.4f} м   степени узлов {d['degrees']}")
    print()

    near = check_close_nodes(g, args.near_m)
    print(f"=== 2. пары узлов ближе {args.near_m} м: {len(near)} ===")
    for d in near[:30]:
        flag = "уже связаны" if d["already_joined"] else "НЕ связаны"
        print(f"  {d['a']:<6} ↔ {d['b']:<6} {d['meters']:7.4f} м   {flag}   "
              f"степени {d['degrees']}")
    if len(near) > 30:
        print(f"  … и ещё {len(near) - 30}")
    print()

    through = check_edge_through_node(g, args.through_m)
    print(f"=== 3. рёбра, проходящие мимо узла (мимо {args.through_m} м): {len(through)} ===")
    for d in through[:30]:
        flag = "между концами" if d["between"] else "в стороне"
        print(f"  ребро {d['edge']:<14} ({d['edge_len_m']:6.2f} м) мимо узла "
              f"{d['node']:<6} на {d['off_line_m']:5.3f} м, степень узла "
              f"{d['node_degree']}  [{flag}]")
    if len(through) > 30:
        print(f"  … и ещё {len(through) - 30}")
    print()

    de = check_dead_ends(g)
    print(f"=== 4. тупики ===")
    print(f"  узлов со степенью 1: {de['n_degree_1']}")
    print(f"  узлов без рёбер:     {de['n_degree_0']}")
    print(f"  {', '.join(de['degree_1'][:24])}")
    print()

    comps = check_components(g)
    print(f"=== 5. связные куски: {len(comps)} ===")
    for c in comps[:8]:
        print(f"  {c['size']:>4} узлов: {', '.join(c['nodes'])}")
    if len(comps) > 8:
        print(f"  … и ещё {len(comps) - 8}")
    print()

    chains = check_tiny_chains(g, args.tiny_m)
    print(f"=== 6. скопления на коротких рёбрах: {len(chains)} ===")
    for c in chains[:12]:
        print(f"  {c['n']} узлов: {', '.join(c['nodes'])}")
    print()

    if args.json:
        Path(args.json).write_text(json.dumps({
            "graph": args.graph,
            "tiny_edges": tiny,
            "close_nodes": near,
            "edge_through_node": through,
            "dead_ends": de,
            "components": comps,
            "tiny_chains": chains,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"записано: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
