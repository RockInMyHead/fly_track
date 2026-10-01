#!/usr/bin/env python3
"""P08.4A — repair the map, change nothing about the fly.

The nine audited junctions left a question that the tracker cannot answer about itself: when
the route is wrong, is the fly wrong or is the map wrong? At J37 the map offered two ways on
that differ by 8.8 degrees and end 1.46 m apart. That is not a fork; it is one corridor drawn
as two, and the tracker spent a decision on it. Two other places had passages ten centimetres
long. None of this is the fly's fault, so this script fixes the map and leaves MaleCNS, the
weights, novelty, the pace model and every threshold exactly as they were frozen in P08.2.

Two repairs, both of them things the building can be asked about rather than things the route
suggests:

  1. Two nodes closer than MERGE_M are one place. The plan settles the number: the median
     corridor here is about 1.6 m wide, so two points half a metre apart cannot have a wall
     between them. Merged into the better-connected node, which keeps its position; nothing is
     moved, so no geometry is invented.

  2. A node offering two edges that leave within DUP_DEG of each other and arrive within DUP_M
     of each other is offering the same passage twice. The edge to the worse-placed end goes:
     a junction belongs in the corridor, not on the skirting. This is what ends the fake fork at
     J37 — after it, J37 is a degree-2 point on a straight corridor and the choice moves to T54,
     where three real ways meet.

Both repairs are applied to the graph alone. The tracker is not touched, so any change in the
route is attributable to the map.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

KIND_RANK = {"junction": 3, "manual": 2, "turn": 1, "endpoint": 0}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wall_distance(plan_path: Path) -> np.ndarray:
    """Distance from every pixel to the nearest wall, in pixels.

    Walls are the dark ink of the plan. The distance answers "how far is this point from the
    skirting", which is what tells a junction placed mid-corridor from one placed against a
    wall, and it is the only measurement here that does not come from the graph itself.
    """
    from scipy import ndimage

    a = np.asarray(Image.open(plan_path).convert("L")).astype(float)
    return ndimage.distance_transform_edt(a >= 140)


class Graph:
    def __init__(self, doc: dict, dist: np.ndarray, mpp: float):
        self.doc = doc
        self.dist = dist
        self.mpp = mpp
        self.nodes = {n["id"]: dict(n) for n in doc["nodes"]}
        self.edges = {e["id"]: dict(e) for e in doc["edges"]}
        self.w = float(doc["img_w"])
        self.h = float(doc["img_h"])

    # -- geometry -----------------------------------------------------------
    def xy(self, nid: str) -> tuple[float, float]:
        n = self.nodes[nid]
        return n["x"] * self.w, n["y"] * self.h

    def px_len(self, eid: str) -> float:
        e = self.edges[eid]
        (x1, y1), (x2, y2) = self.xy(e["from"]), self.xy(e["to"])
        return math.hypot(x2 - x1, y2 - y1)

    def m(self, px: float) -> float:
        return px * self.mpp

    def edges_at(self, nid: str) -> list[str]:
        return [eid for eid, e in self.edges.items() if nid in (e["from"], e["to"])]

    def other(self, eid: str, nid: str) -> str:
        e = self.edges[eid]
        return e["to"] if e["from"] == nid else e["from"]

    def degree(self, nid: str) -> int:
        return len(self.edges_at(nid))

    def off_wall_m(self, nid: str) -> float:
        x, y = self.xy(nid)
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < self.dist.shape[0] and 0 <= xi < self.dist.shape[1]):
            return 0.0
        return float(self.dist[yi, xi]) * self.mpp

    def kind_rank(self, nid: str) -> int:
        return KIND_RANK.get(self.nodes[nid].get("kind", ""), -1)


# --------------------------------------------------------------------- repair 1
def merge_close_nodes(g: Graph, merge_m: float, log: list) -> None:
    """Fuse groups of nodes that lie closer together than a corridor is wide."""
    ids = sorted(g.nodes)
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(ids):
        ax, ay = g.xy(a)
        for b in ids[i + 1:]:
            bx, by = g.xy(b)
            if math.hypot(bx - ax, by - ay) * g.mpp < merge_m:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)

    remap: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        # survivor: best connected, then closest to the kind we trust, then best placed in the
        # corridor, then the plainest name. Order matters only for reproducibility.
        survivor = max(members, key=lambda n: (
            g.degree(n), g.kind_rank(n), g.off_wall_m(n), [-ord(c) for c in n]))
        for n in members:
            if n != survivor:
                remap[n] = survivor
                log.append({"repair": "merge", "removed": n, "into": survivor,
                            "distance_m": round(g.m(math.hypot(
                                g.xy(n)[0] - g.xy(survivor)[0],
                                g.xy(n)[1] - g.xy(survivor)[1])), 4),
                            "removed_degree": g.degree(n),
                            "removed_off_wall_m": round(g.off_wall_m(n), 4)})

    # rebuild edges through the remap
    for eid in list(g.edges):
        e = g.edges[eid]
        a, b = remap.get(e["from"], e["from"]), remap.get(e["to"], e["to"])
        if a == b:
            log.append({"repair": "drop_self_loop", "edge": eid,
                        "was": f'{e["from"]}__{e["to"]}', "kept_node": a})
            del g.edges[eid]
            continue
        e["from"], e["to"] = a, b
        new_id = f"{a}__{b}"
        if new_id != eid:
            log.append({"repair": "rename_edge", "edge": eid, "to": new_id})
            e["id"] = new_id
            g.edges[new_id] = e
            del g.edges[eid]

    for n in remap:
        del g.nodes[n]

    # an edge may now duplicate another; the shorter one survives, and the lengths re-decide
    seen: dict[tuple[str, str], str] = {}
    for eid in list(g.edges):
        e = g.edges[eid]
        key = tuple(sorted((e["from"], e["to"])))
        if key in seen:
            keep, drop = seen[key], eid
            if g.px_len(eid) < g.px_len(seen[key]):
                keep, drop = eid, seen[key]
            log.append({"repair": "drop_duplicate_edge", "edge": drop, "kept": keep,
                        "pair": f"{key[0]}__{key[1]}",
                        "dropped_m": round(g.m(g.px_len(drop)), 4),
                        "kept_m": round(g.m(g.px_len(keep)), 4)})
            del g.edges[drop]
            seen[key] = keep
        else:
            seen[key] = eid

    for nid in g.nodes:
        g.nodes[nid]["degree"] = g.degree(nid)

    for eid in g.edges:
        g.edges[eid]["length_meters"] = round(g.m(g.px_len(eid)), 4)


# --------------------------------------------------------------------- repair 2
def drop_parallel_duplicates(g: Graph, dup_deg: float, dup_m: float, log: list) -> None:
    """Remove an edge when the same node already offers a near-identical way on.

    The two edges must leave almost together and arrive almost together. Both conditions are
    needed: two edges can leave a junction at a wide angle and still end near each other (a
    genuine short loop), and two edges can leave together and end far apart (a corridor that
    opens slowly). Only together do they mean one passage counted twice.
    """
    changed = True
    while changed:
        changed = False
        for nid in sorted(g.nodes):
            eids = g.edges_at(nid)
            if len(eids) < 2:
                continue
            for i, e1 in enumerate(eids):
                for e2 in eids[i + 1:]:
                    a, b = g.other(e1, nid), g.other(e2, nid)
                    if a == b:
                        continue
                    (x0, y0) = g.xy(nid)
                    (x1, y1) = g.xy(a)
                    (x2, y2) = g.xy(b)
                    ang1 = math.atan2(y1 - y0, x1 - x0)
                    ang2 = math.atan2(y2 - y0, x2 - x0)
                    gap = abs(((math.degrees(ang2 - ang1) + 180) % 360) - 180)
                    near = math.hypot(x2 - x1, y2 - y1) * g.mpp
                    if gap >= dup_deg or near >= dup_m:
                        continue
                    # same passage twice. Keep the way whose far end is better placed in the
                    # corridor; a junction belongs on the floor, not against the skirting.
                    drop = e1
                    if g.off_wall_m(a) > g.off_wall_m(b):
                        drop = e2
                    keep = e2 if drop == e1 else e1
                    log.append({"repair": "drop_parallel_edge", "edge": drop, "kept": keep,
                                "at_node": nid, "angle_deg": round(gap, 2),
                                "ends_m_apart": round(near, 3),
                                "dropped_off_wall_m": round(g.off_wall_m(
                                    g.other(drop, nid)), 4),
                                "kept_off_wall_m": round(g.off_wall_m(
                                    g.other(keep, nid)), 4)})
                    del g.edges[drop]
                    for n in (nid,) + (a, b):
                        if n in g.nodes:
                            g.nodes[n]["degree"] = g.degree(n)
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--plan", default=str(ROOT / "data/p08/plan.png"))
    ap.add_argument("--out", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--merge-m", type=float, default=0.5)
    ap.add_argument("--dup-deg", type=float, default=12.0)
    ap.add_argument("--dup-m", type=float, default=1.5)
    ap.add_argument("--log", default=str(ROOT / "output/p084a/graph_repair_log.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.graph)
    doc = load(src)
    g = Graph(doc, wall_distance(Path(args.plan)), doc["meters_per_pixel"])

    n0, e0 = len(g.nodes), len(g.edges)
    # snapshot before anything is replaced, so the report can show what moved
    degrees_before = {n["id"]: n.get("degree") for n in doc["nodes"] if "degree" in n}
    log: list = []
    merge_close_nodes(g, args.merge_m, log)
    drop_parallel_duplicates(g, args.dup_deg, args.dup_m, log)

    for nid in g.nodes:
        g.nodes[nid].setdefault("geometry_source", "")
    g.doc["nodes"] = [g.nodes[n] for n in sorted(g.nodes)]
    g.doc["edges"] = [g.edges[e] for e in sorted(g.edges)]

    if args.dry_run:
        print("сухой прогон, файл не тронут")
    else:
        Path(args.out).write_text(json.dumps(g.doc, indent=2, ensure_ascii=False),
                                  encoding="utf-8")

    by_kind: dict[str, int] = {}
    for item in log:
        by_kind[item["repair"]] = by_kind.get(item["repair"], 0) + 1

    print(f"узлов {n0} → {len(g.nodes)}   рёбер {e0} → {len(g.edges)}")
    print(f"порог слияния {args.merge_m} м, дубли рёбер: угол < {args.dup_deg}°, концы < {args.dup_m} м")
    print()
    for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print()
    print("=== слияния ===")
    for item in log:
        if item["repair"] == "merge":
            print(f"  {item['removed']:<5} → {item['into']:<5}  "
                  f"{item['distance_m']:5.3f} м, степень была {item['removed_degree']}, "
                  f"до стены {item['removed_off_wall_m']:5.3f} м")
    print()
    print("=== убранные рёбра ===")
    for item in log:
        if item["repair"] == "drop_parallel_edge":
            print(f"  {item['edge']:<14} при {item['at_node']:<5} "
                  f"угол {item['angle_deg']:5.2f}°, концы в {item['ends_m_apart']:.3f} м  "
                  f"(оставлено {item['kept']})")
        if item["repair"] == "drop_duplicate_edge":
            print(f"  {item['edge']:<14} дубль {item['pair']}  "
                  f"{item['dropped_m']:.3f} м (оставлено {item['kept']}, {item['kept_m']:.3f} м)")
        if item["repair"] == "drop_self_loop":
            print(f"  {item['edge']:<14} петля после слияния (узел {item['kept_node']})")
    print()
    print("=== узлы с изменившейся степенью ===")
    for nid in sorted(g.nodes):
        new = g.nodes[nid]["degree"]
        was = degrees_before.get(nid)
        if was is not None and was != new and new >= 2:
            print(f"  {nid:<6} {was} → {new}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps({
        "source_graph": str(src), "merge_m": args.merge_m,
        "dup_deg": args.dup_deg, "dup_m": args.dup_m,
        "nodes_before": n0, "nodes_after": len(g.nodes),
        "edges_before": e0, "edges_after": len(g.edges),
        "changes": log,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nжурнал: {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
