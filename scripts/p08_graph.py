#!/usr/bin/env python3
"""
P08 — the graph of passages, and the geometry of a junction.

Kept in one module so that the page and the tracker cannot drift apart: the editor
classifies a turn for display, the tracker classifies it to decide, and both must agree.
The page's copy lives in `webapp/p08_graph_editor.html`; this is the one the analysis uses.

Angles, and which way is left
-----------------------------
Positions are normalized over the plan, but angles are taken in the plan's own pixels
because a plan is rarely square and a normalized angle would be sheared. With `x` to the
right and `y` downward, as in the image, the angle of a direction vector is
`atan2(dy, dx)`. Turning left in the room means turning counter-clockwise seen from
above, and with `y` flipped for the image that is a *decrease* in this angle. So

    delta = angle(after) - angle(before), wrapped to (-180, 180]

    delta <  0   LEFT
    delta >  0   RIGHT
    |delta| < 45 STRAIGHT
    |delta| > 135 BACK

which is the same convention the yaw signal uses: `yaw < 0` is a left turn.

One-way edges
-------------
`bidirectional: false` means travel runs from `from` to `to` only. The tracker asks
`can_travel` before considering an edge, so a one-way passage is never taken backwards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

STRAIGHT_MAX_DEG = 45.0
BACK_MIN_DEG = 135.0


class GraphError(ValueError):
    pass


class Graph:
    """Nodes, edges and their geometry."""

    def __init__(self, doc: dict):
        self.doc = doc or {}
        self.nodes: dict[str, dict] = {}
        for n in self.doc.get("nodes") or []:
            nid = str(n["id"])
            if nid in self.nodes:
                raise GraphError(f"duplicate node id {nid}")
            self.nodes[nid] = {"id": nid, "x": float(n["x"]), "y": float(n["y"]),
                               "kind": n.get("kind"), "degree": n.get("degree"),
                               "orig_id": n.get("orig_id")}

        self.edges: dict[str, dict] = {}
        for e in self.doc.get("edges") or []:
            eid = str(e["id"])
            a, b = str(e["from"]), str(e["to"])
            if a not in self.nodes or b not in self.nodes:
                raise GraphError(f"edge {eid} refers to an unknown node")
            if a == b:
                raise GraphError(f"edge {eid} joins a node to itself")
            if eid in self.edges:
                raise GraphError(f"duplicate edge id {eid}")
            lm = e.get("length_meters")
            self.edges[eid] = {"id": eid, "from": a, "to": b,
                               "bidirectional": bool(e.get("bidirectional", True)),
                               # a measured length is kept when the file gives one, so the
                               # route can be reported in metres rather than in pixels
                               "length_meters": float(lm) if lm is not None else None,
                               "width_median_m": e.get("width_median_m"),
                               "orig_id": e.get("orig_id")}

        self.img_w = float(self.doc.get("img_w") or 1600)
        self.img_h = float(self.doc.get("img_h") or 1000)
        self.start = self.doc.get("start")
        self.meters_per_pixel = self.doc.get("meters_per_pixel")
        self.known_distance = self.doc.get("known_distance")

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ lookups
    def pos(self, node: str) -> tuple[float, float]:
        n = self.nodes[node]
        return (n["x"] * self.img_w, n["y"] * self.img_h)

    def other(self, edge: str, node: str) -> str:
        e = self.edges[edge]
        if e["from"] == node:
            return e["to"]
        if e["to"] == node:
            return e["from"]
        raise GraphError(f"node {node} is not on edge {edge}")

    def edges_at(self, node: str) -> list[str]:
        return [eid for eid, e in self.edges.items()
                if e["from"] == node or e["to"] == node]

    def degree(self, node: str) -> int:
        return len(self.edges_at(node))

    def is_choice(self, node: str) -> bool:
        """True where more than one way on exists, i.e. where MaleCNS is consulted."""
        return self.degree(node) >= 3

    def choice_nodes(self) -> list[str]:
        return [nid for nid in self.nodes if self.is_choice(nid)]

    def dead_ends(self) -> list[str]:
        return [nid for nid in self.nodes if self.degree(nid) <= 1]

    def length_px(self, edge: str) -> float:
        e = self.edges[edge]
        ax, ay = self.pos(e["from"])
        bx, by = self.pos(e["to"])
        return math.hypot(bx - ax, by - ay)

    def length_m(self, edge: str) -> float | None:
        """The edge in metres: the recorded one when present, otherwise from the scale."""
        rec = self.edges[edge].get("length_meters")
        if rec is not None:
            return float(rec)
        if not self.meters_per_pixel:
            return None
        return self.length_px(edge) * float(self.meters_per_pixel)

    def route_meters(self, edges: list[str]) -> float | None:
        vals = [self.length_m(e) for e in edges]
        if any(v is None for v in vals):
            return None
        return float(sum(vals))

    # ------------------------------------------------------------------ geometry
    def dir_away(self, edge: str, node: str) -> tuple[float, float]:
        """Unit vector pointing away from `node` along `edge`."""
        ox, oy = self.pos(node)
        px, py = self.pos(self.other(edge, node))
        dx, dy = px - ox, py - oy
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return (0.0, 0.0)
        return (dx / L, dy / L)

    def heading_into(self, edge: str, node: str) -> tuple[float, float]:
        """Unit vector of travel when arriving at `node` along `edge`."""
        dx, dy = self.dir_away(edge, node)
        return (-dx, -dy)

    @staticmethod
    def _angle(v: tuple[float, float]) -> float:
        return math.atan2(v[1], v[0])

    def turn(self, in_edge: str, node: str, out_edge: str) -> dict:
        """How the turn from `in_edge` onto `out_edge` looks at a shared node."""
        if node not in (self.edges[in_edge]["from"], self.edges[in_edge]["to"]):
            raise GraphError(f"node {node} is not on edge {in_edge}")
        if node not in (self.edges[out_edge]["from"], self.edges[out_edge]["to"]):
            raise GraphError(f"node {node} is not on edge {out_edge}")
        a1 = self._angle(self.heading_into(in_edge, node))
        a2 = self._angle(self.dir_away(out_edge, node))
        d = math.degrees(a2 - a1)
        while d > 180.0:
            d -= 360.0
        while d <= -180.0:
            d += 360.0
        ad = abs(d)
        if ad < STRAIGHT_MAX_DEG:
            side = "STRAIGHT"
        elif ad > BACK_MIN_DEG:
            side = "BACK"
        else:
            side = "RIGHT" if d > 0 else "LEFT"
        return {"side": side, "deg": d}

    # ------------------------------------------------------------------ travel
    def can_travel(self, edge: str, from_node: str) -> bool:
        e = self.edges[edge]
        if e["bidirectional"]:
            return from_node in (e["from"], e["to"])
        return from_node == e["from"]

    def candidates(self, node: str, came_from_edge: str | None,
                   allow_back: bool = False) -> list[str]:
        """Edges that may be taken when standing at `node`.

        The edge just travelled is normally dropped: it is only offered again when the
        caller explicitly allows a reversal, which is what keeps an ordinary glance
        backwards from turning the route around.
        """
        out = []
        for eid in self.edges_at(node):
            if eid == came_from_edge and not allow_back:
                continue
            if self.can_travel(eid, node):
                out.append(eid)
        return out

    def classify_candidates(self, node: str, in_edge: str,
                            allow_back: bool = True) -> list[dict]:
        out = []
        for eid in self.candidates(node, in_edge, allow_back=allow_back):
            t = self.turn(in_edge, node, eid)
            out.append({"edge": eid, "to": self.other(eid, node),
                        "side": t["side"], "deg": t["deg"],
                        "length_px": self.length_px(eid)})
        return out

    # ------------------------------------------------------------------ checks
    def validate(self) -> list[str]:
        """Problems worth reporting before anything is tracked."""
        problems = []
        for nid in self.nodes:
            if not self.edges_at(nid):
                problems.append(f"узел {nid} ни с чем не соединён")
        # a node that can only be left but never entered is a defect in the drawing
        for nid in self.nodes:
            ok = False
            for eid in self.edges_at(nid):
                e = self.edges[eid]
                if e["bidirectional"] or e["to"] == nid:
                    ok = True
                    break
            if not ok:
                problems.append(f"в узел {nid} нельзя войти ни по одному ребру")
        # connectivity from the start, if a start is given
        start_edge = None
        if self.start:
            if self.start.get("type") == "edge":
                start_edge = self.start.get("edge")
            elif self.start.get("type") == "node":
                at = self.edges_at(self.start["node"])
                start_edge = at[0] if at else None
        if start_edge:
            seen = {start_edge}
            frontier = [start_edge]
            while frontier:
                cur = frontier.pop()
                for nid in (self.edges[cur]["from"], self.edges[cur]["to"]):
                    for eid in self.edges_at(nid):
                        if eid not in seen:
                            seen.add(eid)
                            frontier.append(eid)
            missing = [e for e in self.edges if e not in seen]
            if missing:
                problems.append("рёбра вне компоненты старта: " + ", ".join(missing))
        return problems


def load(path: str | Path = None) -> Graph:
    from pathlib import Path as _P
    p = _P(path) if path else _P(__file__).resolve().parents[1] / "data/p08/graph.json"
    return Graph.load(p)
