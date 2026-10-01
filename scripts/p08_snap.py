#!/usr/bin/env python3
"""
P08.2 — reading a hand-drawn line as a route through the graph.

What this is for
----------------
The reference route a person can produce most easily is a line over the plan. This module
turns that line into the ordered list of passages it follows, so it can be compared with what
the tracker produced.

The reading follows the graph
-----------------------------
Choosing the nearest passage for each sample independently does not work, and the reason is
not a detail. Near a node, a hand-drawn line is equally close to every passage that meets
there, so an independent reading alternates between them and turns hand shake into a
there-and-back the person never made. Measured on a synthetic line drawn along a known route
with four pixels of noise, the independent reading gave 28 passages for a route of 12.

So the reading is sticky and it follows the graph:

    a switch is considered only to a passage sharing a node with the current one,
    only once the line has reached that node,
    and only when the other passage is closer by more than a hysteresis margin.

Direction is read, not assumed
------------------------------
Which way a passage was walked matters, because the first passage fixes what is left and right
for every decision after it. The line says which way, through the movement of the projection
along the passage, and that is what is used. Taking the direction from the file instead reads
a route backwards half the time, which the self-test caught on its first case.

A reversal is two passages
--------------------------
Walking into a dead end and back out is two passes along one passage, and a route with a dead
end is full of them. The line shows it as the projection turning round, and that is detected
and kept. Without it an out-and-back reads as a single pass, as if the walker had gone in and
stopped.

What it refuses to do
---------------------
Samples further than `max_dist_m` from every passage are dropped rather than attached to
something implausible, and the count is reported. A line drawn over a wall is not a route, and
saying so is better than inventing one.

Usage:
    PYTHONPATH=. python scripts/p08_snap.py       # self-test on synthetic lines
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402

STEP_PX = 8.0           # resampling interval, about 0.4 m on this plan
MIN_DWELL = 3           # samples a passage must hold before it is accepted
MAX_DIST_M = 4.0        # beyond this a sample is not on any passage and is dropped
BRIDGE_MAX_EDGES = 6    # how far to look for a connection between two passages
HYST_PX = 6.0           # a passage is left only for another that is this much closer
N_ANCHORS = 8           # how many candidate starting passages are tried
REVERSE_EPS = 0.03      # movement along a passage needed before a reversal is believed
RESET_PENALTY = 30.0    # a re-anchor costs this much against the fit error, in pixels


def _project(g: Graph, eid: str, px: float, py: float) -> tuple[float, float]:
    """Distance from a point to a passage, and where along it the closest point falls."""
    e = g.edges[eid]
    ax, ay = g.pos(e["from"])
    bx, by = g.pos(e["to"])
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy or 1.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), t


def _nearest(g: Graph, px: float, py: float) -> tuple[str, float, float]:
    best = (None, 0.0, float("inf"))
    for eid in g.edges:
        d, t = _project(g, eid, px, py)
        if d < best[2]:
            best = (eid, t, d)
    return best


def _resample(pts: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    """Points along the line at even spacing, so mouse speed does not change the reading."""
    if len(pts) < 2:
        return list(pts)
    out = [pts[0]]
    carry = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        pos = carry
        while pos + step <= seg:
            pos += step
            f = pos / seg
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
        carry = pos - seg
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def _dwell(runs: list[list], min_dwell: int) -> list[list]:
    """Fold passages held for fewer than `min_dwell` samples into a neighbouring passage.

    A run is `[edge, samples, first_sample, t_first, t_last]`. Keeping the sample counts is
    what makes the filter work at all: an earlier version was handed only the passage names,
    so every run looked one sample long, every run fell below the threshold, and the entire
    route collapsed into one passage. The information a filter works on has to travel with it.
    """
    if not runs:
        return []
    work = [list(r) for r in runs]
    while len(work) > 1:
        idx = None
        for i, r in enumerate(work):
            if r[1] < min_dwell and (idx is None or r[1] < work[idx][1]):
                idx = i
        if idx is None:
            break
        i = idx
        left = work[i - 1] if i > 0 else None
        right = work[i + 1] if i + 1 < len(work) else None
        if left is None:
            into = right[0]
        elif right is None:
            into = left[0]
        else:
            into = left[0] if left[1] >= right[1] else right[0]
        work[i][0] = into
        merged: list[list] = []
        for r in work:
            if merged and merged[-1][0] == r[0]:
                merged[-1][1] += r[1]
                merged[-1][4] = r[4]          # the later t becomes the end of the merged run
            else:
                merged.append(list(r))
        work = merged
    return work


def _bridge(g: Graph, a: str, b: str) -> list[str] | None:
    """Shortest run of passages joining two passages, or None if too far apart."""
    starts = [g.edges[a]["from"], g.edges[a]["to"]]
    targets = {g.edges[b]["from"], g.edges[b]["to"]}
    q = deque((n, []) for n in starts)
    seen = set(starts)
    while q:
        node, path = q.popleft()
        if node in targets:
            return path
        if len(path) >= BRIDGE_MAX_EDGES:
            continue
        for eid in g.edges_at(node):
            nxt = g.other(eid, node)
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [eid]))
    return None


def _orient(g: Graph, runs: list[list]) -> list[dict]:
    """Passages in order, each oriented the way the line moved along it.

    Direction comes from the movement of the projection parameter, which is what the line
    actually shows. Where a passage carries no parameter — the ones inserted to bridge a gap —
    direction is taken from the node the previous passage ended at, which is the only thing
    known about them.
    """
    out: list[dict] = []
    cur_from = None
    for r in runs:
        eid, _, _, t_first, t_last = r[0], r[1], r[2], r[3], r[4]
        e = g.edges[eid]
        if t_first is not None and t_last is not None and abs(t_last - t_first) > 1e-6:
            frm, to = (e["from"], e["to"]) if t_last > t_first else (e["to"], e["from"])
        elif cur_from is not None and cur_from in (e["from"], e["to"]):
            frm, to = (cur_from, g.other(eid, cur_from))
        else:
            frm, to = e["from"], e["to"]
        entry = {"edge": eid, "from": frm, "to": to}
        if cur_from is not None and frm != cur_from:
            entry["break"] = True
        out.append(entry)
        cur_from = to
    return out


def _track(g: Graph, dense: list[tuple[float, float]], anchor: str, max_px: float,
           hysteresis_px: float) -> dict:
    """Follow the line along the graph from a given starting passage."""
    cur = anchor
    dropped = resets = 0
    # [edge, samples, first_sample, t_first, t_last]
    runs: list[list] = [[cur, 0, 0, None, None]]
    tot_d = 0.0
    n_d = 0

    def record(eid: str, i: int, t: float | None) -> None:
        if runs and runs[-1][0] == eid:
            if runs[-1][3] is None:
                runs[-1][3] = t          # the first t of this run, which gives direction
            runs[-1][1] += 1
            runs[-1][4] = t
        else:
            runs.append([eid, 1, i, t, t])

    run_dir = None      # +1 or -1: the way the line is moving along the current passage

    for i, (x, y) in enumerate(dense):
        d_cur, t_cur = _project(g, cur, x, y)
        if d_cur > max_px:
            # The line has left this passage. A neighbouring passage is tried first, because a
            # gap usually continues through a node; only if none fits is the nearest passage
            # anywhere taken. Going straight to the nearest anywhere is what sent an earlier
            # reading onto a passage that merely lies close by and is not a continuation.
            cand = []
            for node in (g.edges[cur]["from"], g.edges[cur]["to"]):
                for eid in g.edges_at(node):
                    if eid != cur:
                        d2, _ = _project(g, eid, x, y)
                        cand.append((d2, eid))
            cand.sort()
            picked = None
            if cand and cand[0][0] <= max_px:
                picked = cand[0][1]
            else:
                eid, _, d2 = _nearest(g, x, y)
                if d2 <= max_px:
                    picked = eid
            if picked is None:
                dropped += 1
                continue
            if picked != cur:
                resets += 1
                cur = picked
                runs.append([cur, 1, i, None, None])
                continue
        tot_d += d_cur * d_cur
        n_d += 1

        best, best_d, best_t = cur, d_cur, t_cur
        for node in (g.edges[cur]["from"], g.edges[cur]["to"]):
            for eid in g.edges_at(node):
                if eid == cur:
                    continue
                d2, t2 = _project(g, eid, x, y)
                if d2 + hysteresis_px >= best_d:
                    continue
                at_node_end = t_cur > 0.5 if g.edges[cur]["to"] == node else t_cur < 0.5
                from_node_start = t2 < 0.5 if g.edges[eid]["from"] == node else t2 > 0.5
                if at_node_end and from_node_start:
                    best, best_d, best_t = eid, d2, t2

        last_t = runs[-1][4]
        if best != cur:
            cur = best
            runs.append([cur, 1, i, best_t, best_t])
            run_dir = None
            continue
        # Reversal on the same passage. The line turning round is the walker coming back out
        # of a dead end, and it is two passes, not one: without this an out-and-back reads as
        # if the walker had gone in and stopped.
        if (last_t is not None and abs(best_t - last_t) > REVERSE_EPS
                and run_dir is not None and (best_t - last_t) * run_dir < 0):
            runs.append([cur, 1, i, last_t, best_t])
            run_dir = 1 if best_t > last_t else -1
            continue
        record(cur, i, best_t)
        if last_t is not None and abs(best_t - last_t) > REVERSE_EPS:
            run_dir = 1 if best_t > last_t else -1

    mean_d = math.sqrt(tot_d / max(n_d, 1))
    dropped_frac = dropped / max(len(dense), 1)
    # One number to compare attempts by. The fit error is what a wrong reading pays: it has to
    # stretch to reach passages that are not there. A re-anchor is charged a fixed amount
    # because it is a teleport the line did not make.
    score = mean_d * (1.0 + dropped_frac) + RESET_PENALTY * resets
    return {"runs": runs, "dropped": dropped, "resets": resets, "mean_d": mean_d,
            "score": score, "n_runs": len(runs)}


def snap_polyline(g: Graph, pts_norm: list[tuple[float, float]],
                  step_px: float = STEP_PX, min_dwell: int = MIN_DWELL,
                  max_dist_m: float = MAX_DIST_M,
                  hysteresis_px: float = HYST_PX,
                  n_anchors: int = N_ANCHORS) -> dict:
    """Read a drawn line as an ordered list of passages.

    The line is followed from each of the several passages nearest its start, and the attempt
    that fits best wins. Guessing the start from the first sample alone is not reliable,
    because a route usually begins at a node where every passage meeting there is equally
    close, and this graph makes that worse than usual: 23 pairs of its nodes are less than half
    a metre apart, one pair at zero distance, so near such a place a line really is equidistant
    from several passages and only the rest of the line can say which one it means.

    `pts_norm` are points in the plan's own normalized coordinates, the same ones the editor
    stores.
    """
    if len(pts_norm) < 2:
        return {"ok": False, "reason": "линия слишком короткая", "edges": []}
    px = [(float(x) * g.img_w, float(y) * g.img_h) for x, y in pts_norm]
    dense = _resample(px, step_px)
    mpp = float(g.meters_per_pixel or 0.049629166698546515)
    max_px = max_dist_m / mpp

    first = dense[0]
    cand = sorted(((_project(g, eid, *first)[0], eid) for eid in g.edges))[:n_anchors]
    attempts = []
    for d0, eid in cand:
        if d0 > max_px * 2:
            continue
        attempts.append({"anchor": eid, **_track(g, dense, eid, max_px, hysteresis_px)})
    if not attempts:
        return {"ok": False, "reason": "линия проходит далеко от всех проходов",
                "edges": [], "dropped": len(dense), "samples": len(dense)}
    attempts.sort(key=lambda a: a["score"])
    best = attempts[0]

    kept = _dwell(best["runs"], min_dwell)

    # Build the passage list, closing gaps. A repeated passage is a reversal and is kept: safe
    # to keep, because accidental repeats were merged by the dwell filter.
    seq: list[list] = [kept[0]]
    bridges = 0
    unreachable = []
    for r in kept[1:]:
        prev = seq[-1][0]
        eid = r[0]
        if eid == prev:
            seq.append(r)
            continue
        if set([g.edges[eid]["from"], g.edges[eid]["to"]]) & \
           set([g.edges[prev]["from"], g.edges[prev]["to"]]):
            seq.append(r)
            continue
        path = _bridge(g, prev, eid)
        if path:
            for p in path:
                seq.append([p, 0, 0, None, None])
            seq.append(r)
            bridges += 1
        else:
            unreachable.append((prev, eid))
            seq.append(r)

    oriented = _orient(g, seq)
    return {
        "ok": True,
        "edges": oriented,
        "n_edges": len(oriented),
        "samples": len(dense),
        "dropped_off_graph": best["dropped"],
        "bridges_inserted": bridges,
        "reanchors": best["resets"],
        "anchor": best["anchor"],
        "anchors_tried": len(attempts),
        "mean_dist_to_graph_px": round(best["mean_d"], 2),
        "score": round(best["score"], 2),
        "unreachable": unreachable,
        "distinct": len(set(e["edge"] for e in oriented)),
    }


def _draw_along(g: Graph, route: list[tuple[str, str, str]], seed: int = 7,
                noise_px: float = 4.0) -> list[tuple[float, float]]:
    """A hand-like line along a route: the way the walker moved, plus a few pixels of shake."""
    import random
    random.seed(seed)
    pts = []
    for frm, to, _ in route:
        p, q = g.pos(frm), g.pos(to)
        for k in range(6):
            f = k / 6
            pts.append(((p[0] + (q[0] - p[0]) * f + random.uniform(-noise_px, noise_px)) / g.img_w,
                        (p[1] + (q[1] - p[1]) * f + random.uniform(-noise_px, noise_px)) / g.img_h))
    q = g.pos(route[-1][1])
    pts.append((q[0] / g.img_w, q[1] / g.img_h))
    return pts


def main() -> None:
    """Self-test: a noisy line along a known route must read back as that route."""
    import csv
    g = Graph.load(ROOT / "data/p08/graph.json")
    rows = list(csv.DictReader((ROOT / "output/p08/edge_sequence.csv").open()))
    print("P08.2 — проверка при чтении нарисованной линии")
    print(f"  граф: {len(g.nodes)} узлов, {len(g.edges)} рёбер")
    print(f"  линия проводится вдоль маршрута P08 с дрожанием ±4 px")
    print()

    fails = []
    for tag, n in (("первые 12 рёбер", 12), ("весь маршрут", len(rows))):
        route = [(r["from"], r["to"], r["edge"]) for r in rows[:n]]
        pts = _draw_along(g, route)
        res = snap_polyline(g, pts)
        got = [e["edge"] for e in res["edges"]]
        want = [r[2] for r in route]
        m = min(len(want), len(got))
        same = sum(1 for i in range(m) if want[i] == got[i])

        print(f"=== {tag} ===")
        print(f"  проведено {len(want)} рёбер, прочитано {len(got)}")
        print(f"  совпало позиций: {same} из {m}")
        if same < m:
            for i in range(m):
                if want[i] != got[i]:
                    print(f"    первое расхождение на {i}: ждали {want[i]}, прочитали {got[i]}")
                    break
            fails.append(f"{tag}: расхождение после {same} рёбер")

        d0 = (res["edges"][0]["from"], res["edges"][0]["to"])
        want0 = (route[0][0], route[0][1])
        print(f"  первое ребро: {d0[0]} → {d0[1]}, проведено {want0[0]} → {want0[1]} — "
              f"{'совпадает' if d0 == want0 else 'ОБРАТНО'}")
        if d0 != want0:
            fails.append(f"{tag}: направление первого ребра обратное")

        # orientation of every passage, not just the first
        wrong_dir = 0
        for i in range(m):
            if want[i] == got[i]:
                if (res["edges"][i]["from"], res["edges"][i]["to"]) != \
                   (route[i][0], route[i][1]):
                    wrong_dir += 1
        print(f"  направление неверно у {wrong_dir} рёбер из {m}")
        if wrong_dir:
            fails.append(f"{tag}: {wrong_dir} рёбер прочитаны в обратную сторону")

        rev_got = sum(1 for i in range(1, len(got)) if got[i] == got[i - 1])
        rev_want = sum(1 for i in range(1, len(want)) if want[i] == want[i - 1])
        print(f"  разворотов: прочитано {rev_got}, проведено {rev_want}")
        if rev_got != rev_want:
            fails.append(f"{tag}: разворотов {rev_got} против {rev_want}")

        print(f"  якорь {res['anchor']}, перезахватов {res['reanchors']}, "
              f"rms {res['mean_dist_to_graph_px']} px, счёт {res['score']}")
        if res["dropped_off_graph"]:
            print(f"  отброшено точек вне графа: {res['dropped_off_graph']}")
        print()

    if fails:
        for f in fails:
            print(f"  НЕ ПРОШЛО: {f}")
        sys.exit(1)
    print("  обе линии прочитаны верно: порядок рёбер, направление и развороты")


if __name__ == "__main__":
    main()
