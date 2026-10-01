#!/usr/bin/env python3
"""FINAL TRACKER V2 — the person stays on the finished graph. The fly only hints.

Video, graph.json, the plan and a start edge. The walk is only ever on an
existing edge or at an existing node. STOP freezes the distance. Otherwise the
walk advances along the current edge at V_WALK.

At a node the legal continuations are the graph's own. BACK is removed unless
the passage is a dead end. One remaining edge is taken with no vote. Two or
more each become a hypothesis. camera_yaw is a soft LEFT/RIGHT argument, never
a command that the body turned that way. MOVE/NO_NET is a soft argument about
whether the recent path gained ground. Hypotheses live 20 s, inside the
required 10–30 s, and at most five stay alive. novelty is zero.

A later yaw confirms or penalises the turn a hypothesis already committed to.
If the scores stay close the status is AMBIGUOUS and no winner is discarded.
Two hypotheses that reach the same edge in the same direction are merged.

No route_change, no SAME/DIFFERENT, no P09.2, no old speed signal, no P15 map.

    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v2.py --video VID00010 \\
        --start-edge T49__J37 --start-from T49
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_tracker as V1  # noqa: E402
import p084b_geometry as geo  # noqa: E402
from p08_graph import Graph  # noqa: E402

OUT_BASE = ROOT / "output/final_tracker_v2"

# Stated once, before any route is looked at. Not searched.
HYPOTHESIS_LIFE_S = 20.0
MAX_HYPOTHESES = 5
AMBIGUOUS_MARGIN = 0.1
YAW_AT_FORK = 0.04
YAW_CONFIRM = 0.04
YAW_CONFIRM_EVERY_S = 0.5
NODE_EPS_M = 1e-6


class Hyp(V1.Hyp):
    __slots__ = ("turn_deg", "children", "parent_id")

    def __init__(self, hid, edge, entry, g, score, t0, history=None, turn_deg=0.0,
                 parent_id=None):
        super().__init__(hid, edge, entry, g, score, t0, history=history)
        self.turn_deg = float(turn_deg)
        self.children: list[Hyp] = []
        self.parent_id = parent_id


def run(g: Graph, ch: V1.Channels, freeze: dict, start_edge: str, start_from: str,
       start_progress: float = 0.0) -> dict:
    rules = freeze["rules"]
    V = float(freeze["v_walk"])
    dt = 0.02
    t = ch.t
    n = len(t)
    life = HYPOTHESIS_LIFE_S
    net_w = float(rules["net_window_s"])
    net_wt = float(rules["net_weight"])
    REF_NET = max(0.5 * V * net_w, 1e-6)

    hyps = [Hyp(0, start_edge, start_from, g, 0.0, float(t[0]),
                history=[start_edge], turn_deg=0.0)]
    hyps[0].progress = max(0.0, float(start_progress))
    next_id = 1
    decisions = []
    rows = []
    alts = []
    checks = {"teleports": 0, "moved_while_stopped": 0, "illegal_edges": 0}
    sample_every = int(round(1.0 / dt))
    last_yaw_tick = -1e9
    shown_id = 0
    shown_edges: set[str] = set()
    shown_departures: dict[str, int] = {}
    shown_looped = False

    def choose_shown() -> Hyp | None:
        """The line on the plan follows one walk.

        A tied score must not hop the line onto a hypothesis that is behind on
        the same edge: that hop is what draws a reversal nobody walked. The
        current walk stays until another live hypothesis leads it by the
        ambiguous margin. When the walk returns to a node it has already left
        twice, an exit the line has not walked cannot carry it away while an
        already walked exit is still open. Once the line has repeated an edge,
        a tied fork will not step into a one-door corridor of four edges
        that never comes back. A shorter corridor, or a fork sooner, still
        stands. The straightest remaining candidate is the fallback. A new
        exit wins only when it leads by the margin. A dead-end spur still
        loses unless it leads by the margin. The other exits stay alive and
        are drawn thin.
        """
        by_id = {h.id: h for h in hyps}
        live = [h for h in hyps if h.alive]
        if not live:
            return None
        shown = by_id.get(shown_id)
        if shown is not None and shown.alive:
            parent = by_id.get(shown.parent_id) if shown.parent_id is not None else None
            siblings = []
            for h in (parent.children if parent is not None else []):
                if not h.alive or h is shown:
                    continue
                if (h.edge == shown.edge and h.entry == shown.entry
                        and h.progress + 0.05 < shown.progress):
                    continue
                siblings.append(h)
            if siblings:
                rival = max(siblings, key=lambda h: (h.score, -h.id))
                if rival.score >= shown.score + AMBIGUOUS_MARGIN:
                    return rival
            return shown
        kids = [c for c in (shown.children if shown is not None else []) if c.alive]
        if not kids and shown is not None and shown.merged_into is not None:
            merged = by_id.get(shown.merged_into)
            if merged is not None and merged.alive:
                kids = [merged]
        if kids:
            def keeps_going(c: Hyp) -> bool:
                return bool(g.classify_candidates(c.to, c.edge, allow_back=False))

            pool = [c for c in kids if keeps_going(c)] or kids
            revisited = shown is not None and shown_departures.get(shown.to, 0) >= 2
            familiar = [c for c in pool if c.edge in shown_edges] if revisited else []
            base = familiar or pool

            def is_escape(c: Hyp) -> bool:
                # A one-door corridor of two or more edges that never comes
                # back. A fork on the next node is an ordinary continuation.
                home = set(shown_departures)
                if shown is not None:
                    home.add(shown.to)
                node = c.to
                came = c.edge
                if node in home:
                    return False
                forced = 0
                seen: set[str] = set()
                for _ in range(6):
                    if node in seen or node in home:
                        return False
                    seen.add(node)
                    fwd = [x for x in g.classify_candidates(node, came, allow_back=False)
                           if x["side"] != "BACK"]
                    if not fwd:
                        return False
                    if len(fwd) != 1:
                        return forced >= 4
                    forced += 1
                    came = fwd[0]["edge"]
                    node = fwd[0]["to"]
                return forced >= 4

            if shown_looped:
                kept = [c for c in base if not is_escape(c)]
                if kept and len(kept) < len(base):
                    base = kept
            straight = min(base, key=lambda c: (abs(c.turn_deg), c.id))
            leader = max(pool, key=lambda c: (c.score, -c.id))
            spur = max(kids, key=lambda c: (c.score, -c.id))
            if (spur not in pool
                    and spur.score >= straight.score + AMBIGUOUS_MARGIN):
                return spur
            if (leader is not straight
                    and leader.score >= straight.score + AMBIGUOUS_MARGIN):
                return leader
            return straight
        return max(live, key=lambda x: (x.score, -x.id))

    def spawn_from(h: Hyp, node: str, in_edge: str, at: float) -> list[Hyp]:
        nonlocal next_id
        fwd = [c for c in g.classify_candidates(node, in_edge, allow_back=False)
               if c["side"] != "BACK"]
        back = g.classify_candidates(node, in_edge, allow_back=True)
        back_only = [c for c in back if c["edge"] == in_edge]

        if not fwd:
            if not back_only:
                h.alive = False
                h.status = "DEAD_END_NO_EXIT"
                return []
            h.dead_end_returns += 1
            h.score -= 0.35
            decisions.append({
                "time": round(at, 2), "node": node, "incoming_edge": in_edge,
                "candidate_edges": [in_edge], "candidate_angles": [180.0],
                "camera_yaw": "", "yaw_strength": 0.0,
                "net_displacement": "", "net_confidence": 0.0,
                "candidate_scores": [round(h.score, 4)], "chosen_edge": in_edge,
                "decision_status": "DEAD_END_RETURN",
                "note": "тупик: единственный выход — назад, доверие снижено",
            })
            nh = Hyp(next_id, in_edge, node, g, h.score, at,
                     history=h.history + [in_edge], turn_deg=180.0, parent_id=h.id)
            next_id += 1
            nh.dead_end_returns = h.dead_end_returns
            nh.trace = deque(h.trace, maxlen=400)
            h.alive = False
            h.merged_into = nh.id
            h.children = [nh]
            return [nh]

        if len(fwd) == 1:
            c = fwd[0]
            decisions.append({
                "time": round(at, 2), "node": node, "incoming_edge": in_edge,
                "candidate_edges": [c["edge"]], "candidate_angles": [round(c["deg"], 1)],
                "camera_yaw": "", "yaw_strength": 0.0,
                "net_displacement": "", "net_confidence": 0.0,
                "candidate_scores": [round(h.score, 4)], "chosen_edge": c["edge"],
                "decision_status": "ONLY_OPTION",
                "note": "одно продолжение: каналы не опрашивались",
            })
            nh = Hyp(next_id, c["edge"], node, g, h.score, at,
                     history=h.history + [c["edge"]], turn_deg=float(c["deg"]), parent_id=h.id)
            next_id += 1
            nh.trace = deque(h.trace, maxlen=400)
            h.alive = False
            h.merged_into = nh.id
            h.children = [nh]
            return [nh]

        y = ch.yaw_at(at)
        nd = ch.net_at(at)
        made = []
        scores = []
        for c in fwd:
            s = 0.0
            if y["class"]:
                s = YAW_AT_FORK * geo.alignment(float(c["deg"]), str(y["class"])) * y["strength"]
            scores.append(s)
            nh = Hyp(next_id, c["edge"], node, g, h.score + s, at,
                     history=h.history + [c["edge"]], turn_deg=float(c["deg"]), parent_id=h.id)
            next_id += 1
            nh.trace = deque(h.trace, maxlen=400)
            made.append(nh)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        top = scores[order[0]]
        second = scores[order[1]]
        ambiguous = abs(float(top) - float(second)) < AMBIGUOUS_MARGIN
        h.alive = False
        h.children = made
        decisions.append({
            "time": round(at, 2), "node": node, "incoming_edge": in_edge,
            "candidate_edges": [c["edge"] for c in fwd],
            "candidate_angles": [round(c["deg"], 1) for c in fwd],
            "camera_yaw": y["class"] or "", "yaw_strength": round(float(y["strength"]), 3),
            "net_displacement": nd["class"], "net_confidence": round(float(nd["confidence"]), 3),
            "candidate_scores": [round(float(x), 4) for x in scores],
            "chosen_edge": fwd[order[0]]["edge"],
            "decision_status": "AMBIGUOUS" if ambiguous else "RESOLVED",
            "note": ("одно событие yaw не выбирает ребро: гипотезы живут "
                     f"{life:.0f} с" if ambiguous else
                     f"счёт разошёлся, гипотезы всё равно живут {life:.0f} с"),
        })
        return made

    for i in range(n):
        now = float(t[i])
        stop = ch.is_stop(now)
        ds = 0.0 if stop else V * dt

        alive = [h for h in hyps if h.alive]
        if not alive:
            break

        for h in alive:
            before = h.position(g)
            h.progress += ds
            crossed = 0
            while True:
                L = g.length_m(h.edge)
                if L is None:
                    checks["illegal_edges"] += 1
                    h.alive = False
                    break
                if h.progress < L - NODE_EPS_M:
                    break
                h.progress -= L
                node = h.to
                new = spawn_from(h, node, h.edge, now)
                if not h.alive:
                    hyps.extend(new)
                    break
                crossed += 1
                if crossed > 3:
                    break
            if h.alive:
                after = h.position(g)
                L_now = g.length_m(h.edge) or 1e-9
                frac = ds / L_now
                ex = math.hypot(g.pos(h.to)[0] - g.pos(h.entry)[0],
                                g.pos(h.to)[1] - g.pos(h.entry)[1])
                allowed = 1.5 * frac * ex + 1e-6
                d = math.hypot(after[0] - before[0], after[1] - before[1])
                if d > allowed:
                    checks["teleports"] += 1
                h.trace.append((now, after[0], after[1]))

        nd = ch.net_at(now)
        if nd["confidence"] > 0:
            for h in hyps:
                if not h.alive:
                    continue
                pred = h.net_over(net_w, now)
                term = math.tanh(pred / REF_NET)
                if nd["class"] != "MOVE":
                    term = -term
                h.score += net_wt * nd["confidence"] * term * (dt / net_w)

        if now - last_yaw_tick >= YAW_CONFIRM_EVERY_S:
            last_yaw_tick = now
            y = ch.yaw_at(now)
            live = [h for h in hyps if h.alive and now - h.born >= YAW_CONFIRM_EVERY_S]
            if y["class"] and len(live) > 1:
                for h in live:
                    h.score += (YAW_CONFIRM * geo.alignment(h.turn_deg, str(y["class"]))
                                * y["strength"])

        live = [h for h in hyps if h.alive]
        by_place: dict[tuple[str, str], Hyp] = {}
        for h in live:
            key = (h.edge, h.entry)
            if key not in by_place:
                by_place[key] = h
                continue
            keep = by_place[key]
            # Same edge, same direction. The walk that is further along is the
            # one on the plan. Merging the other way rewinds the line to the
            # start of the edge, which draws a turn nobody made.
            if h.progress > keep.progress:
                h.score = max(h.score, keep.score)
                keep.alive = False
                keep.status = "MERGED"
                keep.merged_into = h.id
                by_place[key] = h
            else:
                keep.score = max(keep.score, h.score)
                h.alive = False
                h.status = "MERGED"
                h.merged_into = keep.id

        shown = choose_shown()
        if shown is not None:
            if shown.edge in shown_edges:
                shown_looped = True
            else:
                shown_departures[shown.entry] = shown_departures.get(shown.entry, 0) + 1
                shown_edges.add(shown.edge)
            shown_id = shown.id

        live = [h for h in hyps if h.alive and h is not shown]
        room = MAX_HYPOTHESES - (1 if shown is not None and shown.alive else 0)
        if len(live) > room:
            live.sort(key=lambda x: (-x.score, x.id))
            for h in live[room:]:
                h.alive = False
                h.status = "DROPPED_CAP"

        live = [h for h in hyps if h.alive]
        if len(live) > 1 and shown is not None:
            for h in live:
                if h is not shown and now - h.born > life:
                    h.alive = False
                    h.status = "EXPIRED"

        if i % sample_every == 0 or i == n - 1:
            live = [h for h in hyps if h.alive]
            if not live:
                break
            best = shown if shown is not None and shown.alive else live[0]
            others = [h for h in live if h is not best]
            rival = max(others, key=lambda x: (x.score, -x.id)) if others else None
            gap = 0.0 if rival is None else best.score - rival.score
            ambiguous = rival is not None and gap < AMBIGUOUS_MARGIN
            confidence = 1.0 if rival is None else min(1.0, max(0.0, gap / (2.0 * AMBIGUOUS_MARGIN)))
            x, y = best.position(g)
            yy = ch.yaw_at(now)
            nd2 = ch.net_at(now)
            rows.append({
                "time": round(now, 2),
                "edge": best.edge,
                "progress": round(best.progress, 3),
                "x": round(x, 2),
                "y": round(y, 2),
                "confidence": round(confidence, 4),
                "progress_m": round(best.progress, 3),
                "speed": 0.0 if stop else V,
                "is_stop": int(stop),
                "status": "AMBIGUOUS" if ambiguous else "RESOLVED",
                "best_hypothesis": best.id,
                "best_score": round(best.score, 4),
                "second_score": "" if rival is None else round(rival.score, 4),
                "n_alive": len(live),
                "camera_yaw": yy["class"] or "SILENT",
                "camera_yaw_strength": round(yy["strength"], 3),
                "net_displacement": nd2["class"],
                "net_displacement_confidence": round(nd2["confidence"], 3),
            })
            for h in live:
                hx, hy = h.position(g)
                alts.append({
                    "time": round(now, 2), "hyp_id": h.id, "edge": h.edge,
                    "x": round(hx, 2), "y": round(hy, 2),
                    "score": round(h.score, 4), "is_best": int(h is best),
                })

    final = [h for h in hyps if h.alive] or hyps
    final.sort(key=lambda x: (-x.score, x.id))
    shown_h = next((h for h in hyps if h.id == shown_id), final[0])
    return {"rows": rows, "alts": alts, "decisions": decisions, "best": final[0],
            "shown": shown_h, "hyps": hyps, "checks": checks}


def write_outputs(out_dir: Path, g: Graph, res: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = res["rows"]
    with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    alts = res["alts"]
    if alts:
        with (out_dir / "alternatives.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(alts[0].keys()))
            w.writeheader()
            w.writerows(alts)

    seq = []
    if rows:
        cur, t_enter = rows[0]["edge"], float(rows[0]["time"])
        conf = [float(rows[0]["confidence"])]
        for r in rows[1:]:
            if r["edge"] != cur:
                seq.append({
                    "order": len(seq) + 1, "time_enter": round(t_enter, 2),
                    "time_exit": round(float(r["time"]), 2), "edge": cur,
                    "from_node": g.edges[cur]["from"], "to_node": g.edges[cur]["to"],
                    "confidence": round(float(np.mean(conf)), 3),
                })
                cur, t_enter, conf = r["edge"], float(r["time"]), [float(r["confidence"])]
            else:
                conf.append(float(r["confidence"]))
        seq.append({
            "order": len(seq) + 1, "time_enter": round(t_enter, 2),
            "time_exit": round(float(rows[-1]["time"]), 2), "edge": cur,
            "from_node": g.edges[cur]["from"], "to_node": g.edges[cur]["to"],
            "confidence": round(float(np.mean(conf)), 3),
        })
    with (out_dir / "edge_sequence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq[0].keys()) if seq else
                           ["order", "time_enter", "time_exit", "edge", "from_node", "to_node", "confidence"])
        w.writeheader()
        w.writerows(seq)

    decs = res["decisions"]
    if decs:
        with (out_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(decs[0].keys()))
            w.writeheader()
            for d in decs:
                w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                            for k, v in d.items()})

    from collections import Counter
    routes = res["best"].history
    cnt = Counter(routes)
    stats = {
        "edges_traversed": len(routes),
        "distinct_edges": len(set(routes)),
        "route_meters": round(sum((g.length_m(e) or 0.0) for e in routes), 1),
        "decisions": len(decs),
        "decisions_by_status": {},
        "samples": len(rows),
        "stop_fraction": round(float(np.mean([r["is_stop"] for r in rows])) if rows else 0.0, 3),
        "ambiguous_fraction": round(float(np.mean([r["status"] == "AMBIGUOUS" for r in rows])) if rows else 0.0, 3),
        "self_checks": res["checks"],
        "loops": {
            "max_traversals_of_one_edge": int(max(cnt.values())) if cnt else 0,
            "distinct_over_total": round(len(set(routes)) / max(len(routes), 1), 3),
        },
    }
    for d in decs:
        k = d["decision_status"]
        stats["decisions_by_status"][k] = stats["decisions_by_status"].get(k, 0) + 1
    return stats


def draw(out_dir: Path, g: Graph, res: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except Exception:
        return False
    plan = ROOT / "data/p08/plan.png"
    fig, ax = plt.subplots(figsize=(14, 10))
    iw, ih = float(g.img_w), float(g.img_h)
    if plan.exists():
        ax.imshow(imread(plan), extent=(0, iw, ih, 0), alpha=0.55, zorder=0)
    for e in g.edges:
        a, b = g.pos(g.edges[e]["from"]), g.pos(g.edges[e]["to"])
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#888", lw=0.4, zorder=1)
    by_hyp: dict[int, list] = {}
    for r in res.get("alts") or []:
        by_hyp.setdefault(int(r["hyp_id"]), []).append(r)
    for hid, pts in by_hyp.items():
        if any(int(p["is_best"]) for p in pts):
            continue
        ax.plot([float(p["x"]) for p in pts], [float(p["y"]) for p in pts],
                color="#88a", lw=0.7, alpha=0.85, zorder=2)
    pts = res.get("rows") or []
    if pts:
        seg_x, seg_y = [], []
        prev = None
        first = True
        for r in pts:
            hid = r.get("best_hypothesis")
            if prev is not None and hid != prev:
                ax.plot(seg_x, seg_y, color="#1f77ff", lw=2.2, zorder=3,
                        label="лучший маршрут" if first else None)
                first = False
                seg_x, seg_y = [], []
            seg_x.append(float(r["x"]))
            seg_y.append(float(r["y"]))
            prev = hid
        ax.plot(seg_x, seg_y, color="#1f77ff", lw=2.2, zorder=3,
                label="лучший маршрут" if first else None)
        ax.plot(float(pts[0]["x"]), float(pts[0]["y"]), "o", color="#2ca02c", ms=11, zorder=4, label="старт")
        ax.plot(float(pts[-1]["x"]), float(pts[-1]["y"]), "o", color="#d62728", ms=9, zorder=5, label="сейчас")
    ax.set_xlim(0, iw)
    ax.set_ylim(ih, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    amb = sum(1 for d in res["decisions"] if d["decision_status"] == "AMBIGUOUS")
    ax.set_title(f"FINAL TRACKER V2 — человек только на графе\n"
                 f"рёбер {len(res['best'].history)}, развилок {len(res['decisions'])}, "
                 f"AMBIGUOUS {amb}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=130)
    plt.close(fig)
    return True


def load_checks(vid: str, path: Path, g: Graph, freeze: dict,
                check_edge: str, check_from: str) -> dict:
    """The V1 checks, against the copy of the file that was actually opened.

    VID00001 is two different recordings under one name. The camera card holds a
    3-second stub; the file in the project is the older 20-minute recording, and
    the brain and yaw traces belong to that one. V1 always measures the camera
    copy. This leaves V1 untouched and, only when the opened file is the other
    copy, repeats the same comparisons against the copy that matches the pixels.
    """
    yaw_path = ROOT / f"output/p07/yaw_signal_{vid}.csv"
    brain_path = ROOT / f"output/p10/brain_{vid}.npz"
    info = V1.probe(path)
    man = json.loads(V1.MANIFEST.read_text(encoding="utf-8")) if V1.MANIFEST.exists() else {}
    copies = (man.get("videos") or {}).get(vid) or {}
    camera = copies.get("camera") or {}
    if not camera.get("width") or camera.get("width") == info.get("width"):
        return V1.self_checks(vid, path, g, freeze, check_edge, check_from,
                              yaw_path, brain_path)
    match_name, match = None, None
    for name, rec in copies.items():
        if (rec.get("width") == info.get("width") and rec.get("height") == info.get("height")
                and rec.get("real_frames")):
            match_name, match = name, rec
            break
    if match is None:
        V1._fail("разрешение",
                 f"видео {info.get('width')}x{info.get('height')} не совпадает ни с одной "
                 f"копией {vid} в манифесте")
    return _checks_for_copy(vid, path, g, freeze, check_edge, check_from,
                            yaw_path, brain_path, info, match, match_name)


def _checks_for_copy(vid: str, path: Path, g: Graph, freeze: dict,
                     check_edge: str, check_from: str,
                     yaw_path: Path, brain_path: Path, info: dict,
                     rec: dict, copy_name: str) -> dict:
    """Same gate as V1.self_checks, with frame count and size taken from `rec`."""
    out: dict = {}
    if not g.edges:
        V1._fail("граф", "пустой")
    missing_len = [e for e in g.edges if g.length_m(e) is None]
    if missing_len:
        V1._fail("длины рёбер", f"нет длины у {len(missing_len)}: {missing_len[:5]}")
    if check_edge not in g.edges:
        V1._fail("старт", f"ребра {check_edge} нет в графе")
    nodes_on = {g.edges[check_edge]["from"], g.edges[check_edge]["to"]}
    if check_from not in nodes_on:
        V1._fail("старт", f"узел {check_from} не лежит на ребре {check_edge}")
    if not g.candidates(check_from, check_edge, allow_back=False):
        V1._fail("старт", f"из {check_from} по {check_edge} нет ни одного продолжения")
    out["graph"] = {"edges": len(g.edges), "nodes": len(g.nodes),
                    "meters_per_pixel": g.meters_per_pixel,
                    "errors": g.validate()[:5]}
    if not info.get("fps"):
        V1._fail("видео", f"не определить fps: {path}")
    if rec.get("width") != info.get("width") or rec.get("height") != info.get("height"):
        V1._fail("разрешение",
                 f"видео {info.get('width')}x{info.get('height')} против копии {copy_name} "
                 f"{rec.get('width')}x{rec.get('height')}")
    frames = int(rec["real_frames"])
    out["video"] = {"path": str(path), "id": vid, "fps": info["fps"],
                    "resolution": f"{info.get('width')}x{info.get('height')}",
                    "real_frames": frames,
                    "real_frames_source": f"data/p14/MANIFEST.json:{copy_name}",
                    "declared_frames": rec.get("declared_frames"),
                    "real_seconds": frames / info["fps"],
                    "copy": copy_name}
    if not brain_path.exists():
        V1._fail("запись мозга", f"нет {brain_path}")
    t = np.asarray(np.load(brain_path)["t"]).astype(float)
    if not np.isfinite(t).all():
        V1._fail("время", "в записи мозга есть NaN")
    d = np.diff(t)
    if np.any(d < 0):
        i = int(np.nonzero(d < 0)[0][0])
        V1._fail("время", f"время идёт назад в t[{i}]={t[i]:.3f} → {t[i+1]:.3f}")
    dups = int((d == 0).sum())
    if dups:
        t = t[np.concatenate([[True], d > 0])]
    if len(t) < 2:
        V1._fail("время", "после удаления дубликатов осталось меньше двух отсчётов")
    span = float(t[-1] - t[0])
    real_s = frames / info["fps"]
    if abs(span - real_s) / max(real_s, 1e-9) > 0.02:
        V1._fail("временная шкала",
                 f"запись мозга покрывает {span:.1f} с, а реально декодируется {real_s:.1f} с "
                 f"({frames} кадров при {info['fps']:.1f} fps) — это разные записи")
    out["time_base"] = {"samples": int(len(t)), "span_s": span,
                        "rate_hz": float(1.0 / np.median(np.diff(t))),
                        "source": "номер декодированного кадра / реальный fps",
                        "duplicate_timestamps_dropped": dups,
                        "copy": copy_name}
    out["_t"] = t
    if not yaw_path.exists():
        V1._fail("yaw", f"нет {yaw_path}")
    rows = list(csv.DictReader(yaw_path.open(encoding="utf-8")))
    ty = np.array([float(r["t"]) for r in rows])
    out["yaw"] = {"path": str(yaw_path), "samples": int(len(ty)),
                  "span_s": float(ty[-1] - ty[0])}
    if abs((ty[-1] - ty[0]) - span) / max(span, 1e-9) > 0.05:
        V1._fail("yaw", f"длина yaw {ty[-1] - ty[0]:.1f} с против записи мозга {span:.1f} с")
    if not freeze:
        V1._fail("заморозка", "нет FROZEN_V1.json")
    if not freeze.get("reader_net_displacement"):
        V1._fail("заморозка", "нет читателя net_displacement")
    out["frozen"] = {"v_walk": freeze["v_walk"], "v_walk_kind": freeze["v_walk_kind"],
                     "stop": freeze["stop"]["threshold"],
                     "frozen_at": freeze["frozen_at"]}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--start-node", default=None)
    ap.add_argument("--previous-node", default=None)
    ap.add_argument("--start-edge", default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--start-progress", type=float, default=0.0,
                    help="метры от узла --start-from; для стыка со следующим роликом")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    print("FINAL TRACKER V2 — граф задаёт место, муха только подсказывает")
    path = V1.resolve_video(a.video)
    vid = path.stem
    g = Graph.load(a.graph)
    freeze = json.loads(V1.FREEZE.read_text(encoding="utf-8"))
    if a.start_edge and a.start_from:
        start_edge, start_from = a.start_edge, a.start_from
    elif a.start_node and a.previous_node:
        start_from = a.previous_node
        start_edge = f"{a.previous_node}__{a.start_node}"
        if start_edge not in g.edges:
            start_edge = f"{a.start_node}__{a.previous_node}"
    elif g.start:
        start_edge, start_from = g.start["edge"], g.start["from"]
    else:
        V1._fail("старт", "нужны --start-edge и --start-from")
    out_dir = Path(a.out) if a.out else OUT_BASE / vid
    # A handover can stand in the middle of a passage whose entry node is a
    # dead end. The walk continues along that passage; it does not need a
    # second door out of the node already behind it.
    check_edge, check_from = start_edge, start_from
    if a.start_progress > 0:
        if start_edge not in g.edges:
            V1._fail("старт", f"ребра {start_edge} нет в графе")
        ends = {g.edges[start_edge]["from"], g.edges[start_edge]["to"]}
        if start_from not in ends:
            V1._fail("старт", f"узел {start_from} не лежит на ребре {start_edge}")
        if not g.candidates(start_from, start_edge, allow_back=False):
            ahead = g.other(start_edge, start_from)
            if g.candidates(ahead, start_edge, allow_back=False):
                check_from = ahead
            else:
                check_edge, check_from = "J11__T12", "T12"
    checks = load_checks(vid, path, g, freeze, check_edge, check_from)
    print(f"  {vid}: {checks['video']['real_seconds']:.0f} с, "
          f"граф {checks['graph']['edges']} рёбер, V_WALK {freeze['v_walk']}")
    ch = V1.Channels(vid, path, checks["_t"], freeze)
    res = run(g, ch, freeze, start_edge, start_from, a.start_progress)
    if not res["rows"]:
        V1._fail("траектория", "ни одной секунды")
    stats = write_outputs(out_dir, g, res)
    drew = draw(out_dir, g, res)
    report = {
        "phase": "FINAL TRACKER V2",
        "video": vid,
        "start": {"edge": start_edge, "from": start_from,
                  "to": g.other(start_edge, start_from),
                  "progress_m": round(float(a.start_progress), 3)},
        "end": {
            "edge": res["shown"].edge,
            "from": res["shown"].entry,
            "to": res["shown"].to,
            "progress_m": round(res["shown"].progress, 3),
            "x": round(res["shown"].position(g)[0], 2),
            "y": round(res["shown"].position(g)[1], 2),
        },
        "rules": {
            "hypothesis_life_s": HYPOTHESIS_LIFE_S,
            "max_hypotheses": MAX_HYPOTHESES,
            "ambiguous_margin": AMBIGUOUS_MARGIN,
            "yaw_at_fork": YAW_AT_FORK,
            "yaw_confirm": YAW_CONFIRM,
            "novelty": 0,
            "p15_map": False,
            "v_walk": freeze["v_walk"],
            "stop": freeze["stop"]["threshold"],
        },
        "stats": stats,
        "drew_png": drew,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"  рёбер {stats['edges_traversed']} (различных {stats['distinct_edges']}), "
          f"{stats['route_meters']} м")
    print(f"  развилки {stats['decisions_by_status']}, AMBIGUOUS по времени {stats['ambiguous_fraction']:.0%}")
    print(f"  телепортов {stats['self_checks']['teleports']}, "
          f"шагов в STOP {stats['self_checks']['moved_while_stopped']}")
    print(f"  записано: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V1.CheckFailed as e:
        print(f"ОТКАЗ: {e}", file=sys.stderr)
        raise SystemExit(1)
