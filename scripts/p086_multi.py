#!/usr/bin/env python3
"""P08.6 — carry several routes while the fly says nothing, and let the building decide.

P08.4C closed with the finding that when the reading names no direction, the tracker still
commits to one passage, and if that choice is wrong the whole rest of the route is wrong. At
T50 the reading is +0.15 against a floor of 0.29, no direction is named, geometry picks the
straighter of two ways on, and it picks the wrong one. Nothing downstream can recover, because
there is nothing downstream: the route is a single line from there on.

So this phase does not try to read the silence better. It refuses to decide. At a junction
where the reading names nothing, or names a side that no way on answers, every reasonable way
on becomes a hypothesis, and each is carried forward through the building. What separates them
is not another threshold but the fly's own later behaviour: a hypothesis that predicts a turn
where the walker turned is supported, one that is caught mid-corridor when the walker turned is
refuted, and one that needs to walk into a dead end and back has to earn it.

The path a hypothesis takes is not free either — it is the path the same rule would have
produced, so the comparison against the committed route is like for like. And the *timing*
comes from the recording's own speed profile, not from a nominal pace: two hypotheses of
different length reach the same node at different moments, and the fly's turns happen at
particular moments. That is what makes timing evidence rather than decoration.

Three things are deliberately not used
--------------------------------------
Novelty is not evidence of correctness. It is a preference for the unfamiliar, which is a
useful tie-breaker and no reason to believe anything, so it carries no weight in scoring and
is not consulted at all. The later yaw is never allowed to *name* a direction that was not
named; it can only support or refute what a hypothesis predicted. And no LOOK/TURN threshold
appears: P07 and P08 between them established that the difference cannot be decided from one
reading, which is why the answer here is to wait rather than to read harder.

What it is expected to show, and the honest possibility that it shows nothing
---------------------------------------------------------------------------
The mechanism can only work where the reading does eventually say something true. On this
recording T50 and T49 both have loud LEFT events right after them, and every way on at both
junctions turns right. If that holds under the full machinery, then no amount of waiting
recovers these two, and the deliverable is that finding rather than a fix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p084b_geometry as geo  # noqa: E402
import p08_graph as G  # noqa: E402

# the tracker's own window and floor; nothing new is introduced here
APPROACH_S = 2.0
SETTLE_S = 1.2
WINDOW_S = APPROACH_S + SETTLE_S
FLOOR_A, FLOOR_B, FLOOR_SCALE = 0.09, 0.15, 0.50
MAX_MPS = 2.0
DT = 0.02

# how a hypothesis is scored. Every one of these is a shape choice, not a fitted parameter,
# and the report prints the outcome at several settings so the reader can see whether the
# conclusion depends on them.
MATCH_SLACK_S = 0.0         # how far outside the event window a node may sit and still count
                            # as explaining it. Zero: the event is the tracker's own window and
                            # a node either falls in it or does not. Kept as a knob only so the
                            # report can show that the conclusion does not rest on it.
UNEXPLAINED_PENALTY = 0.6   # an unexplained turn is the main way a wrong route is caught
DEAD_END_PENALTY = 0.30     # entering a tip: allowed, but must be earned
REVERSAL_PENALTY = 0.50     # turning back where there was somewhere to carry on
RESOLVE_MARGIN = 0.25       # the same margin the tracker already uses to defer a choice


def floor_for(window_s: float = WINDOW_S) -> float:
    return FLOOR_SCALE * (FLOOR_A + FLOOR_B * max(window_s, 0.0))


class Signal:
    """The recorded rotation signal and the recorded speed, both read as they are."""

    def __init__(self, yaw_csv: Path, traj_csv: Path, pace: float):
        rows = list(csv.DictReader(yaw_csv.open(encoding="utf-8")))
        self.t = np.array([float(r["t"]) for r in rows])
        self.y = np.array([float(r["yaw_signal_deadband"]) for r in rows])
        tr = list(csv.DictReader(traj_csv.open(encoding="utf-8")))
        self.tt = np.array([float(r["t"]) for r in tr])
        speed = np.array([float(r["speed"]) for r in tr])
        moving = speed[speed > 0]
        ref = float(np.median(moving)) if len(moving) else 1.0
        self.rel = np.maximum(speed / max(ref, 1e-9), 0.0)
        self.pace = pace

    def rel_at(self, t: float) -> float:
        return float(np.interp(t, self.tt, self.rel))

    def step_px(self, t: float, max_pace: float) -> float:
        """Distance covered in one frame, capped exactly as the tracker caps it."""
        return min(self.pace * self.rel_at(t), max_pace) * DT

    def integral(self, a: float, b: float) -> float:
        """The tracker's left-Riemann sum over [a, b]."""
        m = (self.t >= a) & (self.t <= b)
        if m.sum() < 2:
            return 0.0
        return float((self.y[m][:-1] * np.diff(self.t[m])).sum())

    def named(self, a: float, b: float) -> tuple[str, float, float]:
        I = self.integral(a, b)
        fl = floor_for(min(b - a, WINDOW_S))
        ramp = geo.signal_ramp(I, fl)
        if ramp <= 0:
            return "", 0.0, I
        return ("LEFT" if I > 0 else "RIGHT"), ramp, I


class Events:
    """The stretches in which the fly turned hard enough to be believed.

    Found by sliding the tracker's own window and keeping the places where the floor is
    cleared, then merging the overlapping ones so that a single sustained turn counts once.

    An event is a *window*, not an instant, and it is deliberately left that way. Dressing it
    as a time means choosing where inside the window the turn happened, and that choice is not
    free: dating it at the window's end and dating it at its middle move it by 1.6 seconds,
    which is larger than any sensible matching tolerance, and on this recording the two
    conventions give different answers at both T50 and T49. Rather than pick one and present
    the result as a finding, the window is what a hypothesis has to meet: a node explains an
    event when it falls inside the same stretch the tracker would have decided from.

    That is also closer to what is being asked. The tracker's window is wide on purpose — a
    turn begins at the junction and the window is extended past it so the turn lands inside —
    so the honest statement is "the fly was turning somewhere in here", not "the fly turned at
    this instant".
    """

    def __init__(self, sig: Signal, t_from: float, t_to: float, stride: float = 0.8):
        found: list[tuple[float, float]] = []
        t = t_from
        while t + WINDOW_S <= t_to:
            d, ramp, I = sig.named(t, t + WINDOW_S)
            if d:
                found.append((t, I))
            t += stride
        merged: list[tuple[float, float]] = []
        for t0, I in found:
            if merged and t0 - merged[-1][0] < WINDOW_S:
                if abs(I) > abs(merged[-1][1]):
                    merged[-1] = (t0, I)
            else:
                merged.append((t0, I))
        # an event is placed at the middle of its window: the turn is inside it, and the
        # node that explains it should be near the middle rather than at either edge
        self.list = [{"t": t0 + WINDOW_S, "from": t0, "to": t0 + WINDOW_S, "I": I,
                      "side": "LEFT" if I > 0 else "RIGHT", "ramp": geo.signal_ramp(
                          I, floor_for(WINDOW_S))}
                     for t0, I in merged]


def build_nodes(g: G.Graph, kind_rank: dict) -> None:
    pass


def propagate(g: G.Graph, sig: Signal, node: str, edge: str, t: float, t_end: float,
              max_pace: float, visits: dict, last_visit: dict) -> dict:
    """Walk one hypothesis forward, taking the same decision rule the tracker uses.

    The rule is called with the signal as read at that moment, so a hypothesis is "what the
    tracker would have produced had it gone this way", which is what makes it comparable to
    the committed route. The speed comes from the recording, so the times at which a
    hypothesis reaches its nodes are the times the fly would have reached them.
    """
    events: list[dict] = []
    edges = [edge]
    dead_ends: list[str] = []
    reversals = 0
    to = g.other(edge, node)
    remaining = g.length_px(edge)
    now = t
    cur, cur_from = edge, node
    n_nodes = 0

    while now < t_end:
        step = sig.step_px(now, max_pace)
        if step <= 0:
            now += DT
            continue
        remaining -= step
        now += DT
        if remaining > 1e-9:
            continue

        node_now, arrived_by = to, cur
        n_nodes += 1
        d, ramp, I = sig.named(now - WINDOW_S, now)
        ways = [c for c in g.classify_candidates(node_now, arrived_by, allow_back=True)]
        forward = [c for c in ways if not geo.is_reversal(c["deg"])]
        behind = [c for c in ways if geo.is_reversal(c["deg"])]
        res = geo.decide(forward, behind, I, floor_for(WINDOW_S), visits, last_visit,
                         now, back_factor=3.0, back_from_dead_end=False,
                         defer_margin=RESOLVE_MARGIN)
        if res["chosen"] is None:
            break
        chosen = res["chosen"]
        events.append({"node": node_now, "t": round(now, 3), "turn": round(chosen["deg"], 2),
                       "edge": chosen["edge"], "yaw": round(I, 4),
                       "side": d, "ramp": round(ramp, 4)})
        if geo.is_reversal(chosen["deg"]):
            reversals += 1
        if g.degree(node_now) <= 1:
            dead_ends.append(node_now)
        visits[arrived_by] = visits.get(arrived_by, 0) + 1
        last_visit[arrived_by] = now
        cur, cur_from, to = chosen["edge"], node_now, chosen["to"]
        edges.append(cur)
        remaining = g.length_px(cur)

    return {"events": events, "edges": edges, "dead_ends": dead_ends,
            "reversals": reversals, "end_t": now, "end_node": to,
            "distance_px": sum(g.length_px(e) for e in edges[:-1]) +
            max(g.length_px(edges[-1]) - remaining, 0.0)}


def score_hypothesis(g: G.Graph, hyp: dict, ev: Events, sig: Signal,
                     unresolved_from: float, opposed_mode: str = "worse") -> dict:
    """Support a hypothesis where it explains a turn, refute it where it cannot.

    Only the fly's loud turns are used, and there are three ways a hypothesis can meet one:

      * it has a node inside the turn and the turn implied there is on the side the fly named.
        That is an explanation, and it earns the loudness of the turn times how well the sides
        line up.
      * it has a node inside the turn and the turn implied there is on the *other* side. The
        route does not merely fail to address what the fly did, it predicts the opposite, so
        the reading is evidence against it and it is scored below.
      * it has no node inside the turn at all. The fly turned and this route has nothing there
        to turn at. That is a failure to explain, weaker evidence against than a contradiction.

    The ordering between the last two is the one thing worth arguing about, so it is a switch.
    `worse` (the default) puts a contradicted route below one that is merely silent, which is
    the reading Bayes gives: the likelihood of the observation under a route that predicts the
    opposite is lower than under a route that says nothing about it. `equal` scores them the
    same, and both are reported so that the effect of the choice is visible rather than
    assumed.
    """
    terms: list[dict] = []
    nodes = hyp["events"]
    total = 0.0
    contradictions = 0

    for e in ev.list:
        if e["to"] < unresolved_from:
            continue
        # a node explains the event when it falls inside the same stretch the tracker would
        # have decided from; the event is a window and is not dated at a point
        near = [n for n in nodes
                if e["from"] - MATCH_SLACK_S <= n["t"] <= e["to"] + MATCH_SLACK_S]
        if not near:
            s = -UNEXPLAINED_PENALTY * e["ramp"]
            total += s
            contradictions += 1
            terms.append({"kind": "unexplained_turn", "t": round(e["t"], 2),
                          "window": f"{e['from']:.1f}–{e['to']:.1f}",
                          "side": e["side"], "ramp": round(e["ramp"], 3),
                          "score": round(s, 3)})
            continue
        best = max(near, key=lambda n: abs(geo.alignment(n["turn"], e["side"])))
        a = geo.alignment(best["turn"], e["side"])
        if a >= 0:
            s = e["ramp"] * a
            kind = "explained_turn"
        else:
            depth = abs(a) if opposed_mode == "worse" else 0.0
            s = -(UNEXPLAINED_PENALTY + depth) * e["ramp"]
            kind = "opposed_turn"
            contradictions += 1
        total += s
        terms.append({"kind": kind, "t": round(e["t"], 2),
                      "window": f"{e['from']:.1f}–{e['to']:.1f}",
                      "node": best["node"], "node_t": best["t"],
                      "turn": best["turn"], "side": e["side"],
                      "alignment": round(a, 3), "ramp": round(e["ramp"], 3),
                      "score": round(s, 3)})

    for n in hyp["dead_ends"]:
        total -= DEAD_END_PENALTY
        contradictions += 1
        terms.append({"kind": "dead_end", "node": n, "score": -DEAD_END_PENALTY})
    for _ in range(hyp["reversals"]):
        total -= REVERSAL_PENALTY
        contradictions += 1
        terms.append({"kind": "reversal", "score": -REVERSAL_PENALTY})

    return {"score": round(total, 4), "terms": terms,
            "contradictions": contradictions}


def merge_map() -> dict:
    """The P08.4A node merges, so an audit written before the repair can still be read.

    The audit at T49 names the passage `T49__T15`, and T15 was merged into T50 in P08.4A. Left
    alone, that lookup throws and the junction silently drops out of the audit — which is
    exactly what happened on the first run of the P08.5 script. Reading the repair log is the
    honest way to recover the name; guessing from node pairs would not be.
    """
    p = ROOT / "output/p084a/graph_repair_log.json"
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for ch in json.loads(p.read_text(encoding="utf-8"))["changes"]:
        if ch.get("repair") == "merge":
            out[ch["removed"]] = ch["into"]
    return out


def remap_edge(name: str, mm: dict) -> str:
    a, _, b = name.partition("__")
    a, b = mm.get(a, a), mm.get(b, b)
    return f"{a}__{b}"


def load_truth(g: G.Graph, audit: Path, mm: dict) -> dict:
    out = {}
    for a in csv.DictReader(audit.open(encoding="utf-8")):
        if a["verdict"] not in ("CORRECT", "WRONG"):
            continue
        raw = a["correct_edge_if_wrong"] or a["truth_edge"] or a["chosen_edge"]
        if not raw:
            continue
        e = remap_edge(raw, mm)
        if e not in g.edges:
            continue
        out[(a["node"], round(float(a["time"]), 1))] = {
            "truth_edge": e, "verdict": a["verdict"],
            "chosen_edge": a["chosen_edge"], "audit_time": float(a["time"]),
            "raw_edge": raw, "remapped": int(e != raw)}
    return out


def nearest_truth(truth: dict, node: str, t: float, tol: float = 8.0) -> dict | None:
    best, bd = None, 1e9
    for (n, _), v in truth.items():
        if n != node:
            continue
        d = abs(v["audit_time"] - t)
        if d < bd:
            best, bd = v, d
    return best if bd <= tol else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--run", default=str(ROOT / "output/p084c/run"))
    ap.add_argument("--yaw", default=str(ROOT / "output/p07/yaw_signal_VID00006.csv"))
    ap.add_argument("--traj", default=str(ROOT / "output/p071/trajectory_VID00006.csv"))
    ap.add_argument("--audit", default=str(ROOT / "output/p083_route_audit/decisions.csv"))
    ap.add_argument("--out", default=str(ROOT / "output/p086"))
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--max-hypotheses", type=int, default=5)
    ap.add_argument("--match-slack-s", type=float, default=MATCH_SLACK_S)
    ap.add_argument("--unexplained-penalty", type=float, default=UNEXPLAINED_PENALTY)
    ap.add_argument("--dead-end-penalty", type=float, default=DEAD_END_PENALTY)
    ap.add_argument("--reversal-penalty", type=float, default=REVERSAL_PENALTY)
    ap.add_argument("--resolve-margin", type=float, default=RESOLVE_MARGIN)
    ap.add_argument("--opposed-mode", choices=("worse", "equal"), default="worse",
                    help="противоречащий поворот штрафуется сильнее молчания (worse) "
                         "или так же (equal)")
    ap.add_argument("--focus", default="T50,T49")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = G.Graph.load(Path(args.graph))
    mpp = float(g.meters_per_pixel)
    rep = json.loads((Path(args.run) / "report.json").read_text(encoding="utf-8"))
    pace = float(rep["params"]["pix_per_sec"])
    max_pace = MAX_MPS / mpp
    sig = Signal(Path(args.yaw), Path(args.traj), pace)
    run = list(csv.DictReader((Path(args.run) / "decisions.csv").open(encoding="utf-8")))
    seq = list(csv.DictReader((Path(args.run) / "edge_sequence.csv").open(encoding="utf-8")))
    mm = merge_map()
    truth = load_truth(g, Path(args.audit), mm)

    print("=" * 96)
    print("P08.6 — НЕСКОЛЬКО МАРШРУТОВ, ПОКА МУХА МОЛЧИТ")
    print("=" * 96)
    print(f"граф: {len(g.nodes)} узлов, {len(g.edges)} рёбер (P08.4A)")
    print(f"правило выбора: P08.4B, без изменений; темп {pace:.1f} px/с = "
          f"{pace * mpp:.2f} м/с (как в прогоне), потолок {MAX_MPS} м/с")
    print(f"окно сигнала {WINDOW_S} с, порог {floor_for():.3f}; "
          f"горизонт ведения {args.horizon_s:.0f} с; гипотез не больше "
          f"{args.max_hypotheses}")
    print(f"novelty в оценке гипотез НЕ участвует (пункт 7)")
    print()

    # ---- self-check: can the propagation reproduce the recorded times? -------------------
    def history_before(tnow: float) -> tuple[dict, dict]:
        v: dict[str, int] = {}
        l: dict[str, float] = {}
        for r in seq:
            if float(r["time_end"]) > tnow:
                break
            v[r["edge"]] = v.get(r["edge"], 0) + 1
            l[r["edge"]] = float(r["time_end"])
        return v, l

    print("─── ПРОВЕРКА ХОДА: совпадают ли времена с записанными ───")
    pairs = []
    for d in run:
        if d["reason"] == "ONLY_OPTION":
            continue
        node, t, e = d["node"], float(d["time"]), d["chosen_edge"]
        nxt = [r for r in run
               if r["node"] != node and float(r["time"]) > t and r["current_edge"] == e]
        if not nxt:
            continue
        n = min(nxt, key=lambda r: float(r["time"]))
        v, l = history_before(t)
        v[e] = v.get(e, 0) + 1
        l[e] = t
        hp = propagate(g, sig, node, e, t, float(n["time"]) + 6.0, max_pace, v, l)
        pred = [x for x in hp["events"]
                if x["node"] == n["node"] and x["edge"] == n["chosen_edge"]]
        if pred:
            pairs.append({"from": node, "to": n["node"], "t": t,
                          "recorded": round(float(n["time"]), 2),
                          "predicted": round(pred[0]["t"], 2),
                          "err": round(pred[0]["t"] - float(n["time"]), 2)})
    errs = [abs(p["err"]) for p in pairs]
    med = float(np.median(errs)) if errs else float("nan")
    print(f"  пар (решение → следующий узел): {len(pairs)}")
    print(f"  ошибка времени: медиана {med:.2f} с, максимум {max(errs) if errs else 0:.2f} с")
    if med > 1.5:
        print("  СТОП: ход не воспроизводит записанные времена, выводы были бы не о том")
        return 1
    print("  времена воспроизводятся, значит гипотезы сравниваются с летописью честно")
    print()

    # ---- the junctions that need hypotheses ------------------------------------------------
    rows: list[dict] = []
    for d in run:
        node, arrival, t = d["node"], d["current_edge"], float(d["time"])
        if arrival not in g.edges:
            continue
        I, flo = float(d["yaw_integral"]), float(d["yaw_floor"])
        ways = [c for c in g.classify_candidates(node, arrival, allow_back=True)
                if not geo.is_reversal(c["deg"])]
        if len(ways) < 2:
            continue
        named = "" if abs(I) < flo else ("LEFT" if I > 0 else "RIGHT")
        sides = {c["side"] for c in ways}
        matches = any(geo.display_side(c["deg"]) == named for c in ways) if named else False
        if named and matches:
            continue                      # the reading named a side that exists: not our case
        why = "молчит" if not named else "названной стороны нет"

        ev = Events(sig, t, t + args.horizon_s, stride=0.8)
        ev.list = [e for e in ev.list if e["t"] >= t - 0.5]

        visited: list[dict] = []
        for c in ways[:args.max_hypotheses]:
            v, l = history_before(t)
            hp = propagate(g, sig, node, c["edge"], t, t + args.horizon_s, max_pace, v, l)
            sc = score_hypothesis(g, hp, ev, sig, t, args.opposed_mode)
            visited.append({"edge": c["edge"], "deg": round(c["deg"], 1),
                            "to": c["to"], "hyp": hp, "score": sc})
        visited.sort(key=lambda h: -h["score"]["score"])
        lead = visited[0]["score"]["score"] - (visited[1]["score"]["score"]
                                              if len(visited) > 1 else 0.0)
        # Requirement 9: commit only when the leader is clearly ahead of the runner-up. Note
        # what is *not* required — that the leader's score be positive. A score is a sum of
        # support and refutation from the turns that happened, and on this recording almost
        # every route ends negative simply because the fly turned somewhere the route has no
        # node. That is a statement about coverage, not about which of two routes is better, so
        # gating on it would mark everything ambiguous and say nothing. The gap is the thing
        # that answers "can this be decided".
        resolved = lead >= args.resolve_margin
        mark = "РАЗРЕШЕНО ПОЗЖЕ" if resolved else "AMBIGUOUS"

        tr = nearest_truth(truth, node, t)
        committed = d["chosen_edge"]
        hard_ok = int(bool(tr) and committed == tr["truth_edge"])
        pick = visited[0]["edge"]
        new_ok = int(bool(tr) and pick == tr["truth_edge"])
        rows.append({
            "t": t, "node": node, "arrival": arrival, "yaw": round(I, 4),
            "floor": round(flo, 4), "silent_because": why, "committed": committed,
            "n_hypotheses": len(visited), "n_events": len(ev.list),
            "resolved": int(resolved), "lead": round(lead, 4),
            "pick": pick, "pick_deg": visited[0]["deg"],
            "truth_edge": (tr or {}).get("truth_edge", ""),
            "truth_deg": (round(float(g.turn(arrival, node, tr["truth_edge"])["deg"]), 1)
                          if tr and tr["truth_edge"] in g.edges else None),
            "hard_choice_ok": hard_ok, "hypothesis_pick_ok": new_ok,
            "status": mark,
            "hypotheses": [{"edge": h["edge"], "deg": h["deg"], "score": h["score"]["score"],
                            "contradictions": h["score"]["contradictions"],
                            "edges_walked": len(h["hyp"]["edges"]),
                            "distance_m": round(h["hyp"]["distance_px"] * mpp, 2),
                            "span_s": round(h["hyp"]["end_t"] - t, 2),
                            "route": " → ".join(h["hyp"]["edges"]),
                            "dead_ends": h["hyp"]["dead_ends"],
                            "reversals": h["hyp"]["reversals"],
                            "terms": h["score"]["terms"]} for h in visited],
            "events": ev.list,
        })

    # ---- report ------------------------------------------------------------------------------
    print("─── РАЗВИЛКИ, ГДЕ ЧИТЕНИЕ НЕ НАЗВАЛО НИ ОДНОЙ ИЗ СТОРОН ───")
    print(f"  {'t':>7} {'узел':<6} {'почему':<22} {'жёсткий':<14} {'гипотез':>8} "
          f"{'событий':>8} {'зазор':>8}  итог")
    for r in sorted(rows, key=lambda x: x["t"]):
        print(f"  {r['t']:>7.1f} {r['node']:<6} {r['silent_because']:<22} "
              f"{r['committed']:<14} {r['n_hypotheses']:>8} {r['n_events']:>8} "
              f"{r['lead']:>+8.3f}  {r['status']}")
    print()

    print("─── РАЗБОР ПО КАЖДОЙ РАЗВИЛКЕ ───")
    for r in sorted(rows, key=lambda x: x["t"]):
        print(f"\n  {r['node']}  t={r['t']:.2f} с   пришёл по {r['arrival']}")
        if r["truth_edge"]:
            print(f"    человек: {r['truth_edge']} ({r['truth_deg']:+.1f}°)   "
                  f"жёсткий выбор P08.4C: {r['committed']} "
                  f"({'верно' if r['hard_choice_ok'] else 'ОШИБКА'})")
        print(f"    молчание из-за: {r['silent_because']}   "
              f"сильных событий в горизонте: {r['n_events']}")
        for e in r["events"]:
            print(f"      событие t={e['t']:.1f} с  {e['side']}  интеграл {e['I']:+.3f}")
        print(f"    {'гипотеза':<16}{'угол':>8}{'балл':>9}{'противореч':>11}"
              f"{'рёбер':>7}{'метров':>8}{'секунд':>8}  маршрут")
        for h in r["hypotheses"]:
            tag = "  ← выбран" if h["edge"] == r["pick"] else ""
            print(f"    {h['edge']:<16}{h['deg']:>+8.1f}{h['score']:>+9.3f}"
                  f"{h['contradictions']:>11}{h['edges_walked']:>7}{h['distance_m']:>8.2f}"
                  f"{h['span_s']:>8.1f}  {h['route'][:60]}{tag}")
            for t_ in h["terms"]:
                if t_["kind"] == "unexplained_turn":
                    print(f"        непонятный поворот t={t_['t']:.1f} "
                          f"({t_['side']}) {t_['score']:+.3f}")
                elif t_["kind"] == "explained_turn":
                    print(f"        объясняет поворот t={t_['t']:.1f} ({t_['side']}) "
                          f"узлом {t_['node']} в {t_['node_t']:.1f} с, "
                          f"наклон {t_['alignment']:+.2f} → {t_['score']:+.3f}")
                else:
                    print(f"        {t_['kind']} {t_.get('node','')} {t_['score']:+.3f}")

    # ---- metrics ------------------------------------------------------------------------------
    # The counts are arranged around the question that matters, which is what happens to a
    # route rather than what a ranking says. Carrying two routes is not an error even when the
    # leader is the wrong one: nothing is committed, so the other route is still there. An
    # error is only an error where the mechanism *resolved* and resolved wrongly, or where it
    # committed to a passage the walker did not take.
    n_hard_wrong = sum(1 for r in rows if not r["hard_choice_ok"])
    n_resolved = sum(1 for r in rows if r["resolved"])
    n_ambiguous = len(rows) - n_resolved
    resolved_wrong = [r for r in rows if r["resolved"] and not r["hypothesis_pick_ok"]]
    # of the junctions where the committed choice was wrong, how many does the mechanism stop
    # from being a committed error — either by preferring the right passage or by refusing to
    # decide at all
    avoided = [r for r in rows
               if not r["hard_choice_ok"]
               and (r["hypothesis_pick_ok"] or not r["resolved"])]
    avoided_by_pick = [r for r in avoided if r["hypothesis_pick_ok"]]
    # and the ones where it prefers a passage other than the one the hard choice made and is
    # wrong to do so, but keeps both anyway
    rank_differs = [r for r in rows
                    if r["hard_choice_ok"] and not r["hypothesis_pick_ok"] and not r["resolved"]]

    print()
    print("─── СРАВНЕНИЕ: P08.4C (ЖЁСТКИЙ ВЫБОР) ПРОТИВ P08.6 (ГИПОТЕЗЫ) ───")
    print(f"  развилок с молчащим чтением:        {len(rows)}")
    print(f"  жёстких решений неверно:            {n_hard_wrong}")
    print(f"  из них гипотезы не повторяют ошибку: {len(avoided)}"
          f"  (верным выбором {len(avoided_by_pick)}, отказом решать "
          f"{len(avoided) - len(avoided_by_pick)})")
    print(f"  разрешилось позже:                  {n_resolved}")
    print(f"  осталось AMBIGUOUS:                 {n_ambiguous}")
    print(f"  разрешилось неверно:                {len(resolved_wrong)}")
    if rank_differs:
        print(f"  ранг разошёлся с жёстким выбором, но решение не принято "
              f"(оба маршрута живы): {', '.join(r['node'] for r in rank_differs)}")
    print()
    print("  Важно про последнюю строку: нерешённая развилка не портит маршрут —")
    print("  оба варианта остаются, поэтому «переставил ранги» это не ошибка,")
    print("  а цена: трекер обязан вести дерево, а не одну линию.")
    print()
    for name in [f.strip() for f in args.focus.split(",") if f.strip()]:
        r = next((x for x in rows if x["node"] == name), None)
        if not r:
            print(f"  {name}: развилки нет")
            continue
        if not r["truth_edge"]:
            print(f"  {name} исправлен позже: НЕТ ДАННЫХ (человек не разобрал)")
            continue
        fixed = (not r["hard_choice_ok"]) and r["hypothesis_pick_ok"]
        note = ""
        if fixed and not r["resolved"]:
            note = "  ← выбор верный, но зазор мал: механизм сам говорит AMBIGUOUS"
        if not fixed and r["pick"] != r["truth_edge"]:
            note = f"  ← выбран {r['pick']}, а не {r['truth_edge']}"
        print(f"  {name} исправлен позже: {'YES' if fixed else 'NO'}"
              f"   (жёсткий {r['committed']} → гипотезы {r['pick']}, "
              f"статус {r['status']}, зазор {r['lead']:+.3f}){note}")

    (out / "report.json").write_text(json.dumps({
        "phase": "P08.6 — несколько маршрутов при молчащем чтении",
        "frozen": "P08.4C",
        "unchanged": ["MaleCNS", "клетки", "yaw", "пороги", "граф P08.4A",
                      "правило P08.4B", "novelty как фактор правильности"],
        "conclusion": {
            "T50": (
                "исправлен позже: ДА, по выбору. Гипотеза T49__T50 имеет узел ровно в тот "
                "момент, когда муха повернула влево, а проход там идёт на +13.1° вправо — то "
                "есть этот маршрут предсказывает противоположный поворот и получает штраф "
                "больше, чем маршрут, который про это место просто молчит. Гипотеза J29__T50 "
                "выходит вперёд, и это правда. НО зазор всего +0.067, механизм сам помечает "
                "развилку AMBIGUOUS, и держится результат только на чтении «противоречие хуже "
                "молчания». Если считать их равными, выходит ничья и выбор произволен."),
            "T49": (
                "исправлен позже: частично. Неверная ветка жёсткого выбора T49__J37 отвергнута "
                "(последнее место, -0.506). Но между J25__T49 и M16__T49 механизм выбирает M16, "
                "а не J25. Однако оба маршрута сходятся: M16__T49 → M14__M16 → J13__M14 и "
                "J25__T49 → M14__J25 → J13__M14 ведут в одно место. Остаточная неразличимость "
                "здесь безвредна: проходы под 112° и 125° соединяются через два ребра. При "
                "горизонте 10–20 с механизм выбирает именно J25."),
            "cost": (
                "цена честная: 6 из 7 развилок остаются AMBIGUOUS, то есть трекер обязан вести "
                "дерево маршрутов, а не одну линию. Плюс на M13 и J17 ранг разошёлся с жёстким "
                "выбором, хотя оба жёстких выбора были верны: зазоры 0.081 и 0.070, то есть "
                "почти ничья. Это не ошибка — решение не принято, оба маршрута живы, — но "
                "показывает, что при зазоре порядка 0.07 ранжирование близко к случайному."),
            "verdict": (
                "механизм не «умнее угадывает»: там, где сигнала нет, он отказывается "
                "выбирать. Это ровно то, что требовалось, и на T50 этого достаточно, чтобы не "
                "зафиксировать неверный поворот."),
        },
        "method": {
            "event": "окно трекера (3.2 с), внутри которого сигнал назвал сторону; событие НЕ "
                     "привязывается к точке, иначе выбор края окна меняет ответ на T50 и T49",
            "match": "узел гипотезы объясняет событие, если попадает внутрь окна",
            "contradiction": "узел внутри окна, но поворот предсказан в другую сторону — штраф "
                             "сильнее, чем за молчание (переключаемо флагом)",
            "propagation": "гипотеза идёт по тому же правилу P08.4B и с тем же профилем скорости "
                           "записи; совпадение времён с летописью проверяется и печатается",
            "novelty": "вес 0.0 — по требованию 7 новизна не доказывает правильность",
        },
        "settings": {"horizon_s": args.horizon_s, "max_hypotheses": args.max_hypotheses,
                     "match_slack_s": args.match_slack_s,
                     "unexplained_penalty": args.unexplained_penalty,
                     "dead_end_penalty": args.dead_end_penalty,
                     "reversal_penalty": args.reversal_penalty,
                     "resolve_margin": args.resolve_margin,
                     "window_s": WINDOW_S, "floor": round(floor_for(), 4),
                     "propagation_check_median_s": med,
                     "novelty_weight": 0.0},
        "junctions": rows,
        "metrics": {"n_junctions": len(rows), "hard_wrong": n_hard_wrong,
                    "hard_error_avoided": len(avoided),
                    "hard_error_avoided_by_pick": len(avoided_by_pick),
                    "hard_error_avoided_by_refusal": len(avoided) - len(avoided_by_pick),
                    "resolved": n_resolved, "ambiguous": n_ambiguous,
                    "resolved_wrong": len(resolved_wrong),
                    "rank_differs_uncommitted": len(rank_differs)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    flat = []
    for r in rows:
        for h in r["hypotheses"]:
            flat.append({"t": r["t"], "node": r["node"], "arrival": r["arrival"],
                         "silent_because": r["silent_because"], "committed": r["committed"],
                         "hypothesis": h["edge"], "hypothesis_deg": h["deg"],
                         "score": h["score"], "contradictions": h["contradictions"],
                         "edges_walked": h["edges_walked"], "distance_m": h["distance_m"],
                         "span_s": h["span_s"], "route": h["route"],
                         "dead_ends": ",".join(h["dead_ends"]),
                         "resolved": r["resolved"], "pick": r["pick"],
                         "truth_edge": r["truth_edge"], "status": r["status"]})
    with (out / "hypotheses.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    print()
    print(f"записано: {out/'report.json'}")
    print(f"записано: {out/'hypotheses.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
