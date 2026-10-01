#!/usr/bin/env python3
"""P08.4C — keep two ways on, and let the next few seconds choose between them.

The defect this addresses is not the one P08.4B fixed. Geometry now reads the real angle of
a passage, and that was enough to say at J35 that J35__E5 is 112 degrees off the heading and
J35__J36 is 51. It was not enough to decide, because the fly named LEFT and E5 is the left
way on. The fly really did say LEFT. The human watching the video says the walker went right.

What makes that decidable
-------------------------
A glance and a turn are the same event seen through a two-second window: both are a large
rotation in one direction. They differ in what happens next. A glance goes out and comes
back, so the rotation in the window after the node has the *opposite* sign. A turn is
sustained, so it has the same sign. Measured on VID00006 at the J35 decision:

    window before the node   +0.831   LEFT    <- the glance
    window after the node    -0.418   RIGHT   <- the turn

so the reading that named E5 describes something the fly undid, and the way on that survives
is the one the later rotation actually supports. Nothing new is thresholded here: both
windows are the length the tracker already uses (`APPROACH_S + SETTLE_S`), and the floor is
the one it already applies, including the 0.5 factor that the first attempt at this check
forgot — without it the post-window of 0.418 sits under a floor of 0.570 and the contradiction
is invisible.

This is applied as a *refutation only*: the later window may withdraw a direction the earlier
one named, and is not allowed to name one of its own. Letting it name one was tried and is
wrong — at the junction where the walker comes back out of the dead end the later window is
still full of the reversal it has just performed, and on VID00006 that turned a correct
decision into an incorrect one. Withdrawing is safe because it can only remove evidence, and
removing evidence returns the choice to geometry, which is where the map's own answer lives.

Two hypotheses, and what scores them
-----------------------------------
At a junction with more than one way on, the top two by the P08.4B score are both kept. Each
is simulated forward: for the length of the evaluation window, or until two further nodes
have been passed, whichever comes first. A hypothesis scores on three things, all of them
already defined elsewhere in the pipeline:

  1. the rotation the fly performed at the branch, compared with the turn this hypothesis
     implies there, through the same `alignment` P08.4B uses;
  2. at every further node the hypothesis passes, the turn it implies there against the
     rotation the fly performed in that node's own window;
  3. a demerit if the hypothesis walks into a node with nothing beyond it and turns straight
     round. A dead end is not an error in itself — the walker has to come back — but a
     hypothesis that needs one to be true must earn it, which is what the demerit says.

No LOOK/TURN threshold appears, which is deliberate: P07 and P08 between them established
that this cannot be decided from one yaw reading, and the whole point here is to stop asking
it to.
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

# The tracker's own window and floor. Kept in one place so the reader can see that neither
# was chosen here.
APPROACH_S = 2.0
SETTLE_S = 1.2
WINDOW_S = APPROACH_S + SETTLE_S
FLOOR_A = 0.09
FLOOR_B = 0.15
FLOOR_SCALE = 0.50
KIND_RANK = {"junction": 3, "manual": 2, "turn": 1, "endpoint": 0}


def floor_for(window_s: float, scale: float = FLOOR_SCALE) -> float:
    return scale * (FLOOR_A + FLOOR_B * max(float(window_s), 0.0))


class Yaw:
    """The deadbanded rotation signal, integrated the way the tracker integrates it."""

    def __init__(self, path: Path, column: str):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        self.t = np.array([float(r["t"]) for r in rows])
        self.y = np.array([float(r[column]) for r in rows])

    def integral(self, a: float, b: float) -> float:
        """A left-Riemann sum over [a, b], as the tracker accumulates it frame by frame.

        Worth being explicit: the trapezoid rule gives a different answer on this signal,
        because the deadbanded rate alternates between samples. At the J35 decision the
        rectangle sum gives +0.83 and the trapezoid +0.39, so using the wrong one would have
        hidden the whole effect this phase is about.
        """
        m = (self.t >= a) & (self.t <= b)
        if m.sum() < 2:
            return 0.0
        return float((self.y[m][:-1] * np.diff(self.t[m])).sum())

    def side(self, a: float, b: float, scale: float = FLOOR_SCALE) -> tuple[str, float, float]:
        """The direction named over [a, b], the ramp behind it, and the integral itself."""
        I = self.integral(a, b)
        win = min(b - a, WINDOW_S)
        fl = floor_for(win, scale)
        ramp = geo.signal_ramp(I, fl)
        if ramp <= 0:
            return "", 0.0, I
        return ("LEFT" if I * 1 > 0 else "RIGHT"), ramp, I


def simulate(g: G.Graph, node: str, first_edge: str, t0: float, pace: float,
             window_s: float, max_nodes: int, decide_kwargs: dict) -> dict:
    """Walk one hypothesis forward and record every turn it implies, with its time.

    The pace is the nominal one from the run rather than the fly's measured speed, because a
    hypothesis is a claim about which way the walker went, not about how fast; using the run's
    own speed profile would smuggle in the answer to the question being asked. The assumption
    is stated rather than hidden: `pace` is passed in and printed.
    """
    events = [{"node": node, "time": t0, "edge": first_edge, "deg": None, "kind": "branch"}]
    hits: list[dict] = []
    edge, frm = first_edge, node
    to = g.other(edge, node)
    t = t0
    visited_edges = [edge]
    visits: dict[str, int] = {}
    last_visit: dict[str, float] = {}
    nodes_passed = 0

    while t < t0 + window_s and nodes_passed < max_nodes:
        length = g.length_px(edge)
        remaining = length * (1.0 - 0.0)
        dt = remaining / max(pace, 1e-9)
        t += dt
        nodes_passed += 1
        if t >= t0 + window_s:
            break

        # at the new node, take the way on the same rule would take, and record the turn
        ways = [c for c in g.classify_candidates(to, edge, allow_back=True)]
        forward = [c for c in ways if not geo.is_reversal(c["deg"])]
        behind = [c for c in ways if geo.is_reversal(c["deg"])]
        res = geo.decide(forward, behind, 0.0, floor_for(WINDOW_S), visits, last_visit, t,
                         back_factor=3.0, back_from_dead_end=False, **decide_kwargs)
        if res["chosen"] is None:
            break
        chosen = res["chosen"]
        events.append({"node": to, "time": t, "edge": chosen["edge"],
                       "deg": round(chosen["deg"], 3), "kind": "turn"})
        if g.degree(to) <= 1:
            hits.append({"node": to, "time": t, "why": "тупик"})
        visits[edge] = visits.get(edge, 0) + 1
        last_visit[edge] = t
        edge, frm, to = chosen["edge"], to, chosen["to"]
        visited_edges.append(edge)

    return {"events": events, "dead_ends": hits, "edges": visited_edges,
            "end_time": t, "end_node": to}


def score(g: G.Graph, hyp: dict, branch: dict, yaw: Yaw, t_branch: float,
          demerit: float, revoke_glances: bool, use_node_terms: bool = True) -> dict:
    """Score one hypothesis on the three kinds of evidence listed in the module docstring.

    `use_node_terms` exists because the ablation is the interesting part of this phase and it
    ought to be reproducible rather than described. Turning the later-node terms off leaves
    only the branch, and what remains is the part that can be trusted: the glance is revoked
    and a hypothesis that needs a dead-end excursion is demoted. The later-node terms are the
    fragile part — see the report for what they do to T49.
    """
    terms: list[dict] = []

    # ---- 1. the branch. A glance is revoked before it can be used.
    pre_dir, pre_ramp, pre_I = yaw.side(t_branch - WINDOW_S, t_branch)
    post_dir, post_ramp, post_I = yaw.side(t_branch, t_branch + WINDOW_S)
    refuted = bool(revoke_glances and pre_dir and post_dir and pre_dir != post_dir)
    if refuted:
        used_dir, used_ramp = "", 0.0
    else:
        # no direction is ever *introduced* by the later window: it may only withdraw one
        used_dir, used_ramp = pre_dir, pre_ramp
    align = geo.alignment(branch["deg"], used_dir)
    branch_term = used_ramp * align
    terms.append({"kind": "branch", "node": branch["to"], "deg": branch["deg"],
                  "direction": used_dir, "ramp": round(used_ramp, 4),
                  "alignment": round(align, 4), "score": round(branch_term, 4),
                  "pre": round(pre_I, 4), "post": round(post_I, 4),
                  "refuted": int(refuted)})

    # ---- 2. turns this hypothesis implies at later nodes
    for ev in hyp["events"]:
        if not use_node_terms:
            break
        if ev["kind"] != "turn" or ev["deg"] is None:
            continue
        d, ramp, I = yaw.side(ev["time"] - WINDOW_S, ev["time"])
        a = geo.alignment(ev["deg"], d)
        terms.append({"kind": "node", "node": ev["node"], "deg": ev["deg"],
                      "direction": d, "ramp": round(ramp, 4), "alignment": round(a, 4),
                      "score": round(ramp * a, 4), "pre": round(I, 4), "post": 0.0,
                      "refuted": 0})

    # ---- 3. a dead end the hypothesis needs to be true
    for h in hyp["dead_ends"]:
        terms.append({"kind": "dead_end", "node": h["node"], "deg": 0.0, "direction": "",
                      "ramp": 0.0, "alignment": 0.0, "score": round(-demerit, 4),
                      "pre": 0.0, "post": 0.0, "refuted": 0})

    total = sum(t["score"] for t in terms)
    return {"terms": terms, "score": round(total, 4), "refuted": int(refuted)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--run", default=str(ROOT / "output/p084b/run"),
                    help="прогон P08.4B: его развилки и темп берём как есть")
    ap.add_argument("--yaw", default=str(ROOT / "output/p07/yaw_signal_VID00006.csv"))
    ap.add_argument("--yaw-column", default="yaw_signal_deadband")
    ap.add_argument("--out", default=str(ROOT / "output/p084c"))
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--max-nodes", type=int, default=2)
    ap.add_argument("--dead-end-demerit", type=float, default=0.25)
    ap.add_argument("--no-revoke", action="store_true",
                    help="выключить отмену взгляда (для сравнения)")
    ap.add_argument("--no-node-terms", action="store_true",
                    help="убрать доводы от последующих узлов (только развилка и тупик)")
    ap.add_argument("--male-bonus", type=float, default=1.5)
    ap.add_argument("--back-penalty-min", type=float, default=0.08)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = G.Graph.load(Path(args.graph))
    yaw = Yaw(Path(args.yaw), args.yaw_column)
    rep = json.loads((Path(args.run) / "report.json").read_text(encoding="utf-8"))
    dec = list(csv.DictReader((Path(args.run) / "decisions.csv").open(encoding="utf-8")))
    seq = list(csv.DictReader((Path(args.run) / "edge_sequence.csv").open(encoding="utf-8")))

    pace = float(rep["params"]["pix_per_sec"])
    mpp = float(g.meters_per_pixel)
    decide_kwargs = {"male_bonus": args.male_bonus,
                     "back_penalty_min": args.back_penalty_min}

    def history_before(tnow: float) -> tuple[dict, dict]:
        v: dict[str, int] = {}
        l: dict[str, float] = {}
        for r in seq:
            if float(r["time_end"]) > tnow:
                break
            v[r["edge"]] = v.get(r["edge"], 0) + 1
            l[r["edge"]] = float(r["time_end"])
        return v, l

    rows: list[dict] = []
    for d in dec:
        node, arrival, t = d["node"], d["current_edge"], float(d["time"])
        if arrival not in g.edges:
            continue
        ways = g.classify_candidates(node, arrival, allow_back=True)
        forward = [c for c in ways if not geo.is_reversal(c["deg"])]
        behind = [c for c in ways if geo.is_reversal(c["deg"])]
        visits, last = history_before(t)
        res = geo.decide(forward, behind, float(d["yaw_integral"]), float(d["yaw_floor"]),
                         visits, last, t, back_factor=3.0, back_from_dead_end=False,
                         defer_margin=0.25, **decide_kwargs)
        if len(forward) < 2:
            continue                       # not a real junction: nothing to choose

        top = res["scored"][:2]
        scored = []
        for cand in top:
            hyp = simulate(g, node, cand["edge"], t, pace, args.window_s,
                           args.max_nodes, decide_kwargs)
            sc = score(g, hyp, cand, yaw, t, args.dead_end_demerit, not args.no_revoke,
                       not args.no_node_terms)
            scored.append({"edge": cand["edge"], "deg": cand["deg"],
                           "greedy_score": cand["final_score"], "hyp": hyp, "score": sc})
        # A tie is not evidence. The map's own answer stands unless it is *refuted*, which
        # means the evidence against it is positive, not merely that the alternative edged
        # ahead of it by four hundredths on a term that is noise. Without this the beam
        # overrode T49 on a margin of +0.017 against 0.000 and called it a correction.
        by_edge = {s["edge"]: s for s in scored}
        greedy = scored[0]["edge"]
        alternative = max(scored, key=lambda s: s["score"]["score"])
        g_score = by_edge[greedy]["score"]["score"]
        refuted = g_score < 0 and alternative["score"]["score"] > g_score
        winner = alternative if refuted else by_edge[greedy]
        rows.append({"t": t, "node": node, "arrival": arrival,
                     "greedy": greedy, "greedy_deg": scored[0]["deg"],
                     "beam": winner["edge"], "beam_deg": winner["deg"],
                     "changed": int(winner["edge"] != greedy),
                     "scored": scored, "reason": d["reason"]})

    # ------------------------------------------------------------------ report
    changed = [r for r in rows if r["changed"]]
    print("=" * 96)
    print("P08.4C — ДВЕ ГИПОТЕЗЫ НА РАЗВИЛКЕ, РЕШАЮТ СЛЕДУЮЩИЕ 5–10 СЕКУНД")
    print("=" * 96)
    print(f"граф: {len(g.nodes)} узлов, {len(g.edges)} рёбер (P08.4A)")
    print(f"правило выбора: P08.4B, без изменений; темп {pace:.1f} px/с = "
          f"{pace*mpp:.2f} м/с (номинальный, как в прогоне)")
    print(f"окно оценки: {args.window_s:.0f} с или {args.max_nodes} следующих узла")
    print(f"окно сигнала: {WINDOW_S} с, порог {floor_for(WINDOW_S):.3f} "
          f"(= {FLOOR_SCALE} × (FLOOR_A + FLOOR_B × {WINDOW_S}), как у трекера)")
    print(f"отмена взгляда: {'выключена' if args.no_revoke else 'включена'}")
    print(f"штраф за тупик в гипотезе: {args.dead_end_demerit}")
    print()
    print(f"настоящих развилок: {len(rows)}; гипотезы разошлись с жадным выбором: "
          f"{len(changed)}")
    print()
    for r in rows:
        mark = "  ← ГИПОТЕЗЫ РАСХОДЯТСЯ" if r["changed"] else ""
        print(f"  t={r['t']:7.2f}  {r['node']:<6}  жадный {r['greedy']:<16}"
              f"({r['greedy_deg']:>+7.1f}°)   лучшие 2 →" + mark)
        for s in r["scored"]:
            tag = "ВЫБРАНА" if s["edge"] == r["beam"] else "       "
            de = ", ".join(x["node"] for x in s["hyp"]["dead_ends"]) or "—"
            print(f"        {tag}  {s['edge']:<16}({s['deg']:>+7.1f}°)  "
                  f"балл {s['score']['score']:+.3f}   тупики: {de}")
    print()

    print("─── разбор точки J35 ───")
    for r in rows:
        if r["node"] != "J35":
            continue
        print(f"\n  t={r['t']:.2f} с, пришёл по {r['arrival']}, "
              f"жадный выбор {r['greedy']}, гипотезы выбрали {r['beam']}")
        for s in r["scored"]:
            print(f"\n    ГИПОТЕЗА {s['edge']} (угол {s['deg']:+.1f}°), "
                  f"итог {s['score']['score']:+.3f}"
                  + ("   ← ПОБЕДИЛА" if s["edge"] == r["beam"] else ""))
            print(f"      маршрут в окне: {' → '.join(s['hyp']['edges'])}")
            print(f"      события (узел, время, поворот):")
            for ev in s["hyp"]["events"]:
                deg = "—" if ev["deg"] is None else f"{ev['deg']:+.1f}°"
                print(f"        t={ev['time']:7.2f}  {ev['node']:<6} {deg:>8}  {ev['kind']}")
            print(f"      слагаемые балла:")
            for t_ in s["score"]["terms"]:
                extra = ""
                if t_["kind"] == "branch" and t_["refuted"]:
                    extra = (f"   ОТКАЗ ВЗГЛЯДА: до {t_['pre']:+.3f}, "
                             f"после {t_['post']:+.3f}")
                elif t_["kind"] == "branch":
                    extra = (f"   до {t_['pre']:+.3f}, после {t_['post']:+.3f}, "
                             f"направление {t_['direction'] or 'нет'}")
                print(f"        {t_['kind']:<9} {t_['node']:<6} угол "
                      f"{t_['deg']:>+7.1f}°  наклон {t_['alignment']:+.3f}  "
                      f"вес {t_['ramp']:.3f}  → {t_['score']:+.3f}{extra}")

    # ------------------------------------------------------------------ files
    flat: list[dict] = []
    for r in rows:
        for s in r["scored"]:
            flat.append({
                "t": r["t"], "node": r["node"], "arrival": r["arrival"],
                "greedy": r["greedy"], "beam": r["beam"], "changed": r["changed"],
                "hypothesis": s["edge"], "hypothesis_deg": s["deg"],
                "greedy_score": round(s["greedy_score"], 4),
                "hypothesis_score": s["score"]["score"],
                "refuted": s["score"]["refuted"],
                "route": " → ".join(s["hyp"]["edges"]),
                "dead_ends": ",".join(x["node"] for x in s["hyp"]["dead_ends"]),
                "terms": json.dumps(s["score"]["terms"], ensure_ascii=False),
            })
    with (out / "hypotheses.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    (out / "report.json").write_text(json.dumps({
        "phase": "P08.4C — two hypotheses at a junction, decided by what follows",
        "graph": args.graph, "run": args.run,
        "unchanged": ["MaleCNS", "клетки", "yaw", "novelty", "скорость", "веса",
                      "пороги нейронного сигнала", "геометрия P08.4B"],
        "new": {
            "window_s": args.window_s, "max_nodes": args.max_nodes,
            "signal_window_s": WINDOW_S,
            "floor": round(floor_for(WINDOW_S), 4),
            "dead_end_demerit": args.dead_end_demerit,
            "revoke_glances": not args.no_revoke,
            "node_terms": not args.no_node_terms,
            "override_rule": "жадный выбор остаётся, пока его балл не станет отрицательным",
        },
        "junctions": [{"t": r["t"], "node": r["node"], "greedy": r["greedy"],
                       "beam": r["beam"], "changed": bool(r["changed"]),
                       "scores": {s["edge"]: s["score"]["score"] for s in r["scored"]}}
                      for r in rows],
        "n_junctions": len(rows), "n_changed": len(changed),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано: {out/'hypotheses.csv'}")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
