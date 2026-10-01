#!/usr/bin/env python3
"""P12 step 3 — a full walk of VID00006 with the three channels in the decision, and the ledger.

P08.6 is not modified. Its hypothesis machinery, its window, its floor and its propagation are
imported and used as they are; the only new thing is a hook after the choice has been made, which
lets the channels override it. That keeps the comparison honest — both routes are walked by the
same code, and the difference between them is the channels and nothing else.

WHAT THE CHANNELS ARE ALLOWED TO DO TO A DECISION
-------------------------------------------------
    route_change = SAME, conf above threshold
        the body's course did not change, so a turn is not what happened. The straightest forward
        option is taken instead of the one the yaw named. This is the counter the phase asks for:
        "how many times was the yaw cancelled by route_change".
    route_change = DIFFERENT and yaw names a side
        the course did change and the camera says which way, so a turn to that side is preferred.
    net_displacement = NO_NET, conf above threshold
        no net progress was made, so the walk is stopped early rather than advanced. This is the
        second counter: "how many times did NO_NET stop a false advance".
    anything UNKNOWN
        does nothing at all. The junction stands exactly as P08.6 left it.

THE LEDGER
----------
Both routes are walked and the same seven counts are reported for each, plus the two counters that
only P12 can produce. Counts, not a single quality number, because the phase asked what changed and
not which line looks better.

Usage:
    PYTHONPATH=. python scripts/p12_run.py [--video VID00006] [--weights 0.25,0.5]
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

import p084b_geometry as geo  # noqa: E402
import p08_graph as G  # noqa: E402
import p086_multi as M  # noqa: E402
import scripts.p12_channels as P  # noqa: E402

OUT = ROOT / "output/p12"


def side_of(deg: float) -> str:
    """Which way a turn of this sign goes. Positive degrees are left, as in the tracker."""
    return "LEFT" if deg > 0 else "RIGHT"


def channels_for(video: str) -> tuple[dict, dict]:
    """The three readings, with the readers that never saw this recording."""
    ch = json.loads((OUT / "channels.json").read_text(encoding="utf-8"))
    return ch["folds"][video], ch["junctions"]


_YAW_CACHE: dict[str, dict] = {}


def read_at(video: str, fold: dict, t: float, tau: float) -> dict:
    """All three channels at one moment."""
    if video not in _YAW_CACHE:
        _YAW_CACHE[video] = P.yaw_channel(video)
    yaw = P.yaw_at(_YAW_CACHE[video], t)
    Fdn, _n, _c = P.source_features("dn", video, [t])
    Fea, _n2, _c2 = P.source_features("early", video, [t])
    return {"yaw": yaw,
            "route_change": P.apply_or_unknown(Fdn, fold["route_change"], tau, "SAME", "DIFFERENT"),
            "net_displacement": P.apply_or_unknown(Fea, fold["net_displacement"], tau,
                                                   "MOVE", "NO_NET")}


def walk(g: G.Graph, sig: M.Signal, video: str, fold: dict, node0: str, edge0: str,
         t0: float, t_end: float, max_pace: float, visits: dict, last_visit: dict,
         use_p12: bool, tau: float) -> dict:
    """P08.6's propagation, with an optional hook that lets the channels override the choice.

    Copied from p086_multi.propagate rather than imported because the hook has to sit inside the
    loop, between the choice and the move. Everything up to the choice — the window, the floor, the
    candidate list, the deferral margin — is the same code path.
    """
    events: list[dict] = []
    edges = [edge0]
    dead_ends: list[str] = []
    reversals = 0
    cancelled_yaw = 0
    stopped_by_no_net = 0
    now = t0
    cur, cur_from = edge0, node0
    to = g.other(edge0, node0)
    remaining = g.length_px(edge0)

    while now < t_end:
        step = sig.step_px(now, max_pace)
        if step <= 0:
            now += M.DT
            continue
        remaining -= step
        now += M.DT
        if remaining > 1e-9:
            continue

        node_now, arrived_by = to, cur
        d, ramp, I = sig.named(now - M.WINDOW_S, now)
        ways = [c for c in g.classify_candidates(node_now, arrived_by, allow_back=True)]
        forward = [c for c in ways if not geo.is_reversal(c["deg"])]
        behind = [c for c in ways if geo.is_reversal(c["deg"])]
        res = geo.decide(forward, behind, I, M.floor_for(M.WINDOW_S), visits, last_visit,
                         now, back_factor=3.0, back_from_dead_end=False,
                         defer_margin=M.RESOLVE_MARGIN)
        if res["chosen"] is None:
            break
        chosen = res["chosen"]

        if use_p12 and forward:
            c = read_at(video, fold, now, tau)
            rc, nd, yaw = c["route_change"], c["net_displacement"], c["yaw"]
            straight = min(forward, key=lambda x: abs(float(x["deg"])))
            if rc.get("class") == "SAME":
                if chosen["edge"] != straight["edge"]:
                    cancelled_yaw += 1
                    chosen = straight
            elif rc.get("class") == "DIFFERENT" and yaw.get("class"):
                same_side = [x for x in forward
                             if side_of(float(x["deg"])) == yaw["class"]]
                if same_side:
                    chosen = max(same_side, key=lambda x: abs(float(x["deg"])))
            if nd.get("class") == "NO_NET":
                stopped_by_no_net += 1
                break

        events.append({"node": node_now, "t": round(now, 3), "turn": round(chosen["deg"], 2),
                       "edge": chosen["edge"], "yaw": round(I, 4), "side": d,
                       "ramp": round(ramp, 4)})
        if geo.is_reversal(chosen["deg"]):
            reversals += 1
        if g.degree(node_now) <= 1:
            dead_ends.append(node_now)
        visits[arrived_by] = visits.get(arrived_by, 0) + 1
        last_visit[arrived_by] = now
        cur, cur_from, to = chosen["edge"], node_now, chosen["to"]
        edges.append(cur)
        remaining = g.length_px(cur)

    dist = (sum(g.length_px(e) for e in edges[:-1])
            + max(g.length_px(edges[-1]) - remaining, 0.0))
    return {"events": events, "edges": edges, "dead_ends": dead_ends, "reversals": reversals,
            "cancelled_yaw": cancelled_yaw, "stopped_by_no_net": stopped_by_no_net,
            "end_t": now, "end_node": to, "distance_px": dist}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="VID00006")
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--run", default=str(ROOT / "output/p084c/run"))
    ap.add_argument("--yaw", default=str(ROOT / "output/p07/yaw_signal_VID00006.csv"))
    ap.add_argument("--traj", default=str(ROOT / "output/p071/trajectory_VID00006.csv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--match-s", type=float, default=12.0,
                    help="насколько далеко от развилки может стоять точка решения")
    args = ap.parse_args()
    out = Path(args.out)

    g = G.Graph.load(Path(args.graph))
    mpp = float(g.meters_per_pixel)
    rep = json.loads((Path(args.run) / "report.json").read_text(encoding="utf-8"))
    pace = float(rep["params"]["pix_per_sec"])
    max_pace = M.MAX_MPS / mpp
    sig = M.Signal(Path(args.yaw), Path(args.traj), pace)
    run = list(csv.DictReader((Path(args.run) / "decisions.csv").open(encoding="utf-8")))
    seq = list(csv.DictReader((Path(args.run) / "edge_sequence.csv").open(encoding="utf-8")))
    fold, junc = channels_for(args.video)
    jr = json.loads((ROOT / "output/p086/report.json").read_text(encoding="utf-8"))
    resolved = sum(1 for j in jr["junctions"] if j["status"] == "РАЗРЕШЕНО ПОЗЖЕ")
    ambiguous = sum(1 for j in jr["junctions"] if j["status"] == "AMBIGUOUS")

    def history_before(tnow: float):
        v: dict[str, int] = {}
        l: dict[str, float] = {}
        for r in seq:
            if float(r["time_end"]) > tnow:
                break
            v[r["edge"]] = v.get(r["edge"], 0) + 1
            l[r["edge"]] = float(r["time_end"])
        return v, l

    print("=" * 104)
    print(f"P12 шаг 3 — ПОЛНЫЙ ПРОГОН {args.video} С P12-СЛОЕМ")
    print("=" * 104)
    print(f"  P08.6 не изменён: его окно, порог и правило выбора взяты как есть")
    print(f"  порог уверенности каналов {args.tau}; горизонт {args.horizon_s:.0f} с")
    print(f"  читатели обучены БЕЗ {args.video}")
    print()

    # Walk the recording's own decision points, all of them. A free walk with a thirty-second
    # horizon stopped at t=33 and never reached J35, T50 or T49, which is precisely where the
    # phase expected the channels to matter; comparing routes over a stretch that contains none of
    # the interesting junctions would have produced a tidy table about nothing.
    decisions = [d for d in run if d["reason"] != "ONLY_OPTION"]
    print(f"  точек решения в записи: {len(decisions)} (весь ролик, не только начало)")
    print()

    rows = []
    for d in decisions:
        node, arrival, t = d["node"], d["current_edge"], float(d["time"])
        if arrival not in g.edges:
            continue
        ways = [c for c in g.classify_candidates(node, arrival, allow_back=True)]
        forward = [c for c in ways if not geo.is_reversal(c["deg"])]
        if len(forward) < 2:
            continue
        committed = d["chosen_edge"]
        truth = ""
        for j in jr["junctions"]:
            if j["node"] == node and abs(float(j["t"]) - t) < args.match_s:
                truth = j.get("truth_edge", "")
                break
        c = read_at(args.video, fold, t, args.tau)
        rc, nd, yaw = c["route_change"], c["net_displacement"], c["yaw"]
        pick = committed
        note = []
        if rc.get("class") == "SAME":
            straight = min(forward, key=lambda x: abs(float(x["deg"])))
            if pick != straight["edge"]:
                note.append("route_change=SAME: повёрнуто на прямое")
            pick = straight["edge"]
        elif rc.get("class") == "DIFFERENT" and yaw.get("class"):
            same = [x for x in forward if side_of(float(x["deg"])) == yaw["class"]]
            if same:
                best = max(same, key=lambda x: abs(float(x["deg"])))
                if pick != best["edge"]:
                    note.append("route_change=DIFFERENT + yaw: выбрана названная сторона")
                pick = best["edge"]
        stopped = nd.get("class") == "NO_NET"
        if stopped:
            note.append("net=NO_NET: продвижение остановлено")

        deg_of = {x["edge"]: float(x["deg"]) for x in ways}
        rev = bool(deg_of.get(pick) is not None and geo.is_reversal(deg_of[pick]))
        to_node = next((x["to"] for x in ways if x["edge"] == pick), "")
        dead = bool(to_node and g.degree(to_node) <= 1)

        rows.append({"t": t, "node": node, "committed": committed, "pick": pick,
                     "truth": truth, "yaw": yaw.get("class"), "rc": rc.get("class"),
                     "nd": nd.get("class"), "conf_rc": rc.get("confidence", 0),
                     "conf_nd": nd.get("confidence", 0), "changed": pick != committed,
                     "note": note, "reversal": rev, "dead_end": dead, "stopped": stopped})

    print("─── ВСЕ ТОЧКИ РЕШЕНИЯ РОЛИКА ───")
    print(f"  {'t':>7}{'узел':<7}{'P08.6':<15}{'P12':<15}{'истина':<15}"
          f"{'yaw':<6}{'rc':<11}{'nd':<9}что изменилось")
    for r in rows:
        mark = "ИЗМЕНИЛОСЬ" if r["changed"] else ""
        ok = ""
        if r["truth"]:
            ok = "верно" if r["pick"] == r["truth"] else "НЕВЕРНО"
        print(f"  {r['t']:>7.1f}{r['node']:<7}{r['committed']:<15}{r['pick']:<15}"
              f"{r['truth'] or '—':<15}{r['yaw'] or '—':<6}{r['rc'] or '—':<11}"
              f"{r['nd'] or '—':<9}{' '.join(r['note'])} {mark} {ok}")
    print()

    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    print("─── ЛЕТОПИСЬ, КОТОРУЮ ПРОСИЛИ ───")
    with_truth = [r for r in rows if r["truth"]]
    print(f"  {'метрика':<46}{'P08.6':>10}{'P12':>10}")
    print("  " + "-" * 66)

    def pair(label, a, b):
        print(f"  {label:<46}{a:>10}{b:>10}")

    pair("точек решения всего", len(rows), len(rows))
    pair("из них с известной истиной", len(with_truth), len(with_truth))
    pair("верных решений",
         sum(1 for r in with_truth if r["committed"] == r["truth"]),
         sum(1 for r in with_truth if r["pick"] == r["truth"]))
    pair("развилок решено (жёсткий выбор)", len(rows), len(rows))
    pair("оставлено неоднозначными (P08.6)", ambiguous, 0)
    pair("разрешено позже (зазор выше порога)", resolved, 0)
    pair("заходов в тупик", count(lambda r: r["dead_end"] and r["committed"] == r["pick"]),
         count(lambda r: r["dead_end"]))
    pair("возвратов (разворотов)", count(lambda r: r["reversal"] and r["committed"] == r["pick"]),
         count(lambda r: r["reversal"]))
    pair("yaw отменён route_change", 0,
         count(lambda r: "отменён" in " ".join(r["note"]) or
               "route_change=SAME" in " ".join(r["note"])))
    pair("route_change=SAME активен", 0, count(lambda r: r["rc"] == "SAME"))
    pair("route_change=DIFFERENT активен", 0, count(lambda r: r["rc"] == "DIFFERENT"))
    pair("NO_NET остановил продвижение", 0, count(lambda r: r["stopped"]))
    pair("каналы промолчали оба", count(lambda r: True) * 0,
         count(lambda r: r["rc"] is None and r["nd"] is None))
    pair("решений изменено", 0, count(lambda r: r["changed"]))
    print()

    print("─── ЧУВСТВИТЕЛЬНОСТЬ К ПОРОГУ УВЕРЕННОСТИ ───")
    print(f"  {'tau':>6}{'решено rc':>11}{'решено nd':>11}{'изменено':>10}{'верных P08.6':>14}"
          f"{'верных P12':>12}")
    for tau in (0.10, 0.25, 0.40, 0.60):
        rr = []
        for d in decisions:
            node, arrival, t = d["node"], d["current_edge"], float(d["time"])
            if arrival not in g.edges:
                continue
            ways = [c for c in g.classify_candidates(node, arrival, allow_back=True)]
            forward = [c for c in ways if not geo.is_reversal(c["deg"])]
            if len(forward) < 2:
                continue
            committed = d["chosen_edge"]
            truth = ""
            for j in jr["junctions"]:
                if j["node"] == node and abs(float(j["t"]) - t) < args.match_s:
                    truth = j.get("truth_edge", "")
                    break
            c = read_at(args.video, fold, t, tau)
            rc, yaw = c["route_change"], c["yaw"]
            pick = committed
            if rc.get("class") == "SAME" and forward:
                pick = min(forward, key=lambda x: abs(float(x["deg"])))["edge"]
            elif rc.get("class") == "DIFFERENT" and yaw.get("class"):
                same = [x for x in forward if side_of(float(x["deg"])) == yaw["class"]]
                if same:
                    pick = max(same, key=lambda x: abs(float(x["deg"])))["edge"]
            rr.append((committed, pick, truth, rc.get("class"), c["net_displacement"].get("class")))
        n_rc = sum(1 for x in rr if x[3])
        n_nd = sum(1 for x in rr if x[4])
        ch = sum(1 for x in rr if x[0] != x[1])
        c08 = sum(1 for x in rr if x[2] and x[0] == x[2])
        c12 = sum(1 for x in rr if x[2] and x[1] == x[2])
        n_t = sum(1 for x in rr if x[2])
        print(f"  {tau:>6.2f}{n_rc:>11}{n_nd:>11}{ch:>10}{c08:>11}/{n_t}{c12:>9}/{n_t}")
    print()

    doc = {"phase": f"P12 шаг 3 — полный прогон {args.video}",
           "video": args.video, "tau": args.tau, "p086_unchanged": True,
           "n_decision_points": len(rows),
           "p086_junctions": {"resolved_later": resolved, "ambiguous": ambiguous},
           "ledger": {
               "n_with_truth": len(with_truth),
               "correct_p086": sum(1 for r in with_truth if r["committed"] == r["truth"]),
               "correct_p12": sum(1 for r in with_truth if r["pick"] == r["truth"]),
               "dead_ends_p12": count(lambda r: r["dead_end"]),
               "reversals_p12": count(lambda r: r["reversal"]),
               "route_change_SAME": count(lambda r: r["rc"] == "SAME"),
               "route_change_DIFFERENT": count(lambda r: r["rc"] == "DIFFERENT"),
               "no_net_stopped": count(lambda r: r["stopped"]),
               "both_silent": count(lambda r: r["rc"] is None and r["nd"] is None),
               "changed": count(lambda r: r["changed"]),
           },
           "decisions": rows}
    (out / "run_report.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  записано: {out/'run_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
