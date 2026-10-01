#!/usr/bin/env python3
"""P08.4B — replay the new geometry on the junctions the earlier run already reached.

The finding at the end of P08.4A was that the moment a junction is decided moves when the
graph changes, and the yaw window is anchored at that moment, so the signal read at a node is
a different number on a different graph. Comparing two rules by running two routes would
therefore compare the rules and the readings at once, with no telling them apart.

So the first thing here is a freeze, not a run. `frozen_decisions.csv` holds every junction
the P08.4A route actually reached, with the time, the node, the passage arrived along, every
way on with its real angle, and the yaw integral exactly as it was read then. Nothing about
the video is recomputed. On that fixed input both rules are applied, so any difference between
them is the rule.

Two details the freeze has to get right, because getting them wrong would look like a rule
difference:

  * novelty is counted from the sequence up to the moment of the choice, using the time each
    passage *finished*. A passage that is still being walked at the decision has not been
    walked yet, and counting it makes the way on look spent.
  * the rule's short circuits are rehearsed here too, so `ONLY_OPTION` stays `ONLY_OPTION`
    and a corridor is not reported as a choice.

The full run is a separate program, and only makes sense once this reads correctly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p084b_geometry as geo  # noqa: E402
import p08_graph as G  # noqa: E402

# frozen in P08.2: a reversal must clear this many times the floor before it is believed
BACK_FACTOR = 3.0
MALE_MARGIN = 0.25       # frozen in P08.2: top two scores closer than this defer the choice


def rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, data: list[dict], fields: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


def history_before(seq: list[dict], tnow: float) -> tuple[dict, dict]:
    """Counts and last-visit times for passages finished before `tnow`.

    The time used is the passage's end, not its start. A passage still under the walker at
    the moment of the choice has not been walked yet, and counting it would make the way on
    look spent at exactly the junction where it is the straightest continuation.
    """
    visits: dict[str, int] = {}
    last: dict[str, float] = {}
    for r in seq:
        if float(r["time_end"]) > tnow:
            break
        e = r["edge"]
        visits[e] = visits.get(e, 0) + 1
        last[e] = float(r["time_end"])
    return visits, last


def split_candidates(g: G.Graph, node: str, arrival: str) -> tuple[list[dict], list[dict]]:
    """The ways on, and the passage behind, with only passages that may be travelled."""
    raw = g.classify_candidates(node, arrival, allow_back=True)
    forward = [c for c in raw if not geo.is_reversal(c["deg"])]
    back = [c for c in raw if geo.is_reversal(c["deg"])]
    return forward, back


def dead_end_ahead(g: G.Graph, forward: list[dict]) -> bool:
    """True when every way on leads straight into a node with nothing beyond it.

    The third of the three reasons a reversal may be considered. It is structural — it asks
    the map, not the fly — and it fires only where carrying on means arriving at a tip and
    turning round anyway.
    """
    return bool(forward) and all(g.degree(c["to"]) <= 1 for c in forward)


def freeze(g: G.Graph, dec: list[dict], seq: list[dict]) -> list[dict]:
    """One row per (junction, way on), holding the inputs and the old rule's answer."""
    out = []
    for n, d in enumerate(dec, 1):
        node, arrival = d["node"], d["current_edge"]
        if arrival not in g.edges:
            continue
        integral = float(d["yaw_integral"] or 0.0)
        floor = float(d["yaw_floor"] or 0.0)
        offered = set(filter(None, d["candidate_edges"].split("|")))
        visits, last = history_before(seq, float(d["time"]))

        forward, back = split_candidates(g, node, arrival)
        for c in forward + back:
            out.append({
                "decision": n,
                "time": float(d["time"]),
                "node": node,
                "arrival_edge": arrival,
                "candidate_edge": c["edge"],
                "candidate_to": c["to"],
                "angle_deg": round(c["deg"], 3),
                "candidate_kind": "back" if geo.is_reversal(c["deg"]) else "forward",
                "display": geo.display_side(c["deg"]),
                "old_offered": int(c["edge"] in offered),
                "old_chosen": int(c["edge"] == d["chosen_edge"]),
                "old_reason": d["reason"] if c["edge"] == d["chosen_edge"] else "",
                "yaw_integral": round(integral, 4),
                "yaw_floor": round(floor, 4),
                "yaw_window_s": float(d.get("yaw_window_s") or 0.0),
                "yaw_strength": round(geo.signal_ramp(integral, floor, BACK_FACTOR), 4),
                "male_direction": geo.signal_direction(integral, floor),
                "visits_before": visits.get(c["edge"], 0),
                "geometry_score": round(geo.geometry_score(c["deg"]), 4),
            })
    return out


def replay(g: G.Graph, frozen: list[dict], seq: list[dict], male_bonus: float,
           back_factor: float, back_penalty_min: float) -> list[dict]:
    """Apply the new rule to each frozen junction and record every term of every candidate."""
    by_decision: dict[int, list[dict]] = {}
    for r in frozen:
        by_decision.setdefault(r["decision"], []).append(r)

    out = []
    for n in sorted(by_decision):
        group = by_decision[n]
        first = group[0]
        node, arrival = first["node"], first["arrival_edge"]
        integral, floor, tnow = first["yaw_integral"], first["yaw_floor"], first["time"]

        forward, back = split_candidates(g, node, arrival)
        visits, last = history_before(seq, tnow)
        res = geo.decide(forward, back, integral, floor, visits, last, tnow,
                         back_factor=back_factor,
                         back_from_dead_end=dead_end_ahead(g, forward),
                         male_bonus=male_bonus, back_penalty_min=back_penalty_min,
                         defer_margin=MALE_MARGIN)

        chosen = res["chosen"]["edge"] if res["chosen"] else ""
        old = next((r for r in group if r["old_chosen"]), None)

        for s in res["scored"]:
            out.append({
                "decision": n, "time": tnow, "node": node, "arrival_edge": arrival,
                "candidate_edge": s["edge"], "angle_deg": round(s["deg"], 3),
                "display": s["display"],
                "offered_new": 1,
                "old_choice": (old or {}).get("candidate_edge", ""),
                "old_reason": (old or {}).get("old_reason", ""),
                "new_choice": chosen,
                "chosen": int(s["edge"] == chosen),
                "yaw_integral": round(integral, 4),
                "yaw_floor": round(floor, 4),
                "yaw_strength": round(res["ramp"], 4),
                "male_direction": res["male_direction"],
                "alignment": round(s["alignment"], 4),
                "geometry_score": round(s["geometry_score"], 4),
                "male_score": round(s["male_score"], 4),
                "novelty_score": round(s["novelty"], 4),
                "back_penalty": round(s["back_penalty"], 4),
                "final_score": round(s["final_score"], 4),
                "visits_before": s["visits"],
                "reason": res["reason"] if s["edge"] == chosen else "",
                "ambiguous": int(res["ambiguous"]),
                "deferred": int(res["deferred"]),
                "ambiguous_pair": "|".join(res["ambiguous_pair"] or []),
                "back_rule": res["back_rule"],
            })
    return out


def replay_old_rule(g: G.Graph, frozen: list[dict], seq: list[dict]) -> dict[int, str]:
    """The P08 rule, re-implemented here so the freeze can be checked rather than trusted.

    This is the same passage of code the tracker had before this phase: categories taken
    from 45 and 135 degrees, a flat reversal multiplier, and a full bonus for whichever way
    on happened to carry the named label. If this reproduces the choices the earlier run
    logged, then the freeze holds the right times, the right candidate sets and the right
    novelty, and a difference found by the new rule is a difference of rule.
    """
    by_decision: dict[int, list[dict]] = {}
    for r in frozen:
        by_decision.setdefault(r["decision"], []).append(r)

    out: dict[int, str] = {}
    for n, group in by_decision.items():
        first = group[0]
        node, arrival = first["node"], first["arrival_edge"]
        integral, floor, tnow = first["yaw_integral"], first["yaw_floor"], first["time"]
        forward, back = split_candidates(g, node, arrival)
        visits, last = history_before(seq, tnow)

        if not forward:
            out[n] = back[0]["edge"]
            continue
        if len(forward) == 1:
            out[n] = forward[0]["edge"]
            continue

        ramp = geo.signal_ramp(integral, floor, BACK_FACTOR)
        strong = ramp > 0 and abs(integral) >= BACK_FACTOR * floor
        cands = list(forward) + (list(back) if strong else [])
        if not strong:
            keep = [c for c in cands if not geo.is_reversal(c["deg"])]
            if keep:
                cands = keep
        if len(cands) == 1:
            out[n] = cands[0]["edge"]
            continue

        male = geo.signal_direction(integral, floor)
        best, best_score = None, None
        for c in cands:
            deg = c["deg"]
            ad = abs(deg)
            side = "STRAIGHT" if ad < 45 else ("BACK" if ad > 135 else
                                               ("RIGHT" if deg > 0 else "LEFT"))
            gscore = geo.geometry_score(deg)
            pen = 0.08 if ad > 135 else 1.0
            nov = geo.novelty(c["edge"], visits, last, tnow)
            bonus = 1.5 * ramp if (male and ad <= 135 and side == male) else 0.0
            fs = gscore * pen * nov + bonus
            if best_score is None or fs > best_score:
                best, best_score = c["edge"], fs
        out[n] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=str(ROOT / "output/p084a/new"),
                    help="прогон P08.4A, чьи решения замораживаются")
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--out", default=str(ROOT / "output/p084b"))
    ap.add_argument("--male-bonus", type=float, default=1.5)
    ap.add_argument("--back-factor", type=float, default=3.0)
    ap.add_argument("--back-penalty-min", type=float, default=0.08)
    ap.add_argument("--focus", default="J35,T50,T49,T54")
    args = ap.parse_args()

    out, run = Path(args.out), Path(args.run)
    g = G.Graph.load(Path(args.graph))
    dec, seq = rows(run / "decisions.csv"), rows(run / "edge_sequence.csv")

    frozen = freeze(g, dec, seq)
    write_rows(out / "frozen_decisions.csv", frozen, [
        "decision", "time", "node", "arrival_edge", "candidate_edge", "candidate_to",
        "angle_deg", "candidate_kind", "display", "old_offered", "old_chosen", "old_reason",
        "yaw_integral", "yaw_floor", "yaw_window_s", "yaw_strength", "male_direction",
        "visits_before", "geometry_score"])

    rep = replay(g, frozen, seq, args.male_bonus, args.back_factor, args.back_penalty_min)
    write_rows(out / "replay.csv", rep, [
        "decision", "time", "node", "arrival_edge", "candidate_edge", "angle_deg", "display",
        "offered_new", "old_choice", "old_reason", "new_choice", "chosen", "yaw_integral",
        "yaw_floor", "yaw_strength", "male_direction", "alignment", "geometry_score",
        "male_score", "novelty_score", "back_penalty", "final_score", "visits_before",
        "reason", "ambiguous", "deferred", "ambiguous_pair", "back_rule"])

    junctions = {}
    for r in rep:
        junctions.setdefault(r["decision"], r)

    print("=" * 92)
    print("P08.4B — ПОВТОР НА ЗАМОРОЖЕННЫХ ТОЧКАХ")
    print("=" * 92)
    print(f"точек решений: {len(junctions)}   строк (решение × кандидат): {len(rep)}")
    print("вход заморожен: время, узел, углы, yaw_integral — из прогона P08.4A")
    print()

    # ---- does the freeze hold the old rule's own answers? -------------------------------
    old_replay = replay_old_rule(g, frozen, seq)
    mism = [(d, junctions[d]["old_choice"], old_replay.get(d))
            for d in sorted(junctions) if old_replay.get(d) != junctions[d]["old_choice"]]
    print("─── ПРОВЕРКА ЗАМОРОЗКИ ───")
    print(f"  старое правило поверх freeze воспроизводит выбор: "
          f"{len(junctions) - len(mism)} из {len(junctions)}")
    if mism:
        print("  расхождения (значит freeze или проверка неточны, и выводам верить нельзя):")
        for d, want, got in mism:
            print(f"    №{d} {junctions[d]['node']:<6} лог {want} проверка {got}")
    else:
        print("  значит времена, наборы кандидатов и новизна в freeze верны, и разница")
        print("  ниже — это разница правила, а не входа")
    print()

    changed = [r for r in junctions.values() if r["old_choice"] != r["new_choice"]]
    print(f"─── выбор изменился: {len(changed)} из {len(junctions)} ───")
    for r in sorted(changed, key=lambda x: x["time"]):
        ang = {x["candidate_edge"]: x["angle_deg"] for x in rep
               if x["decision"] == r["decision"]}
        print(f"  t={r['time']:7.2f}  {r['node']:<6}  было {r['old_choice']:<16} "
              f"({ang.get(r['old_choice'], '?')}°)  стало {r['new_choice']:<16} "
              f"({ang.get(r['new_choice'], '?')}°)  {r['reason']}")
    print()

    same = [r for r in junctions.values() if r["old_choice"] == r["new_choice"]]
    diff_reason = [r for r in same if r["reason"] != r["old_reason"]]
    print(f"─── выбор тот же, причина другая: {len(diff_reason)} ───")
    for r in sorted(diff_reason, key=lambda x: x["time"]):
        print(f"  t={r['time']:7.2f}  {r['node']:<6}  {r['new_choice']:<16}"
              f"  {r['old_reason']:<18} → {r['reason']}")
    print()

    amb = [r for r in junctions.values() if int(r["ambiguous"])]
    print(f"─── неразличимые ветки (углы ближе 15°): {len(amb)} ───")
    for r in sorted(amb, key=lambda x: x["time"]):
        ang = {x["candidate_edge"]: x["angle_deg"] for x in rep
               if x["decision"] == r["decision"]}
        detail = ", ".join(f"{e} {ang.get(e)}°" for e in r["ambiguous_pair"].split("|") if e)
        print(f"  t={r['time']:7.2f}  {r['node']:<6}  {detail}   "
              f"MaleCNS {'назван ' + r['male_direction'] if r['male_direction'] else 'молчит'}"
              f"   выбрано {r['new_choice']}")
    if not amb:
        print("  нет: после починки графа ни одна развилка маршрута не имеет двух")
        print("  выходов ближе 15° (та самая пара была у старого J37 и убрана в P08.4A)")
    print()

    tally: dict[str, int] = {}
    old_tally: dict[str, int] = {}
    for r in junctions.values():
        tally[r["reason"]] = tally.get(r["reason"], 0) + 1
        old_tally[r["old_reason"]] = old_tally.get(r["old_reason"], 0) + 1
    print("─── причины решений ───")
    for k in sorted(set(tally) | set(old_tally)):
        print(f"  {k:<20} было {old_tally.get(k, 0):>3}   стало {tally.get(k, 0):>3}")
    print()

    print("─── ТОЧКИ, КОТОРЫЕ ПРОВЕРЯЕМ ОСОБО ───")
    for name in [f.strip() for f in args.focus.split(",") if f.strip()]:
        hits = [r for r in rep if r["node"] == name]
        if not hits:
            print(f"\n  {name}: в замороженных точках нет")
            continue
        for dnum in sorted({r["decision"] for r in hits}):
            grp = [r for r in hits if r["decision"] == dnum]
            h = grp[0]
            print(f"\n  {name}  t={h['time']:.2f}с   пришёл по {h['arrival_edge']}"
                  f"   yaw {h['yaw_integral']:+.3f} (порог {h['yaw_floor']:.3f})"
                  f"   MaleCNS: {h['male_direction'] or 'молчит'}"
                  f"   BACK: {h['back_rule']}")
            print(f"    {'кандидат':<16}{'угол':>8} {'подпись':<10}{'геом':>6}{'накл':>7}"
                  f"{'male':>7}{'новизна':>9}{'штраф':>7}{'итог':>8}   исход")
            for r in sorted(grp, key=lambda x: x["angle_deg"]):
                if r["old_choice"] and r["new_choice"] == r["candidate_edge"]:
                    tag = "выбрано (и раньше)"
                elif r["old_choice"]:
                    tag = "было выбрано раньше"
                else:
                    tag = ""
                print(f"    {r['candidate_edge']:<16}{r['angle_deg']:>+8.1f} "
                      f"{r['display']:<10}{r['geometry_score']:>6.3f}{r['alignment']:>+7.2f}"
                      f"{r['male_score']:>+7.2f}{r['novelty_score']:>9.3f}"
                      f"{r['back_penalty']:>7.2f}{r['final_score']:>8.3f}   {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
