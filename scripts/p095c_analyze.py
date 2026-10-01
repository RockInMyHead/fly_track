#!/usr/bin/env python3
"""P09.5C — what the drawn paths say, on however many events have been drawn so far.

The earlier passes asked for a word and could say nothing until all sixty were in. A drawing does
not need a threshold to be useful: even thirty of them show a pattern, and the pattern is what
this reports. It runs on whatever is present and says how much that is, rather than refusing.

Three things are counted, and the third is the one that was not visible before.

    against the rule      the frozen P09.4 class against the drawn verdict
    against the proxy     P07.4's own labelling, which all of P09.1 to P09.4 was measured against
    the stationary case   events where the walker was not going anywhere at all

The third exists because drawing made it visible. A person asked to choose between "the head
turned" and "the body turned" has two buttons and will press one of them; a person asked to draw
the path will draw nothing when there is no path, and then say so. The first two passes had no way
to express this, and the events where it applies were being forced into one of the two answers.

No threshold is chosen here and nothing is refitted. The drawn verdict uses the same net-heading
boundaries as the page shows the reviewer, and both the raw drawing and its derived angles are
stored, so any of it can be recomputed later.

Usage:
    PYTHONPATH=. python scripts/p095c_analyze.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
OUT = ROOT / "output/p095"

HUMAN = ("LOOK", "TURN", "NO_LOCOMOTION")
MACHINE = ("LOOK", "TURN", "UNKNOWN")
FOCUS = ("VID00005", "VID00009")


def load() -> tuple[dict, dict, dict]:
    v2 = json.loads((DATA / "human_labels_v2.json").read_text(encoding="utf-8"))["labels"]
    s = json.loads((DATA / "review_set_v2.json").read_text(encoding="utf-8"))
    events = {e["event_id"]: e for e in s["events"]}
    first = {l["event_id"]: l for l in v2 if l.get("round", 1) == 1}
    reps = {l["event_id"]: l for l in v2 if l.get("round", 1) == 2}
    v1 = {}
    p1 = DATA / "human_labels_v1.json"
    if p1.exists():
        v1 = {l["event_id"]: l["human_label"]
              for l in json.loads(p1.read_text(encoding="utf-8"))["labels"]
              if l.get("round", 1) == 1}
    return events, first, {"v1": v1, "repeats": reps}


def table(rows: list[dict], key: str) -> dict:
    cells = {(m, h): 0 for m in MACHINE for h in HUMAN}
    for r in rows:
        if r[key] in MACHINE and r["human"] in HUMAN:
            cells[(r[key], r["human"])] += 1
    n_turn = sum(cells[(m, "TURN")] for m in MACHINE)
    n_look = sum(cells[(m, "LOOK")] for m in MACHINE)
    n_stat = sum(cells[(m, "NO_LOCOMOTION")] for m in MACHINE)
    return {"cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
            "n_turn": n_turn, "n_look": n_look, "n_stationary": n_stat,
            "turn_revoked": cells[("LOOK", "TURN")],
            "turn_confirmed": cells[("TURN", "TURN")],
            "turn_unknown": cells[("UNKNOWN", "TURN")],
            "look_confirmed": cells[("LOOK", "LOOK")],
            "stationary_confident_turn": cells[("TURN", "NO_LOCOMOTION")]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events, first, extra = load()
    n_done, n_total = len(first), len(events)
    if not first:
        print("нарисованных событий пока нет")
        return 1

    rows = []
    for eid, l in first.items():
        e = events.get(eid)
        if not e:
            continue
        rows.append({"event_id": eid, "clip": e["clip"], "time": e["time"],
                     "human": l["human_label"], "source": l.get("label_source", ""),
                     "machine_b": e["hidden"]["algo_class_b"],
                     "score_b": e["hidden"]["algo_score_b"],
                     "proxy": e["hidden"]["proxy_label"],
                     "pair": e["hidden"]["pair_status"],
                     "net_deg": l.get("derived_turn_net_deg"),
                     "total_deg": l.get("derived_turn_total_deg"),
                     "points": l.get("drawing_points", 0),
                     "n_strokes": len(l.get("drawing", []))})

    print("=" * 100)
    print("P09.5C — НАРИСОВАННЫЕ ПУТИ: ЧТО ВИДНО НА СЕГОДНЯШНЕМ ОБЪЁМЕ")
    print("=" * 100)
    print(f"  размечено рисунком: {n_done} из {n_total} ({n_done / n_total:.0%})")
    print(f"  повторов: {len(extra['repeats'])}")
    print()

    # ---- distribution ---------------------------------------------------------------------
    c = Counter(r["human"] for r in rows)
    print("─── ЧТО СКАЗАЛ ЧЕЛОВЕК ───")
    for k in HUMAN:
        print(f"  {k:<18}{c.get(k, 0):>4}  ({c.get(k, 0) / n_done:.0%})")
    print()
    prints = [r["points"] for r in rows if r["points"]]
    print(f"  с нарисованным путём: {len(prints)} из {n_done}")
    if prints:
        print(f"  точек на путь: медиана {int(np.median(prints))}, "
              f"мин {min(prints)}, макс {max(prints)}")
    print()

    # ---- the third category, which the previous passes could not express ------------------
    st = [r for r in rows if r["human"] == "NO_LOCOMOTION"]
    print("─── «НЕ ШЁЛ»: КАТЕГОРИЯ, КОТОРОЙ НЕ БЫЛО В ПЕРВЫХ ДВУХ ПРОХОДАХ ───")
    print(f"  событий: {len(st)} из {n_done} ({len(st) / n_done:.0%})")
    if st:
        print()
        print(f"  {'событие':<9}{'клип':<11}{'t':>8}{'правило':<9}{'score':>7}{'proxy':<9}"
              f"{'пара':>15}")
        for r in sorted(st, key=lambda x: x["clip"]):
            sc = "—" if r["score_b"] is None else f"{r['score_b']:.3f}"
            print(f"  {r['event_id']:<9}{r['clip']:<11}{r['time']:>8.1f}"
                  f"{r['machine_b']:<9}{sc:>7}{r['proxy']:<9}{r['pair']:>15}")
        print()
        by_rule = Counter(r["machine_b"] for r in st)
        print(f"  как их назвало правило: " +
              "  ".join(f"{k}:{v}" for k, v in sorted(by_rule.items())))
        print(f"  правило сказало UNKNOWN (не знаю) на {by_rule.get('UNKNOWN', 0)} из {len(st)}"
              f" — то есть на части таких событий оно уже молчит")
        conf = by_rule.get("TURN", 0)
        if conf:
            print(f"  НО на {conf} оно сказало ПОВОРОТ с уверенностью — это ложная уверенность")
            print(f"  там, где человек вообще никуда не шёл")
        print()
        print("  ПОЧЕМУ ЭТО ВАЖНО: если сигнал сильный, а ходьбы нет, вопрос «куда пошло тело»")
        print("  не имеет ответа. Первые два прохода вынуждали выбрать LOOK или TURN, потому")
        print("  что третьей кнопки не было.")
    print()

    # ---- against the rule -----------------------------------------------------------------
    RB = table(rows, "machine_b")
    print("─── ПРАВИЛО P09.4 ПРОТИВ НАРИСОВАННОГО ───")
    print(f"  {'':<14}{'человек LOOK':>14}{'человек TURN':>14}{'НЕ ШЁЛ':>10}")
    for m in MACHINE:
        v = [RB["cells"][f"{m}|{h}"] for h in HUMAN]
        print(f"  {m:<14}{v[0]:>14}{v[1]:>14}{v[2]:>10}")
    print(f"  {'всего':<14}{RB['n_look']:>14}{RB['n_turn']:>14}{RB['n_stationary']:>10}")
    print()
    spoke = RB["turn_revoked"] + RB["turn_confirmed"]
    print(f"  человек сказал ПОВОРОТ: {RB['n_turn']}")
    print(f"    правило подтвердило:        {RB['turn_confirmed']}")
    print(f"    правило СНЯЛО направление:  {RB['turn_revoked']}"
          + (f"   ({(RB['turn_revoked'] / spoke):.0%} от тех, о чём оно говорило)"
             if spoke else ""))
    print(f"    правило не знало (UNKNOWN): {RB['turn_unknown']}")
    print(f"  человек сказал В ТУ ЖЕ СТОРОНУ: {RB['n_look']}, правило согласилось на "
          f"{RB['look_confirmed']}")
    print()

    # ---- v1 vs v2 -------------------------------------------------------------------------
    v1 = extra["v1"]
    src = {e["event_id"]: e["source_event_id"] for e in events.values()}
    both = [r for r in rows if src.get(r["event_id"]) in v1]
    if both:
        same = sum(1 for r in both if v1[src[r["event_id"]]] == r["human"])
        print("─── ПЕРВЫЙ ПРОХОД (СЛОВАМИ) ПРОТИВ ВТОРОГО (РИСУНКОМ) ───")
        print(f"  сравнимых событий: {len(both)}")
        print(f"  метка совпала: {same} ({same / len(both):.0%})")
        tr = Counter((v1[src[r["event_id"]]], r["human"]) for r in both)
        print(f"  {'из':<10}" + "".join(f"{h:>16}" for h in HUMAN))
        for a in ("LOOK", "TURN", "UNCLEAR"):
            if not any(k[0] == a for k in tr):
                continue
            print(f"  {a:<10}" + "".join(f"{tr[(a, h)]:>16}" for h in HUMAN))
        print()
        print(f"  Смена формулировки сдвинула {len(both) - same} ответов из {len(both)}.")
        print("  Значит первый проход отвечал на другой вопрос, и его согласие с правилом")
        print("  нельзя было читать как точность правила.")
        print()

    # ---- against the proxy ----------------------------------------------------------------
    pc = Counter((r["proxy"], r["human"]) for r in rows)
    print("─── PROXY P07.4 ПРОТИВ НАРИСОВАННОГО ───")
    print(f"  {'proxy':<14}" + "".join(f"{h:>16}" for h in HUMAN))
    for m in MACHINE:
        print(f"  {m:<14}" + "".join(f"{pc[(m, h)]:>16}" for h in HUMAN))
    agree = sum(pc[(x, x)] for x in ("LOOK", "TURN"))
    print(f"  совпало по LOOK/TURN: {agree} из {n_done} ({agree / n_done:.0%})")
    print()

    # ---- held out -------------------------------------------------------------------------
    fo = [r for r in rows if r["clip"] in FOCUS]
    if fo:
        f = table(fo, "machine_b")
        print(f"─── ОТДЕЛЬНО VID00005 + VID00009 ({len(fo)} событий, не участвовали в построении) ───")
        print(f"  человек TURN {f['n_turn']}, LOOK {f['n_look']}, НЕ ШЁЛ {f['n_stationary']}")
        print(f"  правило сняло направление у {f['turn_revoked']} из {f['n_turn']} поворотов")
        if f["stationary_confident_turn"]:
            print(f"  ложная уверенность на «не шёл»: {f['stationary_confident_turn']}")
        print()

    # ---- files ----------------------------------------------------------------------------
    with (out / "v2_partial.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["event_id"]):
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in r.items()})
    (out / "report_v2_partial.json").write_text(json.dumps({
        "phase": "P09.5C — анализ нарисованных путей (частично)",
        "n_drawn": n_done, "n_total": n_total, "fraction": n_done / n_total,
        "labels": dict(c),
        "stationary": {"n": len(st), "by_rule": dict(Counter(r["machine_b"] for r in st)),
                       "note": "категории «не шёл» не было в первых двух проходах, поэтому "
                               "такие события вынужденно записывались как LOOK или TURN"},
        "vs_rule": RB,
        "vs_proxy": {f"{k[0]}|{k[1]}": v for k, v in pc.items()},
        "proxy_agreement_lookturn": agree / n_done,
        "v1_vs_v2": {"n_comparable": len(both), "same": same if both else 0,
                     "agreement": (same / len(both)) if both else None},
        "held_out": table(fo, "machine_b") if fo else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'v2_partial.csv'}")
    print(f"записано: {out/'report_v2_partial.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
