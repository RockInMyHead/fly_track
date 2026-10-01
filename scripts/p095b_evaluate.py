#!/usr/bin/env python3
"""P09.5B — the two passes against each other, and the second one against the rule.

There are three questions here and they are different, so they are answered separately.

Did the wording change the answers?
    The same sixty moves were judged twice, by the same person, with a different question and in
    a different order. The share that changed is a measurement of the question, not of the rule. If
    a third of the labels move when the wording changes, then the first pass was answering
    something slightly different, and no agreement figure computed from it can be read as
    accuracy — for the rule or for anything else.

Is the second pass more reliable?
    Ten of the sixty came back without being announced, in both passes. Self-agreement in the
    second pass, against the first, is the only lever available on how stable any of this is.

What does the rule look like against the second pass?
    The frozen P09.4 rule, both variants, against the labels from the corrected question. This is
    the figure the first pass was supposed to produce and could not, because the question it asked
    may not have been the one intended.

The two passes share the events but not the identifiers, so they are joined on the source event
id. Nothing is recomputed about the video, and neither pass is discarded.

Usage:
    PYTHONPATH=. python scripts/p095b_evaluate.py
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

HUMAN = ("LOOK", "TURN", "UNCLEAR")
MACHINE = ("LOOK", "TURN", "UNKNOWN")
FOCUS = ("VID00005", "VID00009")


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("labels", [])


def honest(labels: list[dict]) -> dict[str, str]:
    """First-pass answers, keyed by event id, excluding the unannounced repeats."""
    return {l["event_id"]: l["human_label"] for l in labels if l.get("round", 1) == 1}


def repeats(labels: list[dict]) -> dict[str, str]:
    return {l["event_id"]: l["human_label"] for l in labels if l.get("round", 1) == 2}


def matrix(rows: list[dict], a_key: str, b_key: str,
           a_vals=HUMAN, b_vals=HUMAN) -> dict:
    cells = {(x, y): 0 for x in a_vals for y in b_vals}
    for r in rows:
        x, y = r.get(a_key), r.get(b_key)
        if x in a_vals and y in b_vals:
            cells[(x, y)] += 1
    return {"cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
            "n": sum(cells.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    v1 = read_labels(DATA / "human_labels_v1.json")
    if not v1:
        alt = DATA / "human_labels.json"
        v1 = read_labels(alt)
        if v1:
            print(f"  (первый проход прочитан из {alt.name})")
    v2 = read_labels(DATA / "human_labels_v2.json")
    if not v1 or not v2:
        print("нет обеих разметок:")
        print(f"  human_labels_v1.json: {'есть' if v1 else 'НЕТ'}")
        print(f"  human_labels_v2.json: {'есть' if v2 else 'НЕТ'}")
        return 1

    s2 = json.loads((DATA / "review_set_v2.json").read_text(encoding="utf-8"))
    src_of = {e["event_id"]: e["source_event_id"] for e in s2["events"]}
    by_id = {e["event_id"]: e for e in s2["events"]}

    a1, a2 = honest(v1), honest(v2)
    r1, r2 = repeats(v1), repeats(v2)

    print("=" * 100)
    print("P09.5B — ДВЕ ЧЕЛОВЕЧЕСКИЕ РАЗМЕТКИ И ПРАВИЛО")
    print("=" * 100)
    print(f"  первый проход:  {len(a1)} ответов (вопрос через механизм: голова/тело)")
    print(f"  второй проход:  {len(a2)} ответов (вопрос про результат движения)")
    print(f"  событий общих:  {len(set(a1) & {src_of.get(k, k) for k in a2})}")
    print()

    # ---- 1. did the wording change the answers? -------------------------------------------
    joined = []
    for eid2, h2 in a2.items():
        eid1 = src_of.get(eid2)
        if eid1 in a1:
            joined.append({"event_id_v1": eid1, "event_id_v2": eid2,
                           "clip": by_id[eid2]["clip"], "time": by_id[eid2]["time"],
                           "v1": a1[eid1], "v2": h2,
                           "machine_b": by_id[eid2]["hidden"]["algo_class_b"],
                           "score_b": by_id[eid2]["hidden"]["algo_score_b"],
                           "machine_a": by_id[eid2]["hidden"]["algo_class_a"],
                           "proxy": by_id[eid2]["hidden"]["proxy_label"],
                           "pair": by_id[eid2]["hidden"]["pair_status"]})
    M = matrix(joined, "v1", "v2")
    print("─── 1. ПЕРВЫЙ ПРОХОД ПРОТИВ ВТОРОГО (тот же человек, другой вопрос) ───")
    print(f"  {'v1 \\ v2':<12}{'LOOK':>8}{'TURN':>8}{'UNCLEAR':>10}{'всего':>8}")
    for h in HUMAN:
        vals = [M["cells"][f"{h}|{y}"] for y in HUMAN]
        print(f"  {h:<12}{vals[0]:>8}{vals[1]:>8}{vals[2]:>10}{sum(vals):>8}")
    same = sum(M["cells"][f"{h}|{h}"] for h in HUMAN)
    print(f"  {'совпало':<12}{same:>8} из {M['n']} ({same / max(M['n'], 1):.0%})")
    print()
    flipped = M["cells"]["LOOK|TURN"] + M["cells"]["TURN|LOOK"]
    print(f"  изменённых ответов: {M['n'] - same} из {M['n']} "
          f"({1 - same / max(M['n'], 1):.0%})")
    print(f"    из них прямая смена LOOK ↔ TURN: {flipped}")
    print(f"    ушло в UNCLEAR: {M['cells']['LOOK|UNCLEAR'] + M['cells']['TURN|UNCLEAR']}")
    print(f"    вышло из UNCLEAR: {M['cells']['UNCLEAR|LOOK'] + M['cells']['UNCLEAR|TURN']}")
    print()
    changed = [r for r in joined if r["v1"] != r["v2"]]
    if changed:
        print(f"  {'событие':<9}{'клип':<11}{'t':>9}{'v1':>9}{'v2':>9}{'правило':>9}{'score':>8}")
        for r in sorted(changed, key=lambda x: x["clip"]):
            sc = "—" if r["score_b"] is None else f"{r['score_b']:.3f}"
            print(f"  {r['event_id_v2']:<9}{r['clip']:<11}{r['time']:>9.1f}{r['v1']:>9}"
                  f"{r['v2']:>9}{r['machine_b']:>9}{sc:>8}")
    print()
    if M["n"]:
        share = 1 - same / M["n"]
        if share > 0.20:
            print(f"  {share:.0%} ответов сдвинулись от смены формулировки. Это значит, что")
            print("  первый проход отвечал на несколько иной вопрос, и его согласие с правилом")
            print("  нельзя читать как точность правила — ни в плюс, ни в минус.")
        else:
            print(f"  сдвинулось {share:.0%} — формулировка меняла немного, и первый проход")
            print("  можно считать приблизительно тем же вопросом.")
    print()

    # ---- 2. second pass against the rule ---------------------------------------------------
    def vs_rule(rows: list[dict], key: str) -> dict:
        cells = {(m, h): 0 for m in MACHINE for h in HUMAN}
        for r in rows:
            if r[key] in MACHINE and r["v2"] in HUMAN:
                cells[(r[key], r["v2"])] += 1
        n_turn = sum(cells[(m, "TURN")] for m in MACHINE)
        n_look = sum(cells[(m, "LOOK")] for m in MACHINE)
        rev = cells[("LOOK", "TURN")]
        kept = cells[("TURN", "TURN")]
        return {"cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
                "n_look": n_look, "n_turn": n_turn,
                "n_unclear": sum(cells[(m, "UNCLEAR")] for m in MACHINE),
                "human_turn_machine_look": rev, "human_turn_machine_turn": kept,
                "human_look_machine_look": cells[("LOOK", "LOOK")],
                "human_look_machine_turn": cells[("TURN", "LOOK")],
                "turn_wrongly_revoked_rate": rev / (rev + kept) if (rev + kept) else None,
                "turn_kept_rate": kept / n_turn if n_turn else None,
                "look_caught_rate": (cells[("LOOK", "LOOK")] / n_look) if n_look else None}

    RB = vs_rule(joined, "machine_b")
    RA = vs_rule(joined, "machine_a")
    print("─── 2. ВТОРОЙ ПРОХОД ПРОТИВ ПРАВИЛА P09.4 ───")
    print(f"  {'':<20}{'HUMAN LOOK':>12}{'HUMAN TURN':>12}{'UNCLEAR':>10}")
    for m in MACHINE:
        v = [RB["cells"][f"{m}|{h}"] for h in HUMAN]
        print(f"  {m:<20}{v[0]:>12}{v[1]:>12}{v[2]:>10}")
    print(f"  {'всего':<20}{RB['n_look']:>12}{RB['n_turn']:>12}{RB['n_unclear']:>10}")
    print()
    print(f"  HUMAN TURN → правило LOOK (опасное): {RB['human_turn_machine_look']}")
    print(f"  HUMAN LOOK → правило LOOK (верно):   {RB['human_look_machine_look']}")
    print(f"  HUMAN LOOK → правило TURN:           {RB['human_look_machine_turn']}")
    if RB["turn_wrongly_revoked_rate"] is not None:
        ref = ""
        prev = OUT / "report.json"
        if prev.exists():
            try:
                p1 = json.loads(prev.read_text(encoding="utf-8"))["B_gated"]
                ref = (f"   (в первом проходе было "
                       f"{p1['turn_wrongly_revoked_rate']:.0%})")
            except Exception:
                ref = ""
        print(f"  ошибочно снято поворотов: {RB['turn_wrongly_revoked_rate']:.0%}{ref}")
        print(f"  поворотов сохранено: {RB['turn_kept_rate']:.0%}, "
              f"взглядов снято верно: {RB['look_caught_rate']:.0%}")
    print()

    # ---- informativeness ------------------------------------------------------------------
    tab = np.array([[sum(1 for r in joined if r["machine_b"] == m and r["v2"] == h)
                     for h in ("LOOK", "TURN")] for m in MACHINE], dtype=float)
    row_s, col_s, tot_s = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True), tab.sum()
    exp_ = row_s @ col_s / max(tot_s, 1e-9)
    chi2 = float(((tab - exp_) ** 2 / np.maximum(exp_, 1e-9)).sum())
    from math import exp
    p_chi = exp(-chi2 / 2.0)
    print("─── 3. НЕСЁТ ЛИ КЛАСС ПРАВИЛА ИНФОРМАЦИЮ О ВТОРОМ ПРОХОДЕ ───")
    base = col_s[0][1] / max(tot_s, 1e-9)
    print(f"  базовая доля человек TURN: {base:.0%}")
    for i, m in enumerate(MACHINE):
        rn = tab[i].sum()
        if rn:
            print(f"    при правиле {m:<8} человек TURN {tab[i][1] / rn:.0%} (n={int(rn)})")
    print(f"  хи-квадрат {chi2:.2f}, p ≈ {p_chi:.3f} — "
          f"{'значимо' if p_chi < 0.05 else 'НЕ значимо'}")
    print()

    # ---- self agreement -------------------------------------------------------------------
    def selfag(r1: dict, r2: dict, idmap=lambda x: x) -> dict:
        both = [(e, idmap(e)) for e in r2 if idmap(e) in r1]
        if not both:
            return {"n": 0, "usable": False}
        same = sum(1 for e, e1 in both if r1[e1] == r2[e])
        conf = Counter((r1[e1], r2[e]) for e, e1 in both)
        return {"n": len(both), "usable": True, "same": same, "agreement": same / len(both),
                "switched": conf[("LOOK", "TURN")] + conf[("TURN", "LOOK")],
                "pairs": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items())}}

    sa1 = selfag(a1, r1)
    sa2 = selfag(a2, r2)
    print("─── 4. ПОВТОРНЫЕ ПОКАЗЫ В КАЖДОМ ПРОХОДЕ ───")
    for tag, sa in (("первый проход", sa1), ("второй проход", sa2)):
        if not sa["usable"]:
            print(f"  {tag}: повторов нет")
            continue
        print(f"  {tag}: совпало {sa['same']} из {sa['n']} ({sa['agreement']:.0%}), "
              f"смен LOOK↔TURN {sa['switched']}")
    print()

    (out / "v1_vs_v2.csv").write_text("", encoding="utf-8")
    with (out / "v1_vs_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(joined[0].keys()), extrasaction="ignore")
        w.writeheader()
        for r in joined:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    (out / "report_v2.json").write_text(json.dumps({
        "phase": "P09.5B — две разметки и правило",
        "n_v1": len(a1), "n_v2": len(a2), "n_joined": M["n"],
        "v1_vs_v2": M,
        "words_changed_answers": 1 - same / max(M["n"], 1),
        "direct_look_turn_flips": flipped,
        "changed_events": changed,
        "v2_vs_rule_B": RB, "v2_vs_rule_A": RA,
        "class_informativeness": {"table": tab.astype(int).tolist(), "chi2": chi2,
                                  "p": p_chi, "informative": bool(p_chi < 0.05),
                                  "base_human_turn": base},
        "self_agreement": {"v1": sa1, "v2": sa2},
        "note": (
            "первый проход не удалён и не объявлен неверным: он отвечал на вопрос, который "
            "описывал механизм, и на нём видно, что формулировка меняет ответы. Он сохранён "
            "как human_labels_v1.json, но окончательной проверкой правила не является"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'v1_vs_v2.csv'}")
    print(f"записано: {out/'report_v2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
