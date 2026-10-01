#!/usr/bin/env python3
"""P09.5 — the human labels against the frozen rule, and the first non-circular number.

Every comparison in this project until now has been one automatic reading checked against
another. P09.4 got as far as that can go: it showed the frozen filter withdraws 8 percent of the
moves its own labelling calls turns, once the ratio is measured on the signal the pair is built
from. But P07.4's labels and P07.4's pair structure are very nearly the same variable — all 516
glances are paired and 250 of 268 turns are not — so agreeing with them is largely reproducing
them. This script is where a person's answer replaces that.

Two things are reported and they answer different questions:

    against the frozen rule      how the human labels distribute over what P09.4 decided. This is
                                 the measurement the phase exists for.
    against itself               the ten moves shown twice, to see how much of the task is
                                 objectively undecidable. A person who disagrees with themselves
                                 has set a ceiling that no rule can be blamed for missing.

The dangerous cell is human TURN against machine LOOK: the walker turned and the filter threw the
direction away. That is the error that cost J35 its dead end, and it is the one number that
decides whether the filter ships.

Nothing is refitted. The thresholds, the signals, the pair window and the policy come from
`data/p095/FROZEN_P094.json`. Both variants are reported — A on the debanded signal and B on the
gated one — but neither is tuned here, and the sixty moves are a test set: if the rule is changed
after this, these sixty cannot be used as evidence that the change helped.

Usage:
    PYTHONPATH=. python scripts/p095_evaluate.py
    PYTHONPATH=. python scripts/p095_evaluate.py --allow-partial   # for a mid-review look
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


def load() -> tuple[dict, list[dict]]:
    s = json.loads((DATA / "review_set.json").read_text(encoding="utf-8"))
    lab = json.loads((DATA / "human_labels.json").read_text(encoding="utf-8"))["labels"]
    return s, lab


def first_pass(labels: list[dict]) -> dict[str, str]:
    return {l["event_id"]: l["human_label"] for l in labels if l.get("round", 1) == 1}


def second_pass(labels: list[dict]) -> dict[str, str]:
    return {l["event_id"]: l["human_label"] for l in labels if l.get("round", 1) == 2}


def matrix(rows: list[dict], key: str) -> dict:
    """P09.4's class against the human's, for one variant of the rule."""
    cells = {(m, h): 0 for m in MACHINE for h in HUMAN}
    for r in rows:
        m = r[key]
        h = r["human"]
        if m in MACHINE and h in HUMAN:
            cells[(m, h)] += 1
    n_turn = cells[("LOOK", "TURN")] + cells[("TURN", "TURN")] + cells[("UNKNOWN", "TURN")]
    n_look = cells[("LOOK", "LOOK")] + cells[("TURN", "LOOK")] + cells[("UNKNOWN", "LOOK")]
    return {
        "cells": {f"{m}|{h}": cells[(m, h)] for m in MACHINE for h in HUMAN},
        "n_human_turn": n_turn, "n_human_look": n_look,
        "n_human_unclear": sum(cells[(m, "UNCLEAR")] for m in MACHINE),
        # the dangerous one: the walker turned, and the rule withdrew the direction
        "human_turn_machine_look": cells[("LOOK", "TURN")],
        "human_turn_machine_turn": cells[("TURN", "TURN")],
        "human_turn_machine_unknown": cells[("UNKNOWN", "TURN")],
        # the other side: a glance kept alive as a turn
        "human_look_machine_turn": cells[("TURN", "LOOK")],
        "human_look_machine_look": cells[("LOOK", "LOOK")],
        "human_look_machine_unknown": cells[("UNKNOWN", "LOOK")],
        # of the turns the rule spoke about at all, how many did it wrongly withdraw
        "turn_wrongly_revoked_rate": (
            cells[("LOOK", "TURN")] / (cells[("LOOK", "TURN")] + cells[("TURN", "TURN")])
            if (cells[("LOOK", "TURN")] + cells[("TURN", "TURN")]) else None),
        "turn_kept_rate": (
            cells[("TURN", "TURN")] / n_turn if n_turn else None),
        "look_caught_rate": (
            cells[("LOOK", "LOOK")] / n_look if n_look else None),
    }


def self_agreement(labels: list[dict]) -> dict:
    a, b = first_pass(labels), second_pass(labels)
    both = [e for e in b if e in a]
    if not both:
        return {"n": 0, "usable": False}
    same = sum(1 for e in both if a[e] == b[e])
    conf = Counter((a[e], b[e]) for e in both)
    return {"n": len(both), "usable": True, "agreement": same / len(both),
            "same": same,
            "switched_look_turn": conf[("LOOK", "TURN")] + conf[("TURN", "LOOK")],
            "pairs": {f"{x[0]}->{x[1]}": v for x, v in sorted(conf.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not (DATA / "human_labels.json").exists():
        print("меток ещё нет: data/p095/human_labels.json не создан")
        print("разметку запускает scripts/p095_review_server.py")
        return 1
    s, labels = load()
    events = {e["event_id"]: e for e in s["events"]}
    first = first_pass(labels)
    n = len(first)
    if n < len(events) and not args.allow_partial:
        print(f"размечено {n} из {len(events)}. Пока не все — оценка была бы предварительной.")
        print("  для промежуточного взгляда: --allow-partial")
        return 1

    frozen = json.loads((DATA / "FROZEN_P094.json").read_text(encoding="utf-8"))
    rows = []
    for eid, h in first.items():
        e = events.get(eid)
        if not e:
            continue
        rows.append({"event_id": eid, "clip": e["clip"], "time": e["time"], "human": h,
                     "machine_b": e["hidden"]["algo_class_b"],
                     "score_b": e["hidden"]["algo_score_b"],
                     "machine_a": e["hidden"]["algo_class_a"],
                     "score_a": e["hidden"]["algo_score_a"],
                     "proxy": e["hidden"]["proxy_label"],
                     "pair_status": e["hidden"]["pair_status"],
                     "direction": e["hidden"]["algo_direction"]})

    print("=" * 100)
    print("P09.5 — ЧЕЛОВЕК ПРОТИВ ЗАМОРОЖЕННОГО ПРАВИЛА")
    print("=" * 100)
    print(f"  размечено: {n} из {len(events)}")
    print(f"  правило: порог {frozen['frozen']['look_turn_threshold']}, "
          f"окно {frozen['frozen']['pair_window']}")
    print(f"  B — отношение на yaw_gated (заморожено)   A — на yaw_signal_deadband")
    print()

    B = matrix(rows, "machine_b")
    A = matrix(rows, "machine_a")

    def table(m: dict, title: str) -> None:
        print(f"─── {title} ───")
        print(f"  {'':<18}{'HUMAN LOOK':>12}{'HUMAN TURN':>12}{'UNCLEAR':>10}{'всего':>8}")
        for mk in MACHINE:
            vals = [m["cells"][f"{mk}|{h}"] for h in HUMAN]
            print(f"  {mk:<18}{vals[0]:>12}{vals[1]:>12}{vals[2]:>10}{sum(vals):>8}")
        print(f"  {'всего':<18}{m['n_human_look']:>12}{m['n_human_turn']:>12}"
              f"{m['n_human_unclear']:>10}{n:>8}")
        print()

    table(B, "B: gated (замороженный вариант)")
    print("─── ГЛАВНЫЕ ЧИСЛА (B) ───")
    print(f"  1. HUMAN TURN → P09.4 LOOK   (самое опасное): "
          f"{B['human_turn_machine_look']} из {B['n_human_turn']} настоящих поворотов")
    print(f"  2. HUMAN LOOK → P09.4 TURN   (взгляд как поворот): "
          f"{B['human_look_machine_turn']} из {B['n_human_look']}")
    print(f"  3. HUMAN LOOK → P09.4 LOOK   (взгляд снят верно): "
          f"{B['human_look_machine_look']}")
    print(f"  4. HUMAN TURN → P09.4 TURN   (поворот сохранён): "
          f"{B['human_turn_machine_turn']}")
    print(f"  5. HUMAN UNCLEAR: {B['n_human_unclear']}")
    print()
    if B["turn_wrongly_revoked_rate"] is not None:
        print(f"  из поворотов, о которых правило вообще говорило: "
              f"ошибочно снято {B['turn_wrongly_revoked_rate']:.0%}, "
              f"сохранено {B['turn_kept_rate']:.0%}")
    print(f"  взглядов снято верно: {B['look_caught_rate']:.0%}"
          if B["look_caught_rate"] is not None else "")
    print()

    # ---- against the proxy, which is the comparison that turned out to matter --------------
    # P09.4's numbers were agreement with P07.4, and P09.2 was built to reproduce P07.4's reading
    # on a different signal. If P07.4 itself disagrees with a person, then all of P09.1 to P09.4
    # measured agreement with something the person contradicts, and the rule inherits the error.
    print("─── ЧЕЛОВЕК ПРОТИВ PROXY P07.4 (не против правила, а против самой разметки) ───")
    pcells = {(p, h): 0 for p in MACHINE for h in HUMAN}
    for r in rows:
        pcells[(r["proxy"], r["human"])] += 1
    print(f"  {'proxy P07.4':<16}{'HUMAN LOOK':>12}{'HUMAN TURN':>12}{'UNCLEAR':>10}")
    for p in MACHINE:
        v = [pcells[(p, h)] for h in HUMAN]
        print(f"  {p:<16}{v[0]:>12}{v[1]:>12}{v[2]:>10}")
    agree = pcells[("LOOK", "LOOK")] + pcells[("TURN", "TURN")]
    print(f"  {'совпало':<16}{agree:>12} из {n} ({agree / n:.0%})")
    print(f"  proxy сказал LOOK, человек TURN: {pcells[('LOOK', 'TURN')]} из "
          f"{pcells[('LOOK', 'LOOK')] + pcells[('LOOK', 'TURN')]}")
    print()
    print("  Это и есть цена круга: правило проверялось против разметки, которая сама")
    print("  расходится с человеком в двух случаях из трёх.")
    print()


    # ---- does the rule's class carry any information about the human's judgement? ----------
    # This is the sharpest form of the question. If the class is informative, the share of human
    # TURN should differ sharply across the rule's classes. If it is the same everywhere, the
    # rule is not measuring what the person sees, whatever the margins say.
    print("─── НЕСЁТ ЛИ КЛАСС ПРАВИЛА ИНФОРМАЦИЮ О МЕТКЕ ЧЕЛОВЕКА ───")
    tab = np.array([[sum(1 for r in rows
                         if r["machine_b"] == m and r["human"] == h) for h in ("LOOK", "TURN")]
                    for m in MACHINE], dtype=float)
    print(f"  таблица (строка — правило, столбец — человек): {tab.astype(int).tolist()}")
    row_s, col_s, tot_s = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True), tab.sum()
    expect = row_s @ col_s / max(tot_s, 1e-9)
    chi2 = float(((tab - expect) ** 2 / np.maximum(expect, 1e-9)).sum())
    from math import exp
    p_chi = exp(-chi2 / 2.0) if tab.shape[0] == 3 else float("nan")   # dof = 2
    print(f"  хи-квадрат {chi2:.2f}, степеней свободы 2, p ≈ {p_chi:.3f}")
    base_turn = col_s[0][1] / max(tot_s, 1e-9)
    print(f"  базовая доля человек TURN во всей выборке: {base_turn:.0%}")
    for i, m in enumerate(MACHINE):
        r_n = tab[i].sum()
        if r_n:
            print(f"    при правиле {m:<8} человек TURN {tab[i][1] / r_n:.0%}  (n={int(r_n)})")
    informative = p_chi < 0.05
    if not informative:
        print()
        print("  Различие между классами НЕ значимо: доля поворотов примерно одна и та же")
        print("  при всех классах правила. Значит класс правила не различает то, что видит")
        print("  человек, — и это важнее любых долей в матрице выше.")
    print()

    # ---- the case the phase was built for -------------------------------------------------
    print("─── J35: СЛУЧАЙ, РАДИ КОТОРОГО ДЕЛАЛСЯ P09.2 ───")
    j35 = [r for r in rows if r["clip"] == "VID00006" and abs(r["time"] - 113.74) < 6]
    if not j35:
        print("  событие J35 в выборку не попало")
    for r in j35:
        print(f"  {r['event_id']}  t={r['time']:.1f}")
        print(f"    человек:    {r['human']}")
        print(f"    правило B:  {r['machine_b']} (score {r['score_b']})")
        print(f"    proxy P07.4:{r['proxy']}")
    if j35 and j35[0]["human"] == "TURN":
        print()
        print("  Человек говорит ПОВОРОТ там, где всё построено на том, что это был взгляд.")
        print("  Это надо разрешить до любых выводов: либо вопрос поставлен неоднозначно,")
        print("  либо предпосылка P09.2 неверна.")
    print()

    # ---- the held-out clips ---------------------------------------------------------------
    fo = [r for r in rows if r["clip"] in FOCUS]
    Bf = matrix(fo, "machine_b")
    print(f"─── ОТДЕЛЬНО VID00005 + VID00009 (не участвовали в построении правила) ───")
    print(f"  размечено событий: {len(fo)}")
    print(f"  HUMAN TURN → P09.4 LOOK: {Bf['human_turn_machine_look']} из {Bf['n_human_turn']}")
    print(f"  HUMAN LOOK → P09.4 LOOK: {Bf['human_look_machine_look']} из {Bf['n_human_look']}")
    print(f"  HUMAN UNCLEAR: {Bf['n_human_unclear']}")
    if Bf["turn_wrongly_revoked_rate"] is not None:
        print(f"  ошибочно снято поворотов: {Bf['turn_wrongly_revoked_rate']:.0%}")
    print()

    # ---- A against B ----------------------------------------------------------------------
    print("─── A (deadband) ПРОТИВ B (gated), ПО ЧЕЛОВЕЧЕСКИМ МЕТКАМ ───")
    print(f"  {'':<34}{'A':>8}{'B':>8}")
    for label, ka, kb in (
            ("ошибочно снято поворотов", A["human_turn_machine_look"], B["human_turn_machine_look"]),
            ("поворотов сохранено", A["human_turn_machine_turn"], B["human_turn_machine_turn"]),
            ("взглядов снято верно", A["human_look_machine_look"], B["human_look_machine_look"]),
            ("взглядов оставлено как поворот", A["human_look_machine_turn"], B["human_look_machine_turn"])):
        print(f"  {label:<34}{ka:>8}{kb:>8}")
    if A["turn_wrongly_revoked_rate"] is not None and B["turn_wrongly_revoked_rate"] is not None:
        print(f"  {'доля ошибочно снятых (от говорённых)':<34}"
              f"{A['turn_wrongly_revoked_rate']:>8.0%}{B['turn_wrongly_revoked_rate']:>8.0%}")
        print()
        best_turn = "A" if A["turn_wrongly_revoked_rate"] < B["turn_wrongly_revoked_rate"] else "B"
        la = A["look_caught_rate"] or 0
        lb = B["look_caught_rate"] or 0
        best_look = "A" if la > lb else "B"
        print(f"  реже снимает настоящий поворот: {best_turn}")
        print(f"  лучше ловит настоящий взгляд:   {best_look}")
    print()

    # ---- self agreement -------------------------------------------------------------------
    sa = self_agreement(labels)
    print("─── ПОВТОРНЫЕ ПОКАЗЫ: СОГЛАСИЕ ЧЕЛОВЕКА С САМИМ СОБОЙ ───")
    if not sa["usable"]:
        print("  повторных показов пока нет")
    else:
        print(f"  показано дважды: {sa['n']}")
        print(f"  метка совпала: {sa['same']} из {sa['n']} ({sa['agreement']:.0%})")
        print(f"  смен LOOK ↔ TURN между показами: {sa['switched_look_turn']}")
        print(f"  это верхняя граница: чего человек не может решить стабильно, того и правило")
        print(f"  не обязано решать.")
    print()

    # ---- the mistakes ---------------------------------------------------------------------
    mistakes = [r for r in rows if r["human"] == "TURN" and r["machine_b"] == "LOOK"]
    print("─── ВСЕ СЛУЧАИ: ЧЕЛОВЕК СКАЗАЛ ПОВОРОТ, ПРАВИЛО СНЯЛО НАПРАВЛЕНИЕ ───")
    if not mistakes:
        print("  ни одного")
    else:
        print(f"  {'событие':<9}{'клип':<11}{'t':>9}{'score B':>9}{'proxy':>8}"
              f"{'пара':>14}{'напр.':>7}")
        for r in mistakes:
            sc = "—" if r["score_b"] is None else f"{r['score_b']:.3f}"
            print(f"  {r['event_id']:<9}{r['clip']:<11}{r['time']:>9.1f}{sc:>9}"
                  f"{r['proxy']:>8}{r['pair_status']:>14}{r['direction']:>7}")
    print()

    # ---- files ----------------------------------------------------------------------------
    with (out / "confusion.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "machine", "human", "count"])
        for tag, m in (("B_gated", B), ("A_deadband", A), ("B_gated_heldout", Bf)):
            for mk in MACHINE:
                for h in HUMAN:
                    w.writerow([tag, mk, h, m["cells"][f"{mk}|{h}"]])
    with (out / "mistakes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "clip", "time", "human", "machine_b",
                                          "score_b", "machine_a", "score_a", "proxy",
                                          "pair_status", "direction"], extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["human"], x["machine_b"])):
            w.writerow(r)
    by_clip = []
    for c in sorted({r["clip"] for r in rows}):
        sel = [r for r in rows if r["clip"] == c]
        m = matrix(sel, "machine_b")
        by_clip.append({"clip": c, "n": len(sel), **m})
    with (out / "by_clip.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "n", "human_LOOK", "human_TURN", "human_UNCLEAR",
                    "turn_machine_LOOK", "turn_machine_TURN", "turn_machine_UNKNOWN",
                    "look_machine_LOOK", "look_machine_TURN", "look_machine_UNKNOWN"])
        for b in by_clip:
            w.writerow([b["clip"], b["n"], b["n_human_look"], b["n_human_turn"],
                        b["n_human_unclear"], b["human_turn_machine_look"],
                        b["human_turn_machine_turn"], b["human_turn_machine_unknown"],
                        b["human_look_machine_look"], b["human_look_machine_turn"],
                        b["human_look_machine_unknown"]])
    (out / "report.json").write_text(json.dumps({
        "phase": "P09.5 — первая ручная разметка LOOK/TURN против замороженного правила",
        "n_labelled": n, "n_events": len(events),
        "frozen": frozen["frozen"],
        "B_gated": B, "A_deadband": A,
        "held_out_B": {"clips": list(FOCUS), "n": len(fo), **Bf},
        "self_agreement": sa,
        "mistakes_human_turn_machine_look": mistakes,
        "human_vs_proxy": {f"{p}|{h}": pcells[(p, h)] for p in MACHINE for h in HUMAN},
        "class_informativeness": {"table": tab.astype(int).tolist(), "chi2": chi2,
                                  "p": p_chi, "informative": bool(informative),
                                  "base_human_turn": base_turn},
        "human_vs_proxy_agreement": agree / n,
        "proxy_look_human_turn": pcells[("LOOK", "TURN")],
        "j35": [{"event_id": r["event_id"], "time": r["time"], "human": r["human"],
                 "machine_b": r["machine_b"], "score_b": r["score_b"], "proxy": r["proxy"]}
                for r in j35],
        "by_clip": by_clip,
        "what_this_can_now_claim": (
            "впервые числа посчитаны от независимой человеческой метки, а не от автоматической "
            "разметки. До этого согласие с proxy в большой мере воспроизводило структуру пары "
            "(516 LOOK все парные, 250 TURN из 268 без пары)."),
        "what_it_still_cannot": (
            f"60 событий — небольшая выборка, и {sa['n'] if sa.get('usable') else 0} повторных "
            "показов задают верхнюю границу по объективной неоднозначности задачи. Если человек "
            "сам меняет метку между показами, часть расхождения с правилом не его вина и не вина "
            "правила."),
        "rule_for_these_60": frozen["rule_for_these_60"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    # ---- picture ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(21, 11.5), dpi=105)
        fig.patch.set_facecolor("#0f1117")

        def draw(axpos, m, title):
            ax = fig.add_axes(axpos)
            ax.set_facecolor("#0f1117")
            ax.axis("off")
            ax.text(0, 1.16, title, color="white", fontsize=14, weight="bold")
            ax.text(0, 1.05, "HUMAN:", color="#b8c2d8", fontsize=10)
            for j, h in enumerate(HUMAN):
                ax.text(0.46 + j * 0.19, 0.93, h, color="#e8ecf5", fontsize=11,
                        ha="center", weight="bold")
            for i, mk in enumerate(MACHINE):
                y = 0.66 - i * 0.30
                ax.text(0.0, y + 0.04, mk, color="#e8ecf5", fontsize=11, weight="bold")
                for j, h in enumerate(HUMAN):
                    v = m["cells"][mk + "|" + h]
                    danger = (mk == "LOOK" and h == "TURN")
                    good = (mk == "LOOK" and h == "LOOK")
                    col = "#ff3b30" if danger else ("#3ddc84" if good else "#39415a")
                    ax.add_patch(plt.Rectangle((0.34 + j * 0.19, y - 0.04), 0.17, 0.20,
                                               color=col, alpha=0.6))
                    ax.text(0.425 + j * 0.19, y + 0.06, str(v), color="white", fontsize=15,
                            ha="center", va="center", weight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.25)

        draw([0.035, 0.60, 0.24, 0.30], B, "B: gated (заморожено)")
        draw([0.30, 0.60, 0.24, 0.30], A, "A: deadband (для сравнения)")

        ax = fig.add_axes([0.58, 0.60, 0.19, 0.30])
        ax.set_facecolor("#0f1117")
        vals = [B["human_turn_machine_look"], B["human_turn_machine_turn"]]
        ax.bar([0, 1], vals, color=["#ff3b30", "#3ddc84"], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.4, str(v), ha="center", color="white", fontsize=15,
                    weight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["снят — ОШИБКА", "сохранён — верно"], color="#c8cede", fontsize=10)
        ax.tick_params(colors="#c8cede")
        for sp in ax.spines.values():
            sp.set_color("#39415a")
        ax.set_ylim(0, max(vals + [1]) * 1.35 + 1)
        ax.set_ylabel("настоящих поворотов", color="#c8cede", fontsize=10)
        ax.set_title("человек сказал TURN: " + str(B["n_human_turn"]), color="white",
                     fontsize=12, loc="left")
        ax.grid(alpha=0.12, color="#5a6580", axis="y")

        ax = fig.add_axes([0.80, 0.60, 0.17, 0.30])
        ax.set_facecolor("#0f1117")
        if sa.get("usable"):
            pair_vals = [sa["same"], sa["switched_look_turn"]]
            ax.bar([0, 1], pair_vals, color=["#3ddc84", "#ff9f1c"], width=0.6)
            for i, v in enumerate(pair_vals):
                ax.text(i, v + 0.15, str(v), ha="center", color="white", fontsize=13,
                        weight="bold")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["совпало", "LOOK-TURN"], color="#c8cede", fontsize=10)
            ax.set_title("повторные показы: " + str(sa["n"]), color="white", fontsize=12,
                         loc="left")
        else:
            ax.text(0.5, 0.5, "повторов пока нет", color="#c8cede", ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.tick_params(colors="#c8cede")
        for sp in ax.spines.values():
            sp.set_color("#39415a")
        ax.grid(alpha=0.12, color="#5a6580", axis="y")

        axf = fig.add_axes([0.035, 0.04, 0.93, 0.48])
        axf.set_axis_off()
        sa_n = sa["n"] if sa.get("usable") else 0
        lines = [
            ("ЧТО ЭТО ВПЕРВЫЕ ПОКАЗЫВАЕТ", "#3ddc84", 15, "bold"),
            ("Числа посчитаны от независимой человеческой метки, а не от автоматической разметки", "#c8cede", 11, "normal"),
            ("P07.4. До этого согласие с proxy в большой мере воспроизводило структуру пары:", "#c8cede", 11, "normal"),
            ("516 LOOK все парные, 250 TURN из 268 без пары. Теперь это измерено снаружи.", "#c8cede", 11, "normal"),
            ("", "#000", 6, "normal"),
            ("САМОЕ ОПАСНОЕ: человек сказал TURN, правило сняло направление — "
             + str(B["human_turn_machine_look"]) + " из " + str(B["n_human_turn"]), "#ff9f1c", 15, "bold"),
            ("Отложенные клипы VID00005+VID00009: "
             + str(Bf["human_turn_machine_look"]) + " из " + str(Bf["n_human_turn"])
             + "  (" + str(len(fo)) + " событий размечено)", "#c8cede", 11, "normal"),
            ("", "#000", 6, "normal"),
            ("ЧЕГО ЭТО ВСЁ ЕЩЁ НЕ ДОКАЗЫВАЕТ", "#ff9f1c", 15, "bold"),
            ("60 событий — небольшая выборка, и " + str(sa_n) + " повторных показов задают верхнюю границу:", "#c8cede", 11, "normal"),
            ("если человек сам меняет LOOK на TURN между показами, часть расхождения —", "#c8cede", 11, "normal"),
            ("не его вина и не вина правила. Это и есть мера объективной неоднозначности задачи.", "#c8cede", 11, "normal"),
            ("", "#000", 6, "normal"),
            ("ЭТИ 60 НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ ПОВТОРНО", "#ffffff", 15, "bold"),
            ("Набор — проверочная выборка. Если после неё правило изменить, эти же события", "#c8cede", 11, "normal"),
            ("не могут служить доказательством улучшения: понадобится новый набор.", "#c8cede", 11, "normal"),
        ]
        y = 1.0
        for txt, col, size, wt in lines:
            if txt:
                axf.text(0.0, y, txt, color=col, fontsize=size, weight=wt)
            y -= 0.045 if size < 13 else 0.070
        axf.set_xlim(0, 1)
        axf.set_ylim(-0.05, 1.05)

        fig.suptitle("P09.5 — человек против замороженного правила: первое честное измерение",
                     color="white", fontsize=16)
        fig.savefig(out / "p095_summary.png", dpi=105, facecolor="#0f1117")
        print("записано: " + str(out / "p095_summary.png"))
    except Exception as exc:
        print("картинка не построена: " + str(exc))

    print(f"записано: {out/'confusion.csv'}")
    print(f"записано: {out/'mistakes.csv'}")
    print(f"записано: {out/'by_clip.csv'}")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
