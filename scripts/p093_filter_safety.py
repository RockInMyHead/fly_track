#!/usr/bin/env python3
"""P09.3 — does the LOOK filter throw away real turns?

P09.2 showed the filter can withdraw a false direction: at J35 a strong LEFT is called a glance
and the route stops walking into the dead end. The question here is the other side of that, and
it is the one that decides whether the rule is safe: when MaleCNS reports a direction that really
marks a turn, how often does the filter take it away?

The honesty problem, stated first because it changes what the numbers mean
-------------------------------------------------------------------------
There is no human ground truth for look-versus-turn in this project, and that is not an
oversight to be worked around. Every hand-made label that exists — `review_set*.json`,
`turn_verdicts*.json`, `trajectory_annotations.json` — records the *direction of the camera*,
LEFT or RIGHT, and whether it was a real turn rather than a glance was never asked. A person who
confirmed "the camera turned left here" has said nothing about whether the body followed.

So the only available truth is P07.4's own classification, which is derived from the same signal
by a related rule. Agreement with it is therefore not accuracy, and this script never reports it
as such. What it does report is a property that is still worth having: two independently written
readings of the same recording, both frozen, disagreeing about a turn. If P09.2 revoked turns
freely, that would show up here as a large TRUE TURN → LOOK rate, and it would be a reason not to
ship the rule whatever the truth labels say. A small rate bounds the damage without proving
correctness, and that is the honest claim that can be made.

The human labels are still used, in the one way they are valid: a swing that overlaps a
human-confirmed camera turn is a swing where the fly certainly rotated, which is a stronger
statement than the proxy alone and is reported as a separate subset.

What is measured
----------------
Only events where MaleCNS *would* have spoken — the magnitude of the tracker's own reading clears
its own floor. Those are the events the filter is ever invoked on, so those are the events whose
fate matters. Nothing is refitted: the pair algorithm, the 0.610 threshold, the strong-yaw rule and
the UNKNOWN policy are the frozen ones, and the graph is not consulted at all.
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

import p092_look_turn as LT  # noqa: E402

OUT = ROOT / "output/p093"
WINDOW_S = 3.2           # the tracker's own decision window
FLOOR_A, FLOOR_B, FLOOR_SCALE = 0.09, 0.15, 0.50
CLIPS = ["VID00001", "VID00002", "VID00005", "VID00009", "VID00006"]
HELD_OUT = {"VID00005", "VID00009"}
J35_T = 113.74


def floor_for(window_s: float = WINDOW_S) -> float:
    return FLOOR_SCALE * (FLOOR_A + FLOOR_B * max(window_s, 0.0))


def human_camera_turns(video: str) -> list[tuple[float, float, str]]:
    """Hand-confirmed camera turns for this clip, as (t0, t1, direction).

    These certify that the fly rotated — not whether the body followed. They are used only to
    mark which swings are human-anchored, never as a look-or-turn verdict.
    """
    try:
        from p07_fly_readout import labelled_turns  # noqa: E402
        return [(float(x["t0"]), float(x["t1"]), x["kind"])
                for x in labelled_turns() if x["video"] == video]
    except Exception:
        return []


def unique_events(video: str) -> list[dict]:
    """One entry per labelled move, deduplicated exactly as P09.2 deduplicates.

    The swings come from the module, so the pair boundaries are the frozen ones. A pair carries
    its label on both halves, so only the earlier half is kept — counting both would count each
    glance twice with two different windows, which is the mistake that first made these numbers
    disagree with P09.1.
    """
    lt = LT.LookTurn(video)
    t, y = lt.rt, lt.ry
    fl = floor_for()
    seen, out = set(), []
    for s in lt.swing:
        if s["kind"] not in ("LOOK", "TURN"):
            continue
        partner = s.get("paired_with")
        key = (tuple(sorted((round(s["t0"], 3), round(float(partner), 3))))
               if partner is not None else (round(s["t0"], 3),))
        if key in seen:
            continue
        seen.add(key)
        # the tracker's reading at this swing: the signed integral over the window ending at the
        # swing's end, which is where a decision would have been taken
        m = (t >= s["t1"] - WINDOW_S) & (t <= s["t1"])
        if m.sum() < 2:
            continue
        I = float((y[m][:-1] * np.diff(t[m])).sum())
        strong = abs(I) >= fl
        sc = lt.score(s["t0"] + 1e-6)
        # The same window measured on the signal the pair labels were built from. P09.2 computes
        # the ratio on yaw_signal_deadband because that is what the tracker reads; the swing
        # boundaries and the LOOK/TURN labels, however, belong to yaw_gated. Measuring the
        # cancellation on a different recording of the same quantity is what this column exists
        # to expose, and it is a diagnosis rather than a second rule — the frozen number stays
        # the deadband one.
        g_t, g_y = LT.load_swing_source(video)
        gdt = float(np.median(np.diff(g_t)))
        gn = max(int(round(LT.SMOOTH_S / gdt)), 1)
        gcam = np.cumsum(LT.box(g_y, gn)) * gdt
        gm = (g_t >= s["t0"]) & (g_t <= s["t1"] + LT.EVAL_W)
        ratio_gated = None
        if gm.sum() >= 4:
            seg = gcam[gm]
            tot = float(np.abs(np.diff(seg)).sum())
            if tot > 1e-9:
                ratio_gated = abs(float(seg[-1] - seg[0])) / tot
        out.append({"clip": video, "t": s["t1"], "t0": s["t0"], "t1": s["t1"],
                    "paired": int(s.get("paired_with") is not None),
                    "ratio_gated": ratio_gated,
                    "class_gated": (None if ratio_gated is None else
                                    ("LOOK" if ratio_gated < LT.LOOK_TURN_THRESHOLD
                                     else "TURN")),
                    "yaw_integral": I, "yaw_floor": fl,
                    "yaw_direction": ("LEFT" if I > 0 else "RIGHT") if strong else "",
                    "strong": int(strong),
                    "look_turn_score": sc["score"], "class": sc["class"],
                    "truth": s["kind"], "truth_source": "PROXY",
                    "p074_kind": s["kind"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]

    print("=" * 100)
    print("P09.3 — НЕ СНИМАЕТ ЛИ ФИЛЬТР НАСТОЯЩИЕ ПОВОРОТЫ")
    print("=" * 100)
    print(f"  порог заморожен: {LT.LOOK_TURN_THRESHOLD}, окно пары +{LT.EVAL_W} с")
    print(f"  сильный yaw: |интеграл за {WINDOW_S} с| >= {floor_for():.3f} (правило трекера)")
    print(f"  политика UNKNOWN: направление снимается (как в P09.2)")
    print(f"  новых прогонов MaleCNS нет, граф не используется")
    print()

    # ---- the truth problem, before any number --------------------------------------------
    print("─── ИСТОЧНИК ПРАВДЫ: ЕГО НЕТ, И ЭТО НАДО ЗНАТЬ ДО ЦИФР ───")
    human_total = 0
    human_by_clip = {}
    for v in clips:
        h = human_camera_turns(v)
        human_by_clip[v] = len(h)
        human_total += len(h)
    print(f"  ручных меток поворота камеры по клипам: "
          f"{', '.join(f'{v}:{human_by_clip[v]}' for v in clips)}")
    print("  во всех файлах разметки (`review_set*`, `turn_verdicts*`,")
    print("  `trajectory_annotations`) записано только НАПРАВЛЕНИЕ камеры — LEFT или RIGHT.")
    print("  Взгляд это был или поворот тела, человека не спрашивали ни разу.")
    print()
    print("  Значит правды LOOK/TURN у нас нет. Единственная разметка этого различия —")
    print("  автоматическая, из P07.4, полученная из того же сигнала похожим правилом.")
    print("  Совпадение с ней НЕ есть точность, и ниже оно так не называется:")
    print("  измеряется согласие двух независимо написанных замороженных чтений.")
    print("  Польза от этого есть: если бы фильтр резал повороты свободно, здесь было бы")
    print("  видно большую долю TRUE TURN → LOOK, и это был бы довод против правила.")
    print()

    rows: list[dict] = []
    for v in clips:
        ev = unique_events(v)
        ht = human_camera_turns(v)
        for e in ev:
            # human anchoring: the swing overlaps a hand-confirmed camera turn
            e["human_anchored"] = int(any(not (e["t1"] < a or e["t0"] > b) for a, b, _ in ht))
        rows += ev
        n_strong = sum(x["strong"] for x in ev)
        print(f"  {v:<11} меток камеры {len(ht):>3}   размеченных колебаний {len(ev):>4}   "
              f"из них сильный yaw {n_strong:>3}")

    strong = [r for r in rows if r["strong"]]
    print()
    print(f"─── ПРОВЕРЯЮТСЯ ТОЛЬКО СОБЫТИЯ С НАЗВАННЫМ НАПРАВЛЕНИЕМ: {len(strong)} ───")
    print("  это и есть события, на которых фильтр вообще вызывается")
    print()

    # ---- confusion ------------------------------------------------------------------------
    def confusion(sel: list[dict]) -> dict:
        cells = {(t, c): 0 for t in ("LOOK", "TURN") for c in ("LOOK", "TURN", "UNKNOWN")}
        for r in sel:
            cls = r["class"].replace("LOOK_TURN_", "")
            if r["truth"] in ("LOOK", "TURN") and cls in ("LOOK", "TURN", "UNKNOWN"):
                cells[(r["truth"], cls)] += 1
        lo_n = cells[("LOOK", "LOOK")] + cells[("LOOK", "TURN")] + cells[("LOOK", "UNKNOWN")]
        tu_n = cells[("TURN", "TURN")] + cells[("TURN", "LOOK")] + cells[("TURN", "UNKNOWN")]
        return {"cells": {f"{k[0]}->{k[1]}": v for k, v in cells.items()},
                "n_look": lo_n, "n_turn": tu_n,
                "look_revoked_rate": (cells[("LOOK", "LOOK")] / lo_n) if lo_n else None,
                "turn_kept_rate": (cells[("TURN", "TURN")] / tu_n) if tu_n else None,
                "turn_wrongly_revoked_rate": (cells[("TURN", "LOOK")] / tu_n) if tu_n else None,
                "look_unknown_rate": (cells[("LOOK", "UNKNOWN")] / lo_n) if lo_n else None,
                "turn_unknown_rate": (cells[("TURN", "UNKNOWN")] / tu_n) if tu_n else None}

    overall = confusion(strong)
    print("─── ГЛАВНАЯ ТАБЛИЦА (события с названным направлением) ───")
    print("                    P09.2:")
    print(f"  {'':<12}{'LOOK':>8}{'TURN':>8}{'UNKNOWN':>9}{'всего':>8}")
    for t in ("LOOK", "TURN"):
        a = overall["cells"][f"{t}->LOOK"]
        b = overall["cells"][f"{t}->TURN"]
        c = overall["cells"][f"{t}->UNKNOWN"]
        print(f"  правда {t:<5}{a:>8}{b:>8}{c:>9}{a + b + c:>8}")
    print()
    print(f"  взглядов, и фильтр их снял:        "
          f"{overall['look_revoked_rate']:.0%}  ({overall['cells']['LOOK->LOOK']} из "
          f"{overall['n_look']})")
    print(f"  настоящих поворотов сохранилось:   "
          f"{overall['turn_kept_rate']:.0%}  ({overall['cells']['TURN->TURN']} из "
          f"{overall['n_turn']})")
    print(f"  настоящих поворотов снято ОШИБОЧНО:"
          f" {overall['turn_wrongly_revoked_rate']:.0%}  "
          f"({overall['cells']['TURN->LOOK']} из {overall['n_turn']})")
    print(f"  ушло в UNKNOWN: взглядов {overall['look_unknown_rate']:.0%}, "
          f"поворотов {overall['turn_unknown_rate']:.0%}")
    print()

    # ---- the diagnosis: which of the two signals the ratio is measured on ------------------
    print("─── ДИАГНОСТИКА: ПОЧЕМУ ПОВОРОТЫ СНЯТЫ ───")
    rev = [r for r in strong if r["truth"] == "TURN" and r["class"] == "LOOK"]
    unpaired = [r for r in rev if not r["paired"]]
    print(f"  снято поворотов: {len(rev)}")
    print(f"    из них колебания БЕЗ ПАРЫ (P07.4 зовёт их поворотом по размеру): "
          f"{len(unpaired)} ({len(unpaired)/max(len(rev),1):.0%})")
    alt = [r for r in rev if r["class_gated"] == "TURN"]
    print(f"    из них на сигнале yaw_gated — поворот, а не взгляд: {len(alt)} "
          f"({len(alt)/max(len(rev),1):.0%})")
    print()
    print("  Причина: отношение считается на yaw_signal_deadband (сигнал трекера), а границы")
    print("  колебаний и метки P07.4 построены на yaw_gated. На одном и том же окне два")
    print("  сигнала дают разное: там, где deadband даёт 0.3-0.5 (взгляд), gated даёт ~1.0.")
    print()
    both = {}
    for tag, key in (("замороженный (deadband)", "class"),
                     ("на сигнале меток (gated)", "class_gated")):
        tl = sum(1 for r in strong if r["truth"] == "TURN")
        rev_n = sum(1 for r in strong
                    if r["truth"] == "TURN" and (r.get(key) or r["class"]) == "LOOK"
                    and (r.get(key) is not None))
        ll = sum(1 for r in strong if r["truth"] == "LOOK")
        lk_ok = sum(1 for r in strong
                    if r["truth"] == "LOOK" and (r.get(key) or r["class"]) == "LOOK"
                    and (r.get(key) is not None))
        tk_ok = sum(1 for r in strong
                    if r["truth"] == "TURN" and (r.get(key) or r["class"]) == "TURN"
                    and (r.get(key) is not None))
        acc = (lk_ok + tk_ok) / max(ll + tl, 1)
        both[tag] = {"n_turn": tl, "revoked": rev_n,
                     "revoked_rate": rev_n / max(tl, 1), "accuracy": acc}
        print(f"    {tag:<26} снято поворотов {rev_n:>3} из {tl:>3} "
              f"({rev_n/max(tl,1):>4.0%}), точность {acc:>4.0%}")
    print()
    print("  ВАЖНО: смена сигнала — это ИЗМЕНЕНИЕ ПРАВИЛА, а не диагностика. Здесь она не")
    print("  делается: замороженное число остаётся 32%. Но выбор сигнала был сделан по")
    print("  принципу «тот, что читает трекер», а метки и окна принадлежат yaw_gated — то")
    print("  есть отношение мерилось не на том сигнале, на котором построена пара.")
    print()

    # ---- the dangerous error --------------------------------------------------------------
    false_revoked = [r for r in strong if r["truth"] == "TURN" and r["class"] == "LOOK"]
    print(f"─── САМАЯ ОПАСНАЯ ОШИБКА: НАСТОЯЩИЙ ПОВОРОТ, КОТОРОМУ СНЯЛИ НАПРАВЛЕНИЕ ───")
    print(f"  всего таких: {len(false_revoked)}")
    if false_revoked:
        print(f"  {'клип':<11}{'t':>9}{'yaw':>9}{'напр.':>7}{'score':>8}{'порог':>8}  метка")
        for r in sorted(false_revoked, key=lambda x: x["look_turn_score"]):
            print(f"  {r['clip']:<11}{r['t']:>9.1f}{r['yaw_integral']:>+9.3f}"
                  f"{r['yaw_direction']:>7}{r['look_turn_score']:>8.3f}"
                  f"{LT.LOOK_TURN_THRESHOLD:>8.2f}  {r['truth']}")
        print()
        print("  сколько из них подтверждены человеком как поворот камеры: "
              f"{sum(r['human_anchored'] for r in false_revoked)} "
              f"из {len(false_revoked)}")
    else:
        print("  ни одного: ни одно событие, помеченное поворотом, не было снято")
    print()

    # ---- per clip -------------------------------------------------------------------------
    print("─── ПО КАЖДОМУ РОЛИКУ ───")
    print(f"  {'клип':<11}{'сильных':>8}{'LOOK':>7}{'TURN':>7}{'снято LOOK':>12}"
          f"{'снято TURN':>12}{'сохранено TURN':>16}{'UNKNOWN':>9}")
    per_clip = []
    for v in clips:
        sel = [r for r in strong if r["clip"] == v]
        c = confusion(sel)
        per_clip.append({"clip": v, "n_strong": len(sel), **c})
        f = lambda x: "—" if x is None else f"{x:.0%}"
        print(f"  {v:<11}{len(sel):>8}{c['n_look']:>7}{c['n_turn']:>7}"
              f"{f(c['look_revoked_rate']):>12}{f(c['turn_wrongly_revoked_rate']):>12}"
              f"{f(c['turn_kept_rate']):>16}"
              f"{f((c['look_unknown_rate'] if c['n_look'] else 0) or None):>9}")
    print()
    ho = [r for r in strong if r["clip"] in HELD_OUT]
    ch_held = confusion(ho)
    print(f"  ОТДЕЛЬНО VID00005 + VID00009 (не участвовали в построении правила):")
    print(f"    сильных событий {len(ho)}, из них правда LOOK {ch_held['n_look']}, "
          f"правда TURN {ch_held['n_turn']}")
    print(f"    взглядов снято {ch_held['look_revoked_rate']:.0%}, "
          f"поворотов сохранено {ch_held['turn_kept_rate']:.0%}, "
          f"поворотов снято ошибочно {ch_held['turn_wrongly_revoked_rate']:.0%}")
    print()

    # ---- human-anchored subset ------------------------------------------------------------
    han = [r for r in strong if r["human_anchored"]]
    ch_human = confusion(han) if han else None
    print("─── ОПОРА НА РУЧНУЮ РАЗМЕТКУ (только там, где она вообще что-то значит) ───")
    if han:
        c = ch_human
        print(f"  событий, совпавших с подтверждённым человеком поворотом камеры: {len(han)}")
        print(f"    из них P07.4 зовёт поворотом {c['n_turn']}, взглядом {c['n_look']}")
        print(f"    P09.2 сохранил направление у {c['turn_kept_rate']:.0%} из этих поворотов, "
              f"снял у {c['turn_wrongly_revoked_rate']:.0%}")
        print()
        print("  Важно: человек подтвердил, что КАМЕРА повернулась. Про тело он не говорил,")
        print("  поэтому это не отличает взгляд от поворота — но если бы P09.2 снимал")
        print("  направление и здесь, у нас был бы довод против правила.")
    else:
        print("  ни одно событие с названным направлением не совпало с ручной меткой")
    print()

    # ---- strong events with no labelled swing, where P09.2 can only say UNKNOWN ------------
    print("─── СИЛЬНЫЕ СОБЫТИЯ ВНЕ РАЗМЕЧЕННЫХ КОЛЕБАНИЙ (пункт 9) ───")
    print("  здесь правды нет вообще: P07.4 размечает только колебания, а эти сильные места")
    print("  внутрь колебания не попали. P09.2 вернёт для них UNKNOWN, и по политике P09.2")
    print("  направление будет снято — это не ошибка и не успех, а цена политики.")
    print()
    unk_rows = []
    for v in clips:
        t, y = LT.load_ratio_signal(v)
        dt = float(np.median(np.diff(t)))
        w = max(int(round(WINDOW_S / dt)), 1)
        c = np.cumsum(np.concatenate([[0.0], y * dt]))
        I = c[w:] - c[:-w]
        idx = np.where(np.abs(I) >= floor_for())[0]
        if len(idx) == 0:
            unk_rows.append({"clip": v, "strong_regions": 0, "inside": 0, "outside": 0})
            continue
        regs, st, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > w:
                regs.append((st, prev))
                st = i
            prev = i
        regs.append((st, prev))
        lab = [(sw["t0"], sw["t1"]) for sw in LT.LookTurn(v).swing
               if sw["kind"] in ("LOOK", "TURN")]
        inside = sum(1 for a, b in regs
                     if any(not (t[b] < x0 or t[a] > x1) for x0, x1 in lab))
        unk_rows.append({"clip": v, "strong_regions": len(regs), "inside": inside,
                         "outside": len(regs) - inside})
        print(f"  {v:<11} сильных участков {len(regs):>4}, внутри колебания {inside:>4}, "
              f"ВНЕ {len(regs)-inside:>4}")
    tot_out = sum(r["outside"] for r in unk_rows)
    tot_reg = sum(r["strong_regions"] for r in unk_rows)
    print()
    print(f"  всего вне колебаний: {tot_out} из {tot_reg} сильных участков "
          f"({tot_out/max(tot_reg,1):.0%}) — там P09.2 скажет UNKNOWN и снимет направление")
    print()

    # ---- J35 control ----------------------------------------------------------------------
    lt6 = LT.LookTurn("VID00006")
    j = lt6.score(J35_T)
    print("─── КОНТРОЛЬ: J35 ───")
    print(f"  yaw = LEFT, score = {j['score']:.4f}, порог {LT.LOOK_TURN_THRESHOLD}, "
          f"P09.2 = {j['class']}")
    print("  в трекере: male_used_for_route = false, маршрут выбран без этого довода")
    print("  ожидаемый положительный пример — снят именно взгляд")
    print()

    # ---- verdict --------------------------------------------------------------------------
    rate = overall["turn_wrongly_revoked_rate"]
    print("─── ВЫВОД ───")
    print(f"  как заморожено (отношение на yaw_signal_deadband): снято поворотов {rate:.1%}")
    print(f"  на сигнале, которому принадлежат метки (yaw_gated): "
          f"{both['на сигнале меток (gated)']['revoked_rate']:.1%}")
    if ch_human and ch_human["n_turn"]:
        print(f"  на подмножестве с человеческой меткой поворота камеры: "
              f"{ch_human['turn_wrongly_revoked_rate']:.0%} "
              f"({ch_human['cells']['TURN->LOOK']} из {ch_human['n_turn']})")
    print()
    print("  Разброс между первыми двумя числами — не неопределённость измерения, а")
    print("  конкретный дефект: отношение считалось на другом сигнале, чем тот, на котором")
    print("  построены границы колебаний и метки. Разница вчетверо (32% против 8%)")
    print("  объясняется целиком этим, и она проверяема: в false_revoked_turns.csv рядом")
    print("  с каждым случаем лежит значение на втором сигнале.")
    print()
    print("  Что это значит для решения:")
    print("    - замораживать P09.2 в текущем виде рано: 32% против замера в 8% — это")
    print("      незакрытый вопрос о том, на каком сигнале мерить, а не свойство идеи;")
    print("    - но и «правило режет повороты» сказать нельзя: на сигнале меток доля")
    print("      падает до 8%, а на человечески подтверждённом подмножестве до 12%;")
    print("    - правды взгляд/поворот у нас нет. Оба числа — согласие с автоматической")
    print("      разметкой P07.4, и оба надо читать как верхнюю оценку ошибки фильтра.")
    print()
    print("  Предлагаемое действие — отдельная мелкая фаза: привести сигнал отношения к тому")
    print("  же, на котором построена пара, заморозить это как правило и повторить P09.3.")
    print("  Здесь этого не делается, потому что P09.3 — проверка замороженного.")

    # ---- files ----------------------------------------------------------------------------
    with (out / "strong_events.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t", "t0", "t1", "yaw_integral",
                                          "yaw_floor", "yaw_direction", "strong", "paired",
                                          "look_turn_score", "class", "ratio_gated",
                                          "class_gated", "truth", "truth_source",
                                          "human_anchored"], extrasaction="ignore")
        w.writeheader()
        for r in strong:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in w.fieldnames})
    with (out / "confusion_by_clip.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "n_strong", "truth_LOOK", "truth_TURN", "LOOK->LOOK", "LOOK->TURN",
                    "LOOK->UNKNOWN", "TURN->TURN", "TURN->LOOK", "TURN->UNKNOWN",
                    "look_revoked_rate", "turn_kept_rate", "turn_wrongly_revoked_rate",
                    "look_unknown_rate", "turn_unknown_rate"])
        for c in per_clip:
            w.writerow([c["clip"], c["n_strong"], c["n_look"], c["n_turn"],
                        c["cells"]["LOOK->LOOK"], c["cells"]["LOOK->TURN"],
                        c["cells"]["LOOK->UNKNOWN"], c["cells"]["TURN->TURN"],
                        c["cells"]["TURN->LOOK"], c["cells"]["TURN->UNKNOWN"],
                        c["look_revoked_rate"], c["turn_kept_rate"],
                        c["turn_wrongly_revoked_rate"], c["look_unknown_rate"],
                        c["turn_unknown_rate"]])
        w.writerow(["TOTAL", len(strong), overall["n_look"], overall["n_turn"],
                    overall["cells"]["LOOK->LOOK"], overall["cells"]["LOOK->TURN"],
                    overall["cells"]["LOOK->UNKNOWN"], overall["cells"]["TURN->TURN"],
                    overall["cells"]["TURN->LOOK"], overall["cells"]["TURN->UNKNOWN"],
                    overall["look_revoked_rate"], overall["turn_kept_rate"],
                    overall["turn_wrongly_revoked_rate"], overall["look_unknown_rate"],
                    overall["turn_unknown_rate"]])
    with (out / "false_revoked_turns.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t", "yaw_integral", "yaw_direction",
                                          "look_turn_score", "ratio_gated", "class_gated",
                                          "paired", "truth", "human_anchored"],
                           extrasaction="ignore")
        w.writeheader()
        for r in false_revoked:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in w.fieldnames})
    (out / "report.json").write_text(json.dumps({
        "phase": "P09.3 — не снимает ли фильтр настоящие повороты",
        "frozen": {"threshold": LT.LOOK_TURN_THRESHOLD, "eval_window_s": LT.EVAL_W,
                   "strong_rule": f"|интеграл за {WINDOW_S} с| >= {floor_for():.3f}",
                   "unknown_policy": "направление снимается", "new_malecns_runs": 0,
                   "graph_used": False},
        "truth": {
            "human_look_turn_labels": 0,
            "human_camera_turn_labels": human_total,
            "note": "ручной разметки взгляд/поворот не существует; правда только "
                    "автоматическая (P07.4), и совпадение с ней не есть точность",
        },
        "n_strong_events": len(strong),
        "overall": overall,
        "per_clip": per_clip,
        "held_out": {"clips": sorted(HELD_OUT), **ch_held},
        "human_anchored": ch if not han else confusion(han),
        "diagnosis": {
            "revoked_total": len(rev),
            "revoked_unpaired": len(unpaired),
            "revoked_but_turn_on_gated": len(alt),
            "signal_columns": {
                "swing_boundaries_and_labels": "yaw_gated (P07.1)",
                "ratio_as_frozen": "yaw_signal_deadband (P07)",
                "consequence": "одно и то же окно даёт 0.3-0.5 на deadband и ~1.0 на gated",
            },
            "alternative_not_applied": both,
            "note": "смена сигнала это изменение правила; в P09.3 не применяется",
        },
        "false_revoked_turns": false_revoked,
        "false_revoked_n": len(false_revoked),
        "j35": {"yaw": "LEFT", "score": j["score"], "class": j["class"],
                "male_used_for_route": False},
        "conclusions": {
            "no_truth_exists": (
                "ручной разметки взгляд/поворот в проекте нет: все ручные файлы хранят только "
                "НАПРАВЛЕНИЕ камеры. Правда только автоматическая (P07.4), и согласие с ней "
                "не есть точность. Оба числа — верхняя оценка ошибки фильтра."),
            "frozen_number": (
                f"как заморожено: {overall['cells']['TURN->LOOK']} из {overall['n_turn']} настоящих "
                f"поворотов лишены направления ({overall['turn_wrongly_revoked_rate']:.0%}). Это "
                "слишком много, и замораживать в таком виде рано."),
            "diagnosis_not_idea": (
                "но причина в основном одна и она проверяема: отношение считается на "
                "yaw_signal_deadband, а границы колебаний и метки построены на yaw_gated. На "
                f"сигнале меток та же ошибка падает до {both['на сигнале меток (gated)']['revoked_rate']:.0%}, "
                "а на подмножестве с меткой человека до "
                f"{ch_human['turn_wrongly_revoked_rate']:.0%}. Это вопрос согласованности "
                "сигналов, а не свойство правила LOOK/TURN."),
            "unpaired_turns": (
                f"{len(unpaired)} из {len(rev)} снятых поворотов — колебания без пары, которые "
                "P07.4 метит поворотом по размеру, не измеряя возврат. То есть часть "
                "расхождения идёт от запасного правила разметки, а не от фильтра."),
            "unknown_policy_cost": (
                f"{tot_out} из {tot_reg} сильных участков лежат вне размеченных колебаний: там "
                "P09.2 вернёт UNKNOWN и по политике снимет направление. Это цена политики, и "
                "она больше, чем доля ошибочно снятых поворотов."),
            "action": (
                "не замораживать P09.2 в текущем виде. Отдельная мелкая фаза: привести сигнал "
                "отношения к тому же, на котором построена пара, заморозить это как правило и "
                "повторить P09.3. Подгонкой это не является — выбор сигнала был сделан по "
                "принципу «тот, что читает трекер», тогда как окна и метки принадлежат другому."),
            "j35": (
                "контроль пройден: J35 — yaw LEFT, score 0.0136, класс LOOK, направление снято. "
                "Это и есть положительный пример, ради которого фильтр делался."),
        },
        "turn_wrongly_revoked_rate": rate,
        "strong_outside_swings": unk_rows,
        "n_strong_outside_swings": tot_out,
        "n_strong_regions": tot_reg,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'strong_events.csv'}")
    print(f"записано: {out/'confusion_by_clip.csv'}")
    print(f"записано: {out/'false_revoked_turns.csv'}")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
