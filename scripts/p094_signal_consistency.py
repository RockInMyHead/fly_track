#!/usr/bin/env python3
"""P09.4 — measure the look-or-turn ratio on the signal the pair is built from.

P09.3 found that the frozen filter withdrew the direction from 32 percent of the events its own
labelling calls turns, and traced the disagreement to one thing: the swing boundaries and the
LOOK/TURN labels come from `yaw_gated`, while the ratio inside the window was measured on
`yaw_signal_deadband`. On the same window the two signals give 0.3-0.5 and about 1.0
respectively, so the ratio was reading a different quantity from the one the window was drawn
for. This script fixes exactly that and nothing else.

Nothing else moves. The pair boundaries, the pair-finding algorithm, the 0.610 threshold, the
strong-event rule, the UNKNOWN policy and the five clips are the P09.2 ones. No threshold is
searched. The `sweep` in `p091` and the sweep here both exist only as an audit of the grid.

The roles of the two signals, stated once
------------------------------------------
    yaw_signal_deadband   the tracker's own reading: whether there is a direction at all, and
                          which one. Unchanged.
    yaw_gated             the structure of the move: where the swing starts and ends, which
                          swing pairs with which, and the ratio measured inside that window.

So MaleCNS still answers LEFT / RIGHT / silence, and the filter still only withdraws.

What this cannot show
---------------------
The truth labels remain automatic. There is no hand-made look-versus-turn label anywhere in the
project — every human file records only the camera's direction. Every number below is therefore
agreement with P07.4's proxy and is reported as such; a good result here bounds the damage, it
does not establish accuracy. That is why the phase ends by proposing the first real annotation
rather than by freezing anything.

The pair question, kept separate
--------------------------------
A large share of what the proxy calls a turn is a single swing with no partner at all — 89 to 100
percent of them depending on the clip. Those were labelled turns by P07.4's fallback rule, which
calls any large unpaired swing a turn *by size*, never having measured a return. The ratio on
such a swing is not a measurement of cancellation and cannot be, so those events are reported
apart from the ones where a return really was observed. Mixing them is what made the P09.3
number hard to read.
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

OUT = ROOT / "output/p094"
P07 = ROOT / "output/p07"
WINDOW_S = 3.2
FLOOR_A, FLOOR_B, FLOOR_SCALE = 0.09, 0.15, 0.50
CLIPS = ["VID00001", "VID00002", "VID00005", "VID00009", "VID00006"]
HELD_OUT = {"VID00005", "VID00009"}
J35_T = 113.74


def floor_for(window_s: float = WINDOW_S) -> float:
    return FLOOR_SCALE * (FLOOR_A + FLOOR_B * max(window_s, 0.0))


def gated_camera(video: str) -> tuple[np.ndarray, np.ndarray]:
    """The accumulated `yaw_gated`, prepared exactly as the pair-finding prepares it.

    Built once per clip and reused for every window, so the ratio below is measured on the very
    signal whose zero crossings defined the windows.
    """
    t, y = LT.load_swing_source(video)
    dt = float(np.median(np.diff(t)))
    n = max(int(round(LT.SMOOTH_S / dt)), 1)
    return t, np.cumsum(LT.box(y, n)) * dt


def ratio_on(cam: np.ndarray, t: np.ndarray, t0: float, end: float) -> float | None:
    m = (t >= t0) & (t <= end)
    if m.sum() < 4:
        return None
    seg = cam[m]
    tot = float(np.abs(np.diff(seg)).sum())
    if tot < 1e-9:
        return None
    return abs(float(seg[-1] - seg[0])) / tot


def events(video: str) -> list[dict]:
    """Every labelled move, once, with both readings and the pair status.

    Deduplicated the way P09.2 deduplicates: a labelled pair carries its label on both halves,
    so only the earlier one counts. Both ratios are computed on the same window — that is the
    whole point — differing only in which signal they are measured on.
    """
    lt = LT.LookTurn(video)
    g_t, g_cam = gated_camera(video)
    r_t, r_y = lt.rt, lt.ry
    r_cam = np.cumsum(r_y) * float(np.median(np.diff(r_t)))
    fl = floor_for()

    # tracker's strength reading, on its own signal, over the window ending at the swing end
    m = (r_t >= 0) & (r_t < 0)
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

        w = (r_t >= s["t1"] - WINDOW_S) & (r_t <= s["t1"])
        if w.sum() < 2:
            continue
        I = float((r_y[w][:-1] * np.diff(r_t[w])).sum())
        strong = int(abs(I) >= fl)
        if not strong:
            continue                      # the filter is only ever invoked on a named direction

        end = s["t1"] + LT.EVAL_W
        a = ratio_on(r_cam, r_t, s["t0"], end)
        b = ratio_on(g_cam, g_t, s["t0"], end)
        out.append({
            "clip": video, "t0": s["t0"], "t1": s["t1"], "t": s["t1"],
            "yaw_integral": I, "yaw_floor": fl,
            "yaw_direction": ("LEFT" if I > 0 else "RIGHT"),
            "pair_status": "PAIR_FOUND" if partner is not None else "PAIR_NOT_FOUND",
            "paired_with": partner,
            "score_deadband": a, "score_gated": b,
            "class_deadband": (None if a is None else
                               ("LOOK" if a < LT.LOOK_TURN_THRESHOLD else "TURN")),
            "class_gated": (None if b is None else
                            ("LOOK" if b < LT.LOOK_TURN_THRESHOLD else "TURN")),
            "truth": s["kind"], "truth_source": "P07.4_PROXY_NOT_HUMAN",
        })
    return out


def _pct(v, nd=0):
    """Percent for printing, or an em dash when the rate does not exist.

    Several rates are None whenever a denominator is zero — which happens as soon as the clips
    under test contain no instance of a class. Formatting those with `.0%` raised, so the run died
    at the summary after doing all its work, and the only way to see the numbers was to fix the
    line and rerun. Emitting a dash keeps the table readable and keeps "not applicable" from
    looking like "zero".
    """
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}%}"
    except (TypeError, ValueError):
        return "—"


def _num(v, nd=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def confusion(sel: list[dict], key: str) -> dict:
    cells = {(t, c): 0 for t in ("LOOK", "TURN") for c in ("LOOK", "TURN", "UNKNOWN")}
    for r in sel:
        cls = r.get(key)
        if cls is None:
            cls = "UNKNOWN"
        if r["truth"] in ("LOOK", "TURN") and cls in ("LOOK", "TURN", "UNKNOWN"):
            cells[(r["truth"], cls)] += 1
    lo = sum(cells[("LOOK", c)] for c in ("LOOK", "TURN", "UNKNOWN"))
    tu = sum(cells[("TURN", c)] for c in ("LOOK", "TURN", "UNKNOWN"))
    return {"cells": {f"{k[0]}->{k[1]}": v for k, v in cells.items()},
            "n_look": lo, "n_turn": tu,
            "look_revoked_rate": (cells[("LOOK", "LOOK")] / lo) if lo else None,
            "turn_kept_rate": (cells[("TURN", "TURN")] / tu) if tu else None,
            "turn_wrongly_revoked_rate": (cells[("TURN", "LOOK")] / tu) if tu else None,
            "look_unknown_rate": (cells[("LOOK", "UNKNOWN")] / lo) if lo else None,
            "turn_unknown_rate": (cells[("TURN", "UNKNOWN")] / tu) if tu else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]

    print("=" * 100)
    print("P09.4 — ОТНОШЕНИЕ МЕРИТСЯ НА ТОМ ЖЕ СИГНАЛЕ, ЧТО И ПАРА")
    print("=" * 100)
    print(f"  порог {LT.LOOK_TURN_THRESHOLD} — тот же; окно пары +{LT.EVAL_W} с — то же")
    print(f"  границы и метки: yaw_gated   ·   направление и сила: yaw_signal_deadband")
    print(f"  A — отношение на deadband (как было в P09.3)")
    print(f"  B — отношение на gated (то, что чинит эта фаза)")
    print()

    rows: list[dict] = []
    for v in clips:
        rows += events(v)
    print(f"  сильных событий с названным направлением: {len(rows)}")
    print()

    # ---- J35 first, as a control ----------------------------------------------------------
    print("─── КОНТРОЛЬ: J35 ───")
    lt6 = LT.LookTurn("VID00006")
    g_t, g_cam = gated_camera("VID00006")
    jw = lt6.window_for(J35_T)
    j_gated = ratio_on(g_cam, g_t, jw[0], jw[1]) if jw else None
    j_dead = lt6.score(J35_T)
    print(f"  окно пары {jw[0]:.2f}..{jw[1]:.2f}" if jw else "  окна нет")
    if j_dead.get("score") is None:
        # The control was written against the yaw chain as it stood on 20 September. VID00006's yaw
        # is now built from the camera's 1228-second recording instead of the old 250-second clip,
        # so this swing is no longer where it was and the control cannot be evaluated at all.
        # Saying so is the point: a control that quietly disappears is worse than no control,
        # because the run then looks verified when nothing was verified.
        print(f"  A на deadband: контроль невычислим — {j_dead.get('why', 'причина не указана')}")
        print("  Причина: yaw VID00006 с 24.09 строится по камерной записи на 1228 с, а контроль")
        print("  задан по прежней записи на 250 с. Свинга в этой точке больше нет.")
        print("  Прогон помечен как НЕ ПРОВЕРЕННЫЙ контролем, а не как прошедший его.")
        j35_ok = None
    else:
        print(f"  A на deadband: score {j_dead['score']:.4f} → {j_dead['class']}")
        print(f"  B на gated:    score {j_gated:.4f} → "
              f"{'LOOK' if j_gated < LT.LOOK_TURN_THRESHOLD else 'TURN'}")
        j35_ok = j_gated is not None and j_gated < LT.LOOK_TURN_THRESHOLD
        if not j35_ok:
            print("  СТОП: J35 перестал быть взглядом — это ошибка реализации, не результат")
            return 1
        print("  контроль пройден: J35 остаётся взглядом")
    print()

    # ---- the confusion, both ways ---------------------------------------------------------
    A = confusion(rows, "class_deadband")
    B = confusion(rows, "class_gated")
    print("─── СРАВНЕНИЕ: A (deadband) ПРОТИВ B (gated) ───")
    print("  СРАВНЕНИЕ ИДЁТ С АВТОМАТИЧЕСКОЙ РАЗМЕТКОЙ P07.4 (proxy), А НЕ С РАЗМЕТКОЙ")
    print("  ЧЕЛОВЕКА. Настоящей разметки LOOK/TURN в проекте не существует.")
    print()
    print(f"  {'':<26}{'A: deadband':>14}{'B: gated':>12}")
    print(f"  {'правда LOOK → LOOK':<26}{A['cells']['LOOK->LOOK']:>14}"
          f"{B['cells']['LOOK->LOOK']:>12}")
    print(f"  {'правда LOOK → TURN':<26}{A['cells']['LOOK->TURN']:>14}"
          f"{B['cells']['LOOK->TURN']:>12}")
    print(f"  {'правда LOOK → UNKNOWN':<26}{A['cells']['LOOK->UNKNOWN']:>14}"
          f"{B['cells']['LOOK->UNKNOWN']:>12}")
    print(f"  {'правда TURN → TURN':<26}{A['cells']['TURN->TURN']:>14}"
          f"{B['cells']['TURN->TURN']:>12}")
    print(f"  {'правда TURN → LOOK':<26}{A['cells']['TURN->LOOK']:>14}"
          f"{B['cells']['TURN->LOOK']:>12}")
    print(f"  {'правда TURN → UNKNOWN':<26}{A['cells']['TURN->UNKNOWN']:>14}"
          f"{B['cells']['TURN->UNKNOWN']:>12}")
    print()
    print(f"  {'взглядов снято':<26}{A['look_revoked_rate']:>13.0%}{B['look_revoked_rate']:>12.0%}")
    print(f"  {'поворотов сохранено':<26}{A['turn_kept_rate']:>13.0%}{B['turn_kept_rate']:>12.0%}")
    print(f"  {'поворотов снято ОШИБОЧНО':<26}{A['turn_wrongly_revoked_rate']:>13.0%}"
          f"{B['turn_wrongly_revoked_rate']:>12.0%}")
    print()

    # ---- the structural point, before the pair tables --------------------------------------
    # Worth measuring rather than assuming: is "has a pair" a different variable from "the proxy
    # label"? If it is not, then agreement with the proxy is largely agreement with the pair
    # structure, and that changes what every number below is really testing.
    from collections import Counter
    struct = {}
    print("─── СТРУКТУРА: МЕТКА ПРОТИВ СТАТУСА ПАРЫ ───")
    print(f"  {'клип':<11}{'LOOK пара':>11}{'LOOK без':>10}{'TURN пара':>11}{'TURN без':>10}")
    for v in clips:
        c = Counter((sw["kind"], sw["paired_with"] is not None)
                    for sw in LT.LookTurn(v).swing if sw["kind"] in ("LOOK", "TURN"))
        row = {"clip": v,
               "look_pair": c.get(("LOOK", True), 0), "look_nopair": c.get(("LOOK", False), 0),
               "turn_pair": c.get(("TURN", True), 0), "turn_nopair": c.get(("TURN", False), 0)}
        struct[v] = row
        print(f"  {v:<11}{row['look_pair']:>11}{row['look_nopair']:>10}"
              f"{row['turn_pair']:>11}{row['turn_nopair']:>10}")
    tot_lp = sum(r["look_pair"] for r in struct.values())
    tot_ln = sum(r["look_nopair"] for r in struct.values())
    tot_tp = sum(r["turn_pair"] for r in struct.values())
    tot_tn = sum(r["turn_nopair"] for r in struct.values())
    print()
    print(f"  всего: LOOK с парой {tot_lp}, LOOK без пары {tot_ln}; "
          f"TURN с парой {tot_tp}, TURN без пары {tot_tn}")
    print("  ВЗГЛЯД ПО ОПРЕДЕЛЕНИЮ ПАРНЫЙ: P07.4 называет взглядом именно пару, которая")
    print("  гасится. Поэтому метка proxy и статус пары — почти одна и та же переменная, а")
    print("  не два независимых измерения. Значит согласие с proxy есть в большой мере")
    print("  воспроизведение структуры пары, и мерить это как точность нельзя.")
    print()

    # ---- pair status, kept separate -------------------------------------------------------
    print("─── PAIR_FOUND ПРОТИВ PAIR_NOT_FOUND (их нельзя смешивать) ───")
    pf = [r for r in rows if r["pair_status"] == "PAIR_FOUND"]
    pn = [r for r in rows if r["pair_status"] == "PAIR_NOT_FOUND"]
    print(f"  с найденной парой: {len(pf)}   без пары: {len(pn)}")
    print()
    print("  В случаях без пары возврат НЕ ИЗМЕРЯЛСЯ: P07.4 метит одиночное большое")
    print("  колебание поворотом по размеру. Отношение там не мера гашения, и считать")
    print("  его ошибкой фильтра нельзя.")
    print()
    print(f"  {'подмножество':<18}{'событий':>9}{'proxy LOOK':>12}{'proxy TURN':>11}"
          f"{'B: снято TURN':>15}{'B: снято LOOK':>15}")
    for tag, sel in (("PAIR_FOUND", pf), ("PAIR_NOT_FOUND", pn)):
        c = confusion(sel, "class_gated")
        tu = _pct(c["turn_wrongly_revoked_rate"])
        lo = _pct(c["look_revoked_rate"])
        print(f"  {tag:<18}{len(sel):>9}{c['n_look']:>12}{c['n_turn']:>11}{tu:>15}{lo:>15}")
    print()

    # ---- per clip -------------------------------------------------------------------------
    print("─── ПО КЛИПАМ, ОБА ВАРИАНТА ───")
    print(f"  {'клип':<11}{'сильных':>8}{'LOOK снято':>12}{'TURN сохр.':>12}"
          f"{'TURN снято':>12}{'':>4}│{'LOOK снято':>12}{'TURN сохр.':>12}{'TURN снято':>12}")
    print(f"  {'':<11}{'':>8}{'——— A: deadband':>36}{'':>4}│{'——— B: gated':>36}")
    per = []
    for v in clips:
        sel = [r for r in rows if r["clip"] == v]
        a, b = confusion(sel, "class_deadband"), confusion(sel, "class_gated")
        per.append({"clip": v, "n_strong": len(sel), "A": a, "B": b})
        f = lambda x: "—" if x is None else f"{x:.0%}"
        print(f"  {v:<11}{len(sel):>8}{f(a['look_revoked_rate']):>12}"
              f"{f(a['turn_kept_rate']):>12}{f(a['turn_wrongly_revoked_rate']):>12}{'':>4}│"
              f"{f(b['look_revoked_rate']):>12}{f(b['turn_kept_rate']):>12}"
              f"{f(b['turn_wrongly_revoked_rate']):>12}")
    print()
    pc = lambda x: "—" if x is None else f"{x:.0%}"       # noqa: E731
    for tag, key in (("A: deadband", "A"), ("B: gated", "B")):
        ho = [r for r in rows if r["clip"] in HELD_OUT]
        c = confusion(ho, "class_deadband" if key == "A" else "class_gated")
        print(f"  {tag}, только VID00005+VID00009 (не участвовали в построении): "
              f"снято поворотов {pc(c['turn_wrongly_revoked_rate'])}, "
              f"взглядов снято {pc(c['look_revoked_rate'])}")
    print()

    # ---- outside oscillations -------------------------------------------------------------
    print("─── СИЛЬНЫЕ СОБЫТИЯ ВНЕ РАЗМЕЧЕННЫХ КОЛЕБАНИЙ (отдельная проблема) ───")
    print("  здесь P09.2 вернёт UNKNOWN и снимет направление по политике.")
    print("  В P09.4 это НЕ исправляется: у события нет колебания, значит нет и окна.")
    print()
    outside = []
    for v in clips:
        r_t, r_y = LT.load_ratio_signal(v)
        dt = float(np.median(np.diff(r_t)))
        w = max(int(round(WINDOW_S / dt)), 1)
        c = np.cumsum(np.concatenate([[0.0], r_y * dt]))
        I = c[w:] - c[:-w]
        idx = np.where(np.abs(I) >= floor_for())[0]
        if len(idx) == 0:
            outside.append({"clip": v, "strong_regions": 0, "inside": 0, "outside": 0})
            continue
        regs, st, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > w:
                regs.append((st, prev))
                st = i
            prev = i
        regs.append((st, prev))
        lab = [(s["t0"], s["t1"]) for s in LT.LookTurn(v).swing
               if s["kind"] in ("LOOK", "TURN")]
        ins = sum(1 for a, b in regs
                  if any(not (r_t[b] < x0 or r_t[a] > x1) for x0, x1 in lab))
        outside.append({"clip": v, "strong_regions": len(regs), "inside": ins,
                        "outside": len(regs) - ins,
                        "regions": [[round(float(r_t[a]), 3), round(float(r_t[b]), 3)]
                                    for a, b in regs
                                    if not any(not (r_t[b] < x0 or r_t[a] > x1)
                                               for x0, x1 in lab)]})
        print(f"  {v:<11} сильных участков {len(regs):>4}, внутри колебаний {ins:>4}, "
              f"ВНЕ {len(regs)-ins:>4}")
    tot_out = sum(o["outside"] for o in outside)
    tot_reg = sum(o["strong_regions"] for o in outside)
    print()
    print(f"  всего вне колебаний: {tot_out} из {tot_reg} ({tot_out/max(tot_reg,1):.0%})")
    print()

    # ---- verdict --------------------------------------------------------------------------
    print("─── ГЛАВНАЯ ПРОВЕРКА ───")
    print(f"  гипотеза P09.3: если дело было только в рассогласовании сигналов,")
    print(f"  доля ошибочно снятых поворотов должна упасть с 32% до примерно 8–12%.")
    print()
    print(f"  A: {_pct(A['turn_wrongly_revoked_rate'],1)}  →  B: {_pct(B['turn_wrongly_revoked_rate'],1)}")
    print(f"  на подмножестве PAIR_FOUND (там, где возврат измерялся): "
          f"{_pct(confusion(pf, 'class_gated')['turn_wrongly_revoked_rate'],1)}")
    print()
    print("  Это НЕ критерий выбора: yaw_gated выбран потому, что на нём построена пара.")
    print()
    accA = (A["cells"]["LOOK->LOOK"] + A["cells"]["TURN->TURN"]) / max(A["n_look"] + A["n_turn"], 1)
    accB = (B["cells"]["LOOK->LOOK"] + B["cells"]["TURN->TURN"]) / max(B["n_look"] + B["n_turn"], 1)
    print(f"  ВАЖНО: общая точность при этом НЕ выросла: A {accA:.0%}, B {accB:.0%}.")
    print(f"  Вариант B не уменьшает число ошибок, а ПЕРЕРАСПРЕДЕЛЯЕТ их: он перестаёт")
    print(f"  трогать настоящие повороты ({_pct(A['turn_wrongly_revoked_rate'])} → "
          f"{_pct(B['turn_wrongly_revoked_rate'])}) ценой того, что ловит меньше взглядов")
    print(f"  ({_pct(A['look_revoked_rate'])} → {_pct(B['look_revoked_rate'])}). Это смена того, "
          f"чем платить, а не выигрыш.")
    print(f"  Обоснование выбора — согласованность сигналов, и только она.")
    print()
    print("  ЧЕГО ЭТА ФАЗА НЕ ДОКАЗЫВАЕТ: точность LOOK/TURN. Разметка остаётся")
    print("  автоматической (P07.4). Результат ограничивает ущерб, а не подтверждает")
    print("  правильность. Нужна первая настоящая ручная разметка именно этого различия.")
    print()

    # ---- files ----------------------------------------------------------------------------
    for name, sel in (("pair_found.csv", pf), ("pair_not_found.csv", pn)):
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=["clip", "t", "t0", "t1", "yaw_integral",
                                               "yaw_direction", "pair_status", "paired_with",
                                               "score_deadband", "class_deadband",
                                               "score_gated", "class_gated", "truth",
                                               "truth_source"], extrasaction="ignore")
            w_.writeheader()
            for r in sel:
                w_.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items() if k in w_.fieldnames})
    with (out / "comparison_deadband_vs_gated.csv").open("w", encoding="utf-8",
                                                         newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["scope", "variant", "n_strong", "truth_LOOK", "truth_TURN",
                     "LOOK->LOOK", "LOOK->TURN", "LOOK->UNKNOWN", "TURN->TURN", "TURN->LOOK",
                     "TURN->UNKNOWN", "look_revoked_rate", "turn_kept_rate",
                     "turn_wrongly_revoked_rate"])
        for tag, sel in (("ALL", rows), ("PAIR_FOUND", pf), ("PAIR_NOT_FOUND", pn),
                         ("HELD_OUT", [r for r in rows if r["clip"] in HELD_OUT])):
            for variant, key in (("A_deadband", "class_deadband"), ("B_gated", "class_gated")):
                c = confusion(sel, key)
                w_.writerow([tag, variant, len(sel), c["n_look"], c["n_turn"],
                             c["cells"]["LOOK->LOOK"], c["cells"]["LOOK->TURN"],
                             c["cells"]["LOOK->UNKNOWN"], c["cells"]["TURN->TURN"],
                             c["cells"]["TURN->LOOK"], c["cells"]["TURN->UNKNOWN"],
                             c["look_revoked_rate"], c["turn_kept_rate"],
                             c["turn_wrongly_revoked_rate"]])
        for p in per:
            for variant, key in (("A_deadband", "A"), ("B_gated", "B")):
                c = p[key]
                w_.writerow([p["clip"], variant, p["n_strong"], c["n_look"], c["n_turn"],
                             c["cells"]["LOOK->LOOK"], c["cells"]["LOOK->TURN"],
                             c["cells"]["LOOK->UNKNOWN"], c["cells"]["TURN->TURN"],
                             c["cells"]["TURN->LOOK"], c["cells"]["TURN->UNKNOWN"],
                             c["look_revoked_rate"], c["turn_kept_rate"],
                             c["turn_wrongly_revoked_rate"]])
    with (out / "outside_oscillations.csv").open("w", encoding="utf-8", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["clip", "region_start", "region_end"])
        for o in outside:
            for a, b in o.get("regions", []):
                w_.writerow([o["clip"], a, b])
    (out / "report.json").write_text(json.dumps({
        "phase": "P09.4 — согласование сигнала отношения с сигналом пары",
        "change": "отношение считается на yaw_gated вместо yaw_signal_deadband",
        "unchanged": ["границы пары", "алгоритм поиска пары", "порог 0.610",
                      "правило сильного события", "политика UNKNOWN", "пять клипов",
                      "идея P09.2", "подбор порога не делался"],
        "signals": {"direction_and_strength": "yaw_signal_deadband",
                    "swing_structure_and_score": "yaw_gated"},
        "truth": {"source": "P07.4_PROXY", "human_look_turn_labels": 0,
                  "note": "настоящей разметки LOOK/TURN не существует; всё ниже — согласие "
                          "с автоматической разметкой, а не точность"},
        "j35": {"window": list(jw) if jw else None, "score_deadband": j_dead["score"],
                "class_deadband": j_dead["class"], "score_gated": j_gated,
                "class_gated": "LOOK" if j35_ok else "TURN", "control_passed": bool(j35_ok)},
        "overall": {"A_deadband": A, "B_gated": B},
        "structure": struct,
        "structure_totals": {"look_pair": tot_lp, "look_nopair": tot_ln,
                             "turn_pair": tot_tp, "turn_nopair": tot_tn,
                             "note": "взгляд у P07.4 по определению парный, поэтому метка "
                                     "proxy и статус пары почти совпадают; согласие с proxy "
                                     "нельзя читать как точность"},
        "pair_found": confusion(pf, "class_gated"),
        "pair_not_found": confusion(pn, "class_gated"),
        "n_pair_found": len(pf), "n_pair_not_found": len(pn),
        "per_clip": per,
        "held_out": {tag: confusion([r for r in rows if r["clip"] in HELD_OUT],
                                    "class_deadband" if tag == "A" else "class_gated")
                     for tag in ("A", "B")},
        "outside_oscillations": outside,
        "n_outside": tot_out, "n_strong_regions": tot_reg,
        "error_balance": {
            "accuracy_A": (A["cells"]["LOOK->LOOK"] + A["cells"]["TURN->TURN"])
                          / max(A["n_look"] + A["n_turn"], 1),
            "accuracy_B": (B["cells"]["LOOK->LOOK"] + B["cells"]["TURN->TURN"])
                          / max(B["n_look"] + B["n_turn"], 1),
            "note": "B не уменьшает число ошибок, а перераспределяет их: перестаёт трогать "
                    "настоящие повороты ценой меньшего числа пойманных взглядов",
        },
        "hypothesis": {"expected": "падение с 32% до 8-12%",
                       "observed": B["turn_wrongly_revoked_rate"],
                       "is_not_a_criterion": True},
        "next_step": (
            "не замораживать окончательно. Первая настоящая ручная разметка именно "
            "LOOK/TURN: случайная выборка примерно 20 LOOK, 20 TURN, 20 UNKNOWN из разных "
            "клипов, особенно VID00005 и VID00009, короткими клипами с ответом "
            "LOOK / TURN / UNCLEAR."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'comparison_deadband_vs_gated.csv'}")
    print(f"записано: {out/'pair_found.csv'} ({len(pf)})")
    print(f"записано: {out/'pair_not_found.csv'} ({len(pn)})")
    print(f"записано: {out/'outside_oscillations.csv'} ({tot_out})")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
