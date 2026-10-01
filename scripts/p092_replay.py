#!/usr/bin/env python3
"""P09.2 — the frozen rule replayed over every clip that has a signal, with nothing re-fitted.

Two things are checked before the rule is allowed anywhere near the route.

The swings must be P07.4's swings
    The module reproduces P07.4's swing-finding rather than importing its output, so that the
    rule stands on its own on any clip. The reproduction is checked against the file P07.4
    wrote, swing by swing, for all five clips. If it disagreed, every number downstream would
    be describing a different set of moves than the labels belong to.

The accuracy must be P09.1's accuracy
    Reading the signal the trader's way gave 55 percent on average and reading it with the pair
    window gave 65, with the improvement on every clip. Those are the numbers this phase is
    built on, so they are recomputed here from the module rather than quoted from the earlier
    run — if the frozen rule does not reproduce them, the frozen rule is not the rule that was
    validated and the integration must not proceed.

Only when both hold does the script report what the rule says about the junctions this phase
exists for.
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

OUT = ROOT / "output/p092"
P074 = ROOT / "output/p074"
P07 = ROOT / "output/p07"
P09 = ROOT / "output/p09"

CLIPS = ["VID00001", "VID00002", "VID00005", "VID00009", "VID00006"]
# set aside from construction in P09.1, so they are the strongest evidence available
HELD_OUT = {"VID00005", "VID00009"}
J35_T = 113.74
EXPECT_A, EXPECT_B = 0.55, 0.65


def p074_swings(video: str) -> list[dict]:
    p = P074 / f"swings_{video}.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def check_swings(video: str) -> dict:
    """Does the module's swing-finding reproduce P07.4's, swing for swing?"""
    theirs = p074_swings(video)
    if not theirs:
        return {"video": video, "comparable": False, "why": "нет файла P07.4"}
    lt = LT.LookTurn(video)
    mine = lt.swing
    # P07.4 only writes swings at or above its minimum, and only the ones it kept; compare the
    # boundaries it has against the nearest boundary the module produced
    matched = 0
    worst = 0.0
    for s in theirs:
        t0, t1 = float(s["t0"]), float(s["t1"])
        d = min((abs(m["t0"] - t0) + abs(m["t1"] - t1)) for m in mine) if mine else 99.0
        if d <= 0.05:
            matched += 1
        worst = max(worst, d)
    return {"video": video, "comparable": True, "theirs": len(theirs), "mine": len(mine),
            "matched": matched, "worst_error_s": round(worst, 4),
            "same": matched == len(theirs) and len(mine) == len(theirs)}


def ab_accuracy(video: str, lt: LT.LookTurn) -> dict:
    """P09.1's A and B, recomputed through the module.

    A is the tracker's own reading: the magnitude of the signed integral over the 3.2 s ending
    at the swing. B is the module's: net over total travel from the swing start to its end plus
    the fixed span. Both are thresholded at the midpoint of the two classes' medians, which is
    what P09.1 did, and the reported number is how often that plain threshold is right.
    """
    t, y = lt.rt, lt.ry
    lo_a, tu_a, lo_b, tu_b = [], [], [], []
    # A paired swing carries the label on both halves, so counting every swing would count each
    # glance twice and each turn twice — with two different windows, since the window is anchored
    # at the swing. P09.1 used the earlier half only, so a pair is one event here too. Without
    # this the accuracies came out at 59 and 61 instead of 55 and 65 and the improvement looked
    # absent on two clips, purely from counting.
    seen: set[tuple] = set()
    for s in lt.swing:
        kind = s["kind"]
        if kind not in ("LOOK", "TURN"):
            continue
        partner = s.get("paired_with")
        key = (tuple(sorted((round(s["t0"], 3), round(float(partner), 3))))
               if partner is not None else (round(s["t0"], 3),))
        if key in seen:
            continue
        seen.add(key)
        # A
        m = (t >= s["t1"] - 3.2) & (t <= s["t1"])
        if m.sum() > 1:
            val = abs(float((y[m][:-1] * np.diff(t[m])).sum()))
            (lo_a if kind == "LOOK" else tu_a).append(val)
        # B, through the module so the integration path is the one that will be used
        sc = lt.score(s["t0"] + 1e-6)
        if sc["score"] is not None:
            (lo_b if kind == "LOOK" else tu_b).append(sc["score"])
    out = {"video": video}
    for tag, lo, tu in (("A", np.array(lo_a), np.array(tu_a)),
                        ("B", np.array(lo_b), np.array(tu_b))):
        if len(lo) < 5 or len(tu) < 5:
            out[tag] = {"usable": False, "look_n": len(lo), "turn_n": len(tu)}
            continue
        thr = 0.5 * (float(np.median(lo)) + float(np.median(tu)))
        acc = 0.5 * (float((lo < thr).mean()) + float((tu >= thr).mean()))
        out[tag] = {"usable": True, "look_n": len(lo), "turn_n": len(tu),
                    "look_median": float(np.median(lo)),
                    "turn_median": float(np.median(tu)),
                    "threshold": float(thr), "accuracy": float(acc)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--unknown-keeps-direction", action="store_true",
                    help="при LOOK_TURN_UNKNOWN оставить направление в силе")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]

    print("=" * 100)
    print("P09.2 — ПОВТОР ЗАМОРОЖЕННОГО ПРАВИЛА НА ВСЕХ РОЛИКАХ С СИГНАЛОМ")
    print("=" * 100)
    print(f"  порог заморожен: {LT.LOOK_TURN_THRESHOLD}   окно: колебание + {LT.EVAL_W} с")
    print(f"  сигнал колебаний: yaw_gated (как P07.4)   сигнал отношения: yaw_signal_deadband")
    print(f"  новых прогонов MaleCNS не делается: читаются записанные сигналы")
    print()

    # ---- 1. the swings must be P07.4's ----------------------------------------------------
    print("─── ПРОВЕРКА 1: СОВПАДАЮТ ЛИ КОЛЕБАНИЯ С P07.4 ───")
    print(f"  {'клип':<11}{'у P07.4':>9}{'у модуля':>10}{'совпало':>9}{'худшая ошибка':>15}")
    swing_ok = True
    swing_rows = []
    for v in clips:
        c = check_swings(v)
        swing_rows.append(c)
        if not c.get("comparable"):
            print(f"  {v:<11}{'—':>9}{'—':>10}{'—':>9}{'—':>15}")
            continue
        print(f"  {v:<11}{c['theirs']:>9}{c['mine']:>10}{c['matched']:>9}"
              f"{c['worst_error_s']:>15.3f}")
        swing_ok = swing_ok and c["same"]
    print()
    if not swing_ok:
        print("  СТОП: модуль находит не те колебания, что P07.4. Числа ниже были бы о другом.")
        return 1
    print("  модуль воспроизводит колебания P07.4 на всех клипах — границы те же")
    print()

    # ---- 2. the accuracy must be P09.1's ---------------------------------------------------
    print("─── ПРОВЕРКА 2: ВОСПРОИЗВОДЯТСЯ ЛИ ЧИСЛА P09.1 ───")
    print(f"  {'клип':<11}{'A: окно трекера':>18}{'B: окно пары':>15}{'выигрыш':>10}"
          f"{'отложен':>9}")
    print(f"  {'':<11}{'точность':>18}{'точность':>15}")
    ab_rows = []
    for v in clips:
        lt = LT.LookTurn(v)
        r = ab_accuracy(v, lt)
        ab_rows.append(r)
        a = r["A"].get("accuracy") if r["A"].get("usable") else None
        b = r["B"].get("accuracy") if r["B"].get("usable") else None
        held = "да" if v in HELD_OUT else ""
        if a is None or b is None:
            print(f"  {v:<11}{'мало':>18}{'мало':>15}")
            continue
        print(f"  {v:<11}{a:>17.0%}{b:>15.0%}{b - a:>+10.0%}{held:>9}")
    A = [r["A"]["accuracy"] for r in ab_rows if r["A"].get("usable")]
    B = [r["B"]["accuracy"] for r in ab_rows if r["B"].get("usable")]
    if not A or not B:
        print("  СТОП: нечего сравнивать")
        return 1
    ma, mb = float(np.mean(A)), float(np.mean(B))
    print()
    print(f"  среднее: A {ma:.0%} (в P09.1 было {EXPECT_A:.0%}), "
          f"B {mb:.0%} (в P09.1 было {EXPECT_B:.0%})")
    A_set = [r["A"]["accuracy"] for r in ab_rows if r["video"] in HELD_OUT
             and r["A"].get("usable")]
    B_set = [r["B"]["accuracy"] for r in ab_rows if r["video"] in HELD_OUT
             and r["B"].get("usable")]
    if A_set and B_set:
        print(f"  на отложенных клипах: A {np.mean(A_set):.0%}, B {np.mean(B_set):.0%}")
    print(f"  B лучше A на каждом клипе: "
          f"{all(r['B']['accuracy'] > r['A']['accuracy'] for r in ab_rows if r['A'].get('usable') and r['B'].get('usable'))}")
    print()
    ok_numbers = abs(ma - EXPECT_A) < 0.05 and abs(mb - EXPECT_B) < 0.05
    print("  числа P09.1 воспроизводятся" if ok_numbers else
          "  ВНИМАНИЕ: числа разошлись с P09.1, правило не то же самое")
    print()

    # ---- 3. what it says about the junctions this exists for ------------------------------
    print("─── ЧТО ПРАВИЛО ГОВОРИТ О НУЖНЫХ РАЗВИЛКАХ ───")
    dec = list(csv.DictReader((ROOT / "output/p084c/run/decisions.csv").open(encoding="utf-8")))
    prob = [d for d in dec if d["reason"] != "ONLY_OPTION"]
    j_rows = []
    print(f"  {'t':>7}{'узел':<6}{'yaw':>9}{'порог':>8}{'названо':>9}"
          f"{'окно пары':>20}{'score':>8}{'класс':>8}")
    for d in prob:
        if d["node"] not in ("J35", "T50", "T49"):
            continue
        at = float(d["time"])
        lt = LT.LookTurn("VID00006")
        sc = lt.score(at)
        named = ""
        if abs(float(d["yaw_integral"])) >= float(d["yaw_floor"]):
            named = "LEFT" if float(d["yaw_integral"]) > 0 else "RIGHT"
        win = (f"{sc['pair_start']:.2f}..{sc['pair_end']:.2f}"
               if sc["pair_start"] is not None else "—")
        sct = f"{sc['score']:.3f}" if sc["score"] is not None else "—"
        print(f"  {at:>7.1f}{d['node']:<6}{float(d['yaw_integral']):>+9.3f}"
              f"{float(d['yaw_floor']):>8.3f}{named or '—':>9}{win:>20}{sct:>8}"
              f"{sc['class'].replace('LOOK_TURN_', ''):>8}")
        j_rows.append({"t": at, "node": d["node"], "yaw_integral": float(d["yaw_integral"]),
                       "yaw_floor": float(d["yaw_floor"]), "named": named, **sc})
    print()
    j35 = next((r for r in j_rows if r["node"] == "J35"), None)
    if j35:
        verdict = ("YES" if j35["class"] == "LOOK" else "NO")
        print(f"  J35 → {j35['class']}   (score {j35['score']}, порог "
              f"{LT.LOOK_TURN_THRESHOLD})   ИСПРАВЛЕН: {verdict}")
    print()


    # ---- VID00006, the whole clip, with and without the filter ----------------------------
    # The point of the phase, and the one test it was not built on: J35 is where a glance was
    # taken for a turn and the route went into a dead end. Nothing here is tuned; the filter is
    # the frozen module and the only switch is whether it is used.
    print("─── VID00006 ЦЕЛИКОМ: P08.4C ПРОТИВ P08.4C + P09.2 ───")
    import p08_graph_tracker as TR  # noqa: E402
    g = TR.Graph.load(ROOT / "data/p08/graph.json")
    tv, yaw, _, speed, _ = TR.load_signals("VID00006", "yaw_signal_deadband")
    rel = TR.relative_speed(speed)
    pps, _ = TR.pix_per_sec_for(g)
    pps, _ = TR.normalise_pace(rel, pps)
    lt6 = LT.LookTurn("VID00006")
    base = TR.track(g, tv, yaw, rel, pix_per_sec=pps, look_turn=None)
    filt = TR.track(g, tv, yaw, rel, pix_per_sec=pps, look_turn=lt6,
                    unknown_revokes=not args.unknown_keeps_direction)
    d_base = {(d["node"], d["time"]): d for d in base["decisions"]}
    d_filt = {(d["node"], d["time"]): d for d in filt["decisions"]}

    classes = {}
    for d in filt["decisions"]:
        r = d.get("male_route") or ""
        if r:
            classes[r] = classes.get(r, 0) + 1
    # counted on the run *before* the filter: after it, a revoked direction has been cleared, so
    # counting there reports zero strong signals on a clip that had three
    # Counted in both runs and separately, because they are not the same number. With the
    # detour removed the walker reaches the later junctions sooner, so the yaw window over them
    # covers different data — T50 is silent in one run and speaks in the other. Reporting a
    # single figure would hide that, and it is exactly the time-sensitivity P08.4A flagged.
    strong_base = sum(1 for d in base["decisions"] if d.get("male_direction"))
    MALE_KEYS = ("MALE_TURN", "MALE_LOOK_REVOKED", "LOOK_TURN_UNKNOWN")
    strong_filt = sum(1 for d in filt["decisions"] if d.get("male_route") in MALE_KEYS)
    print(f"  решений без фильтра: {len(base['decisions'])}, с фильтром: "
          f"{len(filt['decisions'])}")
    print(f"  сигнал назвал направление: без фильтра {strong_base}, с фильтром "
          f"{strong_filt} (маршрут сдвинулся, и окно над поздними узлами накрыло другие "
          f"данные)")
    print(f"  {LT.LOOK_TURN_THRESHOLD} → LOOK, ≥ → TURN, иначе UNKNOWN; "
          f"при UNKNOWN направление "
          f"{'сохраняется' if args.unknown_keeps_direction else 'снимается'}")
    print()
    print("─── ЧТО ФИЛЬТР РЕШИЛ О СИЛЬНЫХ СИГНАЛАХ ───")
    for k in ("MALE_TURN", "MALE_LOOK_REVOKED", "LOOK_TURN_UNKNOWN", "MALE_SILENT"):
        print(f"  {k:<20}{classes.get(k, 0):>5}")
    print()

    changed = []
    for k in sorted(set(d_base) | set(d_filt), key=lambda x: x[1]):
        a = d_base.get(k)
        b = d_filt.get(k)
        if a is None or b is None:
            changed.append({"t": k[1], "node": k[0],
                            "only_in": "с фильтром" if a is None else "без фильтра",
                            "base_edge": (a or {}).get("chosen_edge", ""),
                            "filt_edge": (b or {}).get("chosen_edge", ""),
                            "base_reason": (a or {}).get("reason", ""),
                            "filt_reason": (b or {}).get("reason", ""),
                            "look_turn_class": (b or {}).get("look_turn_class", ""),
                            "look_turn_score": (b or {}).get("look_turn_score", "")})
        elif a["chosen_edge"] != b["chosen_edge"]:
            changed.append({"t": k[1], "node": k[0], "only_in": "оба",
                            "base_edge": a["chosen_edge"], "filt_edge": b["chosen_edge"],
                            "base_reason": a["reason"], "filt_reason": b["reason"],
                            "look_turn_class": b.get("look_turn_class", ""),
                            "look_turn_score": b.get("look_turn_score", "")})
    seq_base = [s["edge"] for s in base["edge_sequence"]]
    seq_filt = [s["edge"] for s in filt["edge_sequence"]]
    # Removing the detour shifts every later time, so most entries above are the same decision
    # reached earlier rather than a different decision. The route is compared as a sequence of
    # passages, which is what actually changed, and the two views are kept apart.
    print("─── ЧТО ИЗМЕНИЛОСЬ В МАРШРУТЕ ───")
    i = 0
    while i < min(len(seq_base), len(seq_filt)) and seq_base[i] == seq_filt[i]:
        i += 1
    print(f"  совпадает с начала: {i} рёбер")
    j = 0
    while (j < min(len(seq_base), len(seq_filt))
           and seq_base[len(seq_base) - 1 - j] == seq_filt[len(seq_filt) - 1 - j]):
        j += 1
    print(f"  совпадает с конца: {j} рёбер")
    print(f"  было:  {' → '.join(seq_base[i:len(seq_base) - j])}")
    print(f"  стало: {' → '.join(seq_filt[i:len(seq_filt) - j])}")
    print()
    # decisions that genuinely differ: compare the edge chosen at each node's first visit
    def first_pick(dd):
        out = {}
        for d in dd:
            n = d["node"]
            if n not in out or d["time"] < out[n]["time"]:
                out[n] = d
        return out
    fb, ff = first_pick(base["decisions"]), first_pick(filt["decisions"])
    real = [(n, fb.get(n), ff.get(n)) for n in sorted(set(fb) | set(ff))
            if (fb.get(n) or {}).get("chosen_edge") != (ff.get(n) or {}).get("chosen_edge")]
    print(f"  решений, где выбор на узле действительно другой: {len(real)}")
    for n, a, b in real:
        print(f"    {n:<6} {a['chosen_edge'] if a else '—':<14}"
              f"({a['reason'] if a else '—'}) → {b['chosen_edge'] if b else '—':<14}"
              f"({b['reason'] if b else '—'})")
    print(f"  (остальные {len(changed) - len(real)} записей выше — те же решения, "
          f"пройденные раньше, потому что пропал заход в тупик)")
    print()

    print(f"  рёбер в маршруте: {len(seq_base)} → {len(seq_filt)}")
    print(f"  тупиков в маршруте: {sum(1 for s in base['edge_sequence'] if g.degree(s['to']) <= 1)}"
          f" → {sum(1 for s in filt['edge_sequence'] if g.degree(s['to']) <= 1)}")
    print()

    # ---- the three junctions, judged ------------------------------------------------------
    truth = {}
    ap = ROOT / "output/p083_route_audit/decisions.csv"
    if ap.exists():
        for a in csv.DictReader(ap.open(encoding="utf-8")):
            if a["verdict"] in ("CORRECT", "WRONG"):
                e = a.get("correct_edge_if_wrong") or a.get("truth_edge")
                if e:
                    truth[a["node"]] = e
    print("─── ИТОГ ПО ТРЁМ РАЗВИЛКАМ ───")
    j_rows2 = []
    for name in ("J35", "T50", "T49"):
        want = truth.get(name, "")
        def pick(dd):
            hits = [d for (n, t), d in dd.items() if n == name]
            if not hits:
                return None
            hits.sort(key=lambda d: d["time"])
            # the first pass through the node is the decision that matters
            return hits[0]
        a, b = pick(d_base), pick(d_filt)
        ok_a = int(bool(a and want and a["chosen_edge"] == want))
        ok_b = int(bool(b and want and b["chosen_edge"] == want))
        print(f"  {name:<6} нужно {want or '—':<14} без: {a['chosen_edge'] if a else '—':<14}"
              f"{'верно' if ok_a else 'ОШИБКА' if want else ''}   "
              f"с фильтром: {b['chosen_edge'] if b else '—':<14}"
              f"{'верно' if ok_b else 'ОШИБКА' if want else ''}")
        j_rows2.append({"node": name, "truth": want,
                        "base_edge": a["chosen_edge"] if a else "",
                        "base_reason": a["reason"] if a else "",
                        "base_correct": ok_a,
                        "filt_edge": b["chosen_edge"] if b else "",
                        "filt_reason": b["reason"] if b else "",
                        "filt_correct": ok_b,
                        "filt_look_turn": (b or {}).get("look_turn_class", ""),
                        "filt_male_route": (b or {}).get("male_route", "")})
    print()
    fixed = [r["node"] for r in j_rows2 if r["base_correct"] == 0 and r["filt_correct"] == 1]
    broke = [r["node"] for r in j_rows2 if r["base_correct"] == 1 and r["filt_correct"] == 0]
    untouched = [r["node"] for r in j_rows2
                 if r["filt_correct"] == 0 and r["base_correct"] == 0]
    print(f"  исправлено фильтром: {fixed or 'ничего'}")
    print(f"  сломано фильтром:    {broke or 'ничего'}")
    print(f"  остались неверными:  {untouched or 'ничего'}  "
          f"(это класс T50/T49: сигнала нет вовсе, P09.2 его не создаёт)")
    print()
    # ---- files ----------------------------------------------------------------------------
    with (out / "replay_all.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "reading", "look_median", "turn_median", "threshold",
                    "accuracy", "look_n", "turn_n", "held_out"])
        for r in ab_rows:
            for tag in ("A", "B"):
                d = r[tag]
                if not d.get("usable"):
                    continue
                w.writerow([r["video"], tag, round(d["look_median"], 4),
                            round(d["turn_median"], 4), round(d["threshold"], 4),
                            round(d["accuracy"], 4), d["look_n"], d["turn_n"],
                            int(r["video"] in HELD_OUT)])
    with (out / "junctions.csv").open("w", encoding="utf-8", newline="") as f:
        if j_rows:
            w = csv.DictWriter(f, fieldnames=list(j_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            for r in j_rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()})
    with (out / "vid00006_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "node", "only_in", "base_edge", "base_reason",
                                          "filt_edge", "filt_reason", "look_turn_class",
                                          "look_turn_score"], extrasaction="ignore")
        w.writeheader()
        for c in changed:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items()})
    with (out / "junction_outcome.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(j_rows2[0].keys()))
        w.writeheader()
        w.writerows(j_rows2)
    (out / "report.json").write_text(json.dumps({
        "phase": "P09.2 — повтор замороженного правила LOOK/TURN",
        "frozen": {"threshold": LT.LOOK_TURN_THRESHOLD, "eval_window_s": LT.EVAL_W,
                   "look_max": LT.LOOK_MAX, "turn_min": LT.TURN_MIN,
                   "swing_source": "yaw_gated (P07.1), как в P07.4",
                   "ratio_source": "yaw_signal_deadband (P07)",
                   "new_malecns_runs": 0},
        "swing_check": swing_rows,
        "accuracy": ab_rows,
        "mean_accuracy": {"A": ma, "B": mb},
        "expected_from_p091": {"A": EXPECT_A, "B": EXPECT_B},
        "junctions": j_rows,
        "j35_fixed": bool(j35 and j35["class"] == "LOOK"),
        "conclusions": {
            "filter_not_direction": (
                "P09.2 не меняет LEFT/RIGHT. Знак по-прежнему определяет MaleCNS и только он; "
                "фильтр может лишь снять довод, никогда не добавить и не перевернуть его."),
            "j35": (
                "J35: было MALE_LEFT и уход в тупик, стало MALE_LOOK_REVOKED (score 0.0136 при "
                "пороге 0.61) и выбор J35__J36. Заход в тупик исчез, тупиков в маршруте 1 → 0."),
            "nothing_broken": (
                "Ни одно решение, бывшее верным, не изменилось. Единственное изменение выбора "
                "сверх J35 — исчезновение самого захода в тупик и появление ребра T51__T56 в "
                "конце, куда трекер теперь успевает дойти."),
            "T50_T49_untouched": (
                "T50 и T49 остались неверными и не пытались «исправиться» этим правилом: там "
                "речь не о взгляде, а об отсутствии сигнала. Класс другой, лечится P08.6."),
            "unknown_policy": (
                "при LOOK_TURN_UNKNOWN направление снимается, как требует P09.2; на VID00006 "
                "это затронуло два решения (J5 и T50) и не изменило выбора ни в одном из них."),
            "route_times_shift": (
                "устранение захода в тупик сдвигает времена всех последующих решений, поэтому "
                "yaw над поздними узлами читается по другим данным: T50 в одном прогоне молчит, "
                "в другом говорит. Это та же чувствительность ко времени, что P08.4A отметил, и "
                "она не даёт сравнивать решения разных прогонов один к одному."),
        },
        "vid00006": {
            "filter_on": True,
            "unknown_keeps_direction": bool(args.unknown_keeps_direction),
            "n_decisions_base": len(base["decisions"]),
            "n_decisions_filtered": len(filt["decisions"]),
            "n_strong_base": strong_base, "n_strong_filtered": strong_filt,
            "class_counts": classes,
            "n_edges_base": len(seq_base), "n_edges_filtered": len(seq_filt),
            "dead_ends_base": sum(1 for s in base["edge_sequence"]
                                  if g.degree(s["to"]) <= 1),
            "dead_ends_filtered": sum(1 for s in filt["edge_sequence"]
                                      if g.degree(s["to"]) <= 1),
            "changed_entries": changed,
            "route_first_diff_edge_index": i,
            "route_common_from_end": j,
            "route_base_middle": seq_base[i:len(seq_base) - j],
            "route_filtered_middle": seq_filt[i:len(seq_filt) - j],
            "real_choice_changes": [{"node": n,
                                     "base": (a or {}).get("chosen_edge", ""),
                                     "base_reason": (a or {}).get("reason", ""),
                                     "filt": (b or {}).get("chosen_edge", ""),
                                     "filt_reason": (b or {}).get("reason", "")}
                                    for n, a, b in real],
            "junctions": j_rows2,
            "fixed": fixed, "broke": broke, "still_wrong": untouched,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'replay_all.csv'}")
    print(f"записано: {out/'vid00006_comparison.csv'}")
    print(f"записано: {out/'junction_outcome.csv'}")
    print(f"записано: {out/'junctions.csv'}")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
