#!/usr/bin/env python3
"""
P05.4 — test the frozen descending neurons on freshly labelled turns.

No new simulation. The P05.3 recording already covers the whole clip, so the moment the
second labelling round is finished this scores the frozen cells on turns they have never
been shown.

What is frozen, and when
------------------------
The eight primary cells and three controls were written to `output/p05_frozen_targets.json`
before the second round was labelled, and the rule below was fixed at the same time.
Neither is changed after seeing a result. That is the whole point: P05.3 selected these
cells by looking at these neurons on the first set of turns, so the first set can no
longer test them. Only the second round can.

The pre-registered rule
-----------------------
Per cell, AUC between the firing rate on confirmed LEFT and confirmed RIGHT turns.
A cell passes at AUC >= 0.80. Then:

    found the readout   at least 4 of the 8 primary pass, and at most 1 of the 3 controls
    inconclusive        3 primary pass
    close the branch    2 or fewer primary pass

DNp17 and DNa07 passing on both sides is recorded separately as the strong signal.

Why the controls matter
-----------------------
DNa02, DNg100 and MDN receive 0.81, 0.00 and 0.00 percent of their input from this path
and scored at or near chance in P05.3. If they pass here too, the test is not measuring
the path and the result means nothing.

Uncertainty
-----------
With a few dozen turns an AUC carries a wide interval, so the Hanley-McNeil standard
error is reported next to every value and each pass is labelled as clearly above chance
or not. A cell at 0.82 with a standard error of 0.12 is not evidence.

Usage:
    PYTHONPATH=. python scripts/p054_neuron_test.py
    PYTHONPATH=. python scripts/p054_neuron_test.py --excel-label-noise
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

TRACE = ROOT / "output/p053_threshold/spike_trace.npz"
TARGETS_CSV = ROOT / "output/p053_threshold/targets.csv"
FROZEN = ROOT / "data/p05_frozen_targets.json"
R1_SET = ROOT / "data/p01r/review_set.json"
R1_VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
R2_SET = ROOT / "data/p01r/review_set_v2.json"
R2_VERDICTS = ROOT / "data/p01r/turn_verdicts_v2.json"
OUT = ROOT / "output/p054_neuron_test"

SMOOTH_S = 0.3


def auc_raw(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[order] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    return float((rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def hanley_se(auc: float, n1: int, n2: int) -> float:
    """Standard error of an AUC, Hanley & McNeil 1982."""
    if n1 < 2 or n2 < 2:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc)
         + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2)
    return float(np.sqrt(max(v, 0.0)))


def load_bank(set_path: Path, verdict_path: Path) -> tuple[list[dict], dict]:
    cand = json.loads(set_path.read_text(encoding="utf-8"))["candidates"]
    vd = json.loads(verdict_path.read_text(encoding="utf-8")).get("verdicts", {}) \
        if verdict_path.exists() else {}
    out, repeats = [], {}
    for x in cand:
        if "repeat_of" in x:
            repeats[x["id"]] = x["repeat_of"]
        v = vd.get(x["id"])
        if not v:
            continue
        truth = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "truth": truth,
                    "verdict": v["verdict"], "repeat_of": x.get("repeat_of")})
    return out, vd, repeats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity-source", default="fire",
                    choices=["fire", "v"], help="what the readout sees")
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    col = {i: t for i, t in enumerate(csv.DictReader(TARGETS_CSV.open()))}
    d = np.load(TRACE)
    t = d["t"].astype(np.float64)
    fired = d["fired"].astype(np.float64)
    v_pre = d["v_pre"].astype(np.float64)
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(SMOOTH_S / dt)), 1)

    def box(x: np.ndarray, n: int) -> np.ndarray:
        return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")

    print("P05.4 — проверка зафиксированных нисходящих нейронов")
    print(f"  запись: {len(t):,} шагов, {t[-1] - t[0]:.0f} с")
    print(f"  снимаемая величина: "
          f"{'частота спайков' if args.activity_source == 'fire' else 'мембрана v'}")

    r2_items, r2_vd, r2_rep = load_bank(R2_SET, R2_VERDICTS)
    r1_items, r1_vd, _ = load_bank(R1_SET, R1_VERDICTS)

    def confirmed(items: list[dict]) -> list[dict]:
        # `repeat_of` is always present as a key, so filter on its value, not on the
        # key's existence; otherwise every window is dropped as a repeat
        return [x for x in items if x["truth"] in ("LEFT", "RIGHT")
                and not x.get("repeat_of")]

    c2 = confirmed(r2_items)
    c1 = confirmed(r1_items)
    print(f"\n=== РАЗМЕТКА ===")
    print(f"  раунд 1: вердиктов {len(r1_vd)}, подтверждённых поворотов {len(c1)}")
    print(f"  раунд 2: вердиктов {len(r2_vd)}, подтверждённых поворотов {len(c2)} "
          f"({sum(1 for x in c2 if x['truth'] == 'RIGHT')} RIGHT, "
          f"{sum(1 for x in c2 if x['truth'] == 'LEFT')} LEFT)")

    if not c2:
        if not r2_vd:
            print("\n  ВТОРОЙ РАУНД ЕЩЁ НЕ РАЗМЕЧЕН")
            print("  запустить разметку:  REVIEW_ROUND=2 ./webapp/run.sh")
            print("  затем повторить:     PYTHONPATH=. python scripts/p054_neuron_test.py")
        else:
            print(f"\n  РАЗМЕТКА ЕСТЬ ({len(r2_vd)} вердиктов), НО ПОВОРОТЫ НЕ ПОДТВЕРЖДЕНЫ")
            print(f"  подтверждённых LEFT/RIGHT во втором раунде: 0")
            print("  если так и было задумано — тест невозможен, нужен раунд, где")
            print("  повороты действительно подтверждены")
        return

    # ---- label self-consistency -------------------------------------------
    print(f"\n=== ШУМ РАЗМЕТКИ (повторы) ===")
    if r2_rep:
        agree = same_twin = 0
        twins = 0
        by_id = {x["id"]: x for x in r2_items}
        for rid, src in r2_rep.items():
            a, b = by_id.get(rid), by_id.get(src)
            if not a or not b:
                continue
            if "verdict" in a and "verdict" in b:
                twins += 1
                agree += (a["truth"] == b["truth"])
        if twins:
            print(f"  повторов оценено: {twins}")
            print(f"  совпало с первым показом: {agree}/{twins} = "
                  f"{agree / twins * 100:.0f}%")
            print(f"  -> {'это пол шума во всех числах ниже' if agree < twins else 'расхождений нет'}")
        else:
            print("  повторы ещё не оценены")
    else:
        print("  повторов в банке нет")

    # ---- per-cell AUC ------------------------------------------------------
    def win_med(M: np.ndarray, w: dict) -> np.ndarray:
        m = (t >= w["t0"]) & (t <= w["t1"])
        return np.median(M[m], axis=0) if m.any() else np.full(M.shape[1], np.nan)

    def win_rate(M: np.ndarray, w: dict) -> np.ndarray:
        m = (t >= w["t0"]) & (t <= w["t1"])
        return M[m].mean(axis=0) / dt if m.any() else np.full(M.shape[1], np.nan)

    def build(items: list[dict]) -> np.ndarray:
        M = np.zeros((len(items), fired.shape[1]))
        for k, w in enumerate(items):
            M[k] = win_rate(fired, w) if args.activity_source == "fire" \
                else win_med(v_pre, w)
        return M

    A2 = build(c2)
    A1 = build(c1)
    isR2 = np.array([x["truth"] == "RIGHT" for x in c2])
    isR1 = np.array([x["truth"] == "RIGHT" for x in c1])

    def find_col(cell_type: str, side: str):
        for i, x in col.items():
            if x["cell_type"] == cell_type and x["side"] == side:
                return i
        return None

    n1_2, n2_2 = int(isR2.sum()), int((~isR2).sum())

    def score(items: list[dict], M: np.ndarray, isR: np.ndarray, cell_type: str,
              side: str) -> dict:
        j = find_col(cell_type, side)
        if j is None:
            return {}
        raw = auc_raw(M[isR, j], M[~isR, j])
        auc = max(raw, 1.0 - raw) if len(items) else 0.5
        n1, n2 = int(isR.sum()), int((~isR).sum())
        se = hanley_se(auc, n1, n2)
        return {"auc": auc, "raw": raw, "se": se,
                "ci_low": auc - 1.96 * se if np.isfinite(se) else float("nan"),
                "ci_high": auc + 1.96 * se if np.isfinite(se) else float("nan"),
                "rate_L": float(np.mean(M[~isR, j])) if (~isR).any() else float("nan"),
                "rate_R": float(np.mean(M[isR, j])) if isR.any() else float("nan"),
                "n1": n1, "n2": n2}

    print(f"\n=== ГЛАВНЫЙ ТЕСТ: 8 ОСНОВНЫХ КЛЕТОК НА НОВЫХ ПОВОРОТАХ ===")
    print(f"  {n1_2} RIGHT, {n2_2} LEFT подтверждённых во втором раунде\n")
    print(f"  {'клетка':>14s} {'AUC':>7s} {'±SE':>7s} {'95% интервал':>16s} "
          f"{'LEFT Гц':>8s} {'RIGHT Гц':>9s} {'вердикт':>10s}")
    rows, primary = [], []
    for tgt in frozen["primary"]:
        s = score(c2, A2, isR2, tgt["cell_type"], tgt["side"])
        if not s:
            continue
        passed = s["auc"] >= 0.80
        clear = np.isfinite(s["se"]) and (s["ci_low"] > 0.5)
        verdict = ("ПРОШЛА" if passed else "нет") + ("" if clear else " (шумно)")
        print(f"  {tgt['cell_type'] + ' ' + tgt['side']:>14s} {s['auc']:7.3f} "
              f"{s['se']:7.3f} {s['ci_low']:7.2f}..{s['ci_high']:<7.2f} "
              f"{s['rate_L']:8.2f} {s['rate_R']:9.2f} {verdict:>10s}")
        rows.append({**tgt, **s, "round": 2, "passed": passed, "clear": clear})
        primary.append(passed)

    print(f"\n=== КОНТРОЛЬНЫЕ КЛЕТКИ (должны ПРОВАЛИТЬСЯ) ===")
    print(f"  {'клетка':>14s} {'AUC':>7s} {'±SE':>7s} {'95% интервал':>16s} "
          f"{'LEFT Гц':>8s} {'RIGHT Гц':>9s}")
    controls = []
    for tgt in frozen["controls_expected_to_fail"]:
        s = score(c2, A2, isR2, tgt["cell_type"], tgt["side"])
        if not s:
            continue
        print(f"  {tgt['cell_type'] + ' ' + tgt['side']:>14s} {s['auc']:7.3f} "
              f"{s['se']:7.3f} {s['ci_low']:7.2f}..{s['ci_high']:<7.2f} "
              f"{s['rate_L']:8.2f} {s['rate_R']:9.2f}")
        rows.append({**tgt, **s, "round": 2, "passed": s["auc"] >= 0.80,
                     "clear": False})
        controls.append(s["auc"] >= 0.80)

    # ---- round 1 for reference, clearly not a test ------------------------
    print(f"\n=== ДЛЯ СПРАВКИ: ТЕ ЖЕ КЛЕТКИ НА ПЕРВОМ РАУНДЕ ===")
    print("  этот набор использовался при отборе клеток в P05.3, поэтому он не тест")
    print(f"  {n1_2} RIGHT, {n2_2} LEFT против "
          f"{int(isR1.sum())} RIGHT, {int((~isR1).sum())} LEFT\n")
    print(f"  {'клетка':>14s} {'раунд 1':>9s} {'раунд 2':>9s} {'изменилось':>11s}")
    for r in rows:
        if r.get("round") != 2:
            continue
        s1 = score(c1, A1, isR1, r["cell_type"], r["side"])
        if not s1:
            continue
        delta = r["auc"] - s1["auc"]
        print(f"  {r['cell_type'] + ' ' + r['side']:>14s} {s1['auc']:9.3f} "
              f"{r['auc']:9.3f} {delta:+11.3f}")
        r["auc_round1"] = s1["auc"]

    # ---- the pre-registered decision --------------------------------------
    n_pass = int(np.sum(primary))
    n_ctrl_pass = int(np.sum(controls))
    print(f"\n=== РЕШЕНИЕ ПО ЗАРАНЕЕ ЗАФИКСИРОВАННОМУ ПРАВИЛУ ===")
    print(f"  правило: >=4 из 8 основных прошли и <=1 из 3 контролей прошли")
    print(f"  основные прошли: {n_pass} из 8")
    print(f"  контроли прошли: {n_ctrl_pass} из 3")
    if n_ctrl_pass > 1:
        # when a cell with no input from this path separates the turns, the labels are
        # carrying the answer rather than the neurons, so nothing here is a measurement
        decision = (f"ТЕСТ НЕДЕЙСТВИТЕЛЕН: прошли {n_ctrl_pass} контрольных клетки, "
                    f"которые не получают входа от этого пути. Значит метки согласованы "
                    f"с чем-то общим, и разделение основных клеток ничего не "
                    f"доказывает. Причина почти всегда в том, что метки взяты у "
                    f"классификатора, а не поставлены глазами.")
    elif n_pass >= 4:
        decision = ("НАЙДЕН ВЫХОД: направление поворота устойчиво читается с "
                    "зафиксированных нисходящих нейронов на новых поворотах")
    elif n_pass <= 2:
        decision = ("ВЕТКУ ЗАКРЫВАЕМ: даже лучшие кандидаты не дают устойчивого "
                    "сигнала на новых поворотах")
    else:
        decision = (f"НЕОПРЕДЕЛЁННО: прошли {n_pass} из 8, одного раунда разметки "
                    f"не хватило для решения")
    print(f"\n  {decision}")

    strong = [r for r in rows
              if r["cell_type"] in ("DNp17", "DNa07") and r["passed"]]
    print(f"\n  сильный сигнал (DNp17 и DNa07 на обеих сторонах): "
          f"{len(strong)} из 4 пар прошли")
    for r in rows:
        if r["cell_type"] in ("DNp17", "DNa07"):
            print(f"    {r['cell_type']} {r['side']}: AUC {r['auc']:.3f} "
                  f"({'прошла' if r['passed'] else 'нет'})")

    # ---- outputs -----------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "neuron_test.csv").open("w", newline="") as f:
        keys = ["cell_type", "side", "why", "auc", "raw", "se", "ci_low", "ci_high",
                "rate_L", "rate_R", "n1", "n2", "passed", "clear", "auc_round1"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT / "report.json").write_text(json.dumps({
        "frozen_targets": FROZEN.name, "frozen_before": "второй раунд разметки",
        "n_round1_turns": len(c1), "n_round2_turns": len(c2),
        "n_round2_right": int(isR2.sum()), "n_round2_left": int((~isR2).sum()),
        "activity_source": args.activity_source,
        "n_primary_passed": n_pass, "n_controls_passed": n_ctrl_pass,
        "decision": decision,
        "per_cell": rows,
        "rule": "AUC >= 0.80 на подтверждённых поворотах второго раунда; "
                ">=4 из 8 основных и <=1 из 3 контролей",
        "caveat": "AUC считается по медиане окна; при нескольких десятках поворотов "
                  "стандартная ошибка широкая и указана рядом с каждым значением",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
