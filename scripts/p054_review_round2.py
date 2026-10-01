#!/usr/bin/env python3
"""
P05.4 — build the second review round.

The brain recordings already cover the whole clip, so the only thing missing for a
final test of the descending neurons is more eye-confirmed turns. This assembles a
second labelling bank without touching the first round's verdicts.

What goes in
------------
    turns     36 turn candidates the classifier proposed that were never reviewed
              (20 LEFT, 16 RIGHT by its own guess; the guess is not the label)
    controls  24 forward and stationary windows, so a neuron that simply responds to
              movement cannot pass
    repeats   6 windows shown twice under different ids and in different places

The repeats are the point worth explaining. With roughly thirty turns, a neuron can
look convincing by luck, and there is no way to tell label noise from real disagreement
without measuring it. Showing the same window twice gives that measurement directly: the
share of repeats answered inconsistently is a floor on the noise in every number
computed from this bank.

Order is shuffled with a fixed seed so the run is reproducible but the reviewer does not
face all the left turns together.

The four candidate cells are frozen in a separate file and must not be changed after
looking at results.

Usage:
    PYTHONPATH=. python scripts/p054_review_round2.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "data/p01r/windows_v3.json"
ROUND1 = ROOT / "data/p01r/review_set.json"
OUT_JSON = ROOT / "data/p01r/review_set_v2.json"
FREEZE = ROOT / "data/p05_frozen_targets.json"

# Frozen before any result is seen. Taken from P05.3: the four groups with the largest
# share of input from the depth-1 carriers plus a clean record, and three controls that
# deliberately should FAIL so the test can show a negative.
TARGETS = {
    "primary": [
        {"cell_type": "DNp17", "side": "L", "why": "6-8% входа от пути, AUC спайков 1.000"},
        {"cell_type": "DNp17", "side": "R", "why": "8-12% входа от пути, AUC 0.96-1.000"},
        {"cell_type": "DNa07", "side": "L", "why": "13.8% входа, AUC 1.000"},
        {"cell_type": "DNa07", "side": "R", "why": "14.0% входа, AUC 1.000"},
        {"cell_type": "DNp26", "side": "L", "why": "6.4% входа, AUC 1.000"},
        {"cell_type": "DNp26", "side": "R", "why": "6.7% входа, AUC 0.86-1.000"},
        {"cell_type": "DNp20", "side": "L", "why": "6.1% входа, AUC 0.84-0.99"},
        {"cell_type": "DNp20", "side": "R", "why": "6.1% входа, AUC 0.65-0.97"},
    ],
    "controls_expected_to_fail": [
        {"cell_type": "DNa02", "side": "L", "why": "0.81% входа, AUC 0.77-0.86"},
        {"cell_type": "DNg100", "side": "L", "why": "0.00% входа, AUC 0.61-0.67"},
        {"cell_type": "MDN", "side": "L", "why": "0.00% входа, AUC 0.52-0.66"},
    ],
    "frozen": True,
    "rule": "Эти группы зафиксированы до просмотра результатов второго раунда. "
            "После разметки набор не меняется и не дополняется.",
}


def overlaps(a, b, spans, tol=0.5):
    return any(not (b < w0 - tol or a > w1 + tol) for w0, w1 in spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-controls", type=int, default=24)
    ap.add_argument("--n-repeats", type=int, default=6)
    ap.add_argument("--seed", type=int, default=202)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    w3 = json.loads(WINDOWS.read_text(encoding="utf-8"))["windows"]
    r1 = json.loads(ROUND1.read_text(encoding="utf-8"))["candidates"]
    seen = [(x["t0"], x["t1"]) for x in r1]

    turns = [x for x in w3 if x["kind"] in ("LEFT", "RIGHT")
             and not overlaps(x["t0"], x["t1"], seen)]
    ctrl = [x for x in w3 if x["kind"] in ("FWD", "STATIC")
            and not overlaps(x["t0"], x["t1"], seen)]

    print("P05.4 — банк второго раунда разметки")
    print(f"  round 1 просмотрено: {len(r1)} окон, перекрытие исключено")
    print(f"  свободных поворотов-кандидатов: {len(turns)} "
          f"({dict(Counter(x['kind'] for x in turns))})")
    print(f"  свободных FWD/STATIC: {len(ctrl)} "
          f"({dict(Counter(x['kind'] for x in ctrl))})")

    # controls spread across the clip so they are not all from one stretch
    ctrl_sorted = sorted(ctrl, key=lambda x: x["t0"])
    if len(ctrl_sorted) > args.n_controls:
        idx = np.linspace(0, len(ctrl_sorted) - 1, args.n_controls).round().astype(int)
        ctrl_sel = [ctrl_sorted[i] for i in idx]
    else:
        ctrl_sel = ctrl_sorted
    print(f"  контролей отобрано: {len(ctrl_sel)}")

    # 3 turns and 3 controls, re-presented later under fresh ids
    n_rep = args.n_repeats // 2
    rep_src = ([x for x in sorted(turns, key=lambda z: z["t0"])][:n_rep]
               + [x for x in ctrl_sel][:n_rep])
    print(f"  повторов: {len(rep_src)} окон будут показаны дважды "
          f"для оценки шума разметки")

    def make(x: dict, cid: str, repeat_of: str | None = None) -> dict:
        sys_dir = x.get("camera_direction", x["kind"])
        rec = {
            "kind": x["kind"], "t0": x["t0"], "t1": x["t1"],
            "duration_s": x.get("duration_s", x["t1"] - x["t0"]),
            "a": x.get("a", 0.0), "divergence": x.get("divergence", 0.0),
            "residual": x.get("residual", 0.0),
            "activity": x.get("activity", 0.0),
            "activity_vertical": x.get("activity_vertical", 0.0),
            "camera_direction": sys_dir,
            "confidence": x.get("confidence", "medium"),
            "id": cid,
            "expected": (f"камера повернула {sys_dir}" if sys_dir in ("LEFT", "RIGHT")
                         else f"камера {sys_dir.lower()}" if sys_dir in ("FWD", "STATIC")
                         else "неясно"),
        }
        if repeat_of:
            rec["repeat_of"] = repeat_of
        return rec

    items = []
    for i, x in enumerate(turns):
        items.append(make(x, f"s{i:03d}"))
    for j, x in enumerate(ctrl_sel):
        items.append(make(x, f"c{j:03d}"))
    for k, x in enumerate(rep_src):
        rep = make(x, f"p{k:03d}")
        src_id = next(it["id"] for it in items
                      if abs(it["t0"] - x["t0"]) < 1e-6 and it["kind"] == x["kind"])
        rep["repeat_of"] = src_id
        items.append(rep)

    # shuffle, but keep the repeats away from their twins so memory does not carry over
    order = rng.permutation(len(items))
    shuffled = [items[i] for i in order]
    placed, final = set(), []
    for it in shuffled:
        src = it.get("repeat_of")
        if src and src in placed:
            final.append(it)
        else:
            final.append(it)
        placed.add(it["id"])
    # a second pass that pushes any repeat sitting immediately after its twin further on
    fixed, taken = [], set()
    for i, it in enumerate(final):
        src = it.get("repeat_of")
        if src:
            j = next(k for k, o in enumerate(final) if o["id"] == src)
            if abs(i - j) < 12:
                continue
        fixed.append(it)
        taken.add(it["id"])
    for it in final:
        if it["id"] not in taken:
            fixed.append(it)

    counts = Counter(it["kind"] for it in fixed)
    doc = {
        "purpose": "Второй раунд: расширить набор подтверждённых поворотов, чтобы "
                   "проверить зафиксированные нисходящие нейроны на новых данных. "
                   "Кандидаты предложены классификатором, метка ставится глазами.",
        "source": str(WINDOWS),
        "n": len(fixed),
        "counts": dict(counts),
        "n_turns_candidates": len(turns),
        "n_controls": len(ctrl_sel),
        "n_repeats": len(rep_src),
        "repeat_ids": {it["id"]: it["repeat_of"] for it in fixed if "repeat_of" in it},
        "targets_note": "Проверяемые нейроны зафиксированы в output/p05_frozen_targets.json",
        "candidates": fixed,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    FREEZE.write_text(json.dumps(TARGETS, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  окно просмотра: {min(it['t0'] for it in fixed):.0f} .. "
          f"{max(it['t1'] for it in fixed):.0f} с")
    print(f"  состав: {dict(counts)}")
    print(f"  повторы на позициях: "
          f"{[i for i, it in enumerate(fixed) if 'repeat_of' in it][:8]}")
    print(f"\n  зафиксированные цели: "
          f"{len(TARGETS['primary'])} основных + "
          f"{len(TARGETS['controls_expected_to_fail'])} контрольных")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {FREEZE}")
    print(f"\n  запуск разметки:  REVIEW_ROUND=2 ./webapp/run.sh")
    print(f"  затем:            PYTHONPATH=. python scripts/p054_neuron_test.py")


if __name__ == "__main__":
    main()
