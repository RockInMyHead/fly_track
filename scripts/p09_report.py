#!/usr/bin/env python3
"""P09 — put the phase's numbers in one place, with the controls that make them mean something.

Two results came out of this phase and they point the same way, which is worth stating plainly
because it is not the result the phase was looking for.

The cells were never the problem
--------------------------------
On the two clips that carry hand-confirmed camera turns, the candidates and the four types the
tracker already reads behave the same way: both clear their own floor several times more often
during a real turn than they do at random, and both are almost always right about the direction
when they do. A cell's floor is its own 95th percentile, so it clears that floor in five percent
of all windows by construction; clearing it in twenty percent of the labelled turns is the
measurement that the turn is what the cell responds to. On that measurement the new cells are
comparable to the old ones, not better.

T50 was the wrong question
--------------------------
The tracker decides a junction from camera *rotation*. At T50 the calibrated heading does not
move at all in the decision window — zero degrees, while the walker covers 0.35 m — so there was
no rotation to read. Not a weak response, not a coincidental one: nothing to measure. No set of
cells can carry a signal that is not in the stimulus, and the conclusion is that the tracker's
decision variable, not its cell choice, is what failed there.

J35 was the right answer to the wrong question
----------------------------------------------
At J35 the camera rotates 102 degrees to the left and every cell that speaks says LEFT, old and
new alike. The cells are reading the camera correctly. The walker's body then goes right. That is
the look-versus-turn problem P08.4C found, arriving from the other side: the reading is not wrong
about the head, it is being used as though it were about the path.

Nothing here refits anything. The candidates were frozen on the synthetic stimulus before any
video was opened, and every floor is measured from the clip being judged.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p09_real_holdout as H  # noqa: E402

OUT = ROOT / "output/p09"
CHANCE = 5.0            # a cell clears its own 95th-percentile floor this often by construction
PROBLEM_JUNCTIONS = ("J35", "T50", "T49")


def base_rate_table(video: str, turns: list[dict]) -> dict:
    """Per-cell: how often it speaks at real turns against the 5 percent it speaks at random."""
    d = np.load(OUT / f"trace_{video}.npz")
    t, fired = d["t"].astype(float), d["fired"]
    cells = d["cells"].astype(int)
    ctype = [str(x) for x in d["cell_type"]]
    cside = [str(x) for x in d["side"]]
    dt = float(np.median(np.diff(t)))

    pol_by_cell = {}
    for r in csv.DictReader((OUT / "all_dn_metrics.csv").open(encoding="utf-8")):
        pol_by_cell[int(r["cell"])] = float(r["polarity"])
    frozen_ids = {int(x["cell"]) for x in json.loads(
        (ROOT / "data/p09_frozen_targets.json").read_text(encoding="utf-8"))["targets"]}
    ctrl_ids = {int(x["cell"]) for x in H.control_cells()}

    keep, seen = [], set()
    for k, c in enumerate(cells):
        if int(c) in seen:
            continue
        seen.add(int(c))
        keep.append(k)
    fired = fired[:, keep]
    cells = cells[keep]
    ctype = [ctype[k] for k in keep]
    cside = [cside[k] for k in keep]
    z, integ, floor = H.cell_integrals(fired, t, dt)

    if not turns:
        return {"video": video, "n_turns": 0, "candidates": [], "controls": []}
    rows = []
    for k, c in enumerate(cells):
        p = pol_by_cell.get(int(c), np.nan)
        if not np.isfinite(p):
            continue
        spoke = hit = 0
        for x in turns:
            I = H.window_integral(integ, t, x["t1"])
            if abs(I[k]) < floor[k]:
                continue
            spoke += 1
            if ("LEFT" if (I[k] > 0) == (p > 0) else "RIGHT") == x["kind"]:
                hit += 1
        rows.append({"cell": int(c), "cell_type": ctype[k], "side": cside[k],
                     "group": ("candidate" if int(c) in frozen_ids else
                               ("control" if int(c) in ctrl_ids else "other")),
                     "spoke": spoke, "correct": hit, "n_turns": len(turns),
                     "speak_rate": spoke / len(turns),
                     "accuracy_given_speech": (hit / spoke) if spoke else None})
    return {"video": video, "n_turns": len(turns), "chance_speak_rate": CHANCE,
            "candidates": [r for r in rows if r["group"] == "candidate"],
            "controls": [r for r in rows if r["group"] == "control"]}


def main() -> int:
    frozen = json.loads((ROOT / "data/p09_frozen_targets.json").read_text(encoding="utf-8"))
    try:
        from p07_fly_readout import labelled_turns
        all_turns = labelled_turns()
    except Exception:
        all_turns = []

    print("=" * 100)
    print("P09 — ИТОГ: НОВЫЕ НИСХОДЯЩИЕ КЛЕТКИ ДЛЯ LEFT/RIGHT")
    print("=" * 100)
    print(f"  клеток проверено на синтетике:  {frozen['n_screened']}")
    print(f"  прошли все условия:             {frozen['n_survivors']}")
    print(f"  независимых источников:         {frozen['n_groups']} (корреляция по блокам, "
          f"|r| >= {frozen['cluster_corr']})")
    new_types = [f for f in frozen["targets"]
                 if f["cell_type"] not in ("DNp17", "DNa07", "DNp26", "DNp20")]
    print(f"  типов, которых трекер не читал: {len(new_types)} из {len(frozen['targets'])}")
    print()

    # ---- the cells, on clips with hand-confirmed turns -----------------------------------
    holdouts = []
    print("─── КЛЕТКИ НА РЕАЛЬНОМ ВИДЕО, ПРОТИВ РУЧНОЙ РАЗМЕТКИ ───")
    print("    порог клетки — её собственный 95-й процентиль, поэтому случайно она")
    print("    пробивает его ровно в 5% окон. Больше — значит поворот не случаен.")
    print()
    for video in ("VID00001", "VID00002", "VID00006"):
        turns = [x for x in all_turns if x["video"] == video]
        if not (OUT / f"trace_{video}.npz").exists():
            print(f"  {video}: записи нет")
            continue
        tbl = base_rate_table(video, turns)
        holdouts.append(tbl)
        if not turns:
            print(f"  {video}: ручной разметки поворотов нет ({len(turns)})")
            continue
        print(f"  {video}: {len(turns)} подтверждённых поворотов")
        print(f"    {'группа':<12}{'клеток':>8}{'говорят на поворотах':>22}"
              f"{'верных при говорении':>23}")
        for name in ("candidates", "controls"):
            rs = [r for r in tbl[name] if r["spoke"] >= 3]
            if not rs:
                print(f"    {name:<12}{len(tbl[name]):>8}{'—':>22}{'—':>23}")
                continue
            rate = float(np.mean([r["speak_rate"] for r in tbl[name]]))
            ratem = float(np.median([r["speak_rate"] for r in tbl[name]]))
            acc = [r["accuracy_given_speech"] for r in rs
                   if r["accuracy_given_speech"] is not None]
            print(f"    {name:<12}{len(tbl[name]):>8}{rate:>15.1%} (мед {ratem:.0%})"
                  f"{float(np.mean(acc)):>23.0%}")
        print(f"    (случайный уровень — {CHANCE:.0f}%)")
        print()

    (OUT / "real_holdout.csv").write_text("", encoding="utf-8")
    with (OUT / "real_holdout.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "group", "cell", "cell_type", "side", "spoke", "correct",
                    "n_turns", "speak_rate", "accuracy_given_speech"])
        for tbl in holdouts:
            for name in ("candidates", "controls"):
                for r in tbl[name]:
                    w.writerow([tbl["video"], name, r["cell"], r["cell_type"], r["side"],
                                r["spoke"], r["correct"], r["n_turns"],
                                round(r["speak_rate"], 4),
                                ("" if r["accuracy_given_speech"] is None
                                 else round(r["accuracy_given_speech"], 4))])

    # ---- the junctions that started all this ---------------------------------------------
    print("─── ТРИ ПРОВАЛЬНЫЕ РАЗВИЛКИ (VID00006) ───")
    pt = ROOT / "output/p09/problem_turns_VID00006.csv"
    problem = list(csv.DictReader(pt.open(encoding="utf-8"))) if pt.exists() else []
    rot = {}
    tr = list(csv.DictReader(
        (ROOT / "output/p075/trajectory_VID00006.csv").open(encoding="utf-8")))
    T = np.array([float(r["t"]) for r in tr])
    TH = np.array([float(r["theta_calibrated"]) for r in tr])
    for r in problem:
        t = float(r["t"])
        m = (T >= t - 3.2) & (T <= t)
        rot[r["node"]] = float(TH[m][-1] - TH[m][0]) if m.sum() > 1 else 0.0
    print(f"    {'узел':<6}{'нужен':<8}{'пул сказал':<12}{'кандидатов сказало':<20}"
          f"{'из них нужный знак':<20}{'поворот камеры':<16}")
    for r in problem:
        spk = json.loads(r["speakers"]) if r["speakers"] else []
        print(f"    {r['node']:<6}{r['truth_side']:<8}{r['pooled']:<12}{len(spk):<20}"
              f"{r['n_right_sign']:<20}{rot.get(r['node'], float('nan')):>7.1f}°")
    print()
    print("  J35: камера повернулась на 102°, все говорящие клетки сказали LEFT — и они правы")
    print("       про камеру. Тело пошло вправо. Это LOOK против TURN, а не ошибка клеток.")
    print("  T50: поворот камеры РАВЕН НУЛЮ. Муха проехала 0.35 м, не вращая камеру.")
    print("       Сигнала поворота не существует, поэтому его не может дать ни одна клетка.")
    print("  T49: поворот 34°, говорит одна клетка и говорит LEFT, тогда как нужен RIGHT —")
    print("       тот же случай, что J35, с меньшей амплитудой.")

    (OUT / "report.json").write_text(json.dumps({
        "phase": "P09 — поиск новых нисходящих клеток для LEFT/RIGHT",
        "unchanged": ["P08, граф, правила маршрутизации"],
        "screen": {k: frozen[k] for k in
                   ("n_screened", "n_survivors", "n_groups", "filters", "cluster_corr",
                    "stimulus")},
        "frozen_targets": frozen["targets"],
        "n_new_types": len(new_types),
        "holdout": holdouts,
        "problem_junctions": problem,
        "camera_rotation_deg_at": rot,
        "conclusions": {
            "cells_were_not_the_problem": (
                "на VID00001 и VID00002 кандидаты и контрольные 46 клеток ведут себя одинаково: "
                "обе группы пробивают свой порог в 20 и 12 процентах подтверждённых поворотов "
                "против 5 процентов случайно, и почти всегда правы, когда говорят. Новые клетки "
                "сопоставимы со старыми, а не лучше."),
            "T50_was_the_wrong_question": (
                "в окне решения T50 калиброванный курс не меняется вовсе — 0.0°, при пути "
                "0.35 м. Поворота камеры не было, значит измерять было нечего. Это ошибка "
                "переменной решения трекера, а не подбора клеток."),
            "J35_was_the_right_answer_to_the_wrong_question": (
                "камера повернулась на 102° влево, и все клетки, которые говорят, говорят "
                "LEFT — они правы про камеру. Тело пошло вправо. Это LOOK против TURN."),
            "answer": (
                "расширение набора нисходящих клеток задачу не решает. Существующие четыре "
                "читают повороты не хуже новых, а два из трёх провалов объясняются тем, что "
                "сигнала поворота в стимуле не было (T50) либо он относился к камере, а не к "
                "телу (J35). Дальше нужен либо другой признак, кроме вращения камеры, либо "
                "разделение взгляда и поворота, а не другая выборка клеток."),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"записано: {OUT/'report.json'}")
    print(f"записано: {OUT/'real_holdout.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
