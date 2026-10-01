#!/usr/bin/env python3
"""
P06.1 — analysis of the clean direction test.

The stimulus holds everything but the sign of motion fixed, so any difference between
LEFT and RIGHT is attributable to direction. What the numbers mean:

    delta = rate(RIGHT) - rate(LEFT) per frame

    sign consistency  on how many of the 25 independent frames delta has the same sign.
                      A cell reading direction holds one sign across scenes as different
                      as a corridor and a bench. A cell reading brightness or texture
                      has no reason to hold any sign at all.
    effect size       mean(delta) divided by the spread of delta across frames. Below
                      about 1 the frame-to-frame scatter is larger than the effect.
    auc               pooled over frames, so it asks whether LEFT and RIGHT separate at
                      all when the scene is held fixed.
    versus static     how far both moving conditions sit above the stationary one. A
                      cell whose LEFT and RIGHT both exceed static responds to motion
                      and modulates on top of it, which is a different thing from a
                      pure direction detector.

The controls are not required to fail
-------------------------------------
DNg100, DNa02 and MDN receive little or no input from the 48 carriers of P05, but that
says nothing about whether they read direction by another route. If one of them
separates clean LEFT from clean RIGHT here, that is a fact about the connectome rather
than a flaw in the design. They are included because they show what the design returns
for cells with no established reason to care.

Usage:
    PYTHONPATH=. python scripts/p061_synthetic_analyze.py
    PYTHONPATH=. python scripts/p061_synthetic_analyze.py --seeds 64,65
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

OUT = ROOT / "output/p061_synthetic"
TARGETS = ROOT / "output/p053_threshold/targets.csv"

PRIMARY = [("DNp17", "L"), ("DNp17", "R"), ("DNa07", "L"), ("DNa07", "R"),
           ("DNp26", "L"), ("DNp26", "R"), ("DNp20", "L"), ("DNp20", "R")]
CONTROLS = [("DNa02", "L"), ("DNg100", "L"), ("MDN", "L")]


def auc_raw(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    o = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[o] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    return float((rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="64,65")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    col = {i: x for i, x in enumerate(csv.DictReader(TARGETS.open()))}
    data = {}
    for s in seeds:
        p = OUT / f"synthetic_seed{s}.npz"
        if p.exists():
            d = np.load(p)
            data[s] = {"LEFT": d["left"], "RIGHT": d["right"], "STATIC": d["static"]}
        else:
            print(f"  нет данных для seed {s}: {p}")

    if not data:
        print("нет данных")
        return

    n_frames = next(iter(data.values()))["LEFT"].shape[0]
    print("P06.1 — чистый тест направления, анализ")
    print(f"  seed: {list(data)}, кадров на seed: {n_frames}")
    print(f"  стимул: сдвиг циклический, всё кроме знака одинаково")
    print()

    rows = []
    for nm, side in PRIMARY + CONTROLS:
        j = next((k for k, x in col.items()
                  if x["cell_type"] == nm and x["side"] == side), None)
        if j is None:
            continue
        rec = {"cell_type": nm, "side": side,
               "group": "контроль" if (nm, side) in CONTROLS else "основная"}
        for s in data:
            L = data[s]["LEFT"][:, j]
            R = data[s]["RIGHT"][:, j]
            S = data[s]["STATIC"][:, j]
            d = R - L
            nz = d[np.abs(d) > 1e-9]
            rec[f"delta_{s}"] = float(d.mean())
            rec[f"spread_{s}"] = float(d.std())
            rec[f"effect_{s}"] = float(abs(d.mean()) / d.std()) if d.std() > 0 else 0.0
            # how many frames agree with the mean sign; frames with exactly zero delta
            # are neither for nor against, so they are counted separately
            sign = np.sign(d.mean())
            rec[f"consistency_{s}"] = float(np.mean(np.sign(d) == sign))
            rec[f"n_nonzero_{s}"] = int(len(nz))
            rec[f"auc_{s}"] = max(auc_raw(R, L), 1 - auc_raw(R, L))
            rec[f"auc_raw_{s}"] = auc_raw(R, L)
            rec[f"left_{s}"] = float(L.mean())
            rec[f"right_{s}"] = float(R.mean())
            rec[f"static_{s}"] = float(S.mean())
            # does motion raise the cell above standing still at all?
            rec[f"above_static_{s}"] = bool(min(L.mean(), R.mean()) > S.mean())
        rows.append(rec)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o

    print("=== СРЕДНИЕ И РАЗБРОС ПО КАДРАМ ===")
    hdr = "  {:>12s} {:>10s}".format("клетка", "группа")
    for s in data:
        hdr += f" {'d{s}':>8s} {'±sd':>7s} {'эффект':>7s} {'знак':>7s}"
    print(hdr)
    for r in rows:
        line = "  {:>12s} {:>10s}".format(r["cell_type"] + " " + r["side"], r["group"])
        for s in data:
            line += (f" {r[f'delta_{s}']:+7.2f} {r[f'spread_{s}']:7.2f} "
                     f"{r[f'effect_{s}']:7.2f} {r[f'consistency_{s}'] * 100:6.0f}%")
        print(line)

    print(f"\n  'знак' — доля кадров, где сдвиг совпал по знаку со средним.")
    print(f"  'эффект' — |средний сдвиг| / разброс. Меньше 1 — разброс больше эффекта.")

    print(f"\n=== ОТНОСИТЕЛЬНО НЕПОДВИЖНОГО ===")
    print(f"  {'клетка':>12s} {'STATIC':>8s} {'LEFT':>8s} {'RIGHT':>8s} "
          f"{'движение выше покоя':>20s}")
    for r in rows:
        s0 = list(data)[0]
        ok = "да" if r.get(f"above_static_{s0}") else "НЕТ"
        print(f"  {r['cell_type'] + ' ' + r['side']:>12s} {r[f'static_{s0}']:8.2f} "
              f"{r[f'left_{s0}']:8.2f} {r[f'right_{s0}']:8.2f} {ok:>20s}")

    print(f"\n=== AUC (сцена одна и та же) ===")
    hdr = "  {:>12s} {:>10s}".format("клетка", "группа")
    for s in data:
        hdr += f" {('AUC'+str(s)):>9s}"
    hdr += f" {'на обоих':>10s}"
    print(hdr)
    for r in rows:
        line = "  {:>12s} {:>10s}".format(r["cell_type"] + " " + r["side"], r["group"])
        both = []
        for s in data:
            line += f" {r[f'auc_{s}']:9.3f}"
            both.append(r[f"auc_{s}"])
        line += f" {min(both):10.3f}"
        print(line)

    # ---- reproducibility across seeds --------------------------------------
    if len(data) > 1:
        s1, s2 = list(data)[0], list(data)[1]
        print(f"\n=== ВОСПРОИЗВОДИМОСТЬ МЕЖДУ SEED ===")
        print(f"  {'клетка':>12s} {'знак совпал':>12s} {'d1':>9s} {'d2':>9s} "
              f"{'AUC1':>7s} {'AUC2':>7s}")
        agree = []
        for r in rows:
            a = np.sign(r[f"delta_{s1}"]) == np.sign(r[f"delta_{s2}"])
            agree.append(a)
            print(f"  {r['cell_type'] + ' ' + r['side']:>12s} "
                  f"{('да' if a else 'НЕТ'):>12s} {r[f'delta_{s1}']:+9.3f} "
                  f"{r[f'delta_{s2}']:+9.3f} {r[f'auc_{s1}']:7.3f} {r[f'auc_{s2}']:7.3f}")
        n_ag = int(np.sum(agree))
        print(f"\n  знак совпал у {n_ag} из {len(rows)} каналов")

    # ---- pooled statistics against a null that shuffles within each frame ----------
    # A per-frame criterion is the wrong test here. With 25 frames and an effect this
    # small, the frame-to-frame scatter is larger than the mean shift, by construction:
    # that is what a weak but real effect looks like. What can be tested is whether the
    # separation exceeds what shuffling the labels within each frame produces, and
    # whether the sign of the shift repeats across seeds.
    rng = np.random.default_rng(0)
    null = []
    for (nm, side) in PRIMARY:
        j = next((k for k, x in col.items()
                  if x["cell_type"] == nm and x["side"] == side), None)
        if j is None:
            continue
        for s in data:
            R, L = data[s]["RIGHT"][:, j], data[s]["LEFT"][:, j]
            for _ in range(200):
                m = rng.random(len(R)) < 0.5
                a, b = np.where(m, R, L), np.where(m, L, R)
                null.append(max(auc_raw(a, b), 1 - auc_raw(a, b)))
    null = np.array(null)
    thr = float(np.percentile(null, 95))

    pooled = {}
    for nm, side in PRIMARY + CONTROLS:
        j = next((k for k, x in col.items()
                  if x["cell_type"] == nm and x["side"] == side), None)
        if j is None:
            continue
        R = np.concatenate([data[s]["RIGHT"][:, j] for s in data])
        L = np.concatenate([data[s]["LEFT"][:, j] for s in data])
        S = np.concatenate([data[s]["STATIC"][:, j] for s in data])
        pooled[(nm, side)] = {"auc": max(auc_raw(R, L), 1 - auc_raw(R, L)),
                              "delta": float((R - L).mean()),
                              "spread": float((R - L).std()),
                              "motion_above_static": bool(min(R.mean(), L.mean())
                                                          > S.mean())}

    p_ok = [pooled[k] for k in PRIMARY if k in pooled and pooled[k]["auc"] > thr]
    c_ok = [pooled[k] for k in CONTROLS if k in pooled and pooled[k]["auc"] > thr]
    p_sign = 0
    c_sign = 0
    if len(data) > 1:
        s1, s2 = list(data)[0], list(data)[1]
        for k in PRIMARY:
            if k not in pooled:
                continue
            j = next(x for x, y in col.items()
                     if y["cell_type"] == k[0] and y["side"] == k[1])
            d1 = (data[s1]["RIGHT"][:, j] - data[s1]["LEFT"][:, j]).mean()
            d2 = (data[s2]["RIGHT"][:, j] - data[s2]["LEFT"][:, j]).mean()
            p_sign += int(np.sign(d1) == np.sign(d2))
        for k in CONTROLS:
            if k not in pooled:
                continue
            j = next(x for x, y in col.items()
                     if y["cell_type"] == k[0] and y["side"] == k[1])
            d1 = (data[s1]["RIGHT"][:, j] - data[s1]["LEFT"][:, j]).mean()
            d2 = (data[s2]["RIGHT"][:, j] - data[s2]["LEFT"][:, j]).mean()
            c_sign += int(np.sign(d1) == np.sign(d2))

    print(f"\n=== НУЛЬ: ПЕРЕМЕШИВАНИЕ ВНУТРИ КАДРА ===")
    print(f"  медиана {np.median(null):.3f}, 95-й процентиль {thr:.3f}")
    print(f"  (одна и та же сцена, метки LEFT/RIGHT перемешаны — так выглядит случай)")
    print(f"  основные выше порога: {len(p_ok)} из {len(PRIMARY)}")
    print(f"  контроли выше порога: {len(c_ok)} из {len(CONTROLS)}")
    print(f"\n=== ВЕЛИЧИНА ЭФФЕКТА (смешано по seed) ===")
    print(f"  {'клетка':>12s} {'AUC':>7s} {'delta, Гц':>11s} {'разброс':>9s} "
          f"{'выше нуля':>10s}")
    for nm, side in PRIMARY + CONTROLS:
        k = (nm, side)
        if k not in pooled:
            continue
        v = pooled[k]
        print(f"  {nm + ' ' + side:>12s} {v['auc']:7.3f} {v['delta']:+11.3f} "
              f"{v['spread']:9.3f} {('да' if v['auc'] > thr else 'нет'):>10s}")
    pm = float(np.median([pooled[k]["delta"] for k in PRIMARY if k in pooled]))
    cm = float(np.median([abs(pooled[k]["delta"]) for k in CONTROLS
                          if k in pooled]))
    print(f"  медиана |delta|: {abs(pm):.3f} Гц у основных, {cm:.3f} у контролей "
          f"({abs(pm) / max(cm, 1e-9):.0f}x)")

    print(f"\n=== ИТОГ ===")
    print(f"  основные выше нуля: {len(p_ok)} из {len(PRIMARY)}, "
          f"знак совпал на двух seed: {p_sign}")
    print(f"  контроли выше нуля: {len(c_ok)} из {len(CONTROLS)}, "
          f"знак совпал: {c_sign}")
    if len(p_ok) >= 6 and len(c_ok) == 0:
        verdict = ("ЧИСТОЕ НАПРАВЛЕНИЕ ЧИТАЕТСЯ. На неподвижных кадрах, где яркость, "
                   "текстура и величина смещения одинаковы и отличается только знак "
                   "движения, основные клетки разделяют LEFT и RIGHT выше случайного "
                   f"уровня ({len(p_ok)} из {len(PRIMARY)}), а контроли остаются на "
                   "случайном. Но эффект скромный: медиана AUC "
                   f"{np.median([pooled[k]['auc'] for k in PRIMARY if k in pooled]):.2f} "
                   "против 0.94-1.00 на реальном видео. Значит прежние высокие числа "
                   "были сильно завышены различиями сцен, а доля собственно "
                   "направления в них невелика.")
    elif len(p_ok) >= 4:
        verdict = ("НАПРАВЛЕНИЕ ЧИТАЕТСЯ ЧАСТИЧНО: часть основных клеток устойчива к "
                   "чистой смене знака, часть нет")
    else:
        verdict = ("НАПРАВЛЕНИЕ НЕ ПОДТВЕРЖДАЕТСЯ: на чистом стимуле эффект "
                   "исчезает, значит прежние высокие AUC опирались на различия "
                   "реальных окон, а не на направление")
    print(f"\n  {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (OUT / "synthetic_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    (OUT / "report.json").write_text(json.dumps(clean({
        "seeds": list(data), "n_frames": n_frames,
        "stimulus": "неподвижный кадр, циклический сдвиг влево/вправо/без движения; "
                    "яркость, текстура, длительность и величина смещения одинаковы",
        "per_cell": rows, "verdict": verdict,
        "n_primary_above_null": len(p_ok), "n_control_above_null": len(c_ok),
        "null_median": float(np.median(null)), "null_p95": thr,
        "sign_agreement_primary": p_sign, "sign_agreement_control": c_sign,
        "pooled": {f"{k[0]} {k[1]}": v for k, v in pooled.items()},
        "controls_note": "контроли не обязаны проваливаться: нулевой вход от 48 "
                         "носителей P05 не исключает направления по другому пути",
        "caveat": "клетки различают движение и знак на этом стимуле; это не "
                  "доказывает, что они несут направление поворота камеры на видео",
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
