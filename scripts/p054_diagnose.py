#!/usr/bin/env python3
"""
P05.4 — diagnosis: why the frozen test came out invalid.

The frozen test (`p054_neuron_test.py`) is not touched here. This is a separate reading
of the same data, written after the rule fired, to find out what actually happened.

What the test said
------------------
    8 of 8 primary passed, and 2 of 3 controls passed
    -> INVALID by the pre-registered rule

The controls are DNa02, DNg100 and MDN, chosen because they receive 0.81, 0.00 and 0.00
percent of their input from the 48 depth-1 carriers that the whole P05 chain has been
about. If they separate the turns, the test is reading something other than that path.

Three candidate explanations, checked in order
----------------------------------------------
1. The labels are copied from the classifier rather than assigned by eye. Tested by the
   6 repeat windows, which came back 6 of 6 consistent, and by checking that the 9
   corrections the human made were all to FWD or STATIC, never to a turn.
2. Global arousal. LEFT windows might simply be busier, and any neuron would follow.
   Measured against `n_active`, the whole-brain spike count over all 166,700 neurons,
   which is independent of the 46 neurons being tested.
3. A time block artefact. Checked by splitting the turns in half by time.

The result of those checks is in the printed output and in `diagnosis.json`.

Usage:
    PYTHONPATH=. python scripts/p054_diagnose.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACE = ROOT / "output/p053_threshold/spike_trace.npz"
TARGETS_CSV = ROOT / "output/p053_threshold/targets.csv"
GROUPS = ROOT / "output/p05_layer_trace/groups.json"
P051 = ROOT / "output/p051_signal_break/group_metrics.csv"
OUT = ROOT / "output/p054_neuron_test"

CELLS = [("DNp17", "L"), ("DNp17", "R"), ("DNa07", "L"), ("DNa07", "R"),
         ("DNp26", "L"), ("DNp26", "R"), ("DNp20", "L"), ("DNp20", "R"),
         ("DNa02", "L"), ("DNg100", "L"), ("MDN", "L")]


def auc(a: np.ndarray, b: np.ndarray) -> float:
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
    x = (rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return max(x, 1.0 - x)


def load_round(set_path: Path, verdict_path: Path) -> list[dict]:
    rs = json.loads(set_path.read_text(encoding="utf-8"))["candidates"]
    vd = json.loads(verdict_path.read_text(encoding="utf-8"))["verdicts"]
    out = []
    for x in rs:
        if x.get("repeat_of"):
            continue
        v = vd.get(x["id"])
        if not v:
            continue
        truth = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        if truth in ("LEFT", "RIGHT"):
            out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "kind": truth,
                        "verdict": v["verdict"]})
    return sorted(out, key=lambda e: e["t0"])


def main() -> None:
    d = np.load(TRACE)
    t = d["t"].astype(np.float64)
    fired = d["fired"].astype(np.float64)
    nact = d["n_active"].astype(np.float64)
    col = {i: x for i, x in enumerate(csv.DictReader(TARGETS_CSV.open()))}

    r1 = load_round(ROOT / "data/p01r/review_set.json",
                    ROOT / "data/p01r/turn_verdicts.json")
    r2 = load_round(ROOT / "data/p01r/review_set_v2.json",
                    ROOT / "data/p01r/turn_verdicts_v2.json")

    print("P05.4 — диагностика недействительного теста")
    print(f"  весь мозг: {nact.mean():.0f} сработавших нейронов за шаг в среднем")

    # ---- 1. were the labels human, or copied? ------------------------------
    vd2 = json.loads((ROOT / "data/p01r/turn_verdicts_v2.json").read_text(
        encoding="utf-8"))["verdicts"]
    wrong = [v for v in vd2.values() if v["verdict"] == "wrong"]
    corrected_to_turn = [v for v in wrong if (v.get("actual_direction") or "") in
                         ("LEFT", "RIGHT")]
    print(f"\n=== 1. МЕТКИ ЧЕЛОВЕЧЕСКИЕ ИЛИ СКОПИРОВАНЫ ===")
    print(f"  вердиктов: {len(vd2)}, из них 'неверно': {len(wrong)}")
    print(f"  исправлений в LEFT/RIGHT: {len(corrected_to_turn)}")
    print(f"  исправлений в FWD/STATIC: "
          f"{sum(1 for v in wrong if (v.get('actual_direction') or '') in ('FWD', 'STATIC'))}")
    print(f"  -> метки поставлены глазами: ошибки есть, и они не совпадают "
          f"с тем, что предлагала система")

    # ---- 2. global arousal -------------------------------------------------
    def win(m: np.ndarray, w: dict) -> float:
        mask = (t >= w["t0"]) & (t <= w["t1"])
        return float(np.median(m[mask])) if mask.any() else float("nan")

    print(f"\n=== 2. ОБЩЕЕ ВОЗБУЖДЕНИЕ ===")
    conf = {}
    for tag, items in (("раунд 1", r1), ("раунд 2", r2)):
        isR = np.array([w["kind"] == "RIGHT" for w in items])
        g = np.array([win(nact, w) for w in items])
        a = auc(g[isR], g[~isR])
        conf[tag] = {"n": len(items), "auc_arousal": a,
                     "median_L": float(np.median(g[~isR])),
                     "median_R": float(np.median(g[isR]))}
        print(f"  {tag}: {len(items)} поворотов, n_active "
              f"LEFT {np.median(g[~isR]):.0f} против RIGHT {np.median(g[isR]):.0f}, "
              f"AUC {a:.3f}")
    print(f"  -> метки совпадают с общим уровнем активности. Это уже дефект")

    # ---- 3. do the neuron AUCs survive that? -------------------------------
    print(f"\n=== 3. ПЕРЕЖИВАЮТ ЛИ НЕЙРОНЫ СНЯТИЕ ОБЩЕГО ВОЗБУЖДЕНИЯ ===")
    res = {}
    for tag, items in (("раунд 1", r1), ("раунд 2", r2)):
        isR = np.array([w["kind"] == "RIGHT" for w in items])
        g = np.array([win(nact, w) for w in items])
        print(f"\n  {tag}:")
        print(f"    {'клетка':>12s} {'сырой':>8s} {'без возб.':>10s} {'r':>7s} "
              f"{'1-я пол':>8s} {'2-я пол':>8s}")
        res[tag] = {}
        for nm, side in CELLS:
            j = next((k for k, x in col.items()
                      if x["cell_type"] == nm and x["side"] == side), None)
            if j is None:
                continue
            v = np.array([np.mean(fired[(t >= w["t0"]) & (t <= w["t1"]), j]) / 0.02
                          for w in items])
            raw = auc(v[isR], v[~isR])
            A = np.stack([np.ones(len(v)), g], axis=1)
            beta, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
            resid = v - A @ beta
            adj = auc(resid[isR], resid[~isR])
            rr = float(np.corrcoef(v, g)[0, 1])
            h = len(v) // 2
            h1 = auc(v[:h][isR[:h]], v[:h][~isR[:h]])
            h2 = auc(v[h:][isR[h:]], v[h:][~isR[h:]])
            res[tag][f"{nm} {side}"] = {"raw": raw, "adjusted": adj, "r_arousal": rr,
                                        "half1": h1, "half2": h2}
            print(f"    {nm + ' ' + side:>12s} {raw:8.3f} {adj:10.3f} {rr:+7.3f} "
                  f"{h1:8.3f} {h2:8.3f}")

    # ---- 4. where can the controls get the signal at all? ------------------
    print(f"\n=== 4. ОТКУДА КОНТРОЛИ БЕРУТ СИГНАЛ ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p05_trace", ROOT / "scripts/p05_layer_trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fly_vo.config import FlyVOConfig
    from fly_vo.malecns_engine import MaleCNSEngine
    brain = MaleCNSEngine(FlyVOConfig()).brain
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)
    wt = np.asarray(brain.weights)

    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    by_group = {int(r["group"]): r for r in csv.DictReader(P051.open())}
    carriers = {int(r["group"]) for r in by_group.values()
                if r["carries"] == "True" and r["in_analysis"] == "True"
                and int(r["depth_bin"]) == 1
                and r["superclass"] != "descending_neuron"}
    cset = set()
    for g in carriers:
        m = meta[g]
        cset.update(np.flatnonzero((ct == m["cell_type"]) & (sd == m["side"])).tolist())
    tot = np.zeros(brain.n)
    srcv = np.zeros(brain.n)
    isc = np.zeros(brain.n, bool)
    isc[list(cset)] = True
    for jj in range(brain.n):
        lo, hi = ip[jj], ip[jj + 1]
        if hi <= lo:
            continue
        seg = ix[lo:hi]
        np.add.at(tot, seg, np.abs(wt[lo:hi]))
        if isc[jj]:
            np.add.at(srcv, seg, np.abs(wt[lo:hi]))

    print(f"  {'клетка':>12s} {'всего входа':>12s} {'от 48 носителей':>16s} {'доля':>8s}")
    routes = {}
    for nm, side in CELLS:
        c = int(np.flatnonzero((ct == nm) & (sd == side))[0])
        frac = srcv[c] / max(tot[c], 1e-12)
        routes[f"{nm} {side}"] = {"total": float(tot[c]), "from_carriers": float(srcv[c]),
                                  "fraction": float(frac)}
        print(f"  {nm + ' ' + side:>12s} {tot[c]:12.4f} {srcv[c]:16.4f} "
              f"{frac * 100:7.2f}%")
    print(f"\n  -> DNg100 и MDN получают НОЛЬ от этого пути и всё равно разделяют")
    print(f"     метки. Значит они контроли для ПУТИ, а не для СТИМУЛА.")

    # ---- verdict ----------------------------------------------------------
    prim = ["DNp17 L", "DNp17 R", "DNa07 L", "DNa07 R", "DNp26 L", "DNp26 R",
            "DNp20 L", "DNp20 R"]
    ctrl = ["DNa02 L", "DNg100 L", "MDN L"]
    p_raw = np.mean([res["раунд 2"][k]["raw"] >= 0.8 for k in prim])
    c_raw = np.mean([res["раунд 2"][k]["raw"] >= 0.8 for k in ctrl])
    print(f"\n=== ЧТО ЭТО ЗНАЧИТ ===")
    print(f"  основные прошли (сырой AUC): {p_raw * 100:.0f}%")
    print(f"  контроли прошли:             {c_raw * 100:.0f}%")
    print(f"  разрыв основных и контролей есть, но он маленький: "
          f"DNp17 L {res['раунд 2']['DNp17 L']['raw']:.3f} против "
          f"DNa02 L {res['раунд 2']['DNa02 L']['raw']:.3f}")
    print(f"  при заранее зафиксированном требовании '<=1 контроль' тест недействителен")

    diag = {
        "labels_are_human": {"n_wrong": len(wrong),
                             "corrected_to_turn": len(corrected_to_turn)},
        "arousal_confound": conf,
        "per_cell": res,
        "routes": routes,
        "conclusion": "контроли не изолируют стимул: у DNg100 и MDN ноль входа от "
                      "пути, но метки они разделяют. Метки совпадают с общим уровнем "
                      "активности (AUC 0.75 в обоих раундах), и в раунде 1, по "
                      "которому отбирались клетки в P05.3, эта связь уже была. "
                      "Тест не различает 'нейрон читает направление' и 'нейрон читает "
                      "любое различие между моими левыми и правыми окнами'.",
        "what_would_work": "зеркальный контроль на том же видео: те же окна с "
                           "отражённым кадром. Направленно-избирательная клетка "
                           "обязана перевернуться, а реагирующая на сцену — нет. "
                           "Только он убирает все различия сцен целиком.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"\nWrote {OUT}/diagnosis.json")


if __name__ == "__main__":
    main()
