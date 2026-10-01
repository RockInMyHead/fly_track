#!/usr/bin/env python3
"""
P06.2 — which cells carry the direction, and does removing one break the target?

Two stages, and the second is what makes the first mean anything.

Stage 1: who is directional?
----------------------------
For all 2,392 watched cells, delta = rate(RIGHT) - rate(LEFT) per frame, on the same 25
still frames P06.1 used. A cell is kept if

    the sign of delta repeats across frames
    the sign agrees between seed 64 and seed 65

By type, cells are then ranked by the size of the effect and by how many of them are
directional. Types are then ordered by their total weight into the target cells, so the
list is about the path rather than about activity alone.

Stage 2: does the target still work without it?
----------------------------------------------
Correlation is not causation: a directional cell feeding the target may still be
irrelevant if the target gets the same information elsewhere. The test is removal. Zero
the outgoing weights of one type, rerun the identical stimulus, and watch the target's
own delta:

    target delta collapses   that type is on the path
    target delta survives    the type was a passenger, not a carrier

The comparison is against the base run on the same seed and the same frames, so the only
thing that changed is the ablation.

Usage:
    PYTHONPATH=. python scripts/p062_path_analyze.py --stage rank
    PYTHONPATH=. python scripts/p062_path_analyze.py --stage ablation
    PYTHONPATH=. python scripts/p062_path_analyze.py --stage all
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/p062_path"
TARGET_TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]


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


def load(tag: str, seed: int):
    p = OUT / f"trace_{tag}_seed{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"L": d["left"], "R": d["right"], "S": d["static"],
            "watch": d["watch"], "depth": d["depth"]}


def delta_stats(base: dict, seed: int) -> dict:
    """Per-cell delta, its sign consistency across frames, and its AUC."""
    L, R = base["L"], base["R"]
    d = R - L
    sign = np.sign(d.mean(axis=0))
    nz = np.abs(d) > 1e-9
    with np.errstate(invalid="ignore"):
        cons = np.where(nz.sum(axis=0) > 0,
                        (np.sign(d) == sign[None, :]).sum(axis=0)
                        / np.maximum(nz.sum(axis=0), 1), 0.0)
    auc = np.array([max(auc_raw(R[:, i], L[:, i]), 1 - auc_raw(R[:, i], L[:, i]))
                    for i in range(L.shape[1])])
    return {"delta": d.mean(axis=0), "consistency": cons, "auc": auc,
            "spread": d.std(axis=0), "nz": nz.sum(axis=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["rank", "ablation", "all"])
    ap.add_argument("--types", type=int, default=8,
                    help="how many ablation targets to summarise")
    args = ap.parse_args()

    b64 = load("base", 64)
    b65 = load("base", 65)
    if b64 is None:
        print("нет базовой трассировки")
        return

    from fly_vo.config import FlyVOConfig
    from fly_vo.malecns_engine import MaleCNSEngine
    brain = MaleCNSEngine(FlyVOConfig()).brain
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    watch = b64["watch"]
    tname = np.array([str(x) for x in ct[watch]])
    tside = np.array([str(x) for x in sd[watch]])

    s64 = delta_stats(b64, 64)
    s65 = delta_stats(b65, 65) if b65 is not None else None

    # a cell is directional if it holds a sign across frames and that sign repeats
    good = (s64["consistency"] >= 0.75) & (s64["auc"] > 0.60)
    if s65 is not None:
        good &= (s65["consistency"] >= 0.75) & (np.sign(s64["delta"]) ==
                                                np.sign(s65["delta"]))
        good &= (s65["auc"] > 0.60)

    print("P06.2 — кто несёт направление к целевым нисходящим нейронам")
    print(f"  наблюдаемых клеток: {len(watch)}")
    print(f"  базовая трассировка: seed 64"
          f"{' и 65' if b65 is not None else ''}")
    print(f"  направленных клеток (знак держится на кадрах и совпал между seed): "
          f"{int(good.sum())}")

    # ---- per type ----------------------------------------------------------
    groups = collections.defaultdict(list)
    for i in range(len(watch)):
        groups[(tname[i], tside[i])].append(i)

    # weight each type sends into the target cells, for ordering the path
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)
    wt = np.asarray(brain.weights)
    tgt_cells = np.array(sorted({int(c) for nm in TARGET_TYPES
                                 for c in np.flatnonzero(ct == nm)}), dtype=np.int64)
    tpos = np.full(brain.n, -1, np.int64)
    tpos[tgt_cells] = np.arange(len(tgt_cells))
    to_target = collections.Counter()
    for i in range(brain.n):
        a, z = ip[i], ip[i + 1]
        if z <= a:
            continue
        if np.any(tpos[ix[a:z]] >= 0):
            to_target[(str(ct[i]), str(sd[i]))] += 1

    rows = []
    for (nm, side), idxs in groups.items():
        if nm == "":
            continue
        idxs = np.array(idxs)
        n_dir = int(good[idxs].sum())
        rows.append({
            "cell_type": nm, "side": side, "n_watched": int(len(idxs)),
            "n_directional": n_dir,
            "frac_directional": n_dir / max(len(idxs), 1),
            "mean_abs_delta": float(np.mean(np.abs(s64["delta"][idxs]))),
            "median_auc": float(np.median(s64["auc"][idxs])),
            "median_consistency": float(np.median(s64["consistency"][idxs])),
            "synapses_to_targets": int(to_target.get((nm, side), 0)),
            "is_target": bool(nm in TARGET_TYPES),
        })
    rows.sort(key=lambda r: (-r["n_directional"], -r["synapses_to_targets"]))

    print(f"\n=== ТИПЫ, ГДЕ ЕСТЬ НАПРАВЛЕННЫЕ КЛЕТКИ (топ 20) ===")
    print(f"  {'тип':>16s} {'стор':>5s} {'набл':>6s} {'направл':>8s} {'доля':>7s} "
          f"{'|delta|':>8s} {'AUC':>6s} {'к цели':>7s}")
    for r in rows[:20]:
        if r["n_directional"] == 0 and r["synapses_to_targets"] == 0:
            continue
        print(f"  {r['cell_type'][:16]:>16s} {r['side']:>5s} {r['n_watched']:6d} "
              f"{r['n_directional']:8d} {r['frac_directional'] * 100:6.0f}% "
              f"{r['mean_abs_delta']:8.3f} {r['median_auc']:6.3f} "
              f"{r['synapses_to_targets']:7d}")

    # ---- the targets themselves -------------------------------------------
    print(f"\n=== САМИ ЦЕЛЕВЫЕ НЕЙРОНЫ ===")
    print(f"  {'клетка':>12s} {'delta':>9s} {'знак кадр':>10s} {'AUC':>7s} "
          f"{'движ>покой':>11s}")
    for nm in TARGET_TYPES:
        for side in ("L", "R"):
            for i in np.flatnonzero((tname == nm) & (tside == side)):
                motion = (b64["L"][:, i].mean() + b64["R"][:, i].mean()) / 2
                print(f"  {nm + ' ' + side:>12s} {s64['delta'][i]:+9.3f} "
                      f"{s64['consistency'][i] * 100:9.0f}% {s64['auc'][i]:7.3f} "
                      f"{('да' if motion > b64['S'][:, i].mean() else 'нет'):>11s}")

    # ---- ablation ----------------------------------------------------------
    if args.stage in ("ablation", "all"):
        print(f"\n=== АБЛЯЦИЯ: выключить тип и посмотреть на цель ===")
        print(f"  цель: сдвиг delta у целевых клеток относительно базового прогона")
        print(f"  (тот же seed, тот же стимул, отличается только отключённый тип)\n")

        # align by absolute cell index, not by position, so a difference in the watch
        # list could never be mistaken for an effect of the ablation
        base_pos = {int(c): i for i, c in enumerate(watch)}
        target_idx = {}
        for nm in TARGET_TYPES:
            for side in ("L", "R"):
                for i in range(len(watch)):
                    if tname[i] == nm and tside[i] == side:
                        target_idx.setdefault((nm, side), []).append(i)

        base_delta = s64["delta"]

        abl_rows = []
        for r in rows:
            if r["is_target"]:
                continue
            tag = f"abl_{r['cell_type'][:12]}_{r['side']}"
            tr = load(tag, 64)
            if tr is None:
                continue
            # the watched set must match, otherwise the comparison is meaningless
            if len(tr["watch"]) != len(watch) or not np.array_equal(tr["watch"], watch):
                print(f"  {tag}: набор клеток отличается, пропущен")
                continue
            a = delta_stats(tr, 64)
            rec = {"type": r["cell_type"], "side": r["side"], "tag": tag,
                   "n_directional_in_type": r["n_directional"],
                   "synapses_to_targets": r["synapses_to_targets"],
                   "type_mean_abs_delta": r["mean_abs_delta"]}
            for key, idxs in target_idx.items():
                b = float(np.mean(base_delta[idxs]))
                q = float(np.mean(a["delta"][idxs]))
                rec[f"delta_before_{key[0]}_{key[1]}"] = b
                rec[f"delta_after_{key[0]}_{key[1]}"] = q
                rec[f"retained_{key[0]}_{key[1]}"] = (q / b) if abs(b) > 1e-9 else np.nan
            # one summary number: how much of the target direction survives, averaged
            # over the four target types
            ret = [rec[f"retained_{nm}_{sd2}"] for nm in TARGET_TYPES
                   for sd2 in ("L", "R") if f"retained_{nm}_{sd2}" in rec]
            ret = [x for x in ret if np.isfinite(x)]
            rec["mean_retained"] = float(np.mean(ret)) if ret else np.nan
            abl_rows.append(rec)

        if not abl_rows:
            print("  ни один прогон с абляцией не найден")
            print("  команда: PYTHONPATH=. python scripts/p062_path_trace.py "
                  "--tag abl_<тип>_<стор> --seed 64 --ablate <тип>:<стор>")
        else:
            abl_rows.sort(key=lambda r: r["mean_retained"])
            print(f"  {'отключено':>18s} {'направл':>8s} {'к цели':>7s} "
                  f"{'delta %':>9s} {'вердикт':>22s}")
            for r in abl_rows:
                m = r["mean_retained"]
                if not np.isfinite(m):
                    v = "—"
                elif m < 0.5:
                    v = "СИГНАЛ ПАДАЕТ"
                elif m < 0.8:
                    v = "заметно слабее"
                elif m <= 1.25:
                    v = "не изменился"
                else:
                    v = "стало сильнее"
                print(f"  {(r['type'] + ':' + r['side'])[:18]:>18s} "
                      f"{r['n_directional_in_type']:8d} {r['synapses_to_targets']:7d} "
                      f"{m * 100:8.0f}% {v:>22s}")

            print(f"\n  по каждому целевому типу (сколько delta осталось):")
            hdr = f"  {'отключено':>18s}"
            for nm in TARGET_TYPES:
                hdr += f" {nm:>9s}"
            print(hdr)
            for r in abl_rows:
                line = f"  {(r['type'] + ':' + r['side'])[:18]:>18s}"
                for nm in TARGET_TYPES:
                    vals = [r.get(f"retained_{nm}_{sd2}") for sd2 in ("L", "R")]
                    vals = [x for x in vals if x is not None and np.isfinite(x)]
                    line += (f" {np.mean(vals) * 100:8.0f}%"
                             if vals else f" {'—':>9s}")
                print(line)

            # the control: a type with few directional cells should change little
            ctrl = [r for r in abl_rows if r["n_directional_in_type"] <= 6]
            strong = [r for r in abl_rows if r["n_directional_in_type"] > 20]
            print(f"\n  типы с >20 направленными клетками: "
                  f"медиана сохранившегося {np.median([r['mean_retained'] for r in strong]) * 100:.0f}%"
                  if strong else "")
            if ctrl:
                print(f"  типы с <=6 направленными (контроль): "
                      f"медиана сохранившегося "
                      f"{np.median([r['mean_retained'] for r in ctrl]) * 100:.0f}%")
            print(f"\n  если абляция направленного типа роняет сигнал, а абляция "
                  f"ненаправленного (но связанного с целью) нет —")
            print(f"  значит первый действительно на пути, а разница не от общей "
                  f"поломки сети")

        with (OUT / "ablation.csv").open("w", newline="") as f:
            if abl_rows:
                w = csv.DictWriter(f, fieldnames=list(abl_rows[0].keys()))
                w.writeheader()
                for r in abl_rows:
                    w.writerow(r)

    # ---- outputs -----------------------------------------------------------
    keys = list(rows[0].keys())
    with (OUT / "type_ranking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

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

    (OUT / "ranking.json").write_text(json.dumps(clean({
        "n_watched": int(len(watch)),
        "n_directional_cells": int(good.sum()),
        "n_types_with_directional": int(sum(1 for r in rows
                                            if r["n_directional"] > 0)),
        "top_types": rows[:40],
        "criterion": "знак delta держится на >=75% кадров и совпал между seed 64 и 65; "
                     "AUC > 0.60 на обоих",
        "stimulus": "тот же, что в P06.1: 25 неподвижных кадров, циклический сдвиг",
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
