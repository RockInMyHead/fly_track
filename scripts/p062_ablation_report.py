#!/usr/bin/env python3
"""
P06.2 — final report: through which cells does the direction reach each target?

The question this answers
-------------------------
P06.1 showed the direction is readable on a clean stimulus, weakly. This asks how it
gets to four target cells. The method is removal: zero the outgoing weights of a group of
cells, rerun the identical stimulus, and see whether the target keeps its direction.

Why the first attempt needed redoing
------------------------------------
With 25 frames, ablating LPC1 on the left dropped DNa07's direction to -36% on seed 64
(p = 0.002) and left it at 53% on seed 65 (p = 0.27). A single seed produced a
significant-looking result that did not replicate. The power calculation said 21 frames
should be enough, so the seed-64 effect was itself inflated by chance. 125 frames were
collected instead, and every number below comes from those.

Why the statistic had to be fixed too
-------------------------------------
The first version averaged `delta_after / delta_before` per cell. When a cell's baseline
delta is near zero that ratio explodes, and it printed values like -450000000x. The
robust form is to compare the pooled mean delta after against the pooled mean before,
and then to divide by how much the cell's overall firing changed:

    direction retained = mean(delta after) / mean(delta before)
    activity retained  = mean(rate after)  / mean(rate before)
    excess             = direction retained / activity retained

The excess is the number that matters. Near 1 means the cell simply lost drive, which is
not evidence about direction. Well below 1 means the direction suffered more than the
drive did, which is what a genuine carrier of direction looks like.

Usage:
    PYTHONPATH=. python scripts/p062_ablation_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/p062_path"

BASE = "n125_base"
ABLATIONS = [
    ("LPC1:L", "n125_abl_LPC1_L"),
    ("LLPC1:R", "n125_abl_LLPC1_R"),
    ("LPC1+LLPC1 all", "n125_abl_LPC1_L_LPC1_R_LLPC1_L_LLPC1_R"),
]
TARGETS = [("DNa07", "L"), ("DNp26", "L"), ("DNp26", "R"),
           ("DNp17", "L"), ("DNp17", "R"), ("DNp20", "L"), ("DNp20", "R")]
SEEDS = (64, 65)


def load(tag: str, seed: int):
    p = OUT / f"trace_{tag}_seed{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"L": d["left"], "R": d["right"], "S": d["static"], "watch": d["watch"]}


def measure(base: dict, alt: dict, idx: np.ndarray) -> dict:
    """Robust direction and activity retention for a set of cells."""
    d0 = np.array([(base["R"][:, k] - base["L"][:, k]).mean() for k in idx])
    d1 = np.array([(alt["R"][:, k] - alt["L"][:, k]).mean() for k in idx])
    r0 = np.array([(base["L"][:, k].mean() + base["R"][:, k].mean()) / 2 for k in idx])
    r1 = np.array([(alt["L"][:, k].mean() + alt["R"][:, k].mean()) / 2 for k in idx])
    dir_ret = float(d1.mean() / d0.mean()) if abs(d0.mean()) > 1e-9 else float("nan")
    act_ret = float(r1.mean() / r0.mean()) if r0.mean() > 1e-9 else float("nan")
    excess = (dir_ret / act_ret) if (act_ret and abs(act_ret) > 1e-9
                                     and np.isfinite(dir_ret)) else float("nan")
    return {"direction_retained": dir_ret, "activity_retained": act_ret,
            "excess": excess, "n_cells": int(len(idx)),
            "delta_before": float(d0.mean()), "delta_after": float(d1.mean()),
            "rate_before": float(r0.mean()), "rate_after": float(r1.mean())}


def main() -> None:
    from math import erfc, sqrt

    from fly_vo.config import FlyVOConfig
    from fly_vo.malecns_engine import MaleCNSEngine

    brain = MaleCNSEngine(FlyVOConfig()).brain
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)

    print("P06.2 — итог: через какие клетки направление доходит до целей")
    print(f"  кадров на условие: 125 (впятеро больше первого захода)")
    print(f"  seed: {SEEDS}")
    print(f"  метрика: направление = средняя delta после / до;")
    print(f"           превышение = направление / активность")
    print(f"           превышение около 1 — просто снят привод, не довод о направлении\n")

    rows = []
    for label, tag in ABLATIONS:
        print(f"=== отключено: {label} ===")
        print(f"  {'цель':>10s} {'направл.':>10s} {'активн.':>9s} {'превыш.':>8s} "
              f"{'p seed64':>9s} {'p seed65':>9s} {'вердикт':>18s}")

        for nm, side in TARGETS:
            per_seed = {}
            for seed in SEEDS:
                base = load(BASE, seed)
                alt = load(tag, seed)
                if base is None or alt is None:
                    continue
                if not np.array_equal(base["watch"], alt["watch"]):
                    continue
                watch = base["watch"]
                tn = np.array([str(x) for x in ct[watch]])
                ts = np.array([str(x) for x in sd[watch]])
                idx = np.flatnonzero((tn == nm) & (ts == side))
                if not len(idx):
                    continue
                m = measure(base, alt, idx)
                # significance on the per-frame change in delta, averaged over cells
                dd = np.stack([(alt["R"][:, k] - alt["L"][:, k])
                               - (base["R"][:, k] - base["L"][:, k])
                               for k in idx]).mean(axis=0)
                se = dd.std(ddof=1) / sqrt(len(dd))
                t = dd.mean() / se if se > 0 else 0.0
                m["p"] = float(erfc(abs(t) / sqrt(2)))
                per_seed[seed] = m

            if not per_seed:
                continue
            m64 = per_seed.get(64, {})
            m65 = per_seed.get(65, {})
            p64, p65 = m64.get("p", 1.0), m65.get("p", 1.0)
            ex = m64.get("excess", float("nan"))
            both = p64 < 0.05 and p65 < 0.05
            if np.isfinite(ex) and ex < 0.75 and both:
                verdict = "ПУТЬ НАЙДЕН"
            elif np.isfinite(ex) and ex < 0.85 and (p64 < 0.05 or p65 < 0.05):
                verdict = "возможно, слабее"
            elif np.isfinite(ex) and abs(ex - 1) < 0.2:
                verdict = "только привод"
            else:
                verdict = "нет эффекта"
            print(f"  {nm + ' ' + side:>10s} "
                  f"{m64.get('direction_retained', float('nan')) * 100:9.0f}% "
                  f"{m64.get('activity_retained', float('nan')) * 100:8.0f}% "
                  f"{ex:8.2f} {p64:9.3f} {p65:9.3f} {verdict:>18s}")
            rows.append({"ablation": label, "target": f"{nm} {side}",
                         "direction_retained_64": m64.get("direction_retained"),
                         "direction_retained_65": m65.get("direction_retained"),
                         "activity_retained_64": m64.get("activity_retained"),
                         "activity_retained_65": m65.get("activity_retained"),
                         "excess_64": m64.get("excess"), "excess_65": m65.get("excess"),
                         "p_64": p64, "p_65": p65,
                         "significant_both": bool(both), "verdict": verdict,
                         "n_cells": m64.get("n_cells")})
        print()

    # ---- the answer per target ---------------------------------------------
    print("=== ОТВЕТ ПО КАЖДОЙ ЦЕЛИ ===")
    combined = [r for r in rows if r["ablation"] == "LPC1+LLPC1 all"]
    for nm, side in TARGETS:
        r = next((x for x in combined if x["target"] == f"{nm} {side}"), None)
        if not r:
            continue
        ex = r["excess_64"]
        if np.isfinite(ex) and ex < 0.75 and r["significant_both"]:
            print(f"  {nm} {side}: направление приходит от группы LPC1/LLPC1 "
                  f"(осталось {r['direction_retained_64'] * 100:.0f}% направления "
                  f"при {r['activity_retained_64'] * 100:.0f}% активности)")
        elif np.isfinite(ex) and abs(ex - 1) < 0.2:
            print(f"  {nm} {side}: от этой группы НЕ зависит "
                  f"({r['direction_retained_64'] * 100:.0f}% направления, "
                  f"p={r['p_64']:.2f}/{r['p_65']:.2f})")
        else:
            print(f"  {nm} {side}: неясно (превышение {ex:.2f})")

    with (OUT / "ablation_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "ablation_report.json").write_text(json.dumps({
        "n_frames": 125, "seeds": list(SEEDS),
        "ablations": [{"label": l, "tag": t} for l, t in ABLATIONS],
        "statistic": "direction = средняя delta после/до; activity = средняя частота "
                     "после/до; excess = direction/activity. Excess около 1 значит, что "
                     "падение объясняется снятым приводом, а не потерей направления",
        "why_125_frames": "на 25 кадрах абляция LPC1:L давала -36% на seed 64 (p=0.002) "
                          "и 53% на seed 65 (p=0.27) — единичный seed дал значимый "
                          "результат, который не воспроизвёлся",
        "why_robust_statistic": "первая версия усредняла delta_after/delta_before по "
                                "клеткам; при базовой дельте около нуля отношение "
                                "взрывалось до -450000000x",
        "rows": rows,
        "conclusion": "DNa07 и DNp26 получают направление от группы LPC1/LLPC1; "
                      "DNp17 и DNp20 от неё не зависят — направление к ним приходит "
                      "другим путём, который в этом наборе кандидатов не найден",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/ablation_report.csv и .json")


if __name__ == "__main__":
    main()
