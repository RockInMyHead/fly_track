#!/usr/bin/env python3
"""P09 follow-up — does anything at all happen at T50, or is there simply nothing there?

P09 established that the 25 new descending cells are indistinguishable from the four the
tracker already reads when judged on real camera turns, and that two of the three failures
are not cell failures. One of them still has two readings, and they point to different next
steps, so this script separates them.

At T50 the calibrated heading does not move at all in the decision window — zero degrees,
over 0.35 m of travel. If nothing in the recording changes at that moment, then T50 is not a
cell-selection problem and no rotation-based quantity can ever answer it. If instead cells do
change there, what they are responding to is not rotation, and that would be a lead worth
following. The difference decides whether the next phase is about cells at all.

Nothing is refitted and no P09 artifact is modified; this only reads the traces P09 froze and
asks a different question of them. The question is deliberately direction-free: not "which
way" but "did anything move", because at T50 the sign is not what is in doubt.

The statistic
-------------
For each cell, its rate is smoothed and z-scored over the clip as in P09, and then the
magnitude of its mean deviation is computed for every 3.2 s window. Comparing the window at
the junction against all windows of the clip gives a percentile: a cell at the 95th percentile
deviated more at this junction than in 95 percent of the clip. Under no response, five cells
in a hundred reach that by construction, which is the null this is measured against.
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

OUT = ROOT / "output/p09_followup"
TRACE = ROOT / "output/p09/trace_VID00006.npz"
FROZEN = ROOT / "data/p09_frozen_targets.json"
OLD_FOUR = ("DNp17", "DNa07", "DNp26", "DNp20")
JUNCTIONS = [("J35", 113.74, 102.3, "RIGHT"),
             ("T50", 186.14, 0.0, "RIGHT"),
             ("T49", 197.50, 34.5, "RIGHT")]
NULL_PCT = 95.0            # what "deviated" means, and the rate a null cell reaches it


def load_cell_meta() -> tuple[dict, csv.DictReader]:
    frozen_ids = {int(x["cell"]) for x in json.loads(
        FROZEN.read_text(encoding="utf-8"))["targets"]}
    pol = {}
    for r in csv.DictReader((ROOT / "output/p09/all_dn_metrics.csv").open(encoding="utf-8")):
        pol[int(r["cell"])] = float(r["polarity"])
    return {"frozen_ids": frozen_ids, "pol": pol}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(TRACE)
    t = d["t"].astype(float)
    fired = d["fired"]
    cells = d["cells"].astype(int)
    ctype = [str(x) for x in d["cell_type"]]
    cside = [str(x) for x in d["side"]]
    dt = float(np.median(np.diff(t)))
    meta = load_cell_meta()
    frozen_ids, pol_by_cell = meta["frozen_ids"], meta["pol"]

    # the watch list held the controls first and the candidates after, and a few cells are in
    # both; keep one column per cell so nothing is counted twice
    keep, seen = [], set()
    for k, c in enumerate(cells):
        if int(c) in seen:
            continue
        seen.add(int(c))
        keep.append(k)
    fired, cells = fired[:, keep], cells[keep]
    ctype = [ctype[k] for k in keep]
    cside = [cside[k] for k in keep]
    pol = np.array([pol_by_cell.get(int(c), np.nan) for c in cells])
    group = np.array(["new" if int(c) in frozen_ids else
                      ("old_four" if ctype[k] in OLD_FOUR else "control")
                      for k, c in enumerate(cells)])

    z, integ, floor = H.cell_integrals(fired, t, dt)
    w = int(round(H.WINDOW_S / dt))

    # |mean z| for every window of the clip, per cell — the null distribution each junction is
    # read against
    csum = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), axis=0)
    roll = np.abs((csum[w:] - csum[:-w]) / w)

    print("=" * 96)
    print("P09 — ДОПОЛНИТЕЛЬНО: ЧТО ВООБЩЕ ПРОИСХОДИТ В ЭТИХ ТОЧКАХ")
    print("=" * 96)
    print(f"  клеток в записи VID00006: {len(cells)} "
          f"(новых {int((group == 'new').sum())}, "
          f"четыре типа трекера {int((group == 'old_four').sum())}, "
          f"прочих {int((group == 'control').sum())})")
    print(f"  окно {H.WINDOW_S} с, порог отклонения — собственный {NULL_PCT:.0f}-й процентиль")
    print(f"  нуль: случайная клетка проходит порог в 5% окон, "
          f"то есть {0.05 * len(cells):.1f} клетки из {len(cells)}")
    print()

    rows = []
    for name, t0, rot, want in JUNCTIONS:
        i = int(np.searchsorted(t, t0 - H.WINDOW_S, side="left"))
        i = min(max(i, 0), roll.shape[0] - 1)
        dev = roll[i]
        pct = np.array([(roll[:, k] < dev[k]).mean() * 100.0 for k in range(len(cells))])
        # direction, for the cells that do respond
        Iv = H.window_integral(integ, t, t0)

        n_above = int((pct >= NULL_PCT).sum())
        print(f"─── {name}  t={t0:.1f} с   поворот камеры {rot:.1f}°   нужно {want} ───")
        print(f"  клеток выше порога: {n_above} из {len(cells)}   "
              f"(при отсутствии отклика ожидается ~{0.05 * len(cells):.1f})")
        for g in ("new", "old_four", "control"):
            m = group == g
            nz = int((pct[m] >= NULL_PCT).sum())
            print(f"    {g:<10} {nz:>3} из {int(m.sum()):>3}"
                  f"   {100 * nz / max(int(m.sum()), 1):>5.1f}%")
        top = np.argsort(-pct)[:8]
        print(f"    {'клетка':<16}{'стор.':>6}{'группа':>10}{'отклонение':>12}"
              f"{'процентиль':>12}{'говорит':>10}{'нужный знак':>13}")
        for k in top:
            says = ""
            if abs(Iv[k]) >= floor[k]:
                says = "LEFT" if (Iv[k] > 0) == (pol[k] > 0) else "RIGHT"
            print(f"    {ctype[k]:<16}{cside[k]:>6}{group[k]:>10}{dev[k]:>12.3f}"
                  f"{pct[k]:>11.1f}%{says or '—':>10}"
                  f"{('да' if says == want else 'нет') if says else '—':>13}")
        for k in range(len(cells)):
            rows.append({"node": name, "t": t0, "camera_rotation_deg": rot,
                         "needed": want, "cell": int(cells[k]), "cell_type": ctype[k],
                         "side": cside[k], "group": group[k],
                         "deviation": round(float(dev[k]), 4),
                         "percentile": round(float(pct[k]), 2),
                         "above_null": int(pct[k] >= NULL_PCT),
                         "integral": round(float(Iv[k]), 4),
                         "floor": round(float(floor[k]), 4),
                         "says": ("LEFT" if (Iv[k] > 0) == (pol[k] > 0) else "RIGHT")
                                 if abs(Iv[k]) >= floor[k] else ""})
        print()

    with (OUT / "junction_response.csv").open("w", encoding="utf-8", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)

    # ---- the answer, stated as the two alternatives it separates --------------------------
    summary = {}
    for name, t0, rot, want in JUNCTIONS:
        sub = [r for r in rows if r["node"] == name]
        summary[name] = {
            "camera_rotation_deg": rot,
            "n_cells": len(sub),
            "n_above_null": sum(r["above_null"] for r in sub),
            "expected_by_chance": round(0.05 * len(sub), 2),
            "n_new_above": sum(r["above_null"] for r in sub if r["group"] == "new"),
            "n_old_four_above": sum(r["above_null"] for r in sub if r["group"] == "old_four"),
            "n_saying_needed_sign": sum(1 for r in sub if r["says"] == want),
            "n_saying_anything": sum(1 for r in sub if r["says"]),
        }

    print("─── ВЫВОД ПО КАЖДОЙ ТОЧКЕ ───")
    for name, s in summary.items():
        ratio = s["n_above_null"] / max(s["expected_by_chance"], 1e-9)
        if s["n_above_null"] <= s["expected_by_chance"] * 1.5:
            verdict = ("ничего не происходит: откликов не больше случайных, "
                       "поворот не с чем сравнивать")
        elif s["n_saying_needed_sign"] == 0:
            verdict = ("отклики есть, но ни один не называет нужный знак")
        else:
            verdict = f"есть отклик и нужный знак у {s['n_saying_needed_sign']} клеток"
        print(f"  {name:<6} поворот {s['camera_rotation_deg']:>6.1f}°  "
              f"откликов {s['n_above_null']:>3} (случайно {s['expected_by_chance']:.1f}, "
              f"×{ratio:.1f})  говорят нужное {s['n_saying_needed_sign']}")
        print(f"         {verdict}")

    (OUT / "report.json").write_text(json.dumps({
        "phase": "P09 follow-up — есть ли в этих точках хоть какой-то отклик",
        "reads": [str(TRACE), str(FROZEN)],
        "modifies_nothing_in_p09": True,
        "window_s": H.WINDOW_S, "null_percentile": NULL_PCT,
        "junctions": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"записано: {OUT/'junction_response.csv'}")
    print(f"записано: {OUT/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
