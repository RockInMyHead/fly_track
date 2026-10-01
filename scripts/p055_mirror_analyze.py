#!/usr/bin/env python3
"""
P05.5 — mirror control, analysis.

Control and mirror are the same run with one difference: every frame flipped
horizontally in the second. The labels describe the camera, so they do not change.

The statistic
-------------
For each recorded cell, its per-window rate is taken in both passes. Across the same
windows the two traces must disagree for a cell that reads the direction of image
motion:

    control pass, RIGHT turn   image moves right   cell responds fully
    mirror pass,  RIGHT turn   image moves left    cell responds weakly

so the traces go the other way and their correlation is negative. A cell that reads the
scene rather than the motion sees the same scene, only flipped, so its traces follow the
same windows and correlate positively. The window pairing is the test: shuffling it
gives the null.

Why the arousal measure is not the null
---------------------------------------
An earlier version of this script used `n_active`, the whole-brain spike count, as a
measure that "cannot know about direction", and required it not to flip. That premise is
wrong. The connectome is lateralised, so a mirrored image drives the two halves
differently and changes global activity as well. `n_active` is blind to the *label*, not
to the *direction of motion*, and it is therefore not a valid control.

The valid null is the control cells themselves: DNa02 and DNg100 receive 0.81 and 0.00
percent of their input from the depth-1 carriers, so whatever they do is what this
method produces in the absence of the path.

What the test does and does not establish
----------------------------------------
It establishes that the mirror pass reverses the response of the primary cells and not
of the controls. That is what direction selectivity looks like.

It does not separate "reads motion direction" from "reads a left-right asymmetry of the
static scene", because flipping the image reverses both. A cell locked to a static
left-side feature would flip the same way. The test that would separate those is a
motion-free stimulus or a moving-texture control, and neither exists for this clip.

Usage:
    PYTHONPATH=. python scripts/p055_mirror_analyze.py
    PYTHONPATH=. python scripts/p055_mirror_analyze.py --n-perm 20000
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

CONTROL = ROOT / "output/p055_mirror/trace_control.npz"
MIRROR = ROOT / "output/p055_mirror/trace_mirror.npz"
P053 = ROOT / "output/p053_threshold/spike_trace.npz"
TARGETS_CSV = ROOT / "output/p053_threshold/targets.csv"
OUT = ROOT / "output/p055_mirror"

PRIMARY = [("DNp17", "L"), ("DNp17", "R"), ("DNa07", "L"), ("DNa07", "R"),
           ("DNp26", "L"), ("DNp26", "R"), ("DNp20", "L"), ("DNp20", "R")]
CONTROLS = [("DNa02", "L"), ("DNg100", "L"), ("MDN", "L"), ("DNp01", "L")]


def load_windows() -> list[dict]:
    out = []
    for setp, verp in ((ROOT / "data/p01r/review_set.json",
                        ROOT / "data/p01r/turn_verdicts.json"),
                       (ROOT / "data/p01r/review_set_v2.json",
                        ROOT / "data/p01r/turn_verdicts_v2.json")):
        rs = json.loads(setp.read_text(encoding="utf-8"))["candidates"]
        vd = json.loads(verp.read_text(encoding="utf-8"))["verdicts"]
        for x in rs:
            if x.get("repeat_of"):
                continue
            v = vd.get(x["id"])
            if not v:
                continue
            truth = x["camera_direction"] if v["verdict"] == "correct" \
                else (v.get("actual_direction") or "REJECT")
            if truth in ("LEFT", "RIGHT"):
                out.append({"id": x["id"],
                            "round": 2 if "v2" in setp.name else 1,
                            "t0": x["t0"], "t1": x["t1"], "kind": truth})
    return sorted(out, key=lambda e: e["t0"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--metric", default="fire", choices=["fire", "v"])
    args = ap.parse_args()

    c = np.load(CONTROL)
    m = np.load(MIRROR)
    t = c["t"].astype(np.float64)
    dt = float(np.median(np.diff(t)))
    col = {i: x for i, x in enumerate(csv.DictReader(TARGETS_CSV.open()))}
    wins = load_windows()
    isR = np.array([w["kind"] == "RIGHT" for w in wins])
    n = len(wins)

    print("P05.5 — зеркальный контроль на реальном видео")
    print(f"  контроль {len(c['t']):,} шагов, зеркало {len(m['t']):,} шагов")
    print(f"  окон {n} ({int(isR.sum())} RIGHT, {int((~isR).sum())} LEFT), "
          f"величина: {'частота спайков' if args.metric == 'fire' else 'мембрана'}")

    # ---- the pipeline must be the same one that produced P05.3 -------------
    if P053.exists():
        p = np.load(P053)
        same = (len(p["t"]) == len(c["t"]) and
                np.array_equal(p["fired"], c["fired"]))
        print(f"\n=== 0. КОНТРОЛЬНЫЙ ПРОГОН ПРОТИВ P05.3 ===")
        print(f"  побитово совпадает: {same}")
        if not same:
            rs = []
            nn = min(len(p["t"]), len(c["t"]))
            for j in range(c["fired"].shape[1]):
                a, b = p["fired"][:nn, j].astype(float), c["fired"][:nn, j].astype(float)
                if a.std() > 0 and b.std() > 0:
                    rs.append(float(np.corrcoef(a, b)[0, 1]))
            print(f"  (не совпало, медиана корреляции {np.median(rs):+.3f})")

    def vals(arr: np.ndarray, j: int) -> np.ndarray:
        out = np.empty(n)
        for k, w in enumerate(wins):
            mask = (t >= w["t0"]) & (t <= w["t1"])
            if not mask.any():
                out[k] = np.nan
            elif args.metric == "fire":
                out[k] = float(np.mean(arr[mask, j])) / dt
            else:
                out[k] = float(np.median(arr[mask, j]))
        return out

    rng = np.random.default_rng(7)
    rows = []
    print(f"\n=== ТЕСТ: СОВПАДАЮТ ЛИ ОТКЛИКИ МЕЖДУ ПРОГОНАМИ ===")
    print(f"  нуль: перемешивание пар окон. p — доля перемешиваний не ближе к нулю,")
    print(f"  чем наблюдённая корреляция. {args.n_perm} перемешиваний\n")
    print(f"  {'клетка':>12s} {'группа':>10s} {'r':>8s} {'p (r<0)':>10s} "
          f"{'AUC(разн.)':>11s} {'p':>10s} {'вердикт':>14s}")

    for group, cells in (("основная", PRIMARY), ("контроль", CONTROLS)):
        for nm, side in cells:
            j = next((k for k, x in col.items()
                      if x["cell_type"] == nm and x["side"] == side), None)
            if j is None:
                continue
            vc, vm = vals(c["fired"], j), vals(m["fired"], j)
            ok = np.isfinite(vc) & np.isfinite(vm)
            if ok.sum() < 8 or vc[ok].std() == 0 or vm[ok].std() == 0:
                continue
            r = float(np.corrcoef(vc[ok], vm[ok])[0, 1])
            # null: break the window pairing between the two passes
            a = vc[ok] - vc[ok].mean()
            b = vm[ok] - vm[ok].mean()
            null = np.empty(args.n_perm)
            for k in range(args.n_perm):
                null[k] = float(np.dot(a, rng.permutation(b))
                                / (np.linalg.norm(a) * np.linalg.norm(b)))
            p_neg = float(np.mean(null <= r))

            # the difference between the passes, with the scene component cancelled
            d = vc - vm
            okd = np.isfinite(d)
            dR, dL = d[okd & isR], d[okd & ~isR]
            if len(dR) > 2 and len(dL) > 2:
                allv = np.concatenate([dR, dL])
                _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
                o = np.argsort(allv, kind="mergesort")
                rk = np.empty(len(allv))
                rk[o] = np.arange(1, len(allv) + 1)
                s = np.zeros(len(cnt))
                np.add.at(s, inv, rk)
                rk = (s / cnt)[inv]
                raw = (rk[: len(dR)].sum() - len(dR) * (len(dR) + 1) / 2) \
                    / (len(dR) * len(dL))
                auc_d = max(raw, 1.0 - raw)
                # null by shuffling the labels
                lab = np.concatenate([np.ones(len(dR), bool),
                                      np.zeros(len(dL), bool)])
                nd = np.concatenate([dR, dL])
                cnt_ge = 0
                for _ in range(min(args.n_perm, 4000)):
                    pl = rng.permutation(lab)
                    aa, bb = nd[pl], nd[~pl]
                    av = np.concatenate([aa, bb])
                    _, iv, cn = np.unique(av, return_inverse=True,
                                          return_counts=True)
                    oo = np.argsort(av, kind="mergesort")
                    rr = np.empty(len(av))
                    rr[oo] = np.arange(1, len(av) + 1)
                    ss = np.zeros(len(cn))
                    np.add.at(ss, iv, rr)
                    rr = (ss / cn)[iv]
                    x = (rr[: len(aa)].sum() - len(aa) * (len(aa) + 1) / 2) \
                        / (len(aa) * len(bb))
                    cnt_ge += (max(x, 1 - x) >= auc_d)
                p_auc = (cnt_ge + 1) / (min(args.n_perm, 4000) + 1)
            else:
                auc_d, p_auc = 0.5, 1.0

            if r < -0.3 and p_neg < 0.01:
                verdict = "НАПРАВЛЕННАЯ"
            elif r > 0.3:
                verdict = "на сцену"
            else:
                verdict = "неясно"
            print(f"  {nm + ' ' + side:>12s} {group:>10s} {r:+8.3f} {p_neg:10.4f} "
                  f"{auc_d:11.3f} {p_auc:10.4f} {verdict:>14s}")
            rows.append({"cell_type": nm, "side": side, "group": group,
                         "r_control_mirror": r, "p_negative": p_neg,
                         "auc_of_difference": auc_d, "p_auc": p_auc,
                         "verdict": verdict})

    # ---- group summary -----------------------------------------------------
    pr = [x for x in rows if x["group"] == "основная"]
    ct = [x for x in rows if x["group"] == "контроль"]
    print(f"\n=== ИТОГ ===")
    print(f"  основные: {len(pr)} каналов, медиана r "
          f"{np.median([x['r_control_mirror'] for x in pr]):+.3f}")
    print(f"  контроли: {len(ct)} каналов, медиана r "
          f"{np.median([x['r_control_mirror'] for x in ct]):+.3f}")
    n_pr_dir = sum(1 for x in pr if x["verdict"] == "НАПРАВЛЕННАЯ")
    n_ct_dir = sum(1 for x in ct if x["verdict"] == "НАПРАВЛЕННАЯ")
    print(f"  значимо направленных: {n_pr_dir} из {len(pr)} основных, "
          f"{n_ct_dir} из {len(ct)} контролей")
    med_pr = float(np.median([x["auc_of_difference"] for x in pr]))
    med_ct = float(np.median([x["auc_of_difference"] for x in ct]))
    print(f"  медиана AUC разности: {med_pr:.3f} у основных, {med_ct:.3f} у контролей")

    if n_pr_dir >= len(pr) - 1 and n_ct_dir == 0:
        verdict = ("РАЗЛИЧИЕ ПОДТВЕРЖДЕНО: отклик основных клеток переворачивается "
                   "вместе с зеркалом, у контролей не переворачивается. Но это "
                   "отделяет движение от сцены лишь частично — статическая "
                   "лево-правая асимметрия переворачивается так же.")
    elif n_ct_dir >= 1:
        verdict = ("НЕ РАЗЛИЧАЕТ: хотя бы один контроль ведёт себя как направленный, "
                   "значит метод ловит не только направление.")
    else:
        verdict = "СМЕШАННЫЙ РЕЗУЛЬТАТ: смотреть поимённо"
    print(f"\n  {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "mirror_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "report.json").write_text(json.dumps({
        "n_windows": n, "n_right": int(isR.sum()), "n_left": int((~isR).sum()),
        "n_perm": args.n_perm, "metric": args.metric,
        "per_cell": rows,
        "median_r_primary": float(np.median([x["r_control_mirror"] for x in pr])),
        "median_r_control": float(np.median([x["r_control_mirror"] for x in ct])),
        "n_primary_directional": n_pr_dir, "n_control_directional": n_ct_dir,
        "verdict": verdict,
        "method": "один и тот же прогон, отличается только горизонтальное отражение "
                  "кадра; метки описывают камеру и не меняются; нуль — перемешивание "
                  "пар окон между прогонами",
        "caveat": "тест отделяет направление движения от сцены лишь частично: "
                  "статическая лево-правая асимметрия сцены переворачивается при "
                  "зеркалировании так же, как движение. Разделяющий стимул — "
                  "текстура без движения или движение без смены сцены",
        "not_the_null": "n_active не годится как контроль: коннектом латерализован, "
                        "и зеркальная картинка меняет суммарную активность тоже",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
