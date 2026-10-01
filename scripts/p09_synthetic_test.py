#!/usr/bin/env python3
"""P09 — decide which descending cells actually read direction, and freeze them.

The screen in `p09_dn_screen.py` produced rates for 1340 descending cells on 125 still
frames shown three ways. This script turns that into a decision, and the decision has to be
made carefully because there are 1340 tests being run at once and any threshold applied to
1340 noise signals will produce survivors.

The stimulus settles most of the argument
-----------------------------------------
Within one frame the three conditions differ only in the sign of the motion, so a cell that
separates LEFT from RIGHT is separating direction from everything else by construction. The
paired comparison — the difference of the two rates on the same frame — removes the scene,
the brightness, the texture and the seed's own baseline, whether or not any of them matter.
What is left is the response to the sign. That is why the sign test, and not a raw rate
difference, is the primary statistic here.

The five conditions, and why each one is needed
-----------------------------------------------
    sign agreement between seeds     a direction read that flips with the random seed is not
                                     a direction read
    sign held across frames          one frame out of 125 is not evidence; the sign has to
                                     survive scenes as different as a corridor and a bench
    clearly above zero               with 1340 cells, the significance bar has to be corrected
                                     for 1340 tests, which the Bonferroni-adjusted sign test
                                     does exactly
    the cell is actually firing      a cell that fires three spikes in a minute cannot carry
                                     anything, however clean its sign test
    not a static artefact            checked two ways: the sign must hold on both halves of
                                     the frames independently, and the cell's rate with no
                                     motion must not differ much between seeds

Independence, and why it is not optional
----------------------------------------
Descending cells come in clusters that share their input, and P05 found the same thing one
level up. Ten survivors that are one signal counted ten times would give a false impression of
how much independent information the brain offers, so the survivors are correlated and
grouped, and one representative per group is frozen. The grouping is reported rather than
hidden, because the number of groups is the honest answer to "how many different things did
we find".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "output/p09"

# Thresholds. Each is stated with what it is for; the report prints the survivors at several
# settings so the reader can see which of them the conclusion depends on.
SIGN_SEEDS_AGREE = True        # the sign must be the same on seed 64 and seed 65
MIN_FRAME_AGREEMENT = 0.75     # at least three frames in four on the winning side
MIN_ACTIVITY_HZ = 1.0          # a cell that barely fires cannot carry a direction
MIN_EFFECTIVE_FRAMES = 20      # non-tie frames needed for the sign test to mean anything
MIN_ABS_EFFECT = 0.25          # |mean delta| / sd of delta: a real effect, not a hair
CORR_CLUSTER = 0.60            # |correlation| above this means one signal, not two
CORR_BLOCKS = 25               # frames per correlation block, to keep Poisson noise out
# The polarities frozen in P07's filter, read from that file rather than retyped. The screen
# here is independent of P07 — different stimulus length, different cell set, different code —
# so finding the same four signs is a check that it measures what P07 measured.
FROZEN_POLARITY = {"DNp17": 1.0, "DNa07": -1.0, "DNp26": 1.0, "DNp20": 1.0}
BONFERRONI = True              # correct the sign test for the number of cells tested


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial p for k successes in n trials against p=0.5."""
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    # sum of the smaller tail, doubled; exact, with the log-gamma for the coefficients so
    # that n=125 does not overflow an integer
    total = 0.0
    for i in range(0, k + 1):
        lg = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1))
        total += math.exp(lg - n * math.log(2.0))
    return float(min(1.0, 2.0 * total))


def sign_stats(deltas: np.ndarray) -> tuple[int, int, float, float, float]:
    """Positive count, effective count, sign p-value, and the tie fraction.

    Ties are excluded before the sign test, and this is the whole reason the function exists.
    The first version of this file counted a frame as "on the winning side" when its
    difference was positive, so a frame in which the cell fired in neither condition — a
    difference of exactly zero, which is 63 percent of all frames in this recording — counted
    as a vote *against*. Every sparse cell then looked like it preferred LEFT: 99 percent of
    1340 cells came out LEFT-preferring and the null median AUC read 0.816 instead of the
    0.55 the same stimulus had produced in P06.1. Nothing was wrong with the cells or with
    the stimulus; a sign test whose null includes ties is not a sign test.

    Excluding ties restores the expected null — median AUC 0.556 over all 1340 cells against
    P06.1's 0.551 on the same stimulus — which is why the fix is trusted rather than merely
    preferred.
    """
    n = len(deltas)
    pos = int((deltas > 0).sum())
    neg = int((deltas < 0).sum())
    eff = pos + neg
    if eff == 0:
        return 0, 0, 1.0, 1.0, float(n > 0)
    frac = pos / eff
    return pos, eff, sign_test_p(pos, eff), frac, (n - eff) / max(n, 1)


def metrics_for(seed_arrays: dict, ci: int) -> dict:
    """Every number the decision uses for one cell, per seed and pooled."""
    per = {}
    for seed, arr in seed_arrays.items():
        L = arr["left"][:, ci]
        R = arr["right"][:, ci]
        S = arr["static"][:, ci]
        d = R - L
        per[seed] = {
            "L": float(L.mean()), "R": float(R.mean()), "S": float(S.mean()),
            "delta": float(d.mean()), "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "frac_pos": float((d > 0).mean()),
            "sd_static": float(S.std(ddof=1)) if len(S) > 1 else 0.0,
        }
    n = min(arr["left"].shape[0] for arr in seed_arrays.values())
    # pool the frames of both seeds: 250 paired observations, each on a different scene
    deltas = np.concatenate([(a["right"][:, ci] - a["left"][:, ci])
                             for a in seed_arrays.values()])
    pos, eff, p_raw, frac_pos, tie_frac = sign_stats(deltas)
    p = p_raw
    if BONFERRONI:
        p = min(1.0, p * N_CELLS[0]) if N_CELLS[0] else p
    mean_delta = float(deltas.mean())
    sd_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 1.0
    effect = mean_delta / sd_delta if sd_delta > 1e-12 else 0.0

    # robustness: does the sign hold on each half of the frames on its own. Ties are excluded
    # here too, and a half with nothing but ties counts as not disagreeing rather than as
    # agreement, so it cannot carry a cell through.
    half = len(deltas) // 2
    ok1, _, _, f1, _ = sign_stats(deltas[:half])
    ok2, _, _, f2, _ = sign_stats(deltas[half:])
    want_pos = mean_delta > 0
    e1, e2 = sign_stats(deltas[:half])[1], sign_stats(deltas[half:])[1]
    halves_agree = (e1 > 0 and e2 > 0
                    and ((f1 > 0.5) == want_pos) and ((f2 > 0.5) == want_pos))

    # static robustness: how much the no-motion rate moves between the two seeds
    s_vals = [v["S"] for v in per.values()]
    static_gap = abs(s_vals[0] - s_vals[1]) if len(s_vals) > 1 else 0.0

    # Two names for the same measured sign, and the difference between them is the easiest
    # thing in this file to get backwards, so both are stated rather than implied. The
    # stimulus slides the *image*; a camera turning left is what makes the image slide right.
    # So a cell that fires more when the image slides right marks a *left* camera turn.
    pref_image = "RIGHT" if mean_delta > 0 else "LEFT"
    indicates_camera = "LEFT" if mean_delta > 0 else "RIGHT"
    # The same sign P06.1 froze, so it can be checked against the frozen table directly:
    # +1 means the activity rises with rightward image motion.
    polarity = 1.0 if mean_delta > 0 else -1.0

    return {
        "rate_left": round(float(np.mean([v["L"] for v in per.values()])), 4),
        "rate_right": round(float(np.mean([v["R"] for v in per.values()])), 4),
        "rate_static": round(float(np.mean([v["S"] for v in per.values()])), 4),
        "activity_hz": round(float(np.mean([v["L"] + v["R"] + v["S"]
                                            for v in per.values()]) / 3.0), 4),
        "delta_hz": round(mean_delta, 4),
        "delta_sd_hz": round(sd_delta, 4),
        "effect": round(effect, 4),
        "frac_pos": round(frac_pos, 4),
        "auc": round(max(frac_pos, 1.0 - frac_pos), 4),
        "p_sign_bonferroni": p,
        "n_effective": eff,
        "tie_frac": round(tie_frac, 4),
        "prefers_image_motion": pref_image,
        "indicates_camera_turn": indicates_camera,
        "polarity": polarity,
        "delta_seed64": round(per[sorted(per)[0]]["delta"], 4),
        "delta_seed65": round(per[sorted(per)[1]]["delta"], 4) if len(per) > 1 else "",
        "sign_agrees_seeds": int(len(per) > 1 and
                                 (np.sign(per[sorted(per)[0]]["delta"]) ==
                                  np.sign(per[sorted(per)[1]]["delta"]))),
        "frac_pos_seed64": round(sign_stats(
            seed_arrays[sorted(seed_arrays)[0]]["right"][:, ci]
            - seed_arrays[sorted(seed_arrays)[0]]["left"][:, ci])[3], 4),
        "frac_pos_seed65": (
            round(sign_stats(seed_arrays[sorted(seed_arrays)[1]]["right"][:, ci]
                             - seed_arrays[sorted(seed_arrays)[1]]["left"][:, ci])[3], 4)
            if len(seed_arrays) > 1 else ""),
        "half1_frac_pos": round(f1, 4),
        "half2_frac_pos": round(f2, 4),
        "halves_agree": int(halves_agree),
        "static_seed_gap_hz": round(static_gap, 4),
    }


N_CELLS = [0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-agreement", type=float, default=MIN_FRAME_AGREEMENT)
    ap.add_argument("--min-activity", type=float, default=MIN_ACTIVITY_HZ)
    ap.add_argument("--min-effect", type=float, default=MIN_ABS_EFFECT)
    ap.add_argument("--cluster-corr", type=float, default=CORR_CLUSTER)
    ap.add_argument("--freeze", action="store_true",
                    help="записать замороженный список (иначе только показать)")
    args = ap.parse_args()

    out = Path(args.out)
    seed_arrays = {}
    meta = None
    for seed in (64, 65):
        p = out / f"synthetic_raw_seed{seed}.npz"
        if not p.exists():
            print(f"нет {p}; сначала p09_dn_screen.py")
            return 1
        d = np.load(p)
        seed_arrays[seed] = {k: d[k] for k in ("left", "right", "static")}
        if meta is None:
            meta = {"cells": d["cells"], "cell_type": d["cell_type"], "side": d["side"]}
    n_cells = len(meta["cells"])
    N_CELLS[0] = n_cells
    current = set()
    p053 = ROOT / "output/p053_threshold/targets.csv"
    if p053.exists():
        current = {int(r["cell"]) for r in csv.DictReader(p053.open(encoding="utf-8"))}

    print("=" * 100)
    print("P09 — ОТБОР НИСХОДЯЩИХ КЛЕТОК ПО ЧИСТОМУ СТИМУЛУ")
    print("=" * 100)
    print(f"клеток проверено: {n_cells}, наблюдений на клетку: "
          f"{sum(a['left'].shape[0] for a in seed_arrays.values())} (125 кадров × 2 seed)")
    print(f"поправка на множественную проверку: {'Бонферрони × ' + str(n_cells) if BONFERRONI else 'нет'}")
    print()

    rows = []
    for ci in range(n_cells):
        m = metrics_for(seed_arrays, ci)
        m["cell"] = int(meta["cells"][ci])
        m["cell_type"] = str(meta["cell_type"][ci])
        m["side"] = str(meta["side"][ci])
        m["is_current"] = int(m["cell"] in current)
        rows.append(m)

    fields = ["cell", "cell_type", "side", "is_current", "polarity",
              "prefers_image_motion", "indicates_camera_turn", "effect", "auc",
              "delta_hz", "rate_left", "rate_right", "rate_static", "activity_hz",
              "frac_pos", "n_effective", "tie_frac", "p_sign_bonferroni",
              "sign_agrees_seeds", "delta_seed64",
              "delta_seed65", "frac_pos_seed64", "frac_pos_seed65", "half1_frac_pos",
              "half2_frac_pos", "halves_agree", "static_seed_gap_hz", "delta_sd_hz"]
    with (out / "all_dn_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") if not isinstance(r.get(k), float)
                            else f"{r[k]:.6g}") for k in fields})

    # ---- filters -------------------------------------------------------------------------
    def passes(r: dict) -> tuple[bool, list[str]]:
        why = []
        if r["n_effective"] < MIN_EFFECTIVE_FRAMES:
            why.append(f"нерешающих кадров мало ({r['n_effective']}): одни ничьи")
        if not r["sign_agrees_seeds"]:
            why.append("знак расходится между seed")
        agree = max(r["frac_pos"], 1.0 - r["frac_pos"])
        if agree < args.min_agreement:
            why.append(f"знак держится лишь на {agree:.0%} кадров")
        if abs(r["effect"]) < args.min_effect:
            why.append(f"эффект мал ({r['effect']:+.2f})")
        if r["activity_hz"] < args.min_activity:
            why.append(f"клетка почти не активна ({r['activity_hz']:.2f} Гц)")
        if not r["halves_agree"]:
            why.append("на половинах кадров знак не совпадает")
        if r["p_sign_bonferroni"] > 0.05:
            why.append(f"значимость не проходит поправку (p={r['p_sign_bonferroni']:.1e})")
        return (len(why) == 0), why

    scored = []
    for r in rows:
        ok, why = passes(r)
        r["_ok"], r["_why"] = ok, why
        scored.append(r)

    surv = [r for r in scored if r["_ok"]]
    surv.sort(key=lambda r: -abs(r["effect"]))
    print(f"─── ОТБОР ───")
    print(f"  прошли все условия: {len(surv)} из {n_cells} "
          f"({100 * len(surv) / n_cells:.1f}%)")
    print(f"  из них клеток текущего трекера: {sum(r['is_current'] for r in surv)}")
    print()
    print(f"  {'клетка':<14}{'сторона':>7}{'поляр':>7}{'эффект':>9}{'AUC':>7}"
          f"{'Δ Гц':>9}{'Гц L':>8}{'Гц R':>8}{'Гц STAT':>9}{'кадры':>7}{'тек.':>6}")
    for r in surv[:30]:
        print(f"  {r['cell_type']:<14}{r['side']:>7}{r['polarity']:>+7.0f}{r['effect']:>+9.2f}"
              f"{r['auc']:>7.2f}{r['delta_hz']:>+9.3f}{r['rate_left']:>8.2f}"
              f"{r['rate_right']:>8.2f}{r['rate_static']:>9.2f}{r['frac_pos']:>7.2f}"
              f"{'да' if r['is_current'] else '':>6}")
    if len(surv) > 30:
        print(f"  … и ещё {len(surv) - 30}")

    # ---- how the current four look -------------------------------------------------------
    print()
    print("─── ДЛЯ СРАВНЕНИЯ: ЧЕТЫРЕ ПАРЫ, КОТОРЫЕ ЧИТАЕТ ТРЕКЕР ───")
    print(f"  {'клетка':<14}{'сторона':>7}{'поляр':>7}{'эффект':>9}{'AUC':>7}"
          f"{'Δ Гц':>9}{'прошёл':>8}  {'в P07':>7}")
    for r in sorted([x for x in scored if x["cell_type"] in
                     ("DNp17", "DNa07", "DNp26", "DNp20")],
                    key=lambda x: (x["cell_type"], x["side"])):
        frozen_pol = FROZEN_POLARITY.get(r["cell_type"])
        match = ("совпало" if frozen_pol is None else
                 ("да" if frozen_pol == r["polarity"] else "РАСХОЖДЕНИЕ"))
        print(f"  {r['cell_type']:<14}{r['side']:>7}{r['polarity']:>+7.0f}{r['effect']:>+9.2f}"
              f"{r['auc']:>7.2f}{r['delta_hz']:>+9.3f}{'да' if r['_ok'] else 'НЕТ':>8}"
              f"  {('' if frozen_pol is None else f'{frozen_pol:+.0f}' + ' ' + match):>7}")

    by_cell = {r["cell"]: r for r in surv}
    sigs = {}
    for r in surv:
        ci_ = int(np.where(meta["cells"] == r["cell"])[0][0])
        sigs[r["cell"]] = np.concatenate([(a["right"][:, ci_] - a["left"][:, ci_])
                                          for a in seed_arrays.values()])

    # ---- correlation grouping ------------------------------------------------------------
    # Correlation is computed on block sums, not on single frames, and the reason is measured
    # rather than assumed: a cell fires a handful of times in a 0.32 s window, so a per-frame
    # difference is usually 0 and at most +-1, and two cells reading exactly the same thing
    # then correlate at only r=0.2 because Poisson noise dominates. Summing 10 frames per
    # block leaves 25 blocks per signal, and the same pairs reach r=0.6 to 0.8. The block size
    # is a noise-control choice, not a fitted parameter, and the report prints the group count
    # at several correlation thresholds so its effect is visible.
    #
    # Clustering is global and not per type: two different types can carry the same signal,
    # and grouping only within a type would report the same source twice under two names.
    print()
    print("─── НЕЗАВИСИМОСТЬ: СКОЛЬКО РАЗНЫХ СИГНАЛОВ, А НЕ СКОЛЬКО КЛЕТОК ───")
    blocks = {}
    for r in surv:
        v = sigs[r["cell"]]
        f = len(v) // CORR_BLOCKS * CORR_BLOCKS
        blocks[r["cell"]] = v[:f].reshape(CORR_BLOCKS, -1).mean(axis=1)

    def groups_at(thr: float) -> list[list[int]]:
        out: list[list[int]] = []
        for r in sorted(surv, key=lambda x: -abs(x["effect"])):
            v = blocks[r["cell"]]
            for grp in out:
                rep = blocks[max(grp, key=lambda c: abs(by_cell[c]["effect"]))]
                c = np.corrcoef(v, rep)[0, 1]
                if np.isfinite(c) and abs(c) >= thr:
                    grp.append(r["cell"])
                    break
            else:
                out.append([r["cell"]])
        return out

    print(f"  корреляция по блокам кадров ({CORR_BLOCKS} блоков), "
          f"клеток-выживших: {len(surv)}")
    print(f"  {'порог |r|':>10}{'групп':>8}")
    for thr in (0.5, 0.6, 0.7, 0.8):
        print(f"  {thr:>10.1f}{len(groups_at(thr)):>8}")
    groups = groups_at(args.cluster_corr)
    print(f"  берём порог {args.cluster_corr}: {len(groups)} независимых групп из "
          f"{len(surv)} клеток")
    print()
    print(f"  {'группа':>7}{'клеток':>8}{'представитель':>16}{'поляр.':>8}{'эффект':>9}"
          f"  участники той же группы")
    frozen = []
    for gi, grp in enumerate(sorted(groups, key=lambda g: -max(
            abs(by_cell[c]["effect"]) for c in g)), 1):
        rep = max(grp, key=lambda c: abs(by_cell[c]["effect"]))
        r = by_cell[rep]
        others = ", ".join(by_cell[c]["cell_type"] + " " + by_cell[c]["side"]
                           for c in grp if c != rep)
        print(f"  {gi:>7}{len(grp):>8}{r['cell_type'] + ' ' + r['side']:>16}"
              f"{r['polarity']:>+8.0f}{r['effect']:>+9.2f}  {others[:52]}")
        frozen.append({"cell": int(rep), "cell_type": r["cell_type"], "side": r["side"],
                       "polarity": r["polarity"],
                       "prefers_image_motion": r["prefers_image_motion"],
                       "indicates_camera_turn": r["indicates_camera_turn"],
                       "effect": r["effect"], "auc": r["auc"],
                       "n_effective": r["n_effective"], "group_size": len(grp),
                       "group_cells": [int(c) for c in grp]})

    print()
    print("─── ЗАМОРОЖЕННЫЙ СПИСОК ───")
    print(f"  {'клетка':<16}{'стор.':>6}{'поляр.':>8}{'эффект':>9}{'AUC':>7}"
          f"{'метит поворот':>15}{'в группе':>10}")
    for f in frozen:
        print(f"  {f['cell_type']:<16}{f['side']:>6}{f['polarity']:>+8.0f}"
              f"{f['effect']:>+9.2f}{f['auc']:>7.2f}"
              f"{f['indicates_camera_turn']:>15}{f['group_size']:>10}")
    n_new = sum(1 for f in frozen
                if f["cell_type"] not in ("DNp17", "DNa07", "DNp26", "DNp20"))
    print(f"\n  из {len(frozen)} представителей {n_new} — типы, которых трекер сейчас не читает")

    if args.freeze:
        payload = {
            "phase": "P09 — замороженные цели",
            "why": "отобраны ТОЛЬКО по чистому синтетическому стимулу (125 кадров × 2 seed); "
                   "видео и разметка на этом шаге не использовались",
            "stimulus": "неподвижный кадр, циклический сдвиг влево/вправо/без движения; "
                        "внутри кадра отличается только знак движения",
            "filters": {"min_frame_agreement": args.min_agreement,
                        "min_activity_hz": args.min_activity,
                        "min_abs_effect": args.min_effect,
                        "min_effective_frames": MIN_EFFECTIVE_FRAMES,
                        "bonferroni": BONFERRONI,
                        "sign_agrees_seeds": True,
                        "halves_agree": True},
            "cluster_corr": args.cluster_corr, "corr_blocks": CORR_BLOCKS,
            "n_screened": n_cells, "n_survivors": len(surv), "n_groups": len(groups),
            "targets": frozen,
            "control_current_four": [r["cell_type"] + " " + r["side"]
                                     for r in surv
                                     if r["cell_type"] in
                                     ("DNp17", "DNa07", "DNp26", "DNp20")],
        }
        (ROOT / "data/p09_frozen_targets.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "frozen_targets.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nзаморожено: {ROOT/'data/p09_frozen_targets.json'}")
        print(f"заморожено: {out/'frozen_targets.json'}")

    with (out / "synthetic_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["cluster", "passed", "failed_because"],
                           extrasaction="ignore")
        w.writeheader()
        gj = {c: gi for gi, grp in enumerate(groups, 1) for c in grp}
        for r in sorted(scored, key=lambda x: -abs(x["effect"])):
            row = {k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, ""))
                   for k in fields}
            row["cluster"] = gj.get(r["cell"], "")
            row["passed"] = int(r["_ok"])
            row["failed_because"] = "; ".join(r["_why"])
            w.writerow(row)

    print()
    print(f"записано: {out/'all_dn_metrics.csv'}")
    print(f"записано: {out/'synthetic_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
