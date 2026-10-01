#!/usr/bin/env python3
"""P11.B control — is the procedure that found 0.749 honest, or does it inflate anything?

The candidate screen reported 0.749 out-of-fold with p = 0.0011. A number that good deserves to be
checked against a procedure-identical run on inputs whose answer is already known, because there
are two ways it could be wrong and only the first is caught by a permutation null:

    the screen finds something   the null permutes labels and re-runs everything, so a number
                                 above its p95 is not a fluke of this particular labelling
    the screen always finds      a screen that selects about a hundred pairs per fold and averages
    something                    them will return a tidy number even on data with no signal.
                                 A permutation null does not catch that, because permuting labels
                                 makes the screen select *different* pairs, and the average of
                                 many weakly-chosen pairs can still land above its own null by
                                 construction rather than by content.

The second failure is the dangerous one, and the way to rule it out is not more statistics but the
same statistics on a known-negative input. Three runs, identical in every step:

    the 68 descending cells         P10 measured MOVE/NO_NET at 0.560 here, and at 0.560 with the
                                    full per-cell model. It should stay near chance.
    the 20 early visual populations P10 measured 0.695 here. The procedure should reproduce it.
    the P11.B candidates            the number under test.

If the first comes out near chance and the second near 0.70, the procedure is calibrated and the
candidate number means what it says. If the first also comes out near 0.75, the screen manufactures
signal and the candidate result must be withdrawn.

Usage:
    PYTHONPATH=. python scripts/p11b_control.py [--perm 300] [--null-seeds 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.p11a_route_change import FEATS, auc_cols, balanced_clips, restricted_auc  # noqa
from scripts.p11b_analyze import (BIN_S, FEATS as BFEATS, AFTER, BEFORE, DURING,  # noqa
                                  load_groups, rate_features, screen_pairs, series_of)

OUT = ROOT / "output/p11b"


# ------------------------------------------------------------------ the shared procedure

def readout(X: np.ndarray, y: np.ndarray, clips: np.ndarray, uniq: list[str], nf: int,
            n_cells: np.ndarray, min_auc: float, min_agree: float) -> np.ndarray:
    """Every surviving pair of each fold, each with weight one. No row is ever picked."""
    clip_list = sorted(set(clips.tolist()))
    oof = np.full(len(y), np.nan)
    for c in clip_list:
        train, test = clips != c, clips == c
        sel = screen_pairs(y, X, clips, uniq, nf, n_cells, BFEATS, train, min_auc, min_agree)
        if not sel:
            continue
        Z = np.stack([X[:, uniq.index(s["group"]) * nf + BFEATS.index(s["feature"])]
                      for s in sel], axis=1)
        mu, sd = Z[train].mean(axis=0), Z[train].std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        oof[test] = (((Z[test] - mu) / sd) * np.array([s["sign"] for s in sel])).mean(axis=1)
    return oof


def evaluate(X, y, clips, uniq, nf, n_cells, min_auc, min_agree, perm, seeds, seed) -> dict:
    oof = readout(X, y, clips, uniq, nf, n_cells, min_auc, min_agree)
    m = ~np.isnan(oof)
    obs = float(auc_cols(y[m], oof[m, None])[0]) if m.sum() else float("nan")

    # Per-fold AUC, then averaged. The pooled number above ranks held-out scores from folds that
    # used different feature sets and different standardisation, so it mixes quantities that are
    # not on a common scale; if the two disagree, the pooled one is the one to distrust. Folds
    # holding a single class cannot be scored and are reported as such rather than dropped
    # silently.
    per_fold, fold_skip = {}, {}
    for c in sorted(set(clips.tolist())):
        mm = (clips == c) & ~np.isnan(oof)
        if mm.sum() == 0:
            fold_skip[c] = "нет предсказаний"
            continue
        if len(set(y[mm].tolist())) < 2:
            fold_skip[c] = "один класс"
            continue
        per_fold[c] = float(auc_cols(y[mm], oof[mm, None])[0])
    obs_fold = float(np.mean(list(per_fold.values()))) if per_fold else float("nan")

    bal = balanced_clips(y, clips, 3)
    obs_b = restricted_auc(oof, y, clips, bal)
    null, null_b, null_fold = [], [], []
    for si in range(seeds):
        rn = np.random.default_rng(seed + 3000 + 61 * si)
        yp = y.copy()
        for _ in range(perm):
            for c in sorted(set(clips.tolist())):
                mm = clips == c
                yp[mm] = rn.permutation(y[mm])
            o = readout(X, yp, clips, uniq, nf, n_cells, min_auc, min_agree)
            mm = ~np.isnan(o)
            if mm.sum():
                null.append(float(auc_cols(yp[mm], o[mm, None])[0]))
                b = restricted_auc(o, yp, clips, bal)
                if not np.isnan(b):
                    null_b.append(b)
                vals = []
                for c in sorted(set(clips.tolist())):
                    m2 = (clips == c) & ~np.isnan(o)
                    if m2.sum() and len(set(yp[m2].tolist())) > 1:
                        vals.append(float(auc_cols(yp[m2], o[m2, None])[0]))
                if vals:
                    null_fold.append(float(np.mean(vals)))
    null = np.array(null)
    null_b = np.array(null_b)
    null_fold = np.array(null_fold)
    return {"obs": obs, "obs_fold_mean": obs_fold, "per_fold": per_fold, "fold_skipped": fold_skip,
            "obs_balanced": obs_b, "bal": sorted(bal),
            "null_median": float(np.median(null)) if len(null) else float("nan"),
            "null_p95": float(np.percentile(null, 95)) if len(null) else float("nan"),
            "p": float((np.sum(null >= obs) + 1) / (len(null) + 1)) if len(null) else float("nan"),
            "null_fold_median": float(np.median(null_fold)) if len(null_fold) else float("nan"),
            "p_fold": float((np.sum(null_fold >= obs_fold) + 1) / (len(null_fold) + 1))
            if len(null_fold) else float("nan"),
            "p_balanced": float((np.sum(null_b >= obs_b) + 1) / (len(null_b) + 1))
            if len(null_b) else float("nan"),
            "n_perm": int(len(null))}


# ------------------------------------------------------------------ the three inputs

def labels_and_clips():
    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    ev = frozen["labels"]
    cat = np.array([e["category"] for e in ev])
    clips = np.array([e["clip"] for e in ev])
    y = np.where(cat == "NO_NET_DISPLACEMENT", 0, 1).astype(int)
    return ev, y, clips


def dn_input(ev):
    """The 68 descending cells, same eight features, from the P09 recording."""
    from scripts.p11a_route_change import build

    F, _names = build("dn68", ev)
    ng = F.shape[1]
    uniq = [str(i) for i in range(ng)]
    return F.reshape(len(ev), ng * len(FEATS)), uniq, np.ones(ng, dtype=float)


def rate_features_from_series(t: np.ndarray, R: np.ndarray, at: float) -> np.ndarray | None:
    """The same eight quantities, from a recording that is already a rate rather than counts."""
    win = (t >= at + BEFORE[0]) & (t <= at + AFTER[1])
    mb = (t >= at + BEFORE[0]) & (t <= at + BEFORE[1])
    md = (t >= at + DURING[0]) & (t <= at + DURING[1])
    ma = (t >= at + AFTER[0]) & (t <= at + AFTER[1])
    if mb.sum() < 2 or ma.sum() < 2 or win.sum() < 3:
        return None
    W = R[win]
    b, d_, a = R[mb].mean(axis=0), None, R[ma].mean(axis=0)
    m = (t >= at + DURING[0]) & (t <= at + DURING[1])
    d_ = R[m].mean(axis=0) if m.sum() else a
    return np.stack([
        b, d_, a, a - b,
        W.max(axis=0), W.min(axis=0),
        W.sum(axis=0) * BIN_S,
        np.sign(a - b),
    ], axis=1)


def early_input(ev):
    """The twenty early visual populations, from the P10 recording, same eight features.

    Read from output/p10/brain_*.npz, whose vis_rate columns are the twenty populations in the
    order vis_groups records. The first version of this function called load_groups(), which reads
    the P11.B metadata and therefore describes the candidate recording — so it silently measured
    the candidates twice and reported the same number for both rows. A control that compares an
    input with itself proves nothing, which is why a positive control has to be checked for having
    the right number of channels rather than merely for running.
    """
    rows, groups, pops = [], None, None
    cache: dict[str, tuple] = {}
    for e in ev:
        v = e["clip"]
        if v not in cache:
            d = np.load(ROOT / f"output/p10/brain_{v}.npz")
            cache[v] = (d["t"].astype(float), d["vis_rate"].astype(float),
                        [str(g) for g in d["vis_groups"]], d["vis_pop"].astype(float))
        t, R, g, pop = cache[v]
        if groups is None:
            groups, pops = g, pop
        elif g != groups:
            raise RuntimeError("порядок популяций различается между записями")
        rows.append(rate_features_from_series(t, R, float(e["time"])))
    if any(r is None for r in rows):
        raise RuntimeError("у части событий нет окна")
    if len(groups) != 20:
        raise RuntimeError(f"у ранних популяций должно быть 20 каналов, получено {len(groups)}")
    return np.stack(rows), groups, pops


def forward_input(ev):
    """The tracker's own forward numbers — P10 measured 0.318 here, below chance.

    A control is only as good as its negativity. The 68 descending cells were the negative case in
    P10, but at 0.560 they were not obviously negative, and if this procedure extracts a real but
    previously missed signal from them then they are no control at all. The forward readout is the
    one input P10 measured *below* chance, so it is the input that a working procedure must not
    lift above 0.5 by much.
    """
    import csv as _csv

    cache: dict[str, tuple] = {}
    rows, cols = [], None
    for e in ev:
        v = e["clip"]
        if v not in cache:
            rr = list(_csv.DictReader((ROOT / f"output/p07/forward_signal_{v}.csv").open()))
            names = [c for c in ("forward_signal", "forward_raw", "speed") if c in rr[0]]
            cache[v] = (np.array([float(r["t"]) for r in rr]),
                        np.stack([[float(r[c]) for r in rr] for c in names], axis=1), names)
        t, X, names = cache[v]
        if cols is None:
            cols = names
        rows.append(rate_features_from_series(t, X, float(e["time"])))
    if any(r is None for r in rows):
        raise RuntimeError("у части событий нет окна")
    return np.stack(rows), list(cols), np.ones(len(cols), dtype=float)


def video_input(ev):
    """Raw-video geometry — P10 measured 0.568 here, another weak negative."""
    cache: dict[str, tuple] = {}
    rows, cols = [], None
    for e in ev:
        stem = Path(e["clip_file"]).stem
        if stem not in cache:
            d = np.load(ROOT / f"output/p10/video_feat_{stem}.npz")
            names = sorted(k for k in d.files if k.startswith("f_"))
            cache[stem] = (d["t"].astype(float), np.stack([d[k].astype(float) for k in names], 1),
                           names)
        t, X, names = cache[stem]
        if cols is None:
            cols = names
        rows.append(rate_features_from_series(t - float(e["clip_offset_s"]), X, 0.0))
    if any(r is None for r in rows):
        raise RuntimeError("у части событий нет окна")
    return np.stack(rows), list(cols), np.ones(len(cols), dtype=float)


def candidate_input(ev):
    """The P11.B candidate cells, exactly as p11b_analyze builds them."""
    cache: dict[str, tuple] = {}
    uniq, pos, info = load_groups()
    rows = []
    for e in ev:
        v = e["clip"]
        if v not in cache:
            cache[v] = series_of(v)
        t, counts = cache[v]
        rows.append(rate_features(t, counts, pos, len(uniq), info["n_cells"], float(e["time"])))
    if any(r is None for r in rows):
        raise RuntimeError("у части событий нет окна")
    return np.stack(rows), uniq, info["n_cells"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--null-seeds", type=int, default=3)
    ap.add_argument("--min-auc", type=float, default=0.65)
    ap.add_argument("--min-agree", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--subsample", type=int, default=0,
                    help="сколько случайных наборов из 68 групп кандидатов нарисовать")
    args = ap.parse_args()

    ev, y, clips = labels_and_clips()
    print("=" * 104)
    print("P11.B КОНТРОЛЬ — та же процедура на заведомо известных входах")
    print("=" * 104)
    print(f"  событий {len(y)}, MOVE {int(y.sum())}, NO_NET {int((1-y).sum())}")
    print(f"  условия отбора: AUC >= {args.min_auc}, согласие >= {args.min_agree}, "
          f"нуль {args.perm}x{args.null_seeds}")
    print()

    report = {}
    inputs = [
        ("forward", "forward трекера (P10: 0.318 — НИЖЕ СЛУЧАЯ, главный контроль)", forward_input),
        ("video", "геометрия видео (P10: 0.568)", video_input),
        ("dn68", "68 нисходящих клеток (P10: 0.560)", dn_input),
        ("early", "20 ранних зрительных популяций (P10: 0.695)", early_input),
        ("candidates", "P11.B: 80 типов после ранних (ПОД ВОПРОСОМ)", candidate_input),
    ]
    for key, human, fn in inputs:
        F, uniq, n_cells = fn(ev)
        ng = len(uniq)
        X = F.reshape(len(y), ng * len(BFEATS))
        print(f"─── {human}")
        print(f"    групп {ng}, колонок {X.shape[1]}")
        r = evaluate(X, y, clips, uniq, len(BFEATS), n_cells, args.min_auc, args.min_agree,
                     args.perm, args.null_seeds, args.seed)
        print(f"    out-of-fold AUC = {r['obs']:.3f}   (p = {r['p']:.4f})")
        print(f"    нуль: медиана {r['null_median']:.3f}, p95 {r['null_p95']:.3f}, "
              f"{r['n_perm']} перемешиваний")
        print(f"    по фолдам отдельно, среднее = {r['obs_fold_mean']:.3f}   "
              f"(p = {r['p_fold']:.4f})")
        for c, v in sorted(r["per_fold"].items()):
            print(f"        {c}: AUC={v:.3f}")
        for c, why in sorted(r["fold_skipped"].items()):
            print(f"        {c}: не оценивается ({why})")
        print(f"    сбалансированные {r['bal']}: {r['obs_balanced']:.3f} "
              f"(p = {r['p_balanced']:.4f})")
        print()
        report[key] = {"human": human, "n_groups": ng, **r}

    print("─── ЧТЕНИЕ КОНТРОЛЯ ───")
    fw, vi = report["forward"], report["video"]
    dn, ea, ca = report["dn68"], report["early"], report["candidates"]
    print(f"  {'вход':<30}{'склейка':>9}{'по фолдам':>11}{'P10':>8}")
    for tag, r, p10 in (("forward (ниже случая)", fw, 0.318),
                        ("геометрия видео", vi, 0.568),
                        ("68 нисходящих", dn, 0.560),
                        ("ранние клетки (20)", ea, 0.695),
                        ("кандидаты P11.B (159)", ca, None)):
        s = f"{p10:>8.3f}" if p10 else "       -"
        print(f"  {tag:<30}{r['obs']:>9.3f}{r['obs_fold_mean']:>11.3f}{s}")
    print()
    gap_pool = ca["obs"] - ea["obs"]
    gap_fold = ca["obs_fold_mean"] - ea["obs_fold_mean"]
    print(f"  кандидаты минус ранние: склейка {gap_pool:+.3f}, по фолдам {gap_fold:+.3f}")
    print()
    print(f"  ГЛАВНЫЙ КОНТРОЛЬ: forward, который P10 измерил НИЖЕ случая (0.318), эта процедура")
    print(f"  оценивает в {fw['obs']:.3f} (по фолдам {fw['obs_fold_mean']:.3f}).")
    if fw["obs_fold_mean"] > 0.65:
        print("  Это значит, что процедура поднимает выше 0.65 вход, в котором сигнала нет.")
        print("  Абсолютным числам этой процедуры верить нельзя, включая число кандидатов.")
        verdict = "PROCEDURE_INFLATES"
    else:
        print("  Процедура не поднимает заведомо пустой вход, значит абсолютные числа можно")
        print("  читать.")
        verdict = "CALIBRATED"
    report["verdict"] = verdict
    report["gaps"] = {"candidates_minus_early_pooled": gap_pool,
                      "candidates_minus_early_per_fold": gap_fold}
    print()
    print("  ГЛАВНОЕ ДЛЯ P11.B: важно не абсолютное число, а разрыв между кандидатами и")
    print("  ранними клетками. Если он около нуля — нового выхода не найдено, кандидаты лишь")
    print("  повторяют то, что уже было видно в ранних клетках.")

    # ---- is the candidates' advantage over the descending cells just a size effect? ----
    #
    # The candidates offer 159 groups against the 68 of the descending recording, and the screen
    # judges pairs, so more pairs means more can clear the threshold by chance. Drawing random
    # subsets of the candidate groups at the size of the descending set separates the two.
    if args.subsample:
        print()
        print("─── ПРОВЕРКА РАЗМЕРА: кандидаты, урезанные до 68 групп случайно ───")
        F, uniq, n_cells = candidate_input(ev)
        ng_all = len(uniq)
        nf = len(BFEATS)
        rng = np.random.default_rng(args.seed + 777)
        draws = []
        for _ in range(args.subsample):
            pick = np.sort(rng.choice(ng_all, 68, replace=False))
            sub_uniq = [uniq[i] for i in pick]
            # F is [events, groups, features]; take whole groups, then flatten group-major so the
            # column layout matches what screen_pairs expects
            Xs = F[:, pick, :].reshape(len(y), len(pick) * nf)
            o = readout(Xs, y, clips, sub_uniq, nf, n_cells[pick], args.min_auc, args.min_agree)
            m = ~np.isnan(o)
            if m.sum():
                draws.append(float(auc_cols(y[m], o[m, None])[0]))
        draws = np.array(draws)
        print(f"  кандидаты, 159 групп:            {ca['obs']:.3f}")
        print(f"  кандидаты, случайные 68 групп:   медиана {np.median(draws):.3f}, "
              f"p95 {np.percentile(draws, 95):.3f}, максимум {draws.max():.3f} "
              f"({len(draws)} наборов)")
        print(f"  68 нисходящих (столько же групп): {dn['obs']:.3f}")
        if np.median(draws) <= dn["obs"] + 0.03:
            print("  Значит преимущество кандидатов над 68 нисходящими объясняется числом групп,")
            print("  а не тем, что это другие, более информативные клетки.")
        else:
            print("  Значит преимущество не сводится к числу групп: те же 68 групп кандидатов")
            print("  всё равно выше 68 записанных нисходящих.")
        report["subsample68"] = {"median": float(np.median(draws)),
                                 "p95": float(np.percentile(draws, 95)),
                                 "max": float(draws.max()), "n": int(len(draws)),
                                 "dn_obs": dn["obs"], "candidates_obs": ca["obs"]}

    (OUT / "control_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"  записано: {OUT/'control_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
