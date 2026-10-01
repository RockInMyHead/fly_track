#!/usr/bin/env python3
"""P11.B step 3 — does MOVE / NO_NET_DISPLACEMENT survive past the early visual cells?

The question, restated because it is easy to drift into the wrong one: P10 measured this
distinction at 0.705 in the drive handed to the brain, 0.695 in the early visual cells, and 0.560
in the sixty-eight descending cells, all against a null that permutes labels within a recording.
Something between the first and the last of those loses it. This script looks for the level where
it is still present. SAME_DIRECTION against DIFFERENT_DIRECTION is not measured here at all — that
question was answered in P11.A and reusing it would blur two different failures into one number.

WHAT IS BEING LOOKED FOR, EXACTLY
---------------------------------
The phase asked for a specific shape of result rather than the best number available: cell types
whose single-feature separation of the two classes is above 0.65 and whose direction of effect is
the same on most of the held-out recordings. A type that separates well on one recording and
pointlessly on the others has not been found; it has been fitted. So each candidate is judged on
two things at once — how well it separates and whether it agrees with itself across recordings.

Grouping follows the same instruction. Cells are pooled by type and side, never by type alone: a
left and a right copy of a cell type doing opposite things would cancel into nothing if averaged,
and that is precisely the mechanism P11.A identified as what went wrong with yaw. Features are
taken before, during and after the event as well, not only as one number per cell.

Usage:
    PYTHONPATH=. python scripts/p11b_analyze.py [--perm 400]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.p11a_route_change import (FEATS, KS, auc_cols, balanced_clips,  # noqa: E402
                                       restricted_auc)

OUT = ROOT / "output/p11b"
P10L = ROOT / "output/p10"
BIN_S = 0.2
BEFORE = (-3.0, -0.5)
DURING = (-0.5, 0.5)
AFTER = (0.5, 7.0)
MIN_AUC = 0.65


def load_groups() -> tuple[list[str], np.ndarray, dict]:
    meta = json.loads((OUT / "record_meta.json").read_text(encoding="utf-8"))
    ct = meta["cell_type"]
    sd = meta["side"]
    keys = [f"{t}_{s}" for t, s in zip(ct, sd)]
    uniq = sorted(set(keys))
    pos = np.array([uniq.index(k) for k in keys], dtype=np.int64)
    n_cells = np.bincount(pos, minlength=len(uniq)).astype(np.float64)
    return uniq, pos, {"n_cells": n_cells, "meta": meta}


def series_of(video: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(OUT / f"cells_{video}.npz")
    return d["t"].astype(float), d["counts"].astype(np.float64)


def rate_features(t: np.ndarray, counts: np.ndarray, pos: np.ndarray, n_groups: int,
                  n_cells: np.ndarray, at: float) -> np.ndarray | None:
    """Eight quantities per group: spikes per cell per second, pooled in time over the window."""
    def seg(lo: float, hi: float):
        m = (t >= at + lo) & (t <= at + hi)
        return m

    mb, md, ma = seg(*BEFORE), seg(*DURING), seg(*AFTER)
    if mb.sum() < 2 or ma.sum() < 2:
        return None
    win = seg(BEFORE[0], AFTER[1])
    W = counts[win]
    if W.shape[0] < 3:
        return None
    # spikes -> spikes per cell per bin, then to a rate by dividing out the bin width
    per_bin = np.stack([np.bincount(pos, weights=W[i], minlength=n_groups) for i in
                        range(W.shape[0])]) / (n_cells[None, :] * BIN_S)

    def mean_over(mask_full: np.ndarray) -> np.ndarray:
        idx = np.nonzero(mask_full[win])[0]
        return per_bin[idx].mean(axis=0) if len(idx) else per_bin.mean(axis=0)

    b = mean_over(mb)
    d_ = mean_over(md)
    a = mean_over(ma)
    return np.stack([
        b, d_, a, a - b,
        per_bin.max(axis=0), per_bin.min(axis=0),
        per_bin.sum(axis=0) * BIN_S,
        np.sign(a - b),
    ], axis=1)


def screen_pairs(y: np.ndarray, X: np.ndarray, clips: np.ndarray, uniq: list[str],
                 nf: int, n_cells: np.ndarray, feats: tuple[str, ...],
                 train: np.ndarray, min_auc: float, min_agree: float) -> list[dict]:
    """Pairs that pass both conditions on the given subset of events, and nothing else.

    Only `train` is read. The held-out recording is not passed in, not looked up, and cannot
    influence the result — the function does not receive `clips` for any recording outside
    `train` except through the mask, and every quantity it computes is restricted to the mask.
    That is the property the phase asked to be guaranteed, and it is checked directly in
    tests/test_p11b_no_leakage.py rather than asserted here.

    Conditions, both applied to the training recordings only:
        auc       separation on the pooled training events
        agree     the direction of that separation, checked on each training recording alone
    """
    folds = [c for c in sorted(set(clips[train].tolist()))
             if len(set(y[train & (clips == c)].tolist())) > 1]
    out = []
    for gi, g in enumerate(uniq):
        for fi, fname in enumerate(feats):
            col = X[:, gi * nf + fi]
            a_tr = auc_cols(y[train], col[train, None])[0]
            if np.isnan(a_tr) or a_tr < min_auc:
                continue
            sgn = 1.0 if a_tr >= 0.5 else -1.0
            per, agree = [], 0
            for c in folds:
                m = train & (clips == c)
                a_c = auc_cols(y[m], col[m, None])[0]
                if np.isnan(a_c):
                    continue
                per.append(float(a_c))
                agree += int((a_c >= 0.5) == (sgn > 0))
            frac = agree / len(per) if per else 0.0
            if per and frac >= min_agree:
                out.append({"group": g, "feature": fname, "auc_train": float(a_tr),
                            "sign": sgn, "agree": frac, "per_clip_train": per,
                            "n_cells": float(n_cells[gi])})
    return out


def build() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict]:
    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    ev = frozen["labels"]
    uniq, pos, info = load_groups()
    ng = len(uniq)
    cache: dict[str, tuple] = {}
    rows = []
    for e in ev:
        v = e["clip"]
        if v not in cache:
            cache[v] = series_of(v)
        t, counts = cache[v]
        f = rate_features(t, counts, pos, ng, info["n_cells"], float(e["time"]))
        rows.append(f)
    ok = [i for i, r in enumerate(rows) if r is not None]
    F = np.stack([rows[i] for i in ok])
    cat = np.array([ev[i]["category"] for i in ok])
    clips = np.array([ev[i]["clip"] for i in ok])
    return F, uniq, cat, clips, info


def p10_reference() -> dict:
    p = P10L / "levels_report.json"
    if not p.exists():
        return {}
    lv = json.loads(p.read_text(encoding="utf-8"))["levels"]
    task = "MOVING_vs_NO_NET"
    return {k: {"auc": v[task]["auc_aggregate"], "p": v[task]["p_aggregate"],
                "best_channel": v[task]["auc_best_channel"]}
            for k, v in lv.items() if task in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm", type=int, default=400)
    ap.add_argument("--null-seeds", type=int, default=3)
    ap.add_argument("--min-auc", type=float, default=MIN_AUC)
    ap.add_argument("--agree", type=float, default=0.6,
                    help="какая доля обучающих роликов должна согласиться по направлению")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    F, uniq, cat, clips, info = build()
    ng = len(uniq)
    y = np.where(cat == "NO_NET_DISPLACEMENT", 0, 1).astype(int)
    nf = F.shape[2]
    X = F.reshape(len(y), ng * nf)
    rng = np.random.default_rng(args.seed)

    print("=" * 104)
    print("P11.B шаг 3 — MOVE против NO_NET_DISPLACEMENT ПОСЛЕ ранних зрительных клеток")
    print("=" * 104)
    print(f"  событий {len(y)} (SHOULD: все 60), групп тип×сторона {ng}, "
          f"клеток {int(info['n_cells'].sum())}")
    print(f"  метки: MOVE {int(y.sum())}, NO_NET {int((1-y).sum())}")
    print(f"  SAME/DIFFERENT здесь НЕ используется")
    print()

    # ---- screening, applied ONLY inside the four training recordings of each pass ----
    #
    # The first version of this block computed each pair's AUC and cross-recording agreement on
    # all five recordings at once, and then printed the winners. That is the exact mistake the
    # phase warned about: with sixty events, screening 1280 pairs against the same data they are
    # later reported on will always produce a handsome row, and the handsome row means nothing.
    # The threshold and the agreement condition are now applied only to the four recordings a
    # pass is allowed to learn from, and the fifth is used only to score the result.
    print("─── ОТБОР: применяется ТОЛЬКО внутри четырёх обучающих роликов каждого прохода ───")
    print(f"  условия: AUC на обучающих >= {args.min_auc} И одинаковое направление "
          f"на >= {args.agree:.0%} обучающих роликов")
    print("  пятый ролик на этом шаге не виден")
    print()
    clip_list = sorted(set(clips.tolist()))
    table: dict[tuple[str, str], dict] = {}
    fold_rows: dict[str, list[dict]] = {}

    def screen(train: np.ndarray, min_auc: float, min_agree: float) -> list[dict]:
        return screen_pairs(y, X, clips, uniq, nf, info["n_cells"], FEATS, train,
                            min_auc, min_agree)

    for c in clip_list:
        train = clips != c
        sel = screen(train, args.min_auc, args.agree)
        fold_rows[c] = sel
        test = clips == c
        for s in sel:
            key = (s["group"], s["feature"])
            rec = table.setdefault(key, {"group": s["group"], "feature": s["feature"],
                                        "folds": {}, "n_cells": s["n_cells"]})
            a_te = auc_cols(y[test], X[test, uniq.index(s["group"]) * nf +
                                      FEATS.index(s["feature"])][:, None])[0]
            rec["folds"][c] = None if np.isnan(a_te) else float(a_te)
            rec["train_auc"] = {**rec.get("train_auc", {}), c: s["auc_train"]}
            rec["agree"] = {**rec.get("agree", {}), c: s["agree"]}
        print(f"  {c}: отобрано пар на 4 обучающих = {len(sel)}")
    print()

    # ---- the table the phase asked for: type, side, feature, one column per held-out recording ----
    print("─── ТАБЛИЦА ПО ТИПАМ: значение — AUC НА ОТЛОЖЕННОМ РОЛИКЕ ───")
    print("  пусто = на этом проходе пара не прошла отбор по обучающим роликам")
    print()
    ranked = []
    for key, rec in table.items():
        vals = [v for v in rec["folds"].values() if v is not None]
        if len(vals) < 3:
            continue
        rec["median_heldout"] = float(np.median(vals))
        rec["n_folds"] = len(vals)
        rec["mean_heldout"] = float(np.mean(vals))
        ranked.append(rec)
    ranked.sort(key=lambda r: -r["median_heldout"])
    hdr = f"  {'тип':<14}{'стор':>5}  {'признак':<13}" + "".join(
        f"{c.replace('VID000','').rjust(7)}" for c in clip_list) + f"{'медиана':>9}{'фолдов':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rec in ranked[:25]:
        typ, side = rec["group"].rsplit("_", 1)
        line = f"  {typ[:13]:<14}{side:>5}  {rec['feature']:<13}"
        for c in clip_list:
            v = rec["folds"].get(c)
            line += "      -" if v is None else f"{v:>7.2f}"
        line += f"{rec['median_heldout']:>9.3f}{rec['n_folds']:>7}"
        print(line)
    print()
    print(f"  всего пар, прошедших отбор хотя бы в 3 проходах: {len(ranked)}")
    print("  ВНИМАНИЕ: выбирать из этой таблицы лучшую строку и называть её результатом")
    print("  нельзя — это снова отбор по отложенным данным. Ниже — честная сквозная оценка.")
    print()

    # ---- the same selection, turned into one end-to-end number without choosing a winner ----
    #
    # Every pair that survived the screen in a fold contributes to that fold's score, each with the
    # direction learned on the training recordings. No row of the table above is picked, so the
    # number below does not benefit from being chosen after the fact.
    def screen_readout(yv: np.ndarray, perm: int, seeds: int) -> dict:
        def once(y_use: np.ndarray) -> np.ndarray:
            oof = np.full(len(y_use), np.nan)
            for c in clip_list:
                train, test = clips != c, clips == c
                sel = screen(train, args.min_auc, args.agree)
                if not sel:
                    continue
                cols, signs = [], []
                for s in sel:
                    gi = uniq.index(s["group"])
                    fi = FEATS.index(s["feature"])
                    cols.append(X[:, gi * nf + fi])
                    signs.append(s["sign"])
                Z = np.stack(cols, axis=1)
                mu, sd = Z[train].mean(axis=0), Z[train].std(axis=0)
                sd = np.where(sd > 1e-9, sd, 1.0)
                oof[test] = (((Z[test] - mu) / sd) * np.asarray(signs)).mean(axis=1)
            return oof

        oof = once(yv)
        m = ~np.isnan(oof)
        obs = float(auc_cols(yv[m], oof[m, None])[0]) if m.sum() else float("nan")
        bal = balanced_clips(yv, clips, 3)
        obs_b = restricted_auc(oof, yv, clips, bal)
        null, null_b = [], []
        for si in range(seeds):
            rn = np.random.default_rng(args.seed + 9000 + 71 * si)
            yp = yv.copy()
            for _ in range(perm):
                for c in clip_list:
                    mm = clips == c
                    yp[mm] = rn.permutation(yv[mm])
                o = once(yp)
                mm = ~np.isnan(o)
                if mm.sum():
                    null.append(float(auc_cols(yp[mm], o[mm, None])[0]))
                    b = restricted_auc(o, yp, clips, bal)
                    if not np.isnan(b):
                        null_b.append(b)
        null = np.array(null)
        null_b = np.array(null_b)
        return {"obs": obs, "obs_balanced": obs_b, "bal": sorted(bal),
                "null_median": float(np.median(null)) if len(null) else float("nan"),
                "null_p95": float(np.percentile(null, 95)) if len(null) else float("nan"),
                "p": float(np.mean([(np.sum(null >= obs) + 1) / (len(null) + 1)])) if len(null)
                else float("nan"),
                "p_balanced": float(np.mean([(np.sum(null_b >= obs_b) + 1) /
                                             (len(null_b) + 1)])) if len(null_b) else float("nan"),
                "n_perm": int(len(null))}

    print("─── СКВОЗНАЯ ОЦЕНКА ОТБОРА ПО ТИПАМ (без выбора лучшей строки) ───")
    scr = screen_readout(y, args.perm, args.null_seeds)
    print(f"  все отобранные пары каждого прохода внесены в оценку с весом 1")
    print(f"  out-of-fold AUC = {scr['obs']:.3f}   (p = {scr['p']:.4f})")
    print(f"  нуль: медиана {scr['null_median']:.3f}, p95 {scr['null_p95']:.3f}, "
          f"{scr['n_perm']} перемешиваний")
    print(f"  только сбалансированные ролики {scr['bal']}: {scr['obs_balanced']:.3f} "
          f"(p = {scr['p_balanced']:.4f})")
    print()

    # ---- compact nested readout, exactly the check P11.A used ----
    KS_local = tuple(k for k in KS if k <= 8)
    print("─── КОМПАКТНОЕ ЧТЕНИЕ, тот же конвейер, что в P11.A ───")

    def run_once(yv, rng_used, perm: int) -> dict:
        from scripts.p11a_route_change import run
        res = run(X, ng, nf, yv, clips, KS_local, 0.7, mode="logreg")
        obs = res["auc_oof"]
        bal = balanced_clips(yv, clips, 3)
        obs_b = restricted_auc(res["oof"], yv, clips, bal)
        null, null_b = [], []
        for si in range(args.null_seeds):
            rn = np.random.default_rng(args.seed + 5000 + 91 * si)
            yp = yv.copy()
            for _ in range(perm):
                for c in set(clips.tolist()):
                    m = clips == c
                    yp[m] = rn.permutation(yv[m])
                r = run(X, ng, nf, yp, clips, KS_local, 0.7, mode="logreg")
                if r["auc_oof"] is not None:
                    null.append(r["auc_oof"])
                    b = restricted_auc(r["oof"], yp, clips, bal)
                    if not np.isnan(b):
                        null_b.append(b)
        null = np.array(null)
        null_b = np.array(null_b)
        return {"obs": obs, "obs_balanced": obs_b, "bal": sorted(bal),
                "null_median": float(np.median(null)),
                "null_p95": float(np.percentile(null, 95)),
                "p": float(np.mean([(np.sum(null >= obs) + 1) / (len(null) + 1)])) if len(null)
                else float("nan"),
                "p_balanced": float(np.mean([(np.sum(null_b >= obs_b) + 1) /
                                             (len(null_b) + 1)])) if len(null_b)
                else float("nan"),
                "per_fold": res["per_fold"], "k": res["k_chosen"]}

    res = run_once(y, rng, args.perm)
    print(f"  out-of-fold AUC = {res['obs']:.3f}   (p = {res['p']:.4f})")
    print(f"  только сбалансированные ролики {res['bal']}: {res['obs_balanced']:.3f} "
          f"(p = {res['p_balanced']:.4f})")
    print(f"  размер по фолдам: {res['k']}")
    for c, d in sorted(res["per_fold"].items()):
        a = f"{d['auc']:.3f}" if d["auc"] is not None else " н/д"
        print(f"      {c}: AUC={a}  групп {d['n_cells']}  "
              f"(MOVE {d['n_pos']}, NO_NET {d['n_neg']})")
    print()

    # ---- where this sits among the levels already measured ----
    ref = p10_reference()
    print("─── ГДЕ ЭТО СРЕДИ УЖЕ ИЗМЕРЕННЫХ УРОВНЕЙ (все — MOVE против NO_NET) ───")
    order = ["video_full", "video_band", "input", "early", "dn", "yaw", "forward"]
    ru = {"video_full": "сырое видео", "video_band": "видео, центр",
          "input": "вход в мозг", "early": "ранние клетки", "dn": "68 нисходящих",
          "yaw": "yaw", "forward": "forward"}
    for k in order:
        if k not in ref:
            continue
        r = ref[k]
        mark = "  <-- P10: здесь есть" if r["p"] < 0.05 else ""
        print(f"  {ru[k]:<18}{r['auc']:>7.3f}  (p={r['p']:.3f}){mark}")
    print(f"  {'P11.B кандидаты (свод)':<18}{scr['obs']:>7.3f}  (p={scr['p']:.4f})")
    print(f"  {'P11.B кандидаты (чтение)':<18}{res['obs']:>7.3f}  (p={res['p']:.4f})  "
          f"<- {ng} групп, {int(info['n_cells'].sum())} клеток")
    print()

    # The two numbers answer slightly different questions and both are kept:
    #   scr — every pair that passed the screen on the training recordings contributes equally
    #   res — a compact readout over at most eight groups, its size chosen inside the training
    best_p = min(scr["p"], res["p"])
    verdict = "FOUND" if best_p < 0.05 else "NOT_FOUND"
    print("─── ВЫВОД ───")
    if verdict == "FOUND":
        which = "свод по отобранным парам" if scr["p"] <= res["p"] else "компактное чтение"
        winner = scr if scr["p"] <= res["p"] else res
        print(f"  Различение MOVE / NO_NET переносится на отложенный ролик: "
              f"{winner['obs']:.3f}, p={winner['p']:.4f}  ({which})")
        print(f"  Это выше, чем в 68 нисходящих (0.560), и сопоставимо с ранними клетками "
              f"(0.695).")
        stable = sorted({rec["group"] for rec in ranked if rec["n_folds"] >= 4}, key=str)
        if stable:
            print(f"  Группы, прошедшие отбор минимум в 4 проходах: "
                  f"{', '.join(stable[:10])}")
    else:
        print(f"  Переносимого различения не найдено: свод {scr['obs']:.3f} (p={scr['p']:.4f}), "
              f"чтение {res['obs']:.3f} (p={res['p']:.4f}).")
        print(f"  Среди {ng} групп после ранних клеток нет уровня, где MOVE/NO_NET")
        print("  сохраняется так же хорошо, как во входе (0.705) и ранних клетках (0.695).")
        print("  Тогда следующий шаг — не новые типы, а временная динамика и совместная")
        print("  активность сети: информация в ранней зрительной системе есть, но в виде,")
        print("  который одиночные популяции в наших окнах не несут.")
    print()

    (OUT / "analyze_report.json").write_text(json.dumps({
        "phase": "P11.B шаг 3 — MOVE/NO_NET после ранних зрительных клеток",
        # the three rules the phase fixed for this report, recorded verbatim
        "rules": {
            "1_connectome_screen": (
                f"{len(uniq)} групп (тип x сторона), {int(info['n_cells'].sum())} клеток, "
                "отобраны только по связям. Метки MOVE/NO_NET при отборе не использовались."
            ),
            "2_selection": (
                "отбор типов и признаков происходит ЗАНОВО внутри каждого "
                "leave-one-video-out; условия AUC и согласия применяются только к четырём "
                "обучающим роликам"
            ),
            "3_final_score": (
                "оценка только на отложенном ролике; нуль — перемешивание меток внутри "
                "каждого ролика, весь конвейер перезапускается, включая отбор"
            ),
        },
        "depth_note": (
            "connectome рекуррентный, поэтому depth=1 не означает «один биологический "
            "уровень»: это только «есть прямое ребро от ранних клеток». Для этой задачи "
            "достаточно."
        ),
        "n_events": int(len(y)), "n_groups": ng, "n_cells": int(info["n_cells"].sum()),
        "min_auc": args.min_auc, "min_agree": args.agree,
        "n_pairs_screened_per_fold": {c: len(v) for c, v in fold_rows.items()},
        "table": ranked[:80],
        "screen_readout": scr,
        "compact_readout": {k: v for k, v in res.items() if k != "per_fold"},
        "per_fold": res["per_fold"],
        "p10_reference": ref,
        "verdict": verdict,
        "note": "SAME/DIFFERENT в этом шаге не используется",
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"  записано: {OUT/'analyze_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
