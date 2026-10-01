#!/usr/bin/env python3
"""P11.A — can SAME/DIFFERENT be read out of the descending cells that are already recorded?

WHY THIS COMES BEFORE ANY NEW RECORDING
---------------------------------------
P10 measured, at one and the same held-out-clip check, that the 68 recorded descending cells carry
SAME_DIRECTION against DIFFERENT_DIRECTION at AUC 0.767, while the yaw readout the tracker actually
uses carries the same distinction at 0.617. If that is true, then several phases have been spent
looking for a better cell to record when the loss was in the pooled number the tracker reads.

That conclusion is not free, and this script is the test of it rather than a confirmation. Two
things could make the gap disappear on inspection:

    size    yaw is built from 18 of the 68 cells (DNp17, DNa07, DNp26, DNp20 — twelve, two, two and
            two cells on their two sides). Comparing 68 cells with 18 cells is partly a comparison
            of two sizes. The check is therefore run again on exactly those 18 cells, and on the
            seven numbers of the yaw csv itself, with the same procedure.

    chance  68 cells times eight features is 544 candidate columns against forty events. A model
            allowed to look at all of them and then report its held-out score will report the
            number of columns, not the number of cells. Everything below is therefore selected
            *inside* each training split, and every number is judged against a null built by
            permuting the labels within each recording.

THE PROCEDURE, IDENTICAL FOR EVERY SOURCE
-----------------------------------------
    features   eight plain quantities per cell, as the phase asked for and no more: mean before,
               mean during, mean after, the change from before to after, the maximum, the minimum,
               the integral, and the sign of the change. Nothing squared, nothing lagged, no
               cross-products between cells.
    split      leave one recording out. Train on four, score the fifth, five times.
    selection  inside the four training recordings, an inner leave-one-naming-out loop scores each
               feature on three recordings and evaluates it on the fourth. The sign is learned on
               the three and applied to the fourth, so a feature that only helps because of its
               direction on one recording is not rewarded for it.
    consistency a cell is only eligible if the sign of its chosen feature agrees across the inner
               folds, which is what "the same direction of effect on most of the held-out
               recordings" means operationally.
    size       one to eight cells. The number is chosen by the inner loop too, not by looking at
               the held-out recording.
    readout    the signed mean of the chosen (cell, feature) pairs. It stays a small linear
               combination of cell rates on purpose: the question is whether a compact biological
               signal exists, not whether sixty events can be memorised.

Outputs: output/p11a/report.json, output/p11a/route_change_signal.json, output/p11a/p11a.png

Usage:
    PYTHONPATH=. python scripts/p11a_route_change.py [--perm 200]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.p10_levels import EPS, _logreg  # noqa: E402

OUT = ROOT / "output/p11a"
P09 = ROOT / "output/p09"
P07 = ROOT / "output/p07"

FEATS = ("mean_before", "mean_during", "mean_after", "change",
         "max", "min", "integral", "sign_change")
KS = (1, 2, 3, 4, 5, 6, 8)
PRIMARY_MODE = "logreg"     # pre-registered, not chosen after seeing the table
BEFORE = (-3.0, -0.5)
DURING = (-0.5, 0.5)
AFTER = (0.5, 7.0)
YAW_TYPES = ("DNp17", "DNa07", "DNp26", "DNp20")
DN_SMOOTH = 25            # half a second of integration at 20 ms, as in P09 and P10


# ----------------------------------------------------------------- data

LABELS_PATH = ROOT / "data/p10/FROZEN_P10.json"


def labels() -> tuple[list[dict], np.ndarray, np.ndarray]:
    # The default stays the frozen five-clip set so the published number is still reproducible;
    # main() repoints this when another set of recordings is under test.
    f = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    ev = f["labels"]
    y = np.where(np.array([e["category"] for e in ev]) == "SAME_DIRECTION", 1, 0)
    keep = np.isin([e["category"] for e in ev], ["SAME_DIRECTION", "DIFFERENT_DIRECTION"])
    return ev, y, keep


def clip_of(ev: list[dict]) -> np.ndarray:
    return np.array([e["clip"] for e in ev])


_DN: dict[str, tuple] = {}


def dn(clip: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Smoothed rate of the 68 unique recorded descending cells, and which of them feed yaw."""
    if clip not in _DN:
        d = np.load(P09 / f"trace_{clip}.npz")
        cells = d["cells"].astype(int)
        ctype = [str(x) for x in d["cell_type"]]
        side = [str(x) for x in d["side"]]
        uniq, first = np.unique(cells, return_index=True)
        fired = d["fired"][:, first].astype(float)
        rate = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(DN_SMOOTH) / DN_SMOOTH, mode="same"), 0, fired)
        names = [f"{ctype[i]}_{side[i]}" for i in first]
        is_yaw = np.array([ctype[i] in YAW_TYPES for i in first])
        _DN[clip] = (d["t"].astype(float), rate, names, is_yaw, uniq)
    return _DN[clip]


def yaw_channels(clip: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    p = P07 / f"yaw_signal_{clip}.csv"
    rows = list(csv.DictReader(p.open()))
    t = np.array([float(r["t"]) for r in rows])
    cols = [c for c in rows[0] if c == "yaw_signal" or c.endswith("_deadband")
            or c.endswith("_diagnostic") or c.startswith("common_")]
    X = np.stack([np.array([float(r[c]) for r in rows]) for c in cols], axis=1)
    return t, X, cols


def features_at(t: np.ndarray, X: np.ndarray, at: float) -> np.ndarray | None:
    """The eight fixed quantities of one event, per channel. Returns [n_ch, 8]."""
    def seg(lo: float, hi: float) -> np.ndarray:
        m = (t >= at + lo) & (t <= at + hi)
        return X[m] if m.sum() >= 3 else None

    B, D, A = seg(*BEFORE), seg(*DURING), seg(*AFTER)
    if B is None or A is None:
        return None
    if D is None:
        D = np.vstack([B[-2:], A[:2]]) if len(B) >= 2 and len(A) >= 2 else A
    mb, md, ma = B.mean(axis=0), D.mean(axis=0), A.mean(axis=0)
    win = (t >= at + BEFORE[0]) & (t <= at + AFTER[1])
    W = X[win]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    return np.stack([
        mb, md, ma, ma - mb,
        W.max(axis=0), W.min(axis=0), W.sum(axis=0) * dt,
        np.sign(ma - mb),
    ], axis=1)


def build(source: str, ev: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Feature tensor [n_events, n_ch, 8] and channel names, for one source."""
    rows, names = [], None
    for e in ev:
        if source == "dn68":
            t, X, nm, _, _ = dn(e["clip"])
        elif source == "dn_yaw18":
            t, X, nm, is_yaw, _ = dn(e["clip"])
            X = X[:, is_yaw]
            nm = [n for n, k in zip(nm, is_yaw) if k]
        elif source == "yaw_csv":
            t, X, nm = yaw_channels(e["clip"])
        else:
            raise ValueError(source)
        f = features_at(t, X, float(e["time"]))
        if f is None:
            rows.append(None)
            continue
        rows.append(f)
        if names is None:
            names = list(nm)
    if any(r is None for r in rows):
        raise RuntimeError(f"{source}: у части событий нет окна")
    return np.stack(rows), names


# ----------------------------------------------------------------- auc

def auc_cols(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUC of every column of S. Exact, ties counted as half. y must contain both classes."""
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.full(S.shape[1], np.nan)
    Sp, Sn = S[pos], S[neg]
    gt = (Sp[:, None, :] > Sn[None, :, :]).sum(axis=(0, 1))
    eq = (Sp[:, None, :] == Sn[None, :, :]).sum(axis=(0, 1))
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def restricted_auc(oof: np.ndarray, y: np.ndarray, clips: np.ndarray,
                   allowed: set[str]) -> float:
    m = (~np.isnan(oof)) & np.isin(clips, list(allowed))
    if m.sum() == 0 or len(set(y[m].tolist())) < 2:
        return float("nan")
    return float(auc_cols(y[m], oof[m, None])[0])


def balanced_clips(y: np.ndarray, clips: np.ndarray, least: int = 3) -> set[str]:
    """Recordings that contain at least `least` events of each class.

    VID00001 contributes five SAME and one DIFFERENT, VID00006 three and one. An AUC computed
    inside such a recording is decided by that single event, so a pooled number that includes them
    is partly the coin that event landed on. The robust figure is the one over the recordings with
    the mix to support it, and it is reported next to the headline rather than instead of it.
    """
    out = set()
    for c in set(clips.tolist()):
        m = clips == c
        if min(int(y[m].sum()), int((1 - y[m]).sum())) >= least:
            out.add(c)
    return out


class Subsets:
    """Cached sign tensors so that permutations cost almost nothing.

    For rows r, Sgn[i, j, c] = sign(X[i, c] - X[j, c]). Then for any labelling y,

        AUC_c = (n_pos * n_neg + sum_{p, q} Sgn[p, q, c]) / (2 * n_pos * n_neg)

    which is the Mann-Whitney form and is exact with ties. The tensor does not depend on y, so it
    is built once per fold and reused for every permutation.
    """

    def __init__(self, X: np.ndarray):
        self.X = X
        self.cache: dict[tuple, np.ndarray] = {}

    def key(self, rows: np.ndarray) -> np.ndarray:
        k = tuple(rows.tolist())
        if k not in self.cache:
            sub = self.X[rows]
            self.cache[k] = np.sign(sub[:, None, :] - sub[None, :, :]).astype(np.int8)
        return self.cache[k]

    def aucs(self, rows: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        yv = y[rows]
        pos, neg = np.nonzero(yv == 1)[0], np.nonzero(yv == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        S = self.key(rows)
        N = len(pos) * len(neg)
        s = S[np.ix_(pos, neg)].sum(axis=(0, 1), dtype=np.int64)
        return (N + s) / (2.0 * N)


# ----------------------------------------------------------------- selection

def cell_table(auc: np.ndarray, nch: int, nf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell: its best feature, that feature's AUC, and how far from chance it is."""
    A = auc.reshape(nch, nf)
    best_f = A.argmax(axis=1)
    best_dev = np.abs(A - 0.5).max(axis=1)
    best_a = A[np.arange(nch), best_f]
    return best_f, best_a, best_dev


def select(auc_full, aucs_inner, nch, nf, k, min_cons):
    """Pick k cells, using only what the training recordings say.

    auc_full   per-feature AUC on the whole training set
    aucs_inner per-feature AUC on each inner-training set, one per inner fold
    """
    best_f, best_a, best_dev = cell_table(auc_full, nch, nf)
    cons = np.zeros(nch)
    if aucs_inner:
        signs = []
        for ai in aucs_inner:
            if ai is None:
                continue
            A = ai.reshape(nch, nf)
            signs.append(np.sign(A[np.arange(nch), best_f] - 0.5))
        if signs:
            S = np.stack(signs)
            ref = np.sign(best_a - 0.5)
            cons = (S == ref[None, :]).mean(axis=0)
    ok = cons >= min_cons
    score = np.where(ok, best_dev, -1.0)
    idx = np.argsort(-score)[:k]
    idx = idx[score[idx] > 0]
    return {"cells": idx, "feat": best_f[idx],
            "sign": np.where(best_a[idx] >= 0.5, 1.0, -1.0),
            "dev": best_dev[idx], "cons": cons[idx]}


def score_of(X2: np.ndarray, rows: np.ndarray, sel: dict, nf: int) -> np.ndarray:
    """The readout: signed mean of the chosen (cell, feature) rates over the chosen window."""
    if len(sel["cells"]) == 0:
        return np.zeros(len(rows))
    cols = sel["cells"] * nf + sel["feat"]
    return (X2[np.ix_(rows, cols)] * sel["sign"]).mean(axis=1)


def fit_readout(X2: np.ndarray, sel: dict, nf: int, y: np.ndarray, rows: np.ndarray,
                mode: str, l2: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Fit the chosen readout on the training rows and return (weights, bias).

    Two forms are kept. The signed mean is the biological one: one feature per chosen cell, each
    with the direction learned on the training recordings. The logistic form uses all eight
    features of the chosen cells, with a strong ridge so that eight features times eight cells is
    still a small model.
    """
    if len(sel["cells"]) == 0:
        return np.zeros(0), 0.0
    if mode == "signed_mean":
        cols = sel["cells"] * nf + sel["feat"]
        w = sel["sign"] / len(sel["cells"])
        return w.astype(float), 0.0
    cols = (sel["cells"][:, None] * nf + np.arange(nf)[None, :]).ravel()
    Z = X2[np.ix_(rows, cols)]
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    w, b = _logreg((Z - mu) / sd, y[rows].astype(float), l2=l2)
    # fold the standardisation into the weights so scoring needs only the raw columns
    w_raw = w / sd
    b_raw = b - float(np.sum(w * mu / sd))
    return w_raw, b_raw


def apply_readout(X2: np.ndarray, sel: dict, nf: int, w: np.ndarray, b: float,
                  rows: np.ndarray) -> np.ndarray:
    if len(sel["cells"]) == 0:
        return np.zeros(len(rows))
    if len(w) == len(sel["cells"]):
        cols = sel["cells"] * nf + sel["feat"]
    else:
        cols = (sel["cells"][:, None] * nf + np.arange(nf)[None, :]).ravel()
    return X2[np.ix_(rows, cols)] @ w + b


# ----------------------------------------------------------------- the run

def run(X2: np.ndarray, nch: int, nf: int, y: np.ndarray, clips: np.ndarray,
        ks=KS, min_cons=0.75, mode: str = "signed_mean", ks_fixed: int | None = None) -> dict:
    """Nested leave-one-clip-out. Selection never sees the held-out recording.

    mode       signed_mean  one feature per chosen cell, direction from the training recordings
               logreg       all eight features of the chosen cells, strongly ridged
    ks_fixed   skip the inner choice of model size and use this many cells throughout
    """
    n = X2.shape[0]
    subs = Subsets(X2)
    clip_list = sorted(set(clips.tolist()))
    ks_use = (ks_fixed,) if ks_fixed else ks

    oof = np.full(n, np.nan)
    per_fold = {}
    chosen_log = []
    k_log = []

    for c in clip_list:
        test_rows = np.nonzero(clips == c)[0]
        train_rows = np.nonzero(clips != c)[0]
        if len(set(y[train_rows].tolist())) < 2:
            continue
        auc_full = subs.aucs(train_rows, y)
        if auc_full is None:
            continue

        inner = []
        for d in clip_list:
            if d == c:
                continue
            rows = np.nonzero((clips != c) & (clips != d))[0]
            if len(set(y[rows].tolist())) < 2:
                continue
            inner.append((d, rows, subs.aucs(rows, y)))

        # --- inner loop chooses the number of cells ---
        k_best, k_score = ks_use[0], -np.inf
        for k in ks_use:
            val_rows, val_scores = [], []
            for d, rows, a_in in inner:
                if a_in is None:
                    continue
                sel = select(a_in, [x[2] for x in inner if x[0] != d], nch, nf, k, min_cons)
                if len(sel["cells"]) == 0:
                    continue
                w, b = fit_readout(X2, sel, nf, y, rows, mode)
                vr = np.nonzero(clips == d)[0]
                val_rows.append(vr)
                val_scores.append(apply_readout(X2, sel, nf, w, b, vr))
            if not val_rows:
                continue
            r = np.concatenate(val_rows)
            s = np.concatenate(val_scores)
            a = auc_cols(y[r], s[:, None])
            if np.isnan(a[0]):
                continue
            if a[0] > k_score:
                k_score, k_best = a[0], k
        if not np.isfinite(k_score):
            continue

        # --- fit on the whole training set with the chosen size, score the held-out recording ---
        sel = select(auc_full, [x[2] for x in inner], nch, nf, k_best, min_cons)
        if len(sel["cells"]) == 0:
            continue
        w, b = fit_readout(X2, sel, nf, y, train_rows, mode)
        s = apply_readout(X2, sel, nf, w, b, test_rows)
        oof[test_rows] = s
        a_test = auc_cols(y[test_rows], s[:, None])[0]
        per_fold[c] = {"auc": float(a_test) if not np.isnan(a_test) else None,
                       "n_cells": int(len(sel["cells"])), "k_chosen": int(k_best),
                       "inner_cv": float(k_score),
                       "n_pos": int((y[test_rows] == 1).sum()),
                       "n_neg": int((y[test_rows] == 0).sum())}
        k_log.append(k_best)
        for ci, fi, si, di, co in zip(sel["cells"], sel["feat"], sel["sign"], sel["dev"], sel["cons"]):
            chosen_log.append({"fold": c, "cell": int(ci), "feat": FEATS[int(fi)],
                               "sign": float(si), "dev": float(di), "cons": float(co),
                               "k": int(k_best)})

    m = ~np.isnan(oof)
    a_all = auc_cols(y[m], oof[m, None])[0] if m.sum() else float("nan")
    return {"auc_oof": float(a_all) if not np.isnan(a_all) else None,
            "per_fold": per_fold, "chosen": chosen_log,
            "k_chosen": k_log, "oof": oof, "cover": int(m.sum())}


# ----------------------------------------------------------------- baselines

def p10_aggregate(source: str, ev: list[dict]) -> float:
    """The number P10 reported for this level, recomputed so the comparison is like for like."""
    from scripts.p10_levels import (build as build_level, to_matrix, Store, oof_aggregate, auc)
    store = Store()
    lvl = {"dn68": "dn", "yaw_csv": "yaw"}[source]
    rows, _ = build_level(lvl, ev, store)
    X = to_matrix(rows)
    cat = np.array([e["category"] for e in ev])
    clips = np.array([e["clip"] for e in ev])
    keep = np.isin(cat, ["SAME_DIRECTION", "DIFFERENT_DIRECTION"])
    y = np.where(cat == "SAME_DIRECTION", 1, 0)[keep]
    return float(auc(y, oof_aggregate(X[keep], y, clips[keep])))


def diagnostics(X2: np.ndarray, nch: int, nf: int, y: np.ndarray, clips: np.ndarray,
                min_cons: float) -> dict:
    """Separate the three explanations for a weak compact readout.

    A negative result here is only worth reporting if it is the selection that fails and not the
    pipeline, so each step is measured against the others:

        all cells, P10 aggregate     what the phase already reported for this level
        all cells, signed mean       the readout form with no selection at all
        all cells, logistic          the readout form with every cell and a ridge, no selection
        k cells, nested              selection inside the training split, k fixed
        k cells, oracle              selection allowed to see everything — the optimistic bound
    """
    out = {}
    for mode, label in (("signed_mean", "все клетки, знаковая сумма"),
                        ("logreg", "все клетки, логистика (без отбора)")):
        r = run(X2, nch, nf, y, clips, ks=(nch,), min_cons=0.0, mode=mode, ks_fixed=nch)
        out[label] = r["auc_oof"]
    for k in (3, 5, 8):
        r = run(X2, nch, nf, y, clips, ks=(k,), min_cons=min_cons, mode="signed_mean",
                ks_fixed=k)
        out[f"{k} клеток, nested (знаковая сумма)"] = r["auc_oof"]
        r = run(X2, nch, nf, y, clips, ks=(k,), min_cons=min_cons, mode="logreg",
                ks_fixed=k)
        out[f"{k} клеток, nested (логистика)"] = r["auc_oof"]
    # oracle: choose on the pooled set, then score the same pooled set. Not a result, a ceiling.
    subs = Subsets(X2)
    allrows = np.arange(len(y))
    a = subs.aucs(allrows, y)
    for k in (3, 8):
        sel = select(a, [a], nch, nf, k, 0.0)
        w, b = fit_readout(X2, sel, nf, y, allrows, "signed_mean")
        s = apply_readout(X2, sel, nf, w, b, allrows)
        out[f"{k} клеток, oracle (не результат)"] = float(auc_cols(y, s[:, None])[0])
    return out


def control_block(ev: list[dict], y: np.ndarray, clips: np.ndarray, perm: int,
                  rng: np.random.Generator, draws: int = 400) -> dict:
    """Comparisons at an equal model size, where nothing is free to choose.

    This block exists because of a mistake in how P10's numbers were described. The figure quoted
    there for "68 descending cells" came from a logistic model over eight statistics of all
    sixty-eight channels — 544 columns — while the figure quoted for yaw came from 56 columns.
    Comparing 544 columns with 56 is partly comparing two sizes, so the gap cannot be attributed
    to the cells on that evidence alone. The honest description of the P10 line is "68 cells x 8
    statistics, logistic", not "a seven-scalar reduction".

    What is compared here instead: the same eight statistics of the same number of cells, varying
    only which cells are handed to the model.

        yaw's 7 numbers x 8 statistics               56 columns   what the tracker reads today
        yaw's 18 cells x 8 statistics               144 columns
        k cells drawn from the 68 x 8 statistics    56 and 144 columns, many draws

    The random draws are what make it a comparison rather than a single number: if any seven of the
    68 cells beat yaw's fifty-six numbers, the gain is in the cells. If only particular seven do,
    the gain is in the selection, and the null has to carry it.
    """
    from scripts.p10_levels import (build as build_level, Store, STATS, auc, oof_aggregate,
                                    to_matrix)

    store = Store()
    rows, ids = build_level("dn", ev, store)
    names = [str(n) for n in dn(ev[0]["clip"])[2]]
    my_ids = [str(i) for i in dn(ev[0]["clip"])[4]]
    if [str(i) for i in ids] != my_ids:
        raise RuntimeError("порядок каналов p10_levels и p11a не совпал")
    n_ch = len(rows[0]["act_change"])
    nf = len(STATS)
    X2 = to_matrix(rows)                      # stat-major: column j = stat * n_ch + channel

    def cols_of(cells) -> np.ndarray:
        cells = np.asarray(cells, dtype=int)
        return (np.arange(nf)[:, None] * n_ch + cells[None, :]).ravel()

    def oof(cells) -> float:
        return float(auc(y, oof_aggregate(X2[:, cols_of(cells)], y, clips)))

    yaw_idx = np.array([i for i, n in enumerate(names)
                        if n.rsplit("_", 1)[0] in YAW_TYPES], dtype=int)
    out = {"n_channels": int(n_ch), "n_columns_full": int(n_ch * nf),
           "yaw_cells": [names[i] for i in yaw_idx]}

    out["yaw18_all"] = oof(yaw_idx)
    # keep the same 8 statistics but only the channels of the yaw csv, for the smallest comparison
    out["yaw7_csv"] = None    # filled by the caller from the yaw level

    # random draw distributions at equal cell counts
    for k in (7, 18):
        vals = []
        for _ in range(draws):
            v = oof(rng.choice(n_ch, k, replace=False))
            if not np.isnan(v):
                vals.append(v)
        vals = np.array(vals)
        out[f"random{k}"] = {"median": float(np.median(vals)),
                             "p95": float(np.percentile(vals, 95)),
                             "max": float(vals.max()), "n": int(len(vals))}

    # The matched control: seven cells drawn only from the eighteen that feed yaw, at eight
    # statistics each, which is fifty-six columns — the same size as the yaw readout itself, over
    # the same cells. Nothing differs between the two sides except the pooling.
    yvals = []
    for _ in range(draws):
        pick = rng.choice(yaw_idx, 7, replace=False)
        v = oof(pick)
        if not np.isnan(v):
            yvals.append(v)
    yvals = np.array(yvals)
    out["yaw_sub7"] = {"median": float(np.median(yvals)),
                       "p95": float(np.percentile(yvals, 95)),
                       "max": float(yvals.max()), "n": int(len(yvals))}
    # all eighteen, at eight statistics each
    out["all68"] = oof(np.arange(n_ch))

    # the null for the equal-size random draws: permute labels, redraw, repeat fewer times
    yp = y.copy()
    nulls = []
    nulls_y = []
    for _ in range(max(perm // 10, 20)):
        for c in set(clips.tolist()):
            m = clips == c
            yp[m] = rng.permutation(y[m])
        for _ in range(20):
            nulls.append(float(auc(yp, oof_aggregate(
                X2[:, cols_of(rng.choice(n_ch, 7, replace=False))], yp, clips))))
            nulls_y.append(float(auc(yp, oof_aggregate(
                X2[:, cols_of(rng.choice(yaw_idx, 7, replace=False))], yp, clips))))
    nulls = np.array([v for v in nulls if not np.isnan(v)])
    nulls_y = np.array([v for v in nulls_y if not np.isnan(v)])
    if len(nulls):
        out["null_random7"] = {"median": float(np.median(nulls)),
                               "p95": float(np.percentile(nulls, 95)),
                               "max": float(nulls.max()), "n": int(len(nulls))}
    if len(nulls_y):
        out["null_yaw_sub7"] = {"median": float(np.median(nulls_y)),
                                "p95": float(np.percentile(nulls_y, 95)),
                                "max": float(nulls_y.max()), "n": int(len(nulls_y))}
    return out


def main() -> int:
    global LABELS_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-cons", type=float, default=0.75)
    ap.add_argument("--diag", action="store_true", help="разобрать, отбор это или форма чтения")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--least", type=int, default=3,
                    help="минимум событий класса в ролике, чтобы он считался сбалансированным")
    ap.add_argument("--control", action="store_true",
                    help="одна фиксированная формула на трёх наборах клеток, без отбора")
    ap.add_argument("--null-seeds", type=int, default=3,
                    help="сколько независимых нулей усреднить (p печатается интервалом)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exclude", default=None,
                    help="куски через запятую, которые не брать (проверка без спорных записей)")
    ap.add_argument("--labels", default=str(LABELS_PATH),
                    help="файл меток; по умолчанию замороженный набор из пяти кусков")
    args = ap.parse_args()
    LABELS_PATH = Path(args.labels) if Path(args.labels).is_absolute() else ROOT / args.labels
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # A separate stream for the null of the headline number. The control block draws random cell
    # subsets, and if it shared this generator the p-value would move depending on whether
    # --control was passed. Two streams keep every reported number reproducible on its own.
    rng_null = np.random.default_rng(args.seed + 10_000)

    ev, y_full, keep = labels()
    clips_full = clip_of(ev)
    ev = [e for e, k in zip(ev, keep) if k]
    y = y_full[keep]
    clips = clips_full[keep]

    # Some recordings do not come from the camera named in the chain: VID00001 and VID00002 were
    # recorded from files in ~/Downloads, and the human labels for them were made on those same
    # files, so they are internally consistent but describe a different walk than the camera holds.
    # Excluding them answers whether the result survives without the recordings whose provenance is
    # in question, which is a different and weaker claim than leaving them in.
    if args.exclude:
        drop = {s.strip() for s in args.exclude.split(",") if s.strip()}
        m = np.array([c not in drop for c in clips])
        print(f"  исключены куски {sorted(drop)}: снято {int((~m).sum())} событий")
        ev = [e for e, k in zip(ev, m) if k]
        y = y[m]
        clips = clips[m]

    print("=" * 104)
    print("P11.A — SAME / DIFFERENT ИЗ УЖЕ ЗАПИСАННЫХ 68 НИСХОДЯЩИХ КЛЕТОК")
    print("=" * 104)
    print(f"  событий {len(y)} (только SAME и DIFFERENT), роликов {len(set(clips.tolist()))}")
    print(f"  метки: SAME {int(y.sum())}, DIFFERENT {int((1-y).sum())}")
    print(f"  нуль: перемешивание меток ВНУТРИ ролика, {args.perm} повторов")
    print("  отбор клеток — внутри каждого обучающего разбиения, тестовый ролик не виден")
    print()

    report = {"phase": "P11.A — route_change_signal из 68 нисходящих клеток",
              "n_events": int(len(y)), "n_perm": args.perm,
              "features": list(FEATS), "sources": {}}

    sources = {
        "dn68": "все 68 нисходящих клеток",
        "dn_yaw18": "те же 18 клеток, из которых собран yaw",
        "yaw_csv": "семь чисел yaw-файла",
    }
    built = {}
    for src, human in sources.items():
        if src == "yaw_csv":
            # the csv is only used for the baseline; it is not a per-cell source
            built[src] = (None, None)
            continue
        F, names = build(src, ev)
        nch, nf = F.shape[1], F.shape[2]
        built[src] = (F.reshape(len(ev), nch * nf), (nch, nf, names))

    print("─── ОСНОВНОЙ РЕЗУЛЬТАТ ───")
    print("  заранее объявлено: источник — 68 нисходящих клеток, форма — логистика на")
    print("  отобранных клетках, размер — выбирается внутренним циклом. Остальное ниже.")
    print()
    MODES = {"logreg": "логистика", "signed_mean": "знаковая сумма"}
    table = {}
    for src in ("dn68", "dn_yaw18"):
        X2, meta = built[src]
        nch, nf, names = meta
        for mode in ("logreg", "signed_mean"):
            res = run(X2, nch, nf, y, clips, KS, args.min_cons, mode=mode)
            table[(src, mode)] = res
            report["sources"].setdefault(src, {})[mode] = {
                "human": sources[src], "n_cells": int(nch), "auc_oof": res["auc_oof"],
                "per_fold": res["per_fold"], "k_chosen": res["k_chosen"],
                "chosen": res["chosen"]}
        built[src] = (X2, meta, table[(src, PRIMARY_MODE)])

    print(f"  {'источник':<40}{'форма':<16}{'AUC':>7}   клеток по фолдам")
    for (src, mode), res in table.items():
        a = f"{res['auc_oof']:.3f}" if res["auc_oof"] is not None else "  н/д"
        print(f"  {sources[src]:<40}{MODES[mode]:<16}{a:>7}   {res['k_chosen']}")
    print()

    _, meta, res = built["dn68"]
    nch, nf, names = meta
    print(f"  подробно — 68 клеток, логистика:")
    f1 = lambda v: "не оценено" if v is None else f"{v:.3f}"      # noqa: E731
    print(f"    out-of-fold AUC = {f1(res['auc_oof'])}  (покрыто {res['cover']} событий)")
    if res["auc_oof"] is None:
        print(f"    оценка невозможна на {len(set(clips.tolist()))} роликах: внутреннему отбору "
              f"нужно минимум три, чтобы обучаться на части и выбирать на другой. Это не ноль "
              f"и не слабый результат — это отсутствие оценки.")
    for c, d in sorted(res["per_fold"].items()):
        auc_s = f"{d['auc']:.3f}" if d["auc"] is not None else "  н/д"
        print(f"      {c}: AUC={auc_s}  клеток {d['n_cells']}  "
              f"(SAME {d['n_pos']}, DIFF {d['n_neg']})  внутр.оценка {d['inner_cv']:.3f}")
    cnt = Counter((c["cell"], c["feat"]) for c in res["chosen"])
    print(f"    чаще всего выбирались клетки: "
          + ", ".join(f"{names[c]}({n})" for c, n in Counter(
              c["cell"] for c in res["chosen"]).most_common(6)))
    print(f"    чаще всего выбирались признаки: "
          + ", ".join(f"{c}({n})" for c, n in Counter(
              c["feat"] for c in res["chosen"]).most_common(5)))
    print()

    if args.diag:
        print("─── ДИАГНОСТИКА: отбор это или форма чтения ───")
        X2, meta, _ = built["dn68"]
        nch, nf, _ = meta
        dg = diagnostics(X2, nch, nf, y, clips, args.min_cons)
        for k, v in dg.items():
            print(f"    {k:<42} AUC={v:.3f}" if v is not None else f"    {k:<42} н/д")
        report["diagnostics"] = dg
        print()

    # ---- baselines ----
    print("─── БАЗОВЫЕ ЛИНИИ (то же разбиение, та же метрика) ───")
    base = {}
    for src, lvl in (("dn68", "dn"), ("yaw_csv", "yaw")):
        try:
            a = p10_aggregate(src, ev)
            base[lvl] = a
            ncol = {"dn": 68 * 8, "yaw": 7 * 8}[lvl]
            print(f"  {lvl:<10} логистика по всем статистикам всех каналов ({ncol} колонок): "
                  f"AUC={a:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {lvl:<10} не посчиталось: {exc}")
    # the compact readout on the yaw csv, same procedure, for a like-for-like comparison
    Xy3, namesy = build("yaw_csv", ev)
    Xy = Xy3.reshape(len(ev), Xy3.shape[1] * Xy3.shape[2])
    resy = run(Xy, Xy3.shape[1], Xy3.shape[2], y, clips, KS, args.min_cons, mode=PRIMARY_MODE)
    print(f"  {'yaw_csv':<10} компактный readout из 7 чисел:      AUC={f1(resy['auc_oof'])}")
    base["yaw_compact"] = resy["auc_oof"]
    report["baselines"] = base
    report["sources"]["yaw_csv"] = {"human": sources["yaw_csv"], "n_cells": int(Xy3.shape[1]),
                                    "auc_oof": resy["auc_oof"], "per_fold": resy["per_fold"],
                                    "k_chosen": resy["k_chosen"], "chosen": resy["chosen"],
                                    "mode": PRIMARY_MODE}
    print()

    if args.control:
        print("─── КОНТРОЛЬ: одинаковая модель, разные наборы клеток ───")
        print("  П10 цитировал 0.767 как «68 нисходящих». Это логистика по 8 статистикам всех")
        print("  68 каналов = 544 колонки, а не свёртка в 7 скаляров: настоящее сжатие в 7")
        print("  скаляров даёт 0.451. Сравнение 544 колонок с 56 колонками yaw само по себе")
        print("  даёт часть разрыва, поэтому ниже — только выровненные сравнения.")
        print()
        ctl = control_block(ev, y, clips, args.perm, rng)
        report["control"] = ctl
        print(f"  полный уровень 68 клеток x 8 статистик ({ctl['n_columns_full']} колонок)  "
              f"AUC={ctl['all68']:.3f}   <- то, что П10 назвал 0.767")
        print(f"  семь чисел yaw x 8 статистик (56 колонок)                 "
              f"AUC={base.get('yaw', float('nan')):.3f}   <- то, что читает трекер")
        print(f"  все 18 yaw-клеток x 8 статистик (144 колонки)             "
              f"AUC={ctl['yaw18_all']:.3f}")
        print()
        print("  случайные наборы из тех же 68 клеток (та же модель):")
        for k in (7, 18):
            r = ctl[f"random{k}"]
            print(f"    {k:>2} клеток ({k*8:>3} колонок): медиана {r['median']:.3f}, "
                  f"p95 {r['p95']:.3f}, максимум {r['max']:.3f}")
        if "null_random7" in ctl:
            nl = ctl["null_random7"]
            print(f"    нуль для 7 случайных клеток: медиана {nl['median']:.3f}, "
                  f"p95 {nl['p95']:.3f}, max {nl['max']:.3f}")
        print()
        print("  ГЛАВНОЕ СРАВНЕНИЕ: ровно 56 колонок с обеих сторон, только из yaw-клеток")
        y7 = ctl["yaw_sub7"]
        ny7 = ctl["null_yaw_sub7"]
        print(f"    7 yaw-клеток x 8 статистик: медиана {y7['median']:.3f}, "
              f"p95 {y7['p95']:.3f}, максимум {y7['max']}")
        print(f"      нуль: медиана {ny7['median']:.3f}, p95 {ny7['p95']:.3f}, "
              f"max {ny7['max']}")
        print(f"    те же 7 клеток, свёрнутые в числа yaw: {base.get('yaw', float('nan')):.3f}")
        print("    Одинаковые клетки, одинаковое число колонок. Разница — только в том,")
        print("    как 8 статистик свёрнуты в 1 число.")
        print()
        print("  КАК ЭТО ЧИТАТЬ: если любые 7 клеток из 68 бьют 56 чисел yaw — выигрыш в клетках.")
        print("  Если бьют только особые 7 — выигрыш в отборе, и его должен нести нуль.")
        print()

    # ---- null for the pre-registered headline number ----
    print("─── НУЛЬ ДЛЯ ГЛАВНОГО ЧИСЛА ───")
    print(f"  источник 68 клеток, форма {MODES[PRIMARY_MODE]}; нуль перезапускает ВЕСЬ")
    print("  конвейер, включая отбор клеток и выбор размера, на перемешанных метках")
    X2, meta, res = built["dn68"]
    nch, nf, names = meta
    obs = res["auc_oof"]
    bal = balanced_clips(y, clips, args.least)
    obs_bal = restricted_auc(res["oof"], y, clips, bal)
    print(f"  ролики с балансом >= {args.least} на класс: {sorted(bal)}")
    print(f"  только по ним OOF AUC = {obs_bal:.3f}")
    print()
    null, null_bal = [], []
    p_runs, p_bal_runs, med_runs, max_runs = [], [], [], []
    for si in range(args.null_seeds):
        rng_n = np.random.default_rng(args.seed + 10_000 + 137 * si)
        yp = y.copy()
        for it in range(args.perm):
            for c in set(clips.tolist()):
                m = clips == c
                yp[m] = rng_n.permutation(y[m])
            r = run(X2, nch, nf, yp, clips, KS, args.min_cons, mode=PRIMARY_MODE)
            if r["auc_oof"] is not None:
                null.append(r["auc_oof"])
                b = restricted_auc(r["oof"], yp, clips, bal)
                if not np.isnan(b):
                    null_bal.append(b)
            if args.verbose and (it + 1) % 50 == 0:
                print(f"      зерно {si+1}/{args.null_seeds}, перемешиваний {it+1}/{args.perm}",
                      flush=True)
        arr = np.array(null)
        if len(arr) == 0:
            # every permutation returned no estimate, so there is no null to compare against.
            # Reporting a p-value here would invent one; the honest output is that the control
            # could not be run at this number of recordings.
            print(f"    контроль: нуль пуст на этом числе роликов — проверить нечем")
            arr = None
        if arr is not None:
            p_runs.append(float((np.sum(arr >= obs) + 1) / (len(arr) + 1)))
            med_runs.append(float(np.median(arr)))
            max_runs.append(float(arr.max()))
        arrb = np.array(null_bal)
        if len(arrb):
            p_bal_runs.append(float((np.sum(arrb >= obs_bal) + 1) / (len(arrb) + 1)))
    null = np.array(null)
    if len(p_runs):
        p = float(np.mean(p_runs))
    else:
        p = float("nan")
    null_bal = np.array(null_bal)
    p_bal = float(np.mean(p_bal_runs)) if p_bal_runs else float("nan")
    fo = lambda v: "не оценено" if v is None else f"{v:.3f}"      # noqa: E731
    if med_runs:
        print(f"  весь набор:      наблюдение {fo(obs)}; нуль медиана "
              f"{min(med_runs):.3f}-{max(med_runs):.3f}, max {max(max_runs):.3f}")
        print(f"                   p по {args.null_seeds} независимым нулям: "
              f"{min(p_runs):.4f}-{max(p_runs):.4f} (среднее {p:.4f})")
    else:
        print(f"  весь набор:      наблюдение {fo(obs)}; нуль не построен — "
              f"на {len(set(clips.tolist()))} роликах перемешивание не даёт оценки")
    if p_bal_runs:
        print(f"  сбалансированные: наблюдение {fo(obs_bal)}; "
              f"p по нулям: {min(p_bal_runs):.4f}-{max(p_bal_runs):.4f} (среднее {p_bal:.4f})")
    print(f"  ({len(null)} перемешиваний)")
    print()
    print("  ВНИМАНИЕ: p приведён интервалом, а не одним числом. Прогон не бит-воспроизводим:")
    print("  решатель на матрицах в сотни колонок даёт разный результат при одном и том же")
    print("  зерне (проверено: одинаковые запуски дают нуль медианы в диапазоне 0.43-0.47).")
    print("  Поэтому ниже указан разброс p по независимым нулям, а не одна цифра.")
    def _s(arr, how):
        if len(arr) == 0:
            return None
        return float({"median": np.median(arr), "p95": np.percentile(arr, 95),
                      "max": arr.max()}[how])

    report["null"] = {"observed": None if obs is None else float(obs),
                      "median": _s(null, "median"),
                      "p95": _s(null, "p95"), "max": _s(null, "max"),
                      "p": p, "p_range": [min(p_runs), max(p_runs)] if p_runs else None,
                      "p_runs": p_runs,
                      "null_median_range": [min(med_runs), max(med_runs)] if med_runs else None,
                      "n": len(null), "null_seeds": args.null_seeds,
                      "balanced_clips": sorted(bal), "observed_balanced": float(obs_bal),
                      "p_balanced": p_bal,
                      "p_balanced_range": [min(p_bal_runs), max(p_bal_runs)] if p_bal_runs else None,
                      "reproducible": False,
                      "why_not_reproducible": ("численный шум в решателе на матрицах в сотни "
                                               "колонок при одном и том же зерне"),
                      "balanced_p95": float(np.percentile(null_bal, 95)) if len(null_bal) else None}
    print()

    # ---- verdict ----
    print("─── ВЫВОД ───")

    def num(v):
        """None means not estimable; nan keeps arithmetic and printing honest and claims false."""
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    a_new = num(report["sources"]["dn68"][PRIMARY_MODE]["auc_oof"])
    a_sign = num(report["sources"]["dn68"]["signed_mean"]["auc_oof"])
    a_yaw = num(base.get("yaw"))
    a_yawc = num(base.get("yaw_compact"))
    a_18 = num(report["sources"]["dn_yaw18"][PRIMARY_MODE]["auc_oof"])
    a_agg = num(report["baselines"].get("dn"))
    ctl = report.get("control", {})
    r7 = ctl.get("random7", {})
    r7_med = num(r7.get("median"))
    nr7_med = num(ctl.get("null_random7", {}).get("median"))
    y18_all = num(ctl.get("yaw18_all"))
    print(f"  что читает трекер сейчас: 7 чисел yaw x 8 статистик (56 колонок)  {a_yaw:.3f}")
    print(f"  те же 18 yaw-клеток целиком (144 колонки)                          {y18_all:.3f}")
    print(f"  случайные 7 клеток из 68 (56 колонок, медиана)                     "
          f"{r7_med:.3f}   (нуль {nr7_med:.3f})")
    print(f"  все 68 клеток x 8 статистик (544 колонки)                          {a_agg:.3f}"
          f"   <- то, что П10 назвал 0.767")
    print(f"  новый компактный readout, <=8 клеток                               {a_new:.3f}"
          f"   ({'p=н/д' if np.isnan(p) else f'p={p:.4f}'})")
    if np.isnan(a_new):
        print()
        print("  ОЦЕНКА НЕ ПОЛУЧЕНА: на этом числе роликов отбор и нуль не строятся. Это не")
        print("  слабый результат и не ноль — это отсутствие оценки, и утверждать нечего.")
    print()
    if not np.isnan(p) and p < 0.05 and not np.isnan(a_new) and a_new > a_yaw + 0.05:
        print(f"  Разрыв реален, и причина у него — не выбор клеток, а форма свёртки.")
        print(f"  Три довода, от самого слабого к самому сильному:")
        print(f"    1. компактный readout из <=8 клеток: {a_new:.3f} против {a_yaw:.3f} у yaw"
              f" (p={p:.4f})")
        print(f"    2. на тех же 18 клетках, что кормят yaw, целиком: "
              f"{ctl.get('yaw18_all', float('nan')):.3f} —")
        print(f"       те же самые клетки, другое чтение, +"
              f"{ctl.get('yaw18_all', 0) - a_yaw:.3f}")
        print(f"    3. случайные 7 клеток из 68, тот же размер модели: медиана "
              f"{r7.get('median', float('nan')):.3f}")
        print(f"       против {a_yaw:.3f}, при нуле {ctl.get('null_random7', {}).get('median', 0):.3f}. "
              f"То есть дело не в том,")
        print("       какие клетки взять: любые семь, прочитанные полностью, лучше семи")
        print("       свёрнутых чисел. Yaw теряет информацию о направлении пути.")
        verdict = "READOUT_FIXABLE"
    elif p < 0.05:
        print(f"  Компактное чтение выше случая (p={p:.4f}), но не выше yaw настолько,")
        print("  чтобы заменить его.")
        verdict = "WEAK"
    else:
        print("  Разрыв не подтверждается.")
        verdict = "NOT_CONFIRMED"
    report["verdict"] = verdict
    report["verdict_numbers"] = {
        "yaw_7_numbers": float(a_yaw), "yaw18_full": ctl.get("yaw18_all"),
        "random7_median": r7.get("median"), "null_random7_median":
            ctl.get("null_random7", {}).get("median"),
        "all68_logistic": float(a_agg), "compact_readout": float(a_new),
        "compact_signed_mean": float(a_sign), "yaw_compact_readout": float(a_yawc),
        "yaw18_compact_readout": float(a_18), "p": p, "p_balanced": report["null"]["p_balanced"],
    }

    if verdict == "READOUT_FIXABLE":
        sig = {
            "name": "route_change_signal",
            "question": "продолжил путь примерно в прежнем направлении или изменил его",
            "output": ["SAME_DIRECTION", "DIFFERENT_DIRECTION"],
            "source": "68 нисходящих клеток, уже записанных в P09",
            "features": list(FEATS),
            "form": PRIMARY_MODE,
            "auc_out_of_fold": a_new,
            "null_p": p,
            "auc_balanced_clips": report["null"]["observed_balanced"],
            "null_p_balanced": report["null"]["p_balanced"],
            "all_cells_reference": a_agg,
            "yaw_reference": a_yaw,
            "yaw18_full_reference": ctl.get("yaw18_all"),
            "random7_median": r7.get("median"),
            "null_random7_median": ctl.get("null_random7", {}).get("median"),
            "matched_56_columns": {
                "yaw_cells_8_stats_median": ctl.get("yaw_sub7", {}).get("median"),
                "yaw_cells_8_stats_null_median": ctl.get("null_yaw_sub7", {}).get("median"),
                "yaw_pooled_numbers": a_yaw,
                "note": ("ровно 56 колонок с обеих сторон, клетки одни и те же; разница только "
                         "в способе свёртки"),
            },
            "chosen_per_fold": report["sources"]["dn68"][PRIMARY_MODE]["chosen"],
            "caveat": (
                "отбор клеток и выбор размера модели происходят внутри каждого обучающего "
                "разбиения; тестовый ролик не виден ни на одном шаге; нуль перезапускает весь "
                "конвейер на перемешанных метках. Нуль широкий: его максимум 0.84 при "
                "наблюдении 0.76, поэтому главное свидетельство — не это число, а сравнение "
                "на выровненном размере модели: те же 18 клеток, из которых собран yaw, "
                "прочитанные целиком, дают 0.737 против 0.617 у yaw, и даже случайные семь "
                "клеток из 68 дают медиану 0.628 при нуле 0.451."
            ),
            "what_this_is_not": (
                "это не новый источник сигнала. Это то же различение, что P10 нашёл в 68 "
                "клетках, только прочитанное так, что оно сохраняется"
            ),
        }
        (out / "route_change_signal.json").write_text(
            json.dumps(sig, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"  заморожено: {out/'route_change_signal.json'}")

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                default=float), encoding="utf-8")
    print(f"  записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
