#!/usr/bin/env python3
"""P10 — where the information about body movement is lost, measured one level at a time.

THE CHECK
---------
One procedure, run identically at every level, so that levels can be compared with each other and
not merely with chance:

    window       three seconds before the reviewed moment and seven after. Pre-registered in the
                 review interface and not adjusted here.
    features     eight statistics per channel (signed state before and after, magnitude before and
                 after, the change in activity, the change in the signed pattern, whether the sign
                 flipped, the net change in magnitude), reduced to seven physical scalars per
                 level: did it get busier, did the signed pattern move, did anything reverse, how
                 big was the before state, how big the after state, how coherent was the after
                 direction, how did the ratio move. The same seven at every level.
    split        leave one recording out. Train on the other four, score the fifth, repeat.
                 No window from a held-out recording is ever seen during training. This matters
                 more than it looks: two windows from the same walk share a camera, a wearer, a
                 corridor and a lighting condition, so a split that mixes windows of one recording
                 between the two sides measures recognition of the recording, not of the movement.
    model        L2 logistic regression, standardised on the training folds only. Deliberately
                 small: sixty events over five recordings cannot support anything larger.
    probe two    the single best (channel, statistic) chosen inside the training folds and applied
                 to the held-out one. This asks a different and stricter question — not "is there
                 some combination", but "does any individual channel carry this at all".

WHY THE NULL DECIDES THE ANSWER
-------------------------------
The sixty events are not spread evenly across recordings: VID00001 gave five of its six moves as
SAME_DIRECTION and none as NO_NET_DISPLACEMENT, VID00006 gave three of four as SAME_DIRECTION,
while VID00005 and VID00009 are balanced. Anything that separates recordings from one another will
therefore travel with the label. In this arrangement a held-out AUC of 0.65 can mean nothing at
all, and reporting it against 0.5 would be wrong.

Every number here is therefore reported against a null in which the labels are permuted *within
each recording*, keeping that recording's exact mix of categories. Whatever confounding exists in
the real labels exists identically in the null. A level counts as carrying information only if it
rises above that null, and the p-values say by how much.

Usage:
    PYTHONPATH=. python scripts/p10_levels.py [--perm 300]
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
OUT = ROOT / "output/p10"
EPS = 1e-9

STATS = ("signed_before", "signed_after", "mag_before", "mag_after",
         "act_change", "dir_change", "reversal", "abs_diff")

LEVELS = ("video_full", "video_band", "input", "early", "dn", "yaw", "forward")
LAUNCH = {
    "video_full": "сырое видео, весь кадр",
    "video_band": "сырое видео, центральная полоса",
    "input": "вход в мозг: T4a/b, T5a/b, LC4, LPLC2",
    "early": "ранние зрительные клетки, 20 популяций",
    "dn": "68 нисходящих клеток",
    "yaw": "текущий yaw",
    "forward": "текущий forward",
}


# ------------------------------------------------------------------ signal sources

class Store:
    """One npz/csv per recording, loaded once."""

    def __init__(self):
        self.cache: dict = {}

    def get(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


def _video(store: Store, stem: str):
    return store.get(f"v:{stem}", lambda: (
        np.load(OUT / f"video_feat_{stem}.npz")
        if (OUT / f"video_feat_{stem}.npz").exists() else None))


def video_signals(ev: dict, store: Store, band: bool):
    d = _video(store, Path(ev["clip_file"]).stem)
    if d is None:
        return None
    pre = "c_" if band else "f_"
    keys = sorted(k for k in d.files if k.startswith(pre))
    if not keys:
        return None
    t = d["t"].astype(float) - float(ev["clip_offset_s"])
    return t, np.stack([d[k].astype(float) for k in keys], axis=1), keys


def brain_signals(ev: dict, store: Store, which: str):
    v = ev["clip"]
    d = store.get(f"b:{v}", lambda: (
        np.load(OUT / f"brain_{v}.npz") if (OUT / f"brain_{v}.npz").exists() else None))
    if d is None:
        return None
    ig = [str(x) for x in d["inject_groups"]]
    vg = [str(x) for x in d["vis_groups"]]
    if which == "input":
        X, keys = d["inject"].astype(float), ig
    else:
        X, keys = d["vis_rate"].astype(float), vg
    if X.size == 0:
        return None
    return d["t"].astype(float) - float(ev["time"]), X, keys


def dn_signals(ev: dict, store: Store):
    v = ev["clip"]
    d = store.get(f"d:{v}", lambda: _load_dn(v))
    if d is None:
        return None
    t, X, keys = d
    return t - float(ev["time"]), X, keys


def _load_dn(v: str):
    p = ROOT / f"output/p09/trace_{v}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    cells = d["cells"].astype(int)
    _uniq, first = np.unique(cells, return_index=True)
    fired = d["fired"][:, first].astype(float)
    # Sparse spike trains carry almost no level as 0/1 per step; half a second of integration.
    k = 25
    sm = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode="same"), 0, fired)
    keys = [str(c) for c in cells[first]]
    return d["t"].astype(float), sm, keys


def csv_signals(ev: dict, store: Store, which: str):
    v = ev["clip"]

    def load():
        p = ROOT / f"output/p07/{which}_signal_{v}.csv"
        if not p.exists():
            return None
        rows = list(csv.DictReader(p.open()))
        if not rows:
            return None
        t = np.array([float(r["t"]) for r in rows])
        if which == "yaw":
            cols = ["yaw_signal", "yaw_signal_deadband", "yaw_pair_diagnostic"]
            cols += [k for k in rows[0] if k.startswith("common_")]
        else:
            cols = ["forward_signal", "forward_raw", "speed"]
        cols = [c for c in cols if c in rows[0]]
        X = np.stack([np.array([float(r[c]) for r in rows]) for c in cols], axis=1)
        return (t, X, cols)

    d = store.get(f"{which}:{v}", load)
    if d is None:
        return None
    return d[0] - float(ev["time"]), d[1], d[2]


# ------------------------------------------------------------------ window work

def window_stats(t: np.ndarray, X: np.ndarray, pre: float = 3.0, post: float = 7.0):
    mb = (t >= -pre) & (t <= -0.5)
    ma = (t >= 0.5) & (t <= post)
    if mb.sum() < 3 or ma.sum() < 3:
        return None
    B, A = X[mb], X[ma]
    sb, sa = B.mean(axis=0), A.mean(axis=0)
    ab, aa = np.abs(B).mean(axis=0), np.abs(A).mean(axis=0)
    return {
        "signed_before": sb, "signed_after": sa,
        "mag_before": ab, "mag_after": aa,
        "act_change": (aa - ab) / (aa + ab + EPS),
        "dir_change": np.abs(sa - sb) / (np.abs(sa) + np.abs(sb) + EPS),
        "reversal": (sb * sa < 0).astype(float),
        "abs_diff": np.abs(sa) - np.abs(sb),
    }


def to_matrix(rows: list[dict]) -> np.ndarray:
    """Columns in stat-major order: statistic first, then channel."""
    return np.concatenate([np.stack([r[st] for r in rows]) for st in STATS], axis=1)


def aggregates(w: dict) -> np.ndarray:
    return np.array([
        float(np.median(w["act_change"])),
        float(np.median(w["dir_change"])),
        float(np.mean(w["reversal"])),
        float(np.median(w["mag_after"])),
        float(np.median(w["mag_before"])),
        float(np.median(np.abs(w["signed_after"]))),
        float(np.median(w["mag_after"] / (w["mag_before"] + EPS))),
    ])


AGG_NAMES = ("рост_активности", "смена_знака_паттерна", "доля_разворотов",
             "уровень_после", "уровень_до", "направленность_после", "отношение_после_до")


# ------------------------------------------------------------------ the check

def auc(y: np.ndarray, score: np.ndarray) -> float:
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = float(np.sum(pos[:, None] > neg[None, :]))
    eq = float(np.sum(pos[:, None] == neg[None, :]))
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def _logreg(Z: np.ndarray, y: np.ndarray, l2: float, iters: int = 60) -> tuple[np.ndarray, float]:
    """L2 logistic regression by iteratively reweighted least squares.

    Written out rather than imported: the dependency is not present, the problem is seven
    dimensions and sixty rows, and a closed-form step converges in a handful of iterations. The
    ridge term is added to the diagonal before every solve, so a collinear or constant column
    degrades the fit instead of raising.
    """
    n, d = Z.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = np.clip(Z @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.maximum(p * (1.0 - p), 1e-9)
        grad = Z.T @ (p - y) / n + l2 * w
        gb = float(np.mean(p - y))
        H = (Z * W[:, None]).T @ Z / n
        Hb = Z.T @ W / n
        Hbb = float(np.mean(W))
        H = H + (l2 + 1e-9) * np.eye(d)
        try:
            sol = np.linalg.solve(
                np.block([[H, Hb[:, None]], [Hb[None, :], np.array([[Hbb + 1e-9]])]]),
                np.concatenate([grad, [gb]]))
            dw, db = sol[:d], float(sol[d])
        except np.linalg.LinAlgError:
            dw, db = grad * 0.1, gb * 0.1
        w -= dw
        b -= db
        if np.max(np.abs(dw)) + abs(db) < 1e-9:
            break
    return w, b


def fit_score(Xtr, ytr, Xte, C=1.0):
    """Standardise on the training folds only — the held-out rows never touch mu or sd."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    w, b = _logreg((Xtr - mu) / sd, ytr.astype(float), l2=1.0 / max(C, 1e-6))
    z = np.clip((Xte - mu) / sd @ w + b, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def oof_aggregate(X, y, clips, C=1.0):
    out = np.full(len(y), np.nan)
    for c in np.unique(clips):
        te, tr = clips == c, clips != c
        if len(np.unique(y[tr])) < 2:
            continue
        out[te] = fit_score(X[tr], y[tr], X[te], C)
    return out


def oof_best_signal(X, y, clips, names):
    """Choose the one column that separates the training folds best; apply it to the held-out.

    Returns the held-out scores and, for each fold, which statistic of which channel was chosen.
    The second return value is diagnostic only — it is how we find out *which* quantity carried
    the separation, instead of reporting that some combination of ninety-six columns did.
    """
    out = np.full(len(y), np.nan)
    chosen = []
    for c in np.unique(clips):
        te, tr = clips == c, clips != c
        a = np.array([auc(y[tr], X[tr, j]) for j in range(X.shape[1])])
        if np.all(np.isnan(a)):
            continue
        dev = np.abs(a - 0.5)
        j = int(np.nanargmax(dev))
        sign = 1.0 if a[j] >= 0.5 else -1.0
        out[te] = sign * X[te, j]
        # to_matrix lays the columns out statistic-major: every channel for statistic 0, then
        # every channel for statistic 1. Decoding the column index the other way round would
        # report a real column under a wrong name, which is worse than reporting nothing.
        n_ch = len(names) if names else X.shape[1] // len(STATS)
        stat_i, ch_i = divmod(j, max(n_ch, 1))
        chosen.append({"fold": str(c), "column": int(j), "stat": STATS[stat_i],
                       "channel": str(names[ch_i]) if names and ch_i < len(names) else str(ch_i),
                       "train_auc": float(a[j]), "sign": sign})
    return out, chosen


def permutation_null(X, y, clips, kind, names, rng, n_perm):
    yp = y.copy()
    vals = []
    for _ in range(n_perm):
        for c in np.unique(clips):
            m = clips == c
            yp[m] = rng.permutation(y[m])
        if kind == "agg":
            s = oof_aggregate(X, yp, clips)
        else:
            s = oof_best_signal(X, yp, clips, names)[0]
        a = auc(yp, s)
        if not np.isnan(a):
            vals.append(a)
    return np.array(vals)


def per_recording(y: np.ndarray, score: np.ndarray, clips: np.ndarray) -> dict:
    """AUC inside each recording, from the out-of-fold scores.

    The aggregate alone cannot distinguish an effect that holds across recordings from one that a
    single recording produces. With five recordings that distinction is the whole question, so it
    is reported for every level rather than left to be reconstructed by hand.
    """
    out = {}
    for c in sorted(set(clips.tolist())):
        m = clips == c
        yy, ss = y[m], score[m]
        ok = ~np.isnan(ss)
        a = auc(yy[ok], ss[ok]) if ok.any() else float("nan")
        pos, neg = int((yy[ok] == 1).sum()), int((yy[ok] == 0).sum())
        out[c] = {"auc": None if np.isnan(a) else round(float(a), 4),
                  "n": int(m.sum()), "pos": pos, "neg": neg}
    return out


def evaluate(X, y, clips, names, rng, n_perm):
    oof = oof_aggregate(X, y, clips)
    obs_a = auc(y, oof)
    sig_score, chosen = oof_best_signal(X, y, clips, names)
    obs_b = auc(y, sig_score)
    null_a = permutation_null(X, y, clips, "agg", names, rng, n_perm)
    null_b = permutation_null(X, y, clips, "sig", names, rng, n_perm)

    def p(o, nl):
        if np.isnan(o) or len(nl) == 0:
            return float("nan")
        return float((np.sum(nl >= o) + 1) / (len(nl) + 1))

    def stat(nl, how):
        """An empty permutation null means the estimate does not exist, not that it is zero.

        With two recordings there is nothing to permute in some folds, and the earlier code called
        np.median on the empty result and died with an IndexError. A tool that cannot say "not
        estimable" forces the reader to guess, which is how a mixture of provenances produced a
        confident-looking number in the first place.
        """
        if len(nl) == 0:
            return None
        return float(np.median(nl) if how == "median" else np.percentile(nl, 95))

    return {
        "auc_aggregate": float(obs_a), "p_aggregate": p(obs_a, null_a),
        "null_aggregate_n": int(len(null_a)),
        "null_aggregate_median": stat(null_a, "median"),
        "null_aggregate_p95": stat(null_a, "p95"),
        "auc_best_channel": float(obs_b), "p_best_channel": p(obs_b, null_b),
        "null_channel_n": int(len(null_b)),
        "null_channel_median": stat(null_b, "median"),
        "null_channel_p95": stat(null_b, "p95"),
        "per_recording_aggregate": per_recording(y, oof, clips),
        "per_recording_best_channel": per_recording(y, sig_score, clips),
        "chosen_columns": chosen,
    }


# ------------------------------------------------------------------ main

def build(level: str, events, store):
    rows, names = [], None
    for ev in events:
        if level == "video_full":
            got = video_signals(ev, store, band=False)
        elif level == "video_band":
            got = video_signals(ev, store, band=True)
        elif level in ("input", "early"):
            got = brain_signals(ev, store, level)
        elif level == "dn":
            got = dn_signals(ev, store)
        else:
            got = csv_signals(ev, store, level)
        if got is None:
            rows.append(None)
            continue
        t, X, keys = got
        w = window_stats(t, X)
        if w is None:
            rows.append(None)
            continue
        rows.append(w)
        if names is None:
            names = list(keys)
    return rows, names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--level", default=None)
    ap.add_argument("--exclude", default=None,
                    help="куски через запятую, которые не брать (проверка без спорных записей)")
    ap.add_argument("--labels", default=str(ROOT / "data/p10/FROZEN_P10.json"),
                    help="файл меток; по умолчанию замороженный набор из пяти кусков")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frozen = json.loads(labels_path.read_text(encoding="utf-8"))
    events = frozen["labels"]
    # VID00001 and VID00002 were recorded from files in ~/Downloads rather than the camera, and the
    # human labels for them were made on those same files. They are consistent with each other but
    # describe a different walk than the camera holds, so the result is worth reporting with and
    # without them.
    if args.exclude:
        drop = {s.strip() for s in args.exclude.split(",") if s.strip()}
        before = len(events)
        events = [e for e in events if e["clip"] not in drop]
        print(f"  исключены куски {sorted(drop)}: событий {before} -> {len(events)}")
    clips = np.array([e["clip"] for e in events])
    cat = np.array([e["category"] for e in events])
    store = Store()

    TASKS = {
        "SAME_vs_DIFFERENT": (np.where(cat == "SAME_DIRECTION", 1, 0),
                              np.isin(cat, ["SAME_DIRECTION", "DIFFERENT_DIRECTION"])),
        "MOVING_vs_NO_NET": (np.where(cat == "NO_NET_DISPLACEMENT", 0, 1),
                             np.ones(len(cat), bool)),
    }

    print("=" * 108)
    print("P10 — ЕДИНАЯ ПРОВЕРКА НА ВСЕХ УРОВНЯХ")
    print("=" * 108)
    print(f"  событий {len(events)}, перемешиваний {args.perm}")
    print("  нуль: метки перемешиваются ВНУТРИ каждого ролика — сцепление ролик/метка сохраняется")
    print()

    report = {"phase": "P10 — где теряется информация о движении тела",
              "n_events": int(len(events)), "n_perm": args.perm,
              "levels": {}, "launch": LAUNCH,
              "statistics": list(STATS), "aggregates": list(AGG_NAMES)}

    levels = [args.level] if args.level else list(LEVELS)
    for level in levels:
        rows, names = build(level, events, store)
        bad = sum(1 for r in rows if r is None)
        if bad:
            print(f"  {level}: нет данных у {bad} событий — пропущен")
            report["levels"][level] = {"error": f"{bad} без данных"}
            continue
        X = to_matrix(rows)
        rep = {"n_channels": len(names), "channels": names}
        print(f"─── {level}  ({LAUNCH[level]})  каналов {len(names)}" + (
            f"  [{' '.join(names[:4])}...]" if names and len(names) > 4 else
            (f"  [{names}]" if names else "")))
        for task, (yfull, keep) in TASKS.items():
            y = yfull[keep].astype(int)
            cl = clips[keep]
            Xk = X[keep]
            r = evaluate(Xk, y, cl, names, rng, args.perm)
            rep[task] = {**r, "n_events_used": int(keep.sum()),
                         "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
            flag = ""
            if not np.isnan(r["p_aggregate"]) and r["p_aggregate"] < 0.05:
                flag = "  <-- агрегат выше нуля"
            if not np.isnan(r["p_best_channel"]) and r["p_best_channel"] < 0.05:
                flag += "  <-- отдельный канал выше нуля"
            if r["null_aggregate_n"] == 0:
                flag += "  <-- нуль пуст: оценить нельзя"
            fmt = lambda v: "н/д" if v is None else f"{v:.3f}"      # noqa: E731
            print(f"    {task:<20} n={int(keep.sum()):>3} "
                  f"агрегат AUC={r['auc_aggregate']:.3f} (нуль "
                  f"{fmt(r['null_aggregate_median'])}, p95 {fmt(r['null_aggregate_p95'])}, "
                  f"p={fmt(r['p_aggregate'])})   "
                  f"канал AUC={r['auc_best_channel']:.3f} "
                  f"(нуль {fmt(r['null_channel_median'])}, p={fmt(r['p_best_channel'])}){flag}")
            # Per recording, so an effect carried by one clip is visible as such.
            pr = r["per_recording_aggregate"]
            cells = "  ".join(f"{c.replace('VID','')}:{fmt(v['auc'])}({v['pos']}/{v['neg']})"
                              for c, v in sorted(pr.items()))
            print(f"      по роликам (агрегат): {cells}")
            picks = ", ".join(f"{c['channel']}/{c['stat']}" for c in r["chosen_columns"])
            print(f"      выбранный канал по фолдам: {picks}")
        report["levels"][level] = rep
        print()

    dest = out / "levels_report.json"
    prev = {}
    if dest.exists() and args.level:
        prev = json.loads(dest.read_text(encoding="utf-8"))
        prev.get("levels", {}).update(report["levels"])
        report["levels"] = prev["levels"]
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  записано: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
