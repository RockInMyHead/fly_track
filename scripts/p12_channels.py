#!/usr/bin/env python3
"""P12 step 1 — three separate readings, each answering its own question, each with a confidence.

WHAT IS BEING BUILT, AND WHY IT IS THREE THINGS AND NOT ONE
-----------------------------------------------------------
The project spent several phases looking for a single signal that means "the fly is moving this
way". P11 showed why that failed and what to do instead: the questions are different, and the level
that answers each one best is different too.

    camera_yaw        where the camera turned        the tracker's own yaw, LEFT / RIGHT / SILENT
    route_change      did the body's course change   68 descending cells, SAME / DIFFERENT
    net_displacement  was there net progress         the twenty early visual populations,
                                                     MOVE / NO_NET_DISPLACEMENT

They are kept apart on purpose and never summed. A camera that swung left while the body's course
did not change is a glance, and a glance is not an instruction to turn; a single scalar cannot
express that, because it would have to decide the matter before the graph sees it.

EVERY CHANNEL REPORTS A CONFIDENCE, AND SAYS UNKNOWN RATHER THAN GUESSING
-------------------------------------------------------------------------
Each channel returns a class and a confidence, and returns no class at all when the confidence is
below a threshold. The graphtracker is built to carry several hypotheses while it is unsure, so a
channel that abstains costs it nothing, while a channel that guesses wrong takes a hypothesis out
of the running.

HONESTY OF THE FIT, WHICH IS THE WHOLE POINT OF THE PHASE
---------------------------------------------------------
Every reader applied to a recording is fitted on the *other four only*. Cell selection, feature
selection, the number of cells and the ridge are all decided inside that training set, and the held
out recording contributes nothing — not a cell, not a threshold, not a feature. That is what makes
the junction replay in the next step worth reading, and it is the same rule P11.A and P11.B were
held to. The consequence is that there is no single frozen model: there are five, one per
recording, and the one used on VID00006 never saw VID00006.

Usage:
    PYTHONPATH=. python scripts/p12_channels.py [--min-cons 0.7]
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
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.p11a_route_change import (FEATS, KS, Subsets, apply_readout,  # noqa: E402
                                       auc_cols, dn, features_at, fit_readout)

P10 = ROOT / "output/p10"
P09 = ROOT / "output/p09"
P07 = ROOT / "output/p07"
OUT = ROOT / "output/p12"

EARLY_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "LC4", "LPLC2")
BEFORE = (-3.0, -0.5)
DURING = (-0.5, 0.5)
AFTER = (0.5, 7.0)


# ------------------------------------------------------------------ data access

def review() -> tuple[list[dict], dict]:
    s = json.loads((ROOT / "data/p095/review_set_v2.json").read_text(encoding="utf-8"))
    lab = {l["event_id"]: l["human_label"]
           for l in json.loads((ROOT / "data/p095/human_labels_v2.json").read_text(
               encoding="utf-8"))["labels"] if l.get("round", 1) == 1}
    ev = []
    for e in s["events"]:
        if e["event_id"] not in lab:
            continue
        ev.append({"event_id": e["event_id"], "clip": e["clip"], "time": float(e["time"]),
                   "clip_file": e["clip_file"], "old_label": lab[e["event_id"]]})
    return ev, lab


_EARLY: dict[str, tuple] = {}


def early_series(video: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if video not in _EARLY:
        d = np.load(P10 / f"brain_{video}.npz")
        _EARLY[video] = (d["t"].astype(float), d["vis_rate"].astype(float),
                         [str(g) for g in d["vis_groups"]], d["vis_pop"].astype(float))
    return _EARLY[video]


def source_features(kind: str, video: str, times: list[float]):
    """[len(times), n_groups, 8] for one recording, from the chosen level."""
    if kind == "dn":
        t, X, names, _is_yaw, _ids = dn(video)
        n_cells = np.ones(len(names), dtype=float)
    elif kind == "early":
        t, X, names, n_cells = early_series(video)
        if len(names) != 20:
            raise RuntimeError(f"ранних популяций {len(names)}, ожидалось 20")
    else:
        raise ValueError(kind)
    rows = [features_at(t, X, at) for at in times]
    if any(r is None for r in rows):
        raise RuntimeError(f"{kind}/{video}: у части моментов нет окна")
    return np.stack(rows), list(names), n_cells


# ------------------------------------------------------------------ fitting

def select_sym(auc_full: np.ndarray, aucs_inner: list, nch: int, nf: int, k: int,
               min_cons: float) -> dict:
    """Cell and feature choice that does not care which class was called positive.

    P11.A's selector took, for every cell, the feature with the highest AUC *in the direction of
    the positive class*. The polarity of a label is arbitrary, and the consequence was not
    cosmetic: with SAME_DIRECTION as positive it picked each cell's best SAME-predicting feature,
    with DIFFERENT_DIRECTION as positive its best DIFFERENT-predicting one, and the two choices are
    different features of the same cell. The same data therefore gave two different readers
    depending on an arbitrary sign — which is how route_change came out at 0.714 when P11.A had
    reported 0.764 for what should be the identical problem.

    What a reader needs is a feature that *separates* the classes, in whichever direction. So the
    choice is made on |AUC - 0.5|, and the direction is recorded rather than assumed. Everything
    else — the consistency requirement across inner folds, the ranking, the size — is unchanged.
    """
    A = auc_full.reshape(nch, nf)
    dev = np.abs(A - 0.5)
    best_f = dev.argmax(axis=1)
    ar = np.arange(nch)
    best_a = A[ar, best_f]
    best_dev = dev[ar, best_f]

    cons = np.zeros(nch)
    if aucs_inner:
        rows = [ai.reshape(nch, nf)[ar, best_f] - 0.5 for ai in aucs_inner if ai is not None]
        if rows:
            S = np.stack([np.sign(r) for r in rows])
            ref = np.sign(best_a - 0.5)
            cons = (S == ref[None, :]).mean(axis=0)

    ok = cons >= min_cons
    score = np.where(ok, best_dev, -1.0)
    idx = np.argsort(-score)[:k]
    idx = idx[score[idx] > 0]
    return {"cells": idx, "feat": best_f[idx],
            "sign": np.where(best_a[idx] >= 0.5, 1.0, -1.0),
            "dev": best_dev[idx], "cons": cons[idx]}


def fit_channel(Ftr: np.ndarray, y: np.ndarray, train: np.ndarray, uniq: list[str],
                n_cells: np.ndarray, ks=KS, min_cons: float = 0.7,
                nf: int = len(FEATS)) -> dict:
    """Choose cells, features and size inside `train`, then fit. Nothing else is consulted.

    Selection is `select_sym`, which requires a pairing to separate the two classes without caring
    which one is called positive. Size is chosen by an inner loop that holds out one training
    recording at a time, so the number of cells is not free either.
    """
    ng = len(uniq)
    if train.dtype == bool:
        train = np.nonzero(train)[0]
    X2 = Ftr.reshape(len(y), ng * nf)
    subs = Subsets(X2)
    auc_full = subs.aucs(train, y)
    if auc_full is None:
        return {"ok": False, "why": "в обучающих данных только один класс"}

    inner_rows = []
    tmask = np.zeros(len(y), dtype=bool)
    tmask[train] = True
    for c in np.unique(_CLIPS[train]):
        rows = np.nonzero(tmask & (_CLIPS == c))[0]
        rest = np.nonzero(tmask & (_CLIPS != c))[0]
        if len(set(y[rest].tolist())) < 2:
            continue
        a_in = subs.aucs(rest, y)
        if a_in is not None:
            inner_rows.append((c, rows, rest, a_in))

    k_best, k_score = None, -np.inf
    for k in ks:
        val_rows, val_scores = [], []
        for c, rows, rest, a_in in inner_rows:
            others = [a for cc, _r, _rs, a in inner_rows if cc != c]
            sel = select_sym(a_in, others, ng, nf, k, min_cons)
            if len(sel["cells"]) == 0:
                continue
            w, b = fit_readout(X2, sel, nf, y, rest, "logreg")
            val_rows.append(rows)
            val_scores.append(apply_readout(X2, sel, nf, w, b, rows))
        if not val_rows:
            continue
        r = np.concatenate(val_rows)
        s = np.concatenate(val_scores)
        a = auc_cols(y[r], s[:, None])[0]
        if not np.isnan(a) and a > k_score:
            k_score, k_best = float(a), k
    if k_best is None:
        k_best = min(ks)

    sel = select_sym(auc_full, [a for _c, _r, _rs, a in inner_rows], ng, nf, k_best, min_cons)
    if len(sel["cells"]) == 0:
        return {"ok": False, "why": "отбор не дал ни одной пары"}
    w, b = fit_readout(X2, sel, nf, y, train, "logreg")
    return {"ok": True, "k": int(k_best), "inner_cv": float(k_score),
            "cells": [uniq[i] for i in sel["cells"]],
            "cell_idx": sel["cells"].tolist(), "feat": [FEATS[i] for i in sel["feat"]],
            "feat_idx": sel["feat"].tolist(), "sign": sel["sign"].tolist(),
            "cons": sel["cons"].tolist(), "dev": sel["dev"].tolist(),
            "w": np.asarray(w).tolist(), "b": float(b), "n_groups": ng, "nf": nf}


_CLIPS: np.ndarray | None = None


def channel_scores(Fap: np.ndarray, chtr: dict, nf: int = len(FEATS)) -> np.ndarray:
    """Apply a fitted channel to a feature tensor; returns logits."""
    ng = chtr["n_groups"]
    X2 = Fap.reshape(Fap.shape[0], ng * nf)
    cells = np.asarray(chtr["cell_idx"])
    cols = (cells[:, None] * nf + np.arange(nf)[None, :]).ravel()
    return X2[:, cols] @ np.asarray(chtr["w"]) + chtr["b"]


def to_class_conf(z: float) -> tuple[str | None, float, float]:
    p = 1.0 / (1.0 + np.exp(-float(np.clip(z, -30, 30))))
    conf = abs(2 * p - 1)
    return ("pos" if p >= 0.5 else "neg"), float(conf), float(p)


# ------------------------------------------------------------------ yaw channel

def yaw_channel(video: str) -> dict:
    """The tracker's own reading: LEFT / RIGHT / SILENT, with its ramp as the confidence."""
    import p084b_geometry as geo
    import p086_multi as M
    rows = list(csv.DictReader((P07 / f"yaw_signal_{video}.csv").open(encoding="utf-8")))
    t = np.array([float(r["t"]) for r in rows])
    y = np.array([float(r["yaw_signal_deadband"]) for r in rows])
    return {"t": t, "y": y, "window": M.WINDOW_S, "floor_fn": M.floor_for,
            "ramp": geo.signal_ramp}


def yaw_at(ch: dict, at: float) -> dict:
    a, b = at - ch["window"], at
    m = (ch["t"] >= a) & (ch["t"] <= b)
    if m.sum() < 2:
        I = 0.0
    else:
        I = float((ch["y"][m][:-1] * np.diff(ch["t"][m])).sum())
    fl = float(ch["floor_fn"]())
    ramp = float(ch["ramp"](I, fl))
    if ramp <= 0:
        return {"class": None, "confidence": 0.0, "integral": I, "floor": fl,
                "note": "ниже порога"}
    return {"class": "LEFT" if I > 0 else "RIGHT", "confidence": ramp, "integral": I,
            "floor": fl, "note": ""}


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-cons", type=float, default=0.7)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--junctions", default=str(ROOT / "output/p086/report.json"))
    ap.add_argument("--tau", type=float, default=0.25,
                    help="ниже этой уверенности канал отвечает UNKNOWN")
    args = ap.parse_args()
    global _CLIPS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ev, _lab = review()
    clips = np.array([e["clip"] for e in ev])
    _CLIPS = clips
    videos = sorted(set(clips.tolist()))
    nf = len(FEATS)

    print("=" * 104)
    print("P12 шаг 1 — ТРИ КАНАЛА, каждый обучен ТОЛЬКО на четырёх роликах из пяти")
    print("=" * 104)
    print(f"  размеченных событий {len(ev)}, роликов {len(videos)}")
    print(f"  порог уверенности: ниже {args.tau} канал отвечает UNKNOWN")
    print()

    # Positive class in each channel's own terms, so that a positive logit names the class:
    #   route_change      DIFFERENT means the course changed
    #   net_displacement  NO_NET means there was no net progress
    # Getting this backwards would not change any AUC (it is folded into the sign) but it would
    # mislabel every decision the graph is handed, which is worse than a wrong number.
    y_rc = np.where(np.array([e["old_label"] for e in ev]) == "TURN", 1, 0)
    keep_rc = np.isin([e["old_label"] for e in ev], ["LOOK", "TURN"])
    y_nd = np.where(np.array([e["old_label"] for e in ev]) == "NO_LOCOMOTION", 1, 0)

    # training features, one row per event, built from its own recording.
    # The first version of this cached one feature row per recording and reused it for every event
    # of that recording, so all twenty-three VID00005 events carried the features of whichever one
    # was seen first. The channels still ran and still produced numbers; they were just trained on
    # twenty-three copies of one moment. Features are now built for every event time of a recording
    # in one pass, which is both correct and quicker.
    train_F: dict[str, np.ndarray] = {}
    for kind in ("dn", "early"):
        by_video: dict[str, list[tuple[int, float]]] = {}
        for i, e in enumerate(ev):
            by_video.setdefault(e["clip"], []).append((i, float(e["time"])))
        rows: list = [None] * len(ev)
        for v, items in by_video.items():
            Fv, names, nc = source_features(kind, v, [t for _i, t in items])
            if len(Fv) != len(items):
                raise RuntimeError(f"{kind}/{v}: признаков {len(Fv)}, моментов {len(items)}")
            for (i, _t), row in zip(items, Fv):
                rows[i] = row
        train_F[kind] = np.stack(rows)
    uniq_dn = list(dn(videos[0])[2])
    uniq_ea = list(early_series(videos[0])[2])
    nc_dn = np.ones(len(uniq_dn))
    nc_ea = early_series(videos[0])[3]
    print(f"  route_change:   источник — {len(uniq_dn)} нисходящих клеток")
    print(f"  net_displacement: источник — {len(uniq_ea)} ранних популяций "
          f"({int(nc_ea.sum())} клеток)")
    print()

    # junctions of the tracker run, per recording
    jrep = json.loads(Path(args.junctions).read_text(encoding="utf-8"))
    junc = {j["node"]: {"t": float(j["t"]), "arrival": j["arrival"],
                        "committed": j["committed"], "truth": j.get("truth_edge", ""),
                        "pick_p086": j.get("pick"), "status_p086": j.get("status")}
            for j in jrep["junctions"]}

    result: dict = {"tau": args.tau, "min_cons": args.min_cons, "folds": {}, "junctions": {}}
    loo_rc, loo_nd = {}, {}

    for v in videos:
        train = clips != v
        test = clips == v
        fv = {"video": v}

        # ---- route_change ----
        tr_rc = train & keep_rc
        if len(set(y_rc[tr_rc].tolist())) >= 2:
            ch = fit_channel(train_F["dn"], y_rc, tr_rc, uniq_dn, nc_dn, min_cons=args.min_cons)
            fv["route_change"] = ch
            if ch["ok"]:
                # Only the SAME/DIFFERENT events of this recording are scored: the channel was
                # trained on those two classes and asking it about a NO_NET event is not a test of
                # it. The first version scored every event of the recording and then paired the
                # results with the smaller list of labelled ones, which shifted every pair by the
                # number of events that should never have been in the rowset.
                sub = np.nonzero(keep_rc & (clips == v))[0]
                z = channel_scores(train_F["dn"][sub], ch)
                fv["route_change"]["heldout"] = [
                    {"event_id": ev[i]["event_id"], "old_label": ev[i]["old_label"],
                     "z": float(zz), "class": "DIFFERENT" if zz > 0 else "SAME",
                     "confidence": float(abs(2 / (1 + np.exp(-np.clip(zz, -30, 30))) - 1))}
                    for i, zz in zip(sub, z)]
                loo_rc[v] = fv["route_change"]["heldout"]
        else:
            fv["route_change"] = {"ok": False, "why": "в обучающих нет обоих классов"}

        # ---- net_displacement ----
        if len(set(y_nd[train].tolist())) >= 2:
            ch = fit_channel(train_F["early"], y_nd, train, uniq_ea, nc_ea, min_cons=args.min_cons)
            fv["net_displacement"] = ch
            if ch["ok"]:
                sub = np.nonzero(clips == v)[0]
                z = channel_scores(train_F["early"][sub], ch)
                fv["net_displacement"]["heldout"] = [
                    {"event_id": ev[i]["event_id"], "old_label": ev[i]["old_label"],
                     "z": float(zz), "class": "NO_NET" if zz > 0 else "MOVE",
                     "confidence": float(abs(2 / (1 + np.exp(-np.clip(zz, -30, 30))) - 1))}
                    for i, zz in zip(sub, z)]
                loo_nd[v] = fv["net_displacement"]["heldout"]
        else:
            fv["net_displacement"] = {"ok": False, "why": "в обучающих нет обоих классов"}

        result["folds"][v] = fv

    # ---- how well does each channel do on the recordings it never saw ----
    print("─── ЧЕСТНАЯ ПРОВЕРКА КАНАЛОВ (обучение на 4, применение к пятому) ───")
    print()
    for kind, loo, y_true, name, pos_label in (
            ("dn", loo_rc, y_rc, "route_change (SAME/DIFFERENT)", "DIFFERENT"),
            ("early", loo_nd, y_nd, "net_displacement (MOVE/NO_NET)", "MOVE")):
        rows = [r for v in videos for r in loo.get(v, [])]
        if not rows:
            print(f"  {name}: нет предсказаний")
            continue
        # positive class: DIFFERENT for route_change (logit > 0), NO_NET for net_displacement
        if kind == "dn":
            want = np.array([1 if r["old_label"] == "TURN" else 0 for r in rows])
        else:
            want = np.array([1 if r["old_label"] == "NO_LOCOMOTION" else 0 for r in rows])
        z = np.array([r["z"] for r in rows])
        conf = np.array([r["confidence"] for r in rows])
        pred = (z > 0).astype(int)
        a = auc_cols(want, z[:, None])[0]
        a = max(float(a), 1 - float(a))
        decided = conf >= args.tau
        acc_dec = float((pred[decided] == want[decided]).mean()) if decided.sum() else float("nan")
        ch_key = "route_change" if kind == "dn" else "net_displacement"
        ks_used = [result['folds'][v][ch_key].get('k') for v in videos
                   if result['folds'][v][ch_key].get('ok')]
        print(f"  {name}")
        print(f"    событий {len(rows)}, AUC на отложенных роликах = {a:.3f}")
        print(f"    уверенность >= {args.tau}: решено {int(decided.sum())} из {len(rows)}, "
              f"точность среди решённых {acc_dec:.3f}")
        print(f"    UNKNOWN (отказ): {int((~decided).sum())} ({100*(~decided).mean():.0f}%)")
        print(f"    размер модели по фолдам: {ks_used}")
        print()
        result.setdefault("heldout_metrics", {})[kind] = {
            "auc": a, "n": len(rows), "n_decided": int(decided.sum()),
            "accuracy_among_decided": acc_dec, "tau": args.tau,
            "n_unknown": int((~decided).sum()), "k_per_fold": ks_used}

    # ---- what the channels say at the tracker's junctions ----
    print("─── ЧТО КАНАЛЫ ГОВОРЯТ НА РАЗВИЛКАХ ТРЕКЕРА ───")
    print(f"  {'узел':<6}{'t':>7}  {'camera_yaw':<14}{'route_change':<22}"
          f"{'net_displacement':<22}")
    ych = yaw_channel("VID00006")
    v = "VID00006"
    nodes = list(junc)
    jtimes = [junc[n]["t"] for n in nodes]
    Fdn_j, _n1, _c1 = source_features("dn", v, jtimes)
    Fea_j, _n2, _c2 = source_features("early", v, jtimes)
    for k, node in enumerate(nodes):
        j = junc[node]
        y = yaw_at(ych, j["t"])
        fv = result["folds"][v]
        rc = apply_or_unknown(Fdn_j[k:k + 1], fv["route_change"], args.tau, "SAME", "DIFFERENT")
        nd = apply_or_unknown(Fea_j[k:k + 1], fv["net_displacement"], args.tau, "MOVE", "NO_NET")
        result["junctions"][node] = {"t": j["t"], "arrival": j["arrival"],
                                     "committed_p086": j["committed"], "truth": j["truth"],
                                     "camera_yaw": y, "route_change": rc,
                                     "net_displacement": nd,
                                     "pick_p086": j["pick_p086"], "status_p086": j["status_p086"]}
        print(f"  {node:<6}{j['t']:>7.1f}  {fmt(y):<14}{fmt(rc):<22}{fmt(nd):<22}")
    print()

    (out / "channels.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  записано: {out/'channels.json'}")
    return 0


def apply_or_unknown(Fap: np.ndarray, ch: dict, tau: float, neg: str, pos: str) -> dict:
    if not ch.get("ok"):
        return {"class": None, "confidence": 0.0, "why": ch.get("why", "")}
    z = float(channel_scores(Fap, ch)[0])
    cls, conf, p = to_class_conf(z)
    if conf < tau:
        return {"class": None, "confidence": conf, "logit": z,
                "would_be": pos if cls == "pos" else neg, "why": f"уверенность {conf:.2f} < {tau}"}
    return {"class": pos if cls == "pos" else neg, "confidence": conf, "logit": z, "why": ""}


def fmt(d: dict) -> str:
    c = d.get("class")
    if c is None:
        return f"— ({d.get('confidence', 0):.2f})"
    return f"{c} ({d['confidence']:.2f})"


if __name__ == "__main__":
    raise SystemExit(main())
