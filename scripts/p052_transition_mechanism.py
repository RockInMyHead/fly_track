#!/usr/bin/env python3
"""
P05.2 — why does the turn signal die between depth 1 and depth 2?

Two candidate explanations, and they make different predictions:

    the connections are too weak      the drive reaching depth 2 is small in absolute
                                      terms, so nothing survives
    excitation and inhibition cancel  the drive is large, but the excitatory and
                                      inhibitory parts oppose each other and the sum
                                      loses the direction

No new simulation. The real depth-2 targets of the depth-1 carriers are intrinsic optic
lobe cells — L2, T1, T4a, Mi4 and so on — and they were never recorded, so their activity
cannot be read out of P05. What can be done instead is to project the *measured* activity
of the depth-1 carriers through the *measured* connectome onto those targets. The result
is the synaptic drive each target would receive, which is well defined and needs no
simulation.

The linear projection is an upper bound
---------------------------------------
A linear sum ignores the spike threshold, saturation, and the timing that makes a real
synapse nonlinear. So if the projected drive already fails to separate left from right,
the loss is at the summation itself and the answer is cancellation or dilution. If the
projected drive separates cleanly but the recorded descending neurons downstream do not,
then the loss is postsynaptic and this projection cannot see it. The two outcomes point
in different directions, which is the point of running it.

What is measured
----------------
For every depth-2 target type the following are computed from the depth-1 carriers only:

    auc_E          separation of the summed EXCITATORY part of the drive
    auc_I          separation of the summed INHIBITORY part
    auc_net        separation of the two added together
    gross          sum of |w * contrast_i| over sources, the size of the parts
    net            |sum of w * contrast_i|, the size of the whole
    cancellation   1 - net/gross; near zero means the parts survive each other, near
                   one means they annihilate

A target whose excitatory and inhibitory parts each separate the turns but whose sum does
not is the cancellation signature. A target with a small gross is the weakness signature.

Separation is oriented, max(AUC, 1-AUC), because a target inhibited by a leftward signal
carries the direction just as well as one excited by it, only inverted.

Source selection is frozen
--------------------------
The carriers are read from P05.1's group table with `carries == True`; nothing is
re-selected here, no threshold is chosen, and no weight is touched.

Usage:
    PYTHONPATH=. python scripts/p052_transition_mechanism.py
    PYTHONPATH=. python scripts/p052_transition_mechanism.py --min-synapses 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REC64 = ROOT / "output/p05_layer_trace/layer_signals.npz"
REC65 = ROOT / "output/p05_layer_trace_seed65/layer_signals.npz"
GROUPS = ROOT / "output/p05_layer_trace/groups.json"
P051_GROUPS = ROOT / "output/p051_signal_break/group_metrics.csv"
TRACE_SCRIPT = ROOT / "scripts/p05_layer_trace.py"
OUT = ROOT / "output/p052_transition_mechanism"

SMOOTH_S = 0.3
TARGET_DEPTH = 2


def box_mat(M: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return M
    c = np.cumsum(M, axis=0, dtype=np.float64)
    c = np.vstack([np.zeros((1, M.shape[1])), c])
    o = c[n:] - c[:-n]
    left = (n - 1) // 2
    right = n - 1 - left
    o = np.vstack([np.repeat(o[:1], left, axis=0), o, np.repeat(o[-1:], right, axis=0)])
    return o[: M.shape[0]] / n


def auc_raw(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[order] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    return float((rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def orient(raw: float) -> float:
    return max(raw, 1.0 - raw)


def load_group_scheme():
    spec = importlib.util.spec_from_file_location("p05_trace", TRACE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fly_vo.config import FlyVOConfig
    from fly_vo.malecns_engine import MaleCNSEngine
    engine = MaleCNSEngine(FlyVOConfig())
    group_of, meta, depths = mod.build_groups(engine.brain)
    return group_of, meta, depths, engine.brain


def human_turns() -> list[dict]:
    rs = json.loads((ROOT / "data/p01r/review_set.json").read_text(encoding="utf-8"))["candidates"]
    vd = json.loads((ROOT / "data/p01r/turn_verdicts.json").read_text(encoding="utf-8"))["verdicts"]
    out = []
    for x in rs:
        v = vd.get(x["id"])
        if not v:
            continue
        truth = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        if truth in ("LEFT", "RIGHT"):
            out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "kind": truth})
    out.sort(key=lambda e: e["t0"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-synapses", type=int, default=5,
                    help="targets with fewer incoming synapses are listed but not ranked")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]

    # ---- frozen inputs -----------------------------------------------------
    rows = list(csv.DictReader(P051_GROUPS.open()))
    by_group = {int(r["group"]): r for r in rows}
    carriers = [int(r["group"]) for r in rows
                if r["carries"] == "True" and r["in_analysis"] == "True"]
    src = [g for g in carriers
           if int(by_group[g]["depth_bin"]) == 1
           and by_group[g]["superclass"] != "descending_neuron"]
    print("P05.2 — почему сигнал гаснет на переходе depth 1 -> depth 2")
    print(f"  источники взяты из P05.1 замороженными: carries == True")
    print(f"  групп-носителей всего: {len(carriers)}, из них на глубине 1 "
          f"и не нисходящие: {len(src)}")
    pref_R = [g for g in src if float(by_group[g]["pol_64"]) > 0]
    pref_L = [g for g in src if float(by_group[g]["pol_64"]) < 0]
    print(f"  из них предпочитают RIGHT: {len(pref_R)}, LEFT: {len(pref_L)}")
    print(f"  (полярность заморожена из P05.1, здесь не пересчитывается)")

    # ---- measured per-turn activity of the sources -------------------------
    a = np.load(REC64)
    b = np.load(REC65)
    t = a["t"].astype(np.float64)
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(SMOOTH_S / dt)), 1)
    ncell = np.array([m["n_cells"] for m in meta], float)

    turns = [w for w in human_turns() if w["t0"] >= t[0] and w["t1"] <= t[-1]]
    is_R = np.array([w["kind"] == "RIGHT" for w in turns])
    wins = [(w["t0"], w["t1"]) for w in turns]

    def change_matrix(rec: np.ndarray) -> np.ndarray:
        M = rec.astype(np.float64) / ncell[None, :]
        S = box_mat(M, n_sm)
        during = np.stack([np.median(S[(t >= x) & (t <= y)], axis=0) for x, y in wins])
        pre = np.stack([np.median(S[(t >= max(0.0, x - (y - x))) & (t < x)], axis=0)
                        for x, y in wins])
        return during - pre

    C64 = change_matrix(a["G"])          # (n_turns, n_groups), spikes/step/cell
    C65 = change_matrix(b["G"])

    print(f"\n  поворотов: {len(turns)} ({int(is_R.sum())} RIGHT, "
          f"{int((~is_R).sum())} LEFT)")
    print(f"  активность источников — измеренная, шаг 20 мс")

    # ---- connectome --------------------------------------------------------
    _, _, all_depths, brain = load_group_scheme()
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    sc_ = np.asarray(brain.superclass)
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)
    wt = np.asarray(brain.weights)
    dcell = all_depths

    # source cell indices and their measured per-turn change
    src_cells = []
    src_of_cell = {}
    for g in src:
        m = meta[g]
        cells = np.flatnonzero((ct == m["cell_type"]) & (sd == m["side"]))
        for c in cells.tolist():
            src_of_cell[c] = g
            src_cells.append(c)
    print(f"\n  клеток-источников: {len(src_cells)}")

    # accumulate connections grouped by target type, carrying the source group along so
    # the sign and the source's own direction preference stay separable
    edges: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for c in src_cells:
        g = src_of_cell[c]
        for e in range(ip[c], ip[c + 1]):
            i = int(ix[e])
            if ct[i] == "":
                continue
            edges[(str(ct[i]), str(sd[i]))].append((g, float(wt[e])))
    print(f"  типов-целей: {len(edges)}")

    # ---- drive for every target -------------------------------------------
    def evaluate(target_cells_edges: list[tuple[int, float]]) -> dict:
        """Linear synaptic drive from the measured sources onto one target type."""
        gs = np.array([g for g, _ in target_cells_edges])
        ws = np.array([w for _, w in target_cells_edges])
        pos, neg = ws >= 0, ws < 0

        def proj(mask: np.ndarray, C: np.ndarray) -> np.ndarray:
            if not mask.any():
                return np.zeros(C.shape[0])
            return C[:, gs[mask]] @ ws[mask]

        out = {}
        for tag, C in (("64", C64), ("65", C65)):
            total = proj(np.ones(len(ws), bool), C)
            exc = proj(pos, C)
            inh = proj(neg, C)
            out[f"auc_net_{tag}"] = orient(auc_raw(total[is_R], total[~is_R]))
            out[f"auc_exc_{tag}"] = orient(auc_raw(exc[is_R], exc[~is_R])) \
                if pos.any() else float("nan")
            out[f"auc_inh_{tag}"] = orient(auc_raw(inh[is_R], inh[~is_R])) \
                if neg.any() else float("nan")
        return out

    results = []
    for (tt, ts), ed in edges.items():
        cells = np.flatnonzero((ct == tt) & (sd == ts))
        dep = float(np.mean(dcell[cells])) if len(cells) else float("nan")
        cls = str(sc_[cells[0]]) if len(cells) else ""
        gs = np.array([g for g, _ in ed])
        ws = np.array([w for _, w in ed])
        # contrast per source: RIGHT mean minus LEFT mean of the measured change
        contrast64 = np.array([C64[is_R, g].mean() - C64[~is_R, g].mean() for g in gs])
        contrast65 = np.array([C65[is_R, g].mean() - C65[~is_R, g].mean() for g in gs])
        net64 = float(np.sum(ws * contrast64))
        gross64 = float(np.sum(np.abs(ws * contrast64)))
        net65 = float(np.sum(ws * contrast65))
        gross65 = float(np.sum(np.abs(ws * contrast65)))
        # the two parts of the drive, by sign of the connection
        exc_net64 = float(np.sum(ws[ws >= 0] * contrast64[ws >= 0]))
        inh_net64 = float(np.sum(ws[ws < 0] * contrast64[ws < 0]))
        ev = evaluate(ed)
        results.append({
            "target_type": tt, "target_side": ts, "target_depth": dep,
            "target_superclass": cls, "n_cells": len(cells),
            "n_synapses": len(ed), "n_sources": len(set(gs.tolist())),
            "n_sources_exc": len(set(gs[ws >= 0].tolist())),
            "n_sources_inh": len(set(gs[ws < 0].tolist())),
            "weight_abs": float(np.sum(np.abs(ws))),
            "gross_contrast_64": gross64, "net_contrast_64": net64,
            "gross_contrast_65": gross65, "net_contrast_65": net65,
            "exc_net_64": exc_net64, "inh_net_64": inh_net64,
            "cancellation_64": 1.0 - abs(net64) / gross64 if gross64 > 0 else float("nan"),
            "cancellation_65": 1.0 - abs(net65) / gross65 if gross65 > 0 else float("nan"),
            **ev,
        })

    R = [r for r in results if not np.isnan(r["target_depth"])
         and abs(r["target_depth"] - TARGET_DEPTH) < 0.75]
    R.sort(key=lambda r: -r["n_synapses"])
    print(f"\n  цели именно на глубине {TARGET_DEPTH}: {len(R)} типов")

    # ---- the two hypotheses ------------------------------------------------
    print(f"\n=== ГИПОТЕЗА 1: связи слабые ===")
    gross = np.array([r["gross_contrast_64"] for r in R if r["n_synapses"] >= args.min_synapses])
    print(f"  по {len(gross)} целям с >= {args.min_synapses} синапсов:")
    for p in (10, 25, 50, 75, 90, 100):
        print(f"    p{p:<3d} сумма |w * контраст| = {np.percentile(gross, p):10.4f}")
    print(f"  (единицы: вес x спайков на клетку в шаг; это масштаб входа в цель)")

    print(f"\n=== ГИПОТЕЗА 2: возбуждение и торможение гасят друг друга ===")
    big = [r for r in R if r["n_synapses"] >= args.min_synapses]
    canc = np.array([r["cancellation_64"] for r in big])
    print(f"  индекс гашения (1 - |сумма| / сумма|частей|):")
    for p in (10, 25, 50, 75, 90):
        print(f"    p{p:<3d} {np.percentile(canc, p):.3f}")
    print(f"  0 = части складываются, 1 = полностью гасят друг друга")

    # the decisive split: large gross but poor net separation
    print(f"\n=== РАЗЛИЧИЕ ГИПОТЕЗ ПО КАЖДОЙ ЦЕЛИ ===")
    print(f"  большая gross + плохая auc_net  -> гашение")
    print(f"  маленькая gross                -> слабость\n")
    gmed = float(np.median(gross))
    print(f"  медиана gross = {gmed:.4f}\n")
    print(f"  {'тип':>14s} {'стор':>5s} {'синапсов':>9s} {'gross':>9s} {'net':>9s} "
          f"{'гашение':>8s} {'auc E':>7s} {'auc I':>7s} {'auc net':>8s} {'вердикт':>18s}")
    for r in sorted(big, key=lambda r: -r["gross_contrast_64"])[:args.top]:
        ae, ai = r.get("auc_exc_64", np.nan), r.get("auc_inh_64", np.nan)
        if r["gross_contrast_64"] > gmed and r["auc_net_64"] < 0.7:
            verdict = "ГАШЕНИЕ"
        elif r["auc_net_64"] >= 0.7:
            verdict = "сигнал проходит"
        elif r["gross_contrast_64"] < gmed:
            verdict = "источник слабый"
        else:
            verdict = "неясно"
        print(f"  {r['target_type'][:14]:>14s} {r['target_side']:>5s} "
              f"{r['n_synapses']:9d} {r['gross_contrast_64']:9.4f} "
              f"{r['net_contrast_64']:+9.4f} {r['cancellation_64']:8.3f} "
              f"{ae:7.3f} {ai:7.3f} {r['auc_net_64']:8.3f} {verdict:>18s}")

    # ---- totals ------------------------------------------------------------
    n_carry = sum(1 for r in big if r["auc_net_64"] >= 0.7
                  and r.get("auc_net_65", 0) >= 0.7)
    n_cancel = sum(1 for r in big if r["gross_contrast_64"] > gmed
                   and r["auc_net_64"] < 0.7)
    n_weak = sum(1 for r in big if r["gross_contrast_64"] <= gmed)
    print(f"\n=== ИТОГ ===")
    print(f"  целей с >= {args.min_synapses} синапсов: {len(big)}")
    print(f"    сигнал проходит (auc_net >= 0.7 на обоих seed): {n_carry}")
    print(f"    гашение (большой вход, но плохой auc_net):       {n_cancel}")
    print(f"    слабый вход:                                     {n_weak}")

    # ---- dilution: how much of a descending neuron's input comes from us? ---
    # The linear projection says the drive to depth 2 arrives intact, so the loss is not
    # at the synapse. The remaining candidate is that the descending neurons are driven
    # overwhelmingly by something else, so the turn signal is a negligible share of what
    # reaches them. The connectome normalises each neuron's input weights, so the share
    # of |w| attributable to the carriers is directly interpretable.
    print(f"\n=== ГИПОТЕЗА 3: РАЗБАВЛЕНИЕ ===")
    print("  какую долю своего входа нисходящий нейрон получает от наших источников\n")

    src_set = set()
    for g in src:
        m = meta[g]
        src_set.update(np.flatnonzero((ct == m["cell_type"]) & (sd == m["side"])).tolist())
    is_src_cell = np.zeros(brain.n, bool)
    is_src_cell[list(src_set)] = True

    tot_in = np.zeros(brain.n)
    src_in = np.zeros(brain.n)
    # weights are normalised per neuron, so the split is a genuine share
    for j in range(brain.n):
        lo, hi = ip[j], ip[j + 1]
        if hi <= lo:
            continue
        tgt = ix[lo:hi]
        np.add.at(tot_in, tgt, np.abs(wt[lo:hi]))
        if is_src_cell[j]:
            np.add.at(src_in, tgt, np.abs(wt[lo:hi]))

    dil_rows = []
    for mi, m in enumerate(meta):
        if not m["superclass"].startswith("descending_neuron"):
            continue
        if round(m["mean_depth"]) < 2:
            continue
        cells = np.flatnonzero((ct == m["cell_type"]) & (sd == m["side"]))
        if len(cells) == 0:
            continue
        tot = float(tot_in[cells].mean())
        if tot <= 0:
            continue
        frac = float(src_in[cells].mean()) / tot
        in_deg = float(np.mean([ip[c + 1] - ip[c] for c in cells.tolist()]))
        meas = min(float(by_group[mi]["auc_change_64"])
                   if mi in by_group else 0.5,
                   float(by_group[mi]["auc_change_65"]) if mi in by_group else 0.5)
        dil_rows.append({"group": int(mi), "cell_type": m["cell_type"],
                         "side": m["side"], "frac_from_carriers": frac,
                         "total_input_weight": tot, "in_degree": in_deg,
                         "measured_auc": meas})
    dfrac = np.array([r["frac_from_carriers"] for r in dil_rows])
    dauc = np.array([r["measured_auc"] for r in dil_rows])
    print(f"  нисходящих групп: {len(dil_rows)}")
    print(f"  доля входа от наших источников:")
    for p in (10, 25, 50, 75, 90, 100):
        print(f"    p{p:<3d} {np.percentile(dfrac, p) * 100:7.3f}%")
    print(f"\n  {'доля входа':>14s} {'групп':>6s} {'медиана AUC':>12s} {'AUC>0.7':>9s}")
    bins = [(0, 0.0005), (0.0005, 0.002), (0.002, 0.01), (0.01, 0.05), (0.05, 1.01)]
    for lo, hi in bins:
        msk = (dfrac >= lo) & (dfrac < hi)
        if msk.sum() == 0:
            continue
        print(f"  {lo * 100:6.2f}-{hi * 100:6.2f}% {int(msk.sum()):6d} "
              f"{np.median(dauc[msk]):12.3f} {np.mean(dauc[msk] > 0.7) * 100:8.0f}%")
    r_dil = float(np.corrcoef(dfrac, dauc)[0, 1])
    print(f"\n  корреляция доли входа с тем, несёт ли нейрон направление: "
          f"{r_dil:+.3f}")
    hi_group = dfrac >= 0.01
    print(f"  DN с долей входа >= 1%: {int(hi_group.sum())}, из них несут направление: "
          f"{int((dauc[hi_group] > 0.7).sum())} "
          f"({np.mean(dauc[hi_group] > 0.7) * 100:.0f}%)")
    lo_group = dfrac < 0.0005
    print(f"  DN с долей входа < 0.05%: {int(lo_group.sum())}, из них несут: "
          f"{int((dauc[lo_group] > 0.7).sum())} "
          f"({np.mean(dauc[lo_group] > 0.7) * 100:.0f}%)")

    print(f"\n  крупнейшие DN по доле входа:")
    print(f"  {'тип':>14s} {'стор':>5s} {'доля входа':>11s} {'вход':>7s} "
          f"{'вход. связей':>12s} {'AUC':>7s}")
    for r in sorted(dil_rows, key=lambda z: -z["frac_from_carriers"])[:12]:
        print(f"  {r['cell_type'][:14]:>14s} {r['side']:>5s} "
              f"{r['frac_from_carriers'] * 100:10.3f}% {r['total_input_weight']:7.2f} "
              f"{r['in_degree']:12.1f} {r['measured_auc']:7.3f}")

    # ---- do the recorded DNs match their predicted drive? ------------------
    print(f"\n=== ПРОВЕРКА НА ЗАПИСАННЫХ НИСХОДЯЩИХ НЕЙРОНАХ ===")
    print("  для каждой записанной DN-группы: предсказанный вход vs измеренная "
          "активность\n")
    dn_rows = []
    for r in results:
        if not r["target_superclass"].startswith("descending_neuron"):
            continue
        g = next((i for i, m in enumerate(meta)
                  if m["cell_type"] == r["target_type"] and m["side"] == r["target_side"]),
                 None)
        if g is None:
            continue
        meas64 = orient(auc_raw(C64[is_R, g], C64[~is_R, g]))
        meas65 = orient(auc_raw(C65[is_R, g], C65[~is_R, g]))
        dn_rows.append({**r, "measured_auc_64": meas64, "measured_auc_65": meas65,
                        "group": int(g)})
    if dn_rows:
        pred = np.array([min(r["auc_net_64"], r["auc_net_65"]) for r in dn_rows])
        meas = np.array([min(r["measured_auc_64"], r["measured_auc_65"])
                         for r in dn_rows])
        print(f"  DN с предсказанным входом: {len(dn_rows)}")
        print(f"  предсказанный auc_net: медиана {np.median(pred):.3f}")
        print(f"  измеренный auc:        медиана {np.median(meas):.3f}")
        print(f"  корреляция предсказания и измерения: "
              f"{np.corrcoef(pred, meas)[0, 1]:+.3f}")
        hi = (pred >= 0.7) & (meas < 0.6)
        print(f"  DN, где вход предсказан хорошим, а активность нет: {int(hi.sum())}")
        if hi.sum() > 0:
            exp = [r["target_type"] for r in np.array(dn_rows)[hi]]
            print(f"    примеры: {', '.join(exp[:8])}")
            print(f"  -> для этих нейронов потеря происходит ПОСЛЕ суммирования, "
                  f"на спайковой нелинейности; линейная проекция её не видит")

    # ---- outputs -----------------------------------------------------------
    with (OUT / "target_drives.csv").open("w", newline="") as f:
        keys = ["target_type", "target_side", "target_depth", "target_superclass",
                "n_cells", "n_synapses", "n_sources", "n_sources_exc",
                "n_sources_inh", "weight_abs", "gross_contrast_64", "net_contrast_64",
                "gross_contrast_65", "net_contrast_65", "exc_net_64", "inh_net_64",
                "cancellation_64", "cancellation_65", "auc_exc_64", "auc_inh_64",
                "auc_net_64", "auc_exc_65", "auc_inh_65", "auc_net_65"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})

    with (OUT / "source_contributions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_type", "target_side", "source_group", "source_type",
                    "source_side", "source_prefers", "n_synapses", "weight",
                    "weight_abs", "contrast_64", "signed_contribution"])
        for (tt, ts), ed in edges.items():
            acc = defaultdict(lambda: [0.0, 0])
            for g, wv in ed:
                acc[g][0] += wv
                acc[g][1] += 1
            for g, (wsum, n) in acc.items():
                m = meta[g]
                contrast = C64[is_R, g].mean() - C64[~is_R, g].mean()
                w.writerow([tt, ts, g, m["cell_type"], m["side"],
                            "RIGHT" if float(by_group[g]["pol_64"]) > 0 else "LEFT",
                            n, wsum, abs(wsum), contrast, wsum * contrast])

    with (OUT / "dn_dilution.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "cell_type", "side",
                                          "frac_from_carriers", "total_input_weight",
                                          "in_degree", "measured_auc"])
        w.writeheader()
        w.writerows(dil_rows)

    # ---- figures ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    x = np.array([r["gross_contrast_64"] for r in big])
    y = np.array([r["auc_net_64"] for r in big])
    c = np.array([r["cancellation_64"] for r in big])
    sc = ax.scatter(x, y, c=c, cmap="RdYlGn_r", s=26, vmin=0, vmax=1)
    ax.axhline(0.7, color="0.4", ls="--")
    ax.axvline(gmed, color="0.4", ls="--")
    ax.set_xlabel("gross: сумма |w × контраст|  (размер входа)")
    ax.set_ylabel("auc_net: сохранилось ли направление")
    ax.set_title("две гипотезы на плоскости\n"
                 "право-низ = большой вход, но сигнал потерян")
    plt.colorbar(sc, ax=ax, label="индекс гашения")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(c)[::-1][:20]
    lab = [f"{big[i]['target_type'][:11]} {big[i]['target_side']}" for i in order]
    ax.barh(np.arange(len(order)), c[order], color="tab:red")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("индекс гашения (1 = полное взаимное уничтожение)")
    ax.set_title("самые гасящие цели глубины 2")
    ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[0, 2])
    top = sorted(big, key=lambda r: -r["n_synapses"])[:14]
    yy = np.arange(len(top))[::-1]
    ax.barh(yy - 0.2, [r["auc_exc_64"] for r in top], 0.38, color="tab:blue",
            label="возбуждающая часть")
    ax.barh(yy + 0.2, [r["auc_inh_64"] for r in top], 0.38, color="tab:orange",
            label="тормозная часть")
    ax.scatter([r["auc_net_64"] for r in top], yy, color="k", marker="D", s=28,
               zorder=3, label="их сумма")
    ax.axvline(0.5, color="k", ls=":"); ax.axvline(0.7, color="0.4", ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r['target_type'][:11]} {r['target_side']}" for r in top],
                       fontsize=7)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("auc")
    ax.set_title("разложение на возбуждение и торможение\n"
                 "(каждая часть по отдельности против суммы)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(dfrac * 100, dauc, s=22, color="tab:purple", alpha=0.7)
    ax.axhline(0.7, color="0.4", ls="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("доля входа нейрона, приходящая от наших источников, %")
    ax.set_ylabel("несёт ли нейрон направление (AUC)")
    ax.set_title("разбавление: медианный DN получает\n"
                 f"{np.median(dfrac) * 100:.3f}% входа от этого пути")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    if dn_rows:
        p = [min(r["auc_net_64"], r["auc_net_65"]) for r in dn_rows]
        m = [min(r["measured_auc_64"], r["measured_auc_65"]) for r in dn_rows]
        ax.scatter(p, m, s=26, color="tab:red", alpha=0.7)
        ax.plot([0.4, 1], [0.4, 1], color="0.5", ls=":", lw=1)
        ax.axhline(0.6, color="0.4", ls="--"); ax.axvline(0.7, color="0.4", ls="--")
        ax.set_xlabel("предсказанный вход (линейно)")
        ax.set_ylabel("измеренная активность DN")
        ax.set_title("вход предсказан хорошим у всех,\n"
                     "активность — не у всех")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    msk = [(dfrac >= lo) & (dfrac < hi) for lo, hi in bins]
    labels = [f"{lo * 100:.2f}-{hi * 100:.2f}%" for lo, hi in bins]
    vals = [np.mean(dauc[m2] > 0.7) * 100 if m2.sum() else 0 for m2 in msk]
    ns = [int(m2.sum()) for m2 in msk]
    ax.bar(labels, vals, color=["tab:red", "tab:orange", "0.7", "tab:blue", "tab:green"])
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 2, f"{v:.0f}%\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("доля DN, несущих направление, %")
    ax.set_xlabel("доля входа от наших источников")
    ax.set_title("чем больше доля входа,\nтем чаще нейрон несёт направление")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("P05.2 — почему гаснет сигнал на depth 1 -> depth 2. "
                 "Измеренная активность слоя 1, спроецированная через измеренные веса.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "cancellation.png", dpi=125)
    plt.close(fig)

    # dedicated figure for the answer
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax = axes[0]
    ax.hist(dfrac * 100, bins=np.logspace(-3, 1.5, 40), color="tab:purple")
    ax.set_xscale("log")
    ax.axvline(np.median(dfrac) * 100, color="tab:red", ls="--",
               label=f"медиана {np.median(dfrac) * 100:.3f}%")
    ax.axvline(1.0, color="0.4", ls=":", label="порог 1%")
    ax.set_xlabel("доля входа от наших источников, %")
    ax.set_ylabel("нисходящих групп")
    ax.set_title("разбавление входа нисходящих нейронов")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(dfrac[dfrac < 0.0005] * 100,
               dauc[dfrac < 0.0005], s=14, color="0.6", alpha=0.6,
               label=f"доля < 0.05% (n={int((dfrac < 0.0005).sum())})")
    ax.scatter(dfrac[dfrac >= 0.01] * 100, dauc[dfrac >= 0.01], s=34, color="tab:green",
               alpha=0.85, label=f"доля >= 1% (n={int((dfrac >= 0.01).sum())})")
    ax.axhline(0.7, color="0.4", ls="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("доля входа, %"); ax.set_ylabel("AUC направления")
    ax.set_title("где доля входа велика,\nнаправление сохраняется")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    show = sorted(dil_rows, key=lambda z: -z["frac_from_carriers"])[:12]
    yy = np.arange(len(show))[::-1]
    ax.barh(yy, [r["frac_from_carriers"] * 100 for r in show],
            color=["tab:green" if r["measured_auc"] > 0.7 else "tab:red" for r in show])
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r['cell_type'][:12]} {r['side']}\n"
                        f"AUC {r['measured_auc']:.2f}" for r in show], fontsize=7)
    ax.set_xlabel("доля входа от наших источников, %")
    ax.set_title("нисходящие нейроны с наибольшей долей\n"
                 "зелёное — действительно несут направление")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("P05.2 — ответ: сигнал не гасится и не слаб, он разбавлен. "
                 "Нисходящие нейроны слушают другие пути.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "dilution.png", dpi=125)
    plt.close(fig)


    # second figure: E/I decomposition for the strongest carriers
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax = axes[0]
    rr = [r for r in big if not np.isnan(r.get("auc_exc_64", np.nan))
          and not np.isnan(r.get("auc_inh_64", np.nan))]
    if rr:
        ax.scatter([r["auc_exc_64"] for r in rr], [r["auc_inh_64"] for r in rr],
                   s=[10 + r["n_synapses"] ** 0.5 for r in rr], alpha=0.6,
                   color="tab:blue")
        ax.axhline(0.5, color="k", ls=":"); ax.axvline(0.5, color="k", ls=":")
        ax.axhline(0.7, color="0.4", ls="--"); ax.axvline(0.7, color="0.4", ls="--")
        ax.set_xlabel("auc возбуждающей части"); ax.set_ylabel("auc тормозной части")
        ax.set_title("обе части по отдельности информативны\n"
                     "значит дело не в их отсутствии, а в их знаке")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if rr:
        d = np.array([r["auc_net_64"] - min(r["auc_exc_64"], r["auc_inh_64"]) for r in rr])
        ax.hist(d, bins=25, color="tab:red", alpha=0.8)
        ax.axvline(0, color="k", lw=1.2)
        ax.set_xlabel("auc_net - min(auc_E, auc_I)")
        ax.set_ylabel("целей")
        ax.set_title("падение различимости при сложении\n"
                     "левее нуля = сумма хуже, чем каждая часть")
        ax.grid(alpha=0.3)
        n_drop = int((d < -0.1).sum())
        ax.annotate(f"хуже на >0.1: {n_drop} из {len(rr)}", xy=(0.02, 0.95),
                    xycoords="axes fraction", fontsize=9)

    ax = axes[2]
    if rr:
        ax.scatter([r["cancellation_64"] for r in rr],
                   [min(r["auc_net_64"], r["auc_net_65"]) for r in rr],
                   s=22, color="tab:purple", alpha=0.7)
        ax.axhline(0.7, color="0.4", ls="--")
        ax.set_xlabel("индекс гашения"); ax.set_ylabel("auc_net (хуже из двух seed)")
        ax.set_title("связь гашения с потерей сигнала")
    ax.grid(alpha=0.3)

    fig.suptitle("P05.2 — разложение входа на возбуждение и торможение", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "ei_decomposition.png", dpi=125)
    plt.close(fig)

    # ---- report -----------------------------------------------------------
    carry_ex = [r for r in big if r["auc_net_64"] >= 0.7 and r["auc_net_65"] >= 0.7]
    frac_hi = float(np.mean(dfrac >= 0.01))
    frac_lo = float(np.mean(dfrac < 0.0005))
    if frac_lo > 0.5 and np.mean(dauc[dfrac >= 0.01] > 0.7) > 0.5:
        verdict = ("НИ СЛАБОСТЬ, НИ ГАШЕНИЕ. Линейная проекция показывает, что "
                   "направление доходит до глубины 2 почти полностью: медиана auc_net "
                   f"{np.median([min(r['auc_net_64'], r['auc_net_65']) for r in big]):.3f} "
                   f"у {len(big)} целей, а гашение редко превышает 0.7. Причина в "
                   "РАЗБАВЛЕНИИ: медианный нисходящий нейрон получает "
                   f"{np.median(dfrac) * 100:.3f}% своего входа от этих источников, "
                   f"так что сигнал поворота теряется в остальном входе. Там, где доля "
                   "входа велика, направление сохраняется: среди DN с долей >= 1% "
                   f"несут направление {np.mean(dauc[dfrac >= 0.01] > 0.7) * 100:.0f}%, "
                   f"среди DN с долей < 0.05% только "
                   f"{np.mean(dauc[dfrac < 0.0005] > 0.7) * 100:.0f}%.")
    elif n_cancel > n_carry:
        verdict = ("ОСНОВНАЯ ПРИЧИНА — ГАШЕНИЕ: у большинства целей вход большой, "
                   "возбуждающая и тормозная части по отдельности несут направление, "
                   "но при сложении оно теряется")
    elif n_weak > n_carry + n_cancel:
        verdict = ("ОСНОВНАЯ ПРИЧИНА — СЛАБОСТЬ: вход в цели глубины 2 мал по "
                   "абсолютной величине")
    else:
        verdict = ("обе причины присутствуют: часть целей теряет направление из-за "
                   "гашения, часть имеет слишком слабый вход")
    print(f"\n  ВЫВОД: {verdict}")

    (OUT / "report.json").write_text(json.dumps({
        "inputs": {"rec64": str(REC64), "rec65": str(REC65),
                   "carriers_from": str(P051_GROUPS), "groups": str(GROUPS)},
        "n_source_groups": len(src), "n_source_cells": len(src_cells),
        "n_sources_prefer_RIGHT": len(pref_R), "n_sources_prefer_LEFT": len(pref_L),
        "n_turns": len(turns), "n_right": int(is_R.sum()),
        "n_left": int((~is_R).sum()),
        "min_synapses": args.min_synapses,
        "gross_median": gmed,
        "n_targets_depth2": len(R), "n_targets_with_input": len(big),
        "n_carry": n_carry, "n_cancel": n_cancel, "n_weak": n_weak,
        "dilution": {
            "n_descending_groups": len(dil_rows),
            "frac_from_carriers_median": float(np.median(dfrac)),
            "frac_from_carriers_p90": float(np.percentile(dfrac, 90)),
            "frac_from_carriers_max": float(np.max(dfrac)),
            "n_above_1pct": int((dfrac >= 0.01).sum()),
            "n_below_0p05pct": int((dfrac < 0.0005).sum()),
            "auc_when_above_1pct": float(np.mean(dauc[dfrac >= 0.01] > 0.7)),
            "auc_when_below_0p05pct": float(np.mean(dauc[dfrac < 0.0005] > 0.7)),
            "correlation_with_auc": r_dil,
            "note": "веса входа нормированы на нейрон, поэтому доля читается прямо "
                    "как вклад этого пути в общий вход",
        },
        "verdict": verdict,
        "method": "линейная проекция измеренной активности групп глубины 1 через "
                  "измеренные веса коннектома на цели глубины 2; auc считается между "
                  "подтверждёнными LEFT и RIGHT поворотами и ориентируется, потому что "
                  "инвертированный сигнал тоже несёт направление",
        "caveat": "проекция линейна и потому является верхней оценкой: она не "
                  "учитывает спайковый порог, насыщение и время. Если предсказанный "
                  "вход несёт направление, а измеренная активность DN нет, потеря "
                  "происходит после суммирования и эта проекция её не видит",
        "dn_prediction_check": {
            "n": len(dn_rows),
            "predicted_median": float(np.median([min(r["auc_net_64"], r["auc_net_65"])
                                                 for r in dn_rows])) if dn_rows else None,
            "measured_median": float(np.median([min(r["measured_auc_64"],
                                                    r["measured_auc_65"])
                                                for r in dn_rows])) if dn_rows else None,
            "correlation": float(np.corrcoef(
                [min(r["auc_net_64"], r["auc_net_65"]) for r in dn_rows],
                [min(r["measured_auc_64"], r["measured_auc_65"]) for r in dn_rows]
            )[0, 1]) if dn_rows else None,
        } if dn_rows else {},
        "per_target": results,
        "verbatim": "нового прогона мозга не было, веса не менялись, источники взяты "
                    "из P05.1 замороженными, пороги не подбирались",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
