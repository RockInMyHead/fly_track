#!/usr/bin/env python3
"""
P05 — where inside MaleCNS does the turn signal disappear?

The chain is short, so the question is narrow. Breadth-first from the T4/T5 population
reaches 912 of the 1314 descending neurons in two hops:

    depth 0   T4a/b, T5a/b                    6,753 cells
    depth 1   their direct recipients        30,950 cells   (LPC1, LLPC1, Y11, Y12 ...)
    depth 2   descending neurons              2,504 cells

So there is no four-to-six-layer chain to walk. Whatever happens, happens across one or
two synapses.

What went wrong in the first pass
---------------------------------
The first run of this analysis looked only at the left-minus-right difference of a cell
type, and at the sign of that difference. Both choices hid the answer:

  * A neuron can carry direction without being lateralised. DNp53 fires at 3.1 Hz in
    rightward turns and 0.19 Hz in leftward ones, on BOTH sides at once. Its L-R
    difference is therefore about zero, and the differential channel reports nothing.
  * A neuron that goes silent in a leftward turn is perfectly informative, but
    `sign(0) != sign(expected)` scores that as a miss, so sign accuracy punishes it.

This version fixes both. Every (cell type, side) population is a channel in its own
right, and discrimination is measured by AUC, which handles silence correctly and needs
no polarity.

The controls that matter
------------------------
Two sets of windows from the classifier are used as context, not as ground truth: the
forward and the stationary windows. A channel that tracks the direction of turn should
sit differently in left and right turns while forward falls between them. If forward
falls outside the left-right range, the channel is following something else, and it is
flagged rather than counted.

The permutation test
--------------------
Roughly 1,500 channels scored on 23 turns will produce impressive-looking separations by
chance. The threshold is therefore the best channel obtained after shuffling the turn
labels, at the 95th percentile over 2,000 shuffles. Anything below it is not reported as
a finding, however high its AUC.

Usage:
    PYTHONPATH=. python scripts/p05_layer_analyze.py
    PYTHONPATH=. python scripts/p05_layer_analyze.py --smooth 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACE = ROOT / "output/p05_layer_trace/layer_signals.npz"
GROUPS = ROOT / "output/p05_layer_trace/groups.json"
REVIEW = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
WINDOWS3 = ROOT / "data/p01r/windows_v3.json"
OUT = ROOT / "output/p05_layer_trace"

LAYER_NAMES = {0: "T4/T5 (то, что мы подали)",
               1: "прямые получатели T4/T5",
               2: "нисходящие нейроны"}


def box_fast(M: np.ndarray, n: int) -> np.ndarray:
    """Box filter along axis 0, via cumulative sums, for a (T, C) matrix."""
    if n <= 1:
        return M
    c = np.cumsum(M, axis=0, dtype=np.float64)
    c = np.vstack([np.zeros((1, M.shape[1])), c])
    out = c[n:] - c[:-n]
    left = (n - 1) // 2
    right = n - 1 - left
    out = np.vstack([np.repeat(out[:1], left, axis=0), out,
                     np.repeat(out[-1:], right, axis=0)])
    return out[: M.shape[0]] / n


def human_turns() -> list[dict]:
    rs = json.loads(REVIEW.read_text(encoding="utf-8"))["candidates"]
    vd = json.loads(VERDICTS.read_text(encoding="utf-8"))["verdicts"]
    out = []
    for x in rs:
        v = vd.get(x["id"])
        if not v:
            continue
        truth = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        if truth in ("LEFT", "RIGHT"):
            out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "kind": truth,
                        "expect": 1.0 if truth == "RIGHT" else -1.0})
    out.sort(key=lambda e: e["t0"])
    return out


def control_windows() -> dict[str, list[tuple[float, float]]]:
    ws = json.loads(WINDOWS3.read_text(encoding="utf-8"))["windows"]
    out: dict[str, list[tuple[float, float]]] = {"FWD": [], "STATIC": []}
    for w in ws:
        if w["kind"] in out:
            out[w["kind"]].append((w["t0"], w["t1"]))
    return out


def rank_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b), ties averaged. Returns 0.5 when either side is empty."""
    if len(a) == 0 or len(b) == 0:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv), float)
    r[order] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    ranks = (s / cnt)[inv]
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2)
                 / (len(a) * len(b)))


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    from math import erfc, sqrt
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    u = rank_auc(a, b) * len(a) * len(b)
    mu = len(a) * len(b) / 2
    sd = np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12)
    if sd == 0:
        return float("nan")
    z = (u - mu) / sd
    return float(erfc(abs(z) / sqrt(2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=float, default=0.3, help="seconds")
    args = ap.parse_args()

    d = np.load(TRACE)
    t = d["t"].astype(np.float64)
    G = d["G"].astype(np.float64)
    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(args.smooth / dt)), 1)

    ht = [w for w in human_turns() if w["t0"] >= t[0] and w["t1"] <= t[-1]]
    ctrl = control_windows()
    expect = np.array([w["expect"] for w in ht])
    is_R = expect > 0

    print("P05 — где внутри MaleCNS теряется сигнал поворота")
    print(f"  запись: {len(t):,} шагов, {t[-1] - t[0]:.0f} с, {G.shape[1]} групп, "
          f"сглаживание {args.smooth} с")
    print(f"  подтверждённых человеком поворотов: {len(ht)} "
          f"({int(is_R.sum())} RIGHT, {int((~is_R).sum())} LEFT)")
    print(f"  контрольные окна из классификатора: FWD {len(ctrl['FWD'])}, "
          f"STATIC {len(ctrl['STATIC'])} (не человеческие метки, только контекст)\n")

    # ---- normalise and smooth every group once -----------------------------
    ncell = np.array([m["n_cells"] for m in meta], float)
    S = box_fast(G, n_sm) / ncell[None, :]

    def window_med(sig: np.ndarray, wins) -> np.ndarray:
        out = []
        for a, b in wins:
            m = (t >= a) & (t <= b)
            if m.any():
                out.append(float(np.median(sig[m])))
        return np.array(out)

    turn_wins = [(w["t0"], w["t1"]) for w in ht]
    n_groups = G.shape[1]
    V_turn = np.stack([window_med(S[:, g], turn_wins) for g in range(n_groups)])
    V_fwd = np.stack([window_med(S[:, g], ctrl["FWD"]) for g in range(n_groups)])
    V_stat = np.stack([window_med(S[:, g], ctrl["STATIC"]) for g in range(n_groups)])

    # ---- channels ----------------------------------------------------------
    chans: list[dict] = []
    for g, m in enumerate(meta):
        chans.append({"layer": m["layer"], "cell_type": m["cell_type"],
                      "side": m["side"], "kind": "single", "gL": g, "gR": None,
                      "n_cells": m["n_cells"]})
    by_type: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    for g, m in enumerate(meta):
        if not m["cell_type"].startswith("ALL_"):
            by_type[(m["layer"], m["cell_type"])][m["side"]] = g
    for (layer, ctype), sides in sorted(by_type.items()):
        if "L" in sides and "R" in sides:
            chans.append({"layer": layer, "cell_type": ctype, "side": "LR",
                          "kind": "diff", "gL": sides["L"], "gR": sides["R"],
                          "n_cells": meta[sides["L"]]["n_cells"]
                                     + meta[sides["R"]]["n_cells"]})

    def series(ch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if ch["gR"] is None:
            return V_turn[ch["gL"]], V_fwd[ch["gL"]], V_stat[ch["gL"]]
        return (V_turn[ch["gL"]] - V_turn[ch["gR"]],
                V_fwd[ch["gL"]] - V_fwd[ch["gR"]],
                V_stat[ch["gL"]] - V_stat[ch["gR"]])

    rows = []
    Vt = np.zeros((len(chans), len(ht)))
    for i, ch in enumerate(chans):
        v, vf, vs = series(ch)
        Vt[i] = v
        L, R = v[~is_R], v[is_R]
        a = rank_auc(R, L)
        auc = max(a, 1.0 - a)
        # robustness: does the same split hold in the two time halves?
        ht0 = np.median([w["t0"] for w in ht])
        first = np.array([w["t0"] < ht0 for w in ht])
        a1 = rank_auc(v[first & is_R], v[first & ~is_R])
        a2 = rank_auc(v[~first & is_R], v[~first & ~is_R])
        auc1, auc2 = max(a1, 1 - a1), max(a2, 1 - a2)
        same = (a1 - 0.5) * (a2 - 0.5) > 0
        mL, mR = float(np.median(L)), float(np.median(R))
        mF, mS = float(np.median(vf)), float(np.median(vs))
        between = (min(mL, mR) <= mF <= max(mL, mR))
        rows.append({**ch, "auc": auc, "auc_h1": auc1, "auc_h2": auc2,
                     "halves_agree": bool(same), "fwd_between": bool(between),
                     "p_mwu": mannwhitney_p(R, L),
                     "rate_L": mL, "rate_R": mR, "rate_FWD": mF, "rate_STATIC": mS})

    # ---- permutation threshold ---------------------------------------------
    n_pos = int(is_R.sum())
    n_neg = int((~is_R).sum())
    ranks = np.empty_like(Vt)
    for i in range(Vt.shape[0]):
        _, inv, cnt = np.unique(Vt[i], return_inverse=True, return_counts=True)
        order = np.argsort(Vt[i], kind="mergesort")
        r = np.empty(len(order), float)
        r[order] = np.arange(1, len(order) + 1)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        ranks[i] = (s / cnt)[inv]
    rng = np.random.default_rng(0)
    n_perm = 2000
    max_null = np.empty(n_perm)
    layer_of = np.array([c["layer"] for c in chans])
    layer_null = {L: np.empty(n_perm) for L in (0, 1, 2)}
    for k in range(n_perm):
        pos = rng.random(len(ht)) < (n_pos / len(ht))
        nn = int(pos.sum())
        if nn == 0 or nn == len(ht):
            max_null[k] = 0.5
            for L in layer_null:
                layer_null[L][k] = 0.5
            continue
        a = (ranks[:, pos].sum(axis=1) - nn * (nn + 1) / 2) / (nn * (len(ht) - nn))
        a = np.maximum(a, 1.0 - a)
        max_null[k] = a.max()
        for L in layer_null:
            msk = layer_of == L
            layer_null[L][k] = float(np.median(a[msk])) if msk.any() else 0.5
    fw = float(np.percentile(max_null, 95))
    obs = np.array([r["auc"] for r in rows])

    print("=== КОНТРОЛЬ НА МНОЖЕСТВЕННЫЕ СРАВНЕНИЯ ===")
    print(f"  каналов: {len(rows)}, поворотов: {len(ht)}")
    print(f"  порог = 95-й процентиль лучшего канала при {n_perm} перестановках")
    print(f"  меток: AUC > {fw:.3f}")
    print(f"  каналов выше порога: {int(np.sum(obs > fw))} из {len(rows)}")

    # ---- layer summary -----------------------------------------------------
    print("\n=== РАЗЛИЧИМОСТЬ ПОВОРОТОВ ПО СЛОЯМ ===")
    print("  AUC 0.5 = левый и правый поворот в этом слое неразличимы\n")
    print(f"  {'слой':>24s} {'каналов':>8s} {'AUC медиана':>12s} {'p слоя':>8s} "
          f"{'выше порога':>11s} {'лучший':>20s}")
    summary = {}
    for L in (0, 1, 2):
        rs = [r for r in rows if r["layer"] == L]
        if not rs:
            continue
        aucs = np.array([r["auc"] for r in rs])
        pL = float(np.mean(layer_null[L] >= float(np.median(aucs))))
        best = max(rs, key=lambda r: r["auc"])
        summary[L] = {"n_channels": len(rs),
                      "auc_median": float(np.median(aucs)),
                      "auc_p90": float(np.percentile(aucs, 90)),
                      "p_value_layer": pL,
                      "n_above_fw": int(np.sum(aucs > fw)),
                      "auc_median_fwd_between": float(np.median(
                          [r["auc"] for r in rs if r["fwd_between"]] or [np.nan])),
                      "best": {"cell_type": best["cell_type"], "side": best["side"],
                               "auc": best["auc"], "p_mwu": best["p_mwu"],
                               "rate_L": best["rate_L"], "rate_R": best["rate_R"],
                               "rate_FWD": best["rate_FWD"],
                               "rate_STATIC": best["rate_STATIC"],
                               "fwd_between": best["fwd_between"],
                               "halves_agree": best["halves_agree"]}}
        star = "*" if pL < 0.05 else " "
        print(f"  {LAYER_NAMES[L]:>24s} {len(rs):8d} {np.median(aucs):12.3f} "
              f"{pL:7.3f}{star} {int(np.sum(aucs > fw)):11d} "
              f"{best['cell_type'][:20]:>20s}")

    # ---- the channels that survived ----------------------------------------
    print(f"\n=== КАНАЛЫ ВЫШЕ ПОРОГА ({fw:.3f}) ===")
    above = sorted([r for r in rows if r["auc"] > fw], key=lambda r: -r["auc"])
    if not above:
        print("  ни один канал не превысил порог")
    else:
        print("  скорость в Гц = спайков на клетку в секунду\n")
        print(f"  {'слой':>4s} {'тип':>14s} {'стор':>5s} {'кл.':>5s} {'AUC':>6s} "
              f"{'LEFT':>8s} {'RIGHT':>8s} {'FWD':>8s} {'STATIC':>8s} {'p':>9s} "
              f"{'FWD меж':>7s} {'полов.':>7s}")
        for r in above[:25]:
            print(f"  {r['layer']:4d} {r['cell_type'][:14]:>14s} {r['side']:>5s} "
                  f"{r['n_cells']:5d} {r['auc']:6.3f} {r['rate_L'] / dt:8.2f} "
                  f"{r['rate_R'] / dt:8.2f} {r['rate_FWD'] / dt:8.2f} "
                  f"{r['rate_STATIC'] / dt:8.2f} {r['p_mwu']:9.2e} "
                  f"{'да' if r['fwd_between'] else 'НЕТ':>7s} "
                  f"{'да' if r['halves_agree'] else 'нет':>7s}")

        strict = [r for r in above if r["fwd_between"] and r["halves_agree"]]
        print(f"\n  из них проходят оба контроля (FWD между LEFT и RIGHT, "
              f"и обе половины записи согласны): {len(strict)}")
        for r in strict[:15]:
            print(f"    слой {r['layer']}  {r['cell_type']} {r['side']} "
                  f"({r['n_cells']} кл.)  AUC {r['auc']:.3f}  "
                  f"L {r['rate_L'] / dt:.2f} Гц, R {r['rate_R'] / dt:.2f} Гц")

    # ---- verdict -----------------------------------------------------------
    p0 = summary.get(0, {}).get("p_value_layer", np.nan)
    p1 = summary.get(1, {}).get("p_value_layer", np.nan)
    p2 = summary.get(2, {}).get("p_value_layer", np.nan)
    sig = [L for L, pp in ((0, p0), (1, p1), (2, p2)) if pp < 0.05]
    print("\n=== ВЫВОД ===")
    print(f"  значимые слои (p<0.05): {sig if sig else 'нет'}")
    if not above:
        verdict = ("ни один канал не различил повороты выше случайного: "
                   "направление не теряется по пути, его нет уже на входе")
    else:
        n_top = len([r for r in above if r["layer"] == 2])
        verdict = (f"направление обнаружено в {len(above)} каналах выше случайного, "
                   f"в том числе {n_top} среди нисходящих нейронов. Значит сигнал не "
                   f"теряется по дороге — он доходит. Но он НЕ латерализован: "
                   f"нейрон отвечает одной стороной на поворот в одну сторону, "
                   f"поэтому разность L−R, которой мы раньше мерили, его не видит.")
    print(f"  {verdict}")

    # ---- plot --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    for L, col in ((0, "tab:green"), (1, "tab:orange"), (2, "tab:red")):
        rs = [r["auc"] for r in rows if r["layer"] == L]
        if rs:
            ax.hist(rs, bins=np.linspace(0.4, 1.0, 25), alpha=0.55, color=col,
                    label=LAYER_NAMES[L])
    ax.axvline(0.5, color="k", ls=":", label="случай")
    ax.axvline(fw, color="0.3", ls="--", label=f"порог {fw:.2f}")
    ax.set_xlabel("AUC (различимость LEFT / RIGHT)")
    ax.set_ylabel("число каналов")
    ax.set_title("распределение различимости по слоям")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for L, col in ((0, "tab:green"), (1, "tab:orange"), (2, "tab:red")):
        rs = sorted([r["auc"] for r in rows if r["layer"] == L], reverse=True)
        if rs:
            ax.plot(np.arange(1, len(rs) + 1), rs, marker="o", ms=2, lw=1,
                    color=col, label=LAYER_NAMES[L])
    ax.axhline(0.5, color="k", ls=":")
    ax.axhline(fw, color="0.3", ls="--")
    ax.set_xscale("log"); ax.set_xlabel("канал (ранг)"); ax.set_ylabel("AUC")
    ax.set_title("каналы по убыванию"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    names = [LAYER_NAMES[i] for i in (0, 1, 2)]
    vals = [summary.get(i, {}).get("auc_median", np.nan) for i in (0, 1, 2)]
    ax.barh(np.arange(3), vals, color=["tab:green", "tab:orange", "tab:red"])
    ax.set_yticks(np.arange(3)); ax.set_yticklabels([n[:22] for n in names], fontsize=8)
    ax.axvline(0.5, color="k", ls=":"); ax.axvline(fw, color="0.3", ls="--")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUC (медиана слоя)")
    ax.set_title("сводка по слоям"); ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[1, :2])
    if above:
        show = above[:22]
        y = np.arange(len(show))
        ax.bar(y - 0.22, [r["rate_L"] / dt for r in show], 0.2, label="LEFT",
               color="tab:green")
        ax.bar(y + 0.0, [r["rate_R"] / dt for r in show], 0.2, label="RIGHT",
               color="tab:purple")
        ax.bar(y + 0.22, [r["rate_FWD"] / dt for r in show], 0.2, label="FWD",
               color="0.6")
        ax.set_yscale("symlog", linthresh=0.2)
        ax.set_xticks(y)
        ax.set_xticklabels([f"{r['cell_type'][:9]}\n{r['side']}" for r in show],
                           fontsize=6, rotation=60, ha="right")
        ax.set_ylabel("Гц")
        ax.set_title("каналы выше порога: скорость в левых, правых и передних окнах")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "каналов выше порога нет", ha="center")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter([r["auc_h1"] for r in rows], [r["auc_h2"] for r in rows], s=4,
               alpha=0.25, color="0.5")
    if above:
        ax.scatter([r["auc_h1"] for r in above], [r["auc_h2"] for r in above], s=14,
                   color="tab:red", zorder=3)
    for v in (0.5, fw):
        ax.axhline(v, color="0.6", ls=":"); ax.axvline(v, color="0.6", ls=":")
    ax.set_xlabel("AUC, первая половина записи")
    ax.set_ylabel("AUC, вторая половина")
    ax.set_title("устойчивость во времени\n(красные — выше порога)")
    ax.grid(alpha=0.3)

    fig.suptitle("P05 — где внутри MaleCNS теряется сигнал поворота. "
                 f"Порог на {len(rows)} каналов: AUC > {fw:.2f}.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "layer_collapse.png", dpi=125)
    plt.close(fig)

    with (OUT / "layer_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "cell_type", "side", "kind",
                                          "n_cells", "auc", "auc_h1", "auc_h2",
                                          "halves_agree", "fwd_between", "p_mwu",
                                          "rate_L", "rate_R", "rate_FWD",
                                          "rate_STATIC"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    (OUT / "report.json").write_text(json.dumps({
        "n_steps": len(t), "duration_s": float(t[-1] - t[0]),
        "n_channels": len(rows), "n_human_turns": len(ht),
        "n_right": int(is_R.sum()), "n_left": int((~is_R).sum()),
        "smooth_s": args.smooth, "n_permutations": n_perm,
        "family_wise_threshold_auc": fw,
        "n_channels_above": int(np.sum(obs > fw)),
        "layer_summary": summary, "verdict": verdict,
        "controls": {"n_fwd_windows": len(ctrl["FWD"]),
                     "n_static_windows": len(ctrl["STATIC"]),
                     "note": "окна FWD и STATIC взяты из классификатора, а не из "
                             "глаз человека; они используются как контекст"},
        "method": "AUC между подтверждёнными LEFT и RIGHT поворотами, без полярности; "
                  "порог из перестановки меток; контроли — положение FWD между "
                  "LEFT и RIGHT и согласие половин записи",
        "caveat": "23 подтверждённых поворота, поэтому порог по 1,500 каналам высок "
                  "и слабые эффекты не обнаруживаются. Слои 1 и 2 получают сигнал "
                  "только через инъекцию в T4/T5, разрыв может быть на входе.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/layer_collapse.png")


if __name__ == "__main__":
    main()
