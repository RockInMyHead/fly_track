#!/usr/bin/env python3
"""
P05 — summary figure: what reproduces across seeds and what does not.

The answer to "where does the turn signal die" is decided by reproducibility. A layer
whose left/right separation is identical under two independent random seeds is carrying
a real signal; a layer whose separation vanishes is noise that happened to line up with
the labels once.

    layer 0  T4/T5                 cross-seed r = 0.99
    layer 1  their recipients      cross-seed r = 0.98
    layer 2  descending neurons    cross-seed r = 0.21

Usage:
    PYTHONPATH=. python scripts/p05_summary_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

A = ROOT / "output/p05_layer_trace/layer_signals.npz"
B = ROOT / "output/p05_layer_trace_seed65/layer_signals.npz"
GROUPS = ROOT / "output/p05_layer_trace/groups.json"
REVIEW = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
OUT = ROOT / "output/p05_layer_trace"

LAYER_NAMES = {0: "T4/T5\n(что мы подали)", 1: "прямые получатели\nT4/T5",
               2: "нисходящие\nнейроны"}


def main() -> None:
    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    ncell = np.array([m["n_cells"] for m in meta], float)
    d = np.load(A)
    t = d["t"].astype(np.float64)
    GA = d["G"].astype(np.float64) / ncell[None, :]
    GB = np.load(B)["G"].astype(np.float64) / ncell[None, :]

    rs = json.loads(REVIEW.read_text(encoding="utf-8"))["candidates"]
    vd = json.loads(VERDICTS.read_text(encoding="utf-8"))["verdicts"]
    ht = []
    for x in rs:
        v = vd.get(x["id"])
        if not v:
            continue
        tr = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        if tr in ("LEFT", "RIGHT"):
            ht.append((x["t0"], x["t1"], tr))
    ht.sort()
    wins = [(a, b) for a, b, _ in ht]
    isR = np.array([k == "RIGHT" for _, _, k in ht])

    def meds(M: np.ndarray) -> np.ndarray:
        return np.stack([np.array([np.median(M[(t >= a) & (t <= b), g]) for a, b in wins])
                         for g in range(M.shape[1])])

    VA, VB = meds(GA), meds(GB)

    def corr_per_layer(M: np.ndarray, N: np.ndarray, layer: int) -> np.ndarray:
        ii = [i for i, m in enumerate(meta)
              if m["layer"] == layer and not m["cell_type"].startswith("ALL_")]
        out = []
        for i in ii:
            if np.std(M[i]) > 1e-12 and np.std(N[i]) > 1e-12:
                out.append(float(np.corrcoef(M[i], N[i])[0, 1]))
        return np.array(out)

    def auc_of(v: np.ndarray) -> float:
        R, L = v[isR], v[~isR]
        if len(R) == 0 or len(L) == 0:
            return 0.5
        allv = np.concatenate([R, L])
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        o = np.argsort(allv, kind="mergesort")
        r = np.empty(len(allv))
        r[o] = np.arange(1, len(allv) + 1)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        rk = (s / cnt)[inv]
        a = (rk[: len(R)].sum() - len(R) * (len(R) + 1) / 2) / (len(R) * len(L))
        return max(a, 1 - a)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    cols = ["tab:green", "tab:orange", "tab:red"]
    for L in (0, 1, 2):
        c = corr_per_layer(VA, VB, L)
        if len(c):
            ax.bar(L, np.median(c), color=cols[L], width=0.6)
            ax.scatter(np.full(len(c), L) + np.random.default_rng(0).normal(0, 0.06, len(c)),
                       c, s=12, color="k", alpha=0.5, zorder=3)
            ax.text(L, 1.04, f"медиана {np.median(c):+.2f}", ha="center", fontsize=9)
    ax.axhline(0.8, color="0.4", ls="--", label="порог воспроизводимости 0.8")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([LAYER_NAMES[i] for i in (0, 1, 2)],
                                                 fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("корреляция между двумя seed")
    ax.set_title("воспроизводимость по слоям\n(по 23 поворотам)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    for L in (0, 1, 2):
        ii = [i for i, m in enumerate(meta)
              if m["layer"] == L and not m["cell_type"].startswith("ALL_")]
        a = np.array([auc_of(VA[i]) for i in ii])
        b = np.array([auc_of(VB[i]) for i in ii])
        if len(a):
            ax.scatter(a, b, s=16, color=cols[L], alpha=0.6,
                       label=f"{LAYER_NAMES[L].splitlines()[0]} (n={len(a)})")
    ax.plot([0.4, 1], [0.4, 1], color="0.5", ls=":", lw=1)
    ax.axvline(0.5, color="k", ls=":"); ax.axhline(0.5, color="k", ls=":")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("AUC, seed 64"); ax.set_ylabel("AUC, seed 65")
    ax.set_title("одна и та же мера на двух seed\nточки у 0.5 = шум, у диагонали = сигнал")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    names = ["T4a_L", "T5a_L", "LPC1_L", "LLPC1_L", "Y11_L", "Y12_L", "Y1_L", "Tlp5_L"]
    x = np.arange(len(names))
    for i, nm in enumerate(names):
        typ, sid = nm.rsplit("_", 1)
        gi = next(k for k, m in enumerate(meta)
                  if m["cell_type"] == typ and m["side"] == sid)
        for j, (V, off) in enumerate(((VA, -0.2), (VB, 0.2))):
            ax.bar(i + off, V[gi][~isR].mean() / 0.02, 0.16,
                   color="tab:green", alpha=0.9 if j == 0 else 0.55)
            ax.bar(i + off, (V[gi][isR].mean() - V[gi][~isR].mean()) / 0.02, 0.16,
                   bottom=V[gi][~isR].mean() / 0.02, color="tab:purple",
                   alpha=0.9 if j == 0 else 0.55)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_", " ") for n in names],
                                         rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Гц")
    ax.set_title("примеры: зелёное — LEFT, фиолетовое — RIGHT\n"
                 "слева seed 64, справа seed 65: значения повторяются")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    ii = [i for i, m in enumerate(meta) if m["layer"] == 2]
    a = np.array([auc_of(VA[i]) for i in ii])
    b = np.array([auc_of(VB[i]) for i in ii])
    best = sorted(range(len(ii)), key=lambda k: -(a[k] + b[k]))[:10]
    ax.barh(np.arange(len(best)), [b[k] for k in best], color="tab:red", alpha=0.6,
            label="seed 65")
    ax.barh(np.arange(len(best)), [a[k] for k in best], color="tab:red", height=0.4,
            label="seed 64")
    ax.set_yticks(np.arange(len(best)))
    ax.set_yticklabels([f"{meta[ii[k]]['cell_type'][:12]} {meta[ii[k]]['side']}"
                        for k in best], fontsize=7)
    ax.axvline(0.5, color="k", ls=":")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUC")
    ax.set_title("сильнейшие нисходящие нейроны:\nлучшие на одном seed проваливаются на другом")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[:, 2])
    # rebuild the path from the brain's signs, which agreed with the truth 23/23
    def path(signs):
        th = np.pi / 2
        xx = yy = 0.0
        pe = 0.0
        P = []
        for s, (a, b2, _) in zip(signs, ht):
            gap = max(0.0, a - pe)
            if gap > 0:
                xx += 1.4 * gap * np.cos(th)
                yy += 1.4 * gap * np.sin(th)
                P.append((xx, yy))
            th += np.deg2rad(s * 90.0)
            P.append((xx, yy))
            pe = max(pe, b2)
        return np.array([p[0] for p in P]), np.array([p[1] for p in P])

    bx, by = path(np.where(isR, 1.0, -1.0))
    ax.plot(bx, by, lw=2.4, color="tab:orange",
            label="мозг мухи, слои 0-1 (23/23)")
    ax.plot(bx, by, lw=1.0, color="0.35", ls="--", label="истина (твои метки)")
    ax.plot(bx[0], by[0], "o", color="tab:green", ms=11)
    ax.plot(bx[-1], by[-1], "s", color="tab:red", ms=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.set_title("траектория по 23 подтверждённым поворотам\n"
                 "масштаб поворота — допущение, форма от него не зависит")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("P05 — сигнал поворота есть в T4/T5 и у их прямых получателей, "
                 "и теряется к нисходящим нейронам. Воспроизводимость на двух seed "
                 "разделяет сигнал и шум.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "p05_summary.png", dpi=125)
    plt.close(fig)
    print(f"Wrote {OUT}/p05_summary.png")


if __name__ == "__main__":
    main()
