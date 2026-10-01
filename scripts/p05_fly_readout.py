#!/usr/bin/env python3
"""
P05 — the trajectory the fly draws once the readout stops cancelling itself.

Why the earlier attempt failed
------------------------------
P0.4 concluded the brain does not carry turn direction. It was measuring this:

    t4_lr = (T4a_L + T4b_L) - (T4a_R + T4b_R)

but T4a and T4b prefer opposite directions, so adding them removed almost everything:

    T4a_L   LEFT 16.8 Hz   RIGHT  7.3 Hz
    T4b_L   LEFT  7.7 Hz   RIGHT 18.4 Hz
    sum     LEFT 24.6 Hz   RIGHT 25.8 Hz     a difference of 1.2 Hz, down from 9.5

The signal was never lost. It was averaged away. Twenty five of the fifty four layer-1
populations separate the confirmed turns with no overlap at all in their ranges.

What is done here
-----------------
A steering signal is built from the individual populations instead of their sums:

    steering(t) = sum over chosen types of  polarity_i * z_i(t)

where `z_i` is the smoothed rate of population `i`, standardised over the recording.
Polarity is taken from the FIRST HALF of the confirmed turns and the trajectory is then
scored on the SECOND HALF, which the polarity never saw. Selection of which types to
include is likewise done on the first half only.

This is the same test set as before: 23 turns whose direction came from an eye.

Usage:
    PYTHONPATH=. python scripts/p05_fly_readout.py
    PYTHONPATH=. python scripts/p05_fly_readout.py --signals output/p05_layer_trace_seed65/layer_signals.npz
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

GROUPS = ROOT / "output/p05_layer_trace/groups.json"
REVIEW = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
FRONT_CSV = ROOT / "output/p04_trajectory/trajectory_turns.csv"
FULL_SIGNAL = ROOT / "output/p02_full/full_signal.npz"
OUT = ROOT / "output/p05_fly_readout"

WALK_SPEED = 1.40
FPS = 30.0
STOP_ACTIVITY = 0.60


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


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


def integrate(yaw_dps: np.ndarray, speed: np.ndarray, t: np.ndarray) -> dict:
    dt = float(np.median(np.diff(t)))
    th = np.pi / 2
    X = np.empty(len(t))
    Y = np.empty(len(t))
    x = y = 0.0
    for i in range(len(t)):
        X[i], Y[i] = x, y
        x += speed[i] * dt * np.cos(th)
        y += speed[i] * dt * np.sin(th)
        th += np.deg2rad(yaw_dps[i]) * dt
    return {"x": X, "y": Y, "heading": th, "t": t,
            "final_x": x, "final_y": y,
            "total_turn": float(np.sum(np.abs(yaw_dps)) * dt)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default=str(
        ROOT / "output/p05_layer_trace/layer_signals.npz"))
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--min-sep", type=float, default=0.75,
                    help="AUC required on the calibration half to include a type")
    ap.add_argument("--layers", default="0,1",
                    help="which layers may contribute (descending neurons do not "
                         "reproduce across seeds, see report)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(args.signals)
    t = d["t"].astype(np.float64)
    G = d["G"].astype(np.float64)
    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(args.smooth / dt)), 1)

    ht = [w for w in human_turns() if w["t0"] >= t[0] and w["t1"] <= t[-1]]
    expect = np.array([w["expect"] for w in ht])
    n_cal = len(ht) // 2
    is_R = expect > 0
    cal = np.zeros(len(ht), bool)
    cal[:n_cal] = True

    print("P05 — траектория по одиночным типам вместо суммы")
    print(f"  запись: {len(t):,} шагов, {t[-1] - t[0]:.0f} с  (файл {Path(args.signals).name})")
    print(f"  поворотов: {len(ht)}, калибровка на первых {n_cal}, счёт на {len(ht) - n_cal}\n")

    ncell = np.array([m["n_cells"] for m in meta], float)
    S = box_mat(G, n_sm) / ncell[None, :]

    # standardise each population over the whole recording, so that populations of very
    # different firing rates contribute comparably
    med = np.median(S, axis=0)
    mad = 1.4826 * np.median(np.abs(S - med[None, :]), axis=0)
    mad = np.where(mad > 1e-9, mad, np.nan)
    Z = (S - med[None, :]) / mad[None, :]

    def win_vals(M: np.ndarray) -> np.ndarray:
        return np.stack([np.array([np.median(M[(t >= w["t0"]) & (t <= w["t1"]), g])
                                   for w in ht]) for g in range(M.shape[1])])

    Vz = win_vals(Z)

    # ---- select and orient using the calibration half only -------------------
    keep_layers = {int(x) for x in args.layers.split(",") if x.strip() != ""}
    sel = []
    for g, m in enumerate(meta):
        if m["layer"] not in keep_layers:
            continue
        if not np.isfinite(mad[g]):
            continue
        cR = Vz[g][cal & is_R]
        cL = Vz[g][cal & ~is_R]
        if len(cR) < 3 or len(cL) < 3:
            continue
        allv = np.concatenate([cR, cL])
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        order = np.argsort(allv, kind="mergesort")
        r = np.empty(len(allv))
        r[order] = np.arange(1, len(allv) + 1)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        ranks = (s / cnt)[inv]
        a = (ranks[: len(cR)].sum() - len(cR) * (len(cR) + 1) / 2) / (len(cR) * len(cL))
        auc = max(a, 1.0 - a)
        if auc < args.min_sep:
            continue
        pol = 1.0 if a > 0.5 else -1.0          # higher = rightward turn
        sel.append({"g": g, "layer": m["layer"], "cell_type": m["cell_type"],
                    "side": m["side"], "n_cells": m["n_cells"], "pol": pol,
                    "auc_cal": auc})

    print(f"=== ОТОБРАНО ПО ПЕРВОЙ ПОЛОВИНЕ (порог AUC {args.min_sep}) ===")
    print(f"  типов: {len(sel)}")
    by_layer = {}
    for s in sel:
        by_layer.setdefault(s["layer"], []).append(s)
    for L in sorted(by_layer):
        print(f"    слой {L}: {len(by_layer[L])} "
              f"({', '.join(s['cell_type'] for s in by_layer[L][:8])}"
              f"{' ...' if len(by_layer[L]) > 8 else ''})")

    if not sel:
        print("\n  ничего не отобрано — калибровка не нашла направленных типов")
        return

    # ---- build the continuous steering signal -------------------------------
    steer = np.zeros(len(t))
    for s in sel:
        steer += s["pol"] * np.nan_to_num(Z[:, s["g"]], nan=0.0)
    steer /= len(sel)
    steer_s = box(steer, max(int(round(1.0 / dt)), 1))

    # ---- score on the held-out half -----------------------------------------
    sv = np.array([np.median(steer_s[(t >= w["t0"]) & (t <= w["t1"])]) for w in ht])
    ev = ~cal
    # a sample is correct if its sign matches; silence counts as its own failure
    correct = np.sign(sv[ev]) == np.sign(expect[ev])
    acc_eval = float(np.mean(correct))
    correct_all = float(np.mean(np.sign(sv) == np.sign(expect)))

    # reference: the frontend on the same turns. Its sign convention is opposite to
    # `expect` (a positive uniform component means the content moved right, so the
    # camera went left), so it is oriented by its own calibration half rather than
    # assumed to agree.
    yaw = box(d["yaw_frozen"].astype(np.float64), n_sm)
    yv = np.array([np.median(yaw[(t >= w["t0"]) & (t <= w["t1"])]) for w in ht])
    yaw_pol = float(np.sign(np.sum(yv[cal] * expect[cal]))) or 1.0
    acc_yaw = float(np.mean(np.sign(yv[~cal] * yaw_pol) == np.sign(expect[~cal])))
    acc_yaw_all = float(np.mean(np.sign(yv * yaw_pol) == np.sign(expect)))

    print(f"\n=== ПРОВЕРКА НА ОТЛОЖЕННОЙ ПОЛОВИНЕ ===")
    print(f"  точность знака, мозг (слой 1+2): {acc_eval * 100:.0f}% "
          f"({int(correct.sum())}/{len(correct)})")
    print(f"  то же на всех поворотах:          {correct_all * 100:.0f}%")
    print(f"  для сравнения, фронтенд:          {acc_yaw * 100:.0f}% "
          f"(на всех {acc_yaw_all * 100:.0f}%)")

    # ---- trajectory, assembled turn by turn --------------------------------
    # The readout is validated ON turn windows. Between turns nothing calibrates it, and
    # integrating it there accumulated 19,000 degrees of drift when it was tried, the
    # same trap that sank the naive frontend integration in P0.4. So the path is built
    # from the turns, with the heading held between them.
    def turns_to_path(signs: np.ndarray) -> dict:
        th = np.pi / 2
        x = y = 0.0
        prev_end = 0.0
        P = []
        for s, w in zip(signs, ht):
            gap = max(0.0, w["t0"] - prev_end)
            if gap > 0:
                x += WALK_SPEED * gap * np.cos(th)
                y += WALK_SPEED * gap * np.sin(th)
                P.append((prev_end, x, y, th))
            th += np.deg2rad(s * 90.0)
            P.append((w["t1"], x, y, th))
            prev_end = max(prev_end, w["t1"])
        return {"t": np.array([p[0] for p in P]), "x": np.array([p[1] for p in P]),
                "y": np.array([p[2] for p in P]),
                "total_turn": float(np.sum([abs(s) * 90.0 for s in signs]))}

    brain_sgn = np.where(np.sign(sv) == 0, 1.0, np.sign(sv))
    tr_brain = turns_to_path(brain_sgn)
    tr_truth = turns_to_path(np.sign(expect))

    print(f"\n=== ТРАЕКТОРИЯ (по поворотам, как для фронтенда) ===")
    print(f"  мозг:   конечная точка ({tr_brain['x'][-1]:.0f}, {tr_brain['y'][-1]:.0f}) м, "
          f"суммарный поворот {tr_brain['total_turn']:.0f}°")
    print(f"  истина: конечная точка ({tr_truth['x'][-1]:.0f}, "
          f"{tr_truth['y'][-1]:.0f}) м, суммарный поворот {tr_truth['total_turn']:.0f}°")
    print(f"  совпадение направлений: "
          f"{int(np.sum(brain_sgn == np.sign(expect)))}/{len(ht)}")

    ref_path = None
    if FRONT_CSV.exists():
        rr = list(csv.DictReader(FRONT_CSV.open()))
        ref_path = (np.array([float(r["x"]) for r in rr]),
                    np.array([float(r["y"]) for r in rr]))

    # ---- plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    ax = axes[0]
    ax.plot(tr_brain["x"], tr_brain["y"], lw=1.4, color="tab:orange",
            label=f"мозг мухи ({acc_eval * 100:.0f}% на отложенных)")
    ax.plot(tr_truth["x"], tr_truth["y"], lw=1.4, color="0.35", ls="--",
            label="истина (твои метки)")
    if ref_path is not None:
        ax.plot(ref_path[0], ref_path[1], lw=1.0, color="tab:blue", alpha=0.6,
                label="фронтенд")
    ax.plot(tr_truth["x"][0], tr_truth["y"][0], "o", color="tab:green", ms=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.set_title("траектория"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    idx = np.arange(len(ht))
    ax.bar(idx, sv, color=["tab:green" if e < 0 else "tab:purple" for e in expect])
    ax.plot(idx, [e * 0 for e in expect], color="k", lw=0.8)
    for i, w in enumerate(ht):
        if not cal[i]:
            ax.plot(i, sv[i], "o", mfc="none", mec="k", ms=9)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("поворот (по времени)")
    ax.set_ylabel("сигнал мозга")
    ax.set_title("сигнал на каждом повороте\nкружком — отложенная половина")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    names = [f"{s['cell_type']} {s['side']}" for s in sel][:16]
    aucs = [s["auc_cal"] for s in sel][:16]
    ax.barh(np.arange(len(names)), aucs, color="tab:orange")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0.5, color="k", ls=":")
    ax.set_xlabel("AUC на калибровочной половине")
    ax.set_title(f"отобранные типы ({len(sel)} всего)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("P05 — траектория по одиночным типам. Сумма T4a+T4b гасила сигнал; "
                 "здесь типы складываются с учётом своей полярности.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fly_readout_trajectory.png", dpi=130)
    plt.close(fig)

    with (OUT / "selected_types.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "cell_type", "side", "n_cells",
                                          "pol", "auc_cal"])
        w.writeheader()
        for s in sel:
            w.writerow({k: s[k] for k in w.fieldnames})

    (OUT / "report.json").write_text(json.dumps({
        "signals_file": str(args.signals),
        "n_turns": len(ht), "n_calibration": n_cal,
        "n_selected": len(sel), "smooth_s": args.smooth,
        "min_sep_auc": args.min_sep,
        "acc_eval_brain": acc_eval, "acc_all_brain": correct_all,
        "acc_frontend_eval": acc_yaw, "acc_frontend_all": acc_yaw_all,
        "selected": sel,
        "why": "P0.4 summed T4a+T4b, which prefer opposite directions, so the sum "
               "cancelled most of the signal; individual populations separate cleanly",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/fly_readout_trajectory.png")


if __name__ == "__main__":
    main()
