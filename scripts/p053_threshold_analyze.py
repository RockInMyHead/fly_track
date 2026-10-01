#!/usr/bin/env python3
"""
P05.3 — is the spike threshold what kills the turn signal?

Reads the membrane trace and asks, for every neuron that was watched, three things at
once:

    the drive      does the directional path deliver a different input on left and
                   right turns, and by how much
    the membrane   does v move with it, and where does it sit relative to the threshold
    the spikes     does the firing rate follow

The dynamics make the question sharp. With no input v settles at tonic/(1-decay) = 0.772,
the threshold is 1.0, so the whole usable range is 0.228, and the noise kick is 0.220 per
step at 1.2 Hz. That noise is 97 percent of the range. A directional input has to move v
by a decent fraction of 0.228 to change the firing rate, and it does so against a
perturbation of the same size.

The controlled comparison
------------------------
DNbe001 and DNpe056 receive almost the same weight from the carriers of P05.1, 0.0535
against 0.0529, and yet one carries the turn direction at AUC 0.955 and the other at
0.500. Both are in the set below. Whatever separates them is a mechanism, not a matter of
how much input arrives.

Separation is oriented, max(AUC, 1-AUC), because a neuron inhibited on a leftward turn
carries the direction just as well as one excited by it.

Ground truth is the 23 turns confirmed by eye as LEFT or RIGHT.

Usage:
    PYTHONPATH=. python scripts/p053_threshold_analyze.py
    PYTHONPATH=. python scripts/p053_threshold_analyze.py --bn 8
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

TRACE = ROOT / "output/p053_threshold/spike_trace.npz"
TARGETS = ROOT / "output/p053_threshold/targets.csv"
DYNAMICS = ROOT / "output/p053_threshold/dynamics.json"
REVIEW = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
RUN = ROOT / "scripts/p053_threshold_run.py"
OUT = ROOT / "output/p053_threshold"


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


def orient(v: float) -> float:
    return max(v, 1.0 - v)


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
            out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "kind": truth})
    out.sort(key=lambda e: e["t0"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-s", type=float, default=None,
                    help="seconds before the turn used as baseline; default = same length")
    args = ap.parse_args()

    tg = {i: t for i, t in enumerate(csv.DictReader(TARGETS.open()))}
    # the incoming weight of a neuron is normalised to 1.0 in this connectome (the
    # columns sum to one), so an older targets file without the column is still usable
    for t in tg.values():
        t.setdefault("in_weight", "1.0")
    dyn = json.loads(DYNAMICS.read_text(encoding="utf-8"))
    d = np.load(TRACE)
    t = d["t"].astype(np.float64)
    v_pre = d["v_pre"].astype(np.float64)
    fired = d["fired"].astype(np.float64)
    total_in = d["total_in"].astype(np.float64)
    drive = d["drive_carrier"].astype(np.float64)
    drive_L = d["drive_L"].astype(np.float64)
    drive_R = d["drive_R"].astype(np.float64)

    thr = dyn["threshold"]
    base_v = dyn["baseline_v"]
    noise = dyn["noise_amp"]
    print("P05.3 — спайковый порог")
    print(f"  запись: {len(t):,} шагов, {t[-1] - t[0]:.0f} с, {v_pre.shape[1]} нейронов")
    print(f"  порог {thr}, базовое v {base_v:.4f}, запас {thr - base_v:.4f}, "
          f"шум за шаг {noise:.3f} ({noise / (thr - base_v) * 100:.0f}% запаса)")

    turns = [w for w in human_turns() if w["t0"] >= t[0] and w["t1"] <= t[-1]]
    is_R = np.array([w["kind"] == "RIGHT" for w in turns])
    print(f"  подтверждённых поворотов: {len(turns)} "
          f"({int(is_R.sum())} RIGHT, {int((~is_R).sum())} LEFT)")

    def win(vals: np.ndarray, a: float, b: float) -> np.ndarray:
        m = (t >= a) & (t <= b)
        return np.median(vals[m], axis=0) if m.any() else np.full(vals.shape[1], np.nan)

    def rate(vals: np.ndarray, a: float, b: float) -> np.ndarray:
        m = (t >= a) & (t <= b)
        return vals[m].mean(axis=0) if m.any() else np.full(vals.shape[1], np.nan)

    # per-turn values
    V, F, I, D, DL, DR = (np.zeros((len(turns), v_pre.shape[1])) for _ in range(6))
    Dm, DmE, DmW = (np.zeros((len(turns), v_pre.shape[1])) for _ in range(3))
    for k, w in enumerate(turns):
        V[k] = win(v_pre, w["t0"], w["t1"])
        F[k] = rate(fired, w["t0"], w["t1"])
        I[k] = win(total_in, w["t0"], w["t1"])
        D[k] = win(drive, w["t0"], w["t1"])
        DL[k] = win(drive_L, w["t0"], w["t1"])
        DR[k] = win(drive_R, w["t0"], w["t1"])
        # the carrier drive is sparse, and a median over a mostly-zero window gives zero
        # for every window, which reads as AUC 0.5 no matter how the neuron behaves. The
        # mean keeps the magnitude, so the drive is scored on that as well.
        Dm[k] = rate(drive, w["t0"], w["t1"])
        DmE[k] = rate(drive_L, w["t0"], w["t1"])
        DmW[k] = rate(drive_R, w["t0"], w["t1"])

    gain = float(dyn["gain"])
    decay = float(dyn["decay"])
    steady = gain / (1.0 - decay)      # v contribution of a sustained unit of input
    rows = []
    for j in range(v_pre.shape[1]):
        ti = tg[j]
        vmed = float(np.median(v_pre[:, j]))
        vstd = float(np.std(v_pre[:, j]))
        fmed = float(np.mean(fired[:, j])) / 0.02
        # the noise floor: high-frequency wobble of v, which the slow drive cannot explain
        resid = v_pre[:, j] - np.convolve(v_pre[:, j], np.ones(25) / 25, mode="same")
        nfloor = float(np.std(resid))
        # where a sustained carrier drive would place v, and whether that reaches the
        # threshold on its own. This is the question the threshold hypothesis asks.
        drive_mean = float(np.mean(drive[:, j]))
        drive_v_ss = drive_mean * steady
        drive_v_turn = float(np.mean(D[:, j])) * steady
        # firing is always a threshold crossing, so how often the membrane sits near the
        # threshold decides whether a small bias can change the rate at all
        pre_spike = v_pre[1:, j][fired[1:, j] > 0]
        v_before_spike = float(pre_spike.mean()) if len(pre_spike) else float("nan")
        frac_near = float(np.mean(v_pre[:, j] > 0.9))
        reset_frac = float(np.mean(v_pre[:, j] < 0.1))
        rec = {
            "col": j,
            "cell_type": ti["cell_type"], "side": ti["side"], "set": ti["set"],
            "cell": int(ti["cell"]), "in_degree": int(ti["in_degree"]),
            "carrier_weight": float(ti["carrier_weight"]),
            "carrier_frac": float(ti["carrier_weight"]) / max(float(ti["in_weight"]), 1e-9),
            "w_left": float(ti["w_from_left_carriers"]),
            "w_right": float(ti["w_from_right_carriers"]),
            "w_exc_left": float(ti["w_exc_from_left"]),
            "w_exc_right": float(ti["w_exc_from_right"]),
            "w_inh_left": float(ti["w_inh_from_left"]),
            "w_inh_right": float(ti["w_inh_from_right"]),
            "v_median": vmed,
            "v_sigma": vstd,
            "dist_to_threshold": thr - vmed,
            "noise_floor": nfloor,
            "fire_hz": fmed,
            "auc_v": orient(auc_raw(V[is_R, j], V[~is_R, j])),
            "auc_fire": orient(auc_raw(F[is_R, j], F[~is_R, j])),
            "auc_total_in": orient(auc_raw(I[is_R, j], I[~is_R, j])),
            "auc_drive": orient(auc_raw(D[is_R, j], D[~is_R, j])),
            "auc_drive_mean": orient(auc_raw(Dm[is_R, j], Dm[~is_R, j])),
            "auc_driveL_mean": orient(auc_raw(DmE[is_R, j], DmE[~is_R, j])),
            "auc_driveR_mean": orient(auc_raw(DmW[is_R, j], DmW[~is_R, j])),
            "drive_mean": drive_mean,
            "drive_v_ss": drive_v_ss,
            "drive_v_turn": drive_v_turn,
            "drive_reaches_threshold": bool(np.median(v_pre[:, j]) + abs(drive_v_turn) >= thr),
            "v_before_spike": v_before_spike,
            "frac_near_threshold": frac_near,
            "frac_reset": reset_frac,
            "auc_drive_L": orient(auc_raw(DL[is_R, j], DL[~is_R, j])),
            "auc_drive_R": orient(auc_raw(DR[is_R, j], DR[~is_R, j])),
            "dv_turn": float(np.mean(V[is_R, j]) - np.mean(V[~is_R, j])),
            "dfire_turn": float(np.mean(F[is_R, j]) - np.mean(F[~is_R, j])) / 0.02,
            "d_in_turn": float(np.mean(I[is_R, j]) - np.mean(I[~is_R, j])),
            "d_drive_turn": float(np.mean(D[is_R, j]) - np.mean(D[~is_R, j])),
            "d_driveL_turn": float(np.mean(DL[is_R, j]) - np.mean(DL[~is_R, j])),
            "d_driveR_turn": float(np.mean(DR[is_R, j]) - np.mean(DR[~is_R, j])),
        }
        rec["dv_over_sigma"] = abs(rec["dv_turn"]) / max(vstd, 1e-9)
        rec["dv_over_range"] = abs(rec["dv_turn"]) / (thr - base_v)
        rec["dv_over_noise"] = abs(rec["dv_turn"]) / max(noise, 1e-9)
        rows.append(rec)

    keep = [r for r in rows if r["set"] == "keeps"]
    lose = [r for r in rows if r["set"] == "same_input_loses"]
    little = [r for r in rows if r["set"] == "little_input"]

    def summ(name: str, rs: list[dict]) -> dict:
        if not rs:
            return {}
        f = lambda k: np.array([r[k] for r in rs])
        return {
            "n": len(rs),
            "carrier_frac": float(np.median(f("carrier_frac"))),
            "auc_drive": float(np.median(f("auc_drive"))),
            "auc_drive_mean": float(np.median(f("auc_drive_mean"))),
            "auc_v": float(np.median(f("auc_v"))),
            "auc_fire": float(np.median(f("auc_fire"))),
            "v_median": float(np.median(f("v_median"))),
            "dist_to_threshold": float(np.median(f("dist_to_threshold"))),
            "dv_turn": float(np.median(f("dv_turn"))),
            "dv_over_range": float(np.median(f("dv_over_range"))),
            "dv_over_noise": float(np.median(f("dv_over_noise"))),
            "dv_over_sigma": float(np.median(f("dv_over_sigma"))),
            "fire_hz": float(np.median(f("fire_hz"))),
            "drive_v_ss": float(np.median(f("drive_v_ss"))),
            "v_before_spike": float(np.nanmedian(f("v_before_spike"))),
            "frac_near_threshold": float(np.median(f("frac_near_threshold"))),
            "frac_reset": float(np.median(f("frac_reset"))),
            "n_drive_reaches_threshold": int(np.sum(f("drive_reaches_threshold"))),
        }

    S = {"keeps": summ("keeps", keep), "same_input_loses": summ("loses", lose),
         "little_input": summ("little", little)}

    print(f"\n=== ПО ГРУППАМ ===")
    print(f"  {'группа':>18s} {'кл':>4s} {'доля вх':>8s} {'auc вх':>7s} {'auc вх(ср)':>11s} "
          f"{'auc v':>7s} {'auc спайк':>10s} {'v':>7s} {'до порога':>10s} "
          f"{'dв/запас':>9s} {'вход->v':>8s} {'v перед спайком':>16s} {'у порога':>10s}")
    for k, label in (("keeps", "несут направление"), ("same_input_loses", "тот же вход, теряют"),
                     ("little_input", "мало входа")):
        s = S[k]
        if not s:
            continue
        print(f"  {label:>18s} {s['n']:4d} {s['carrier_frac'] * 100:7.2f}% "
              f"{s['auc_drive']:7.3f} {s['auc_drive_mean']:11.3f} "
              f"{s['auc_v']:7.3f} {s['auc_fire']:10.3f} "
              f"{s['v_median']:7.4f} {s['dist_to_threshold']:10.4f} "
              f"{s['dv_over_range']:8.1%} {s['drive_v_ss']:8.4f} "
              f"{s['v_before_spike']:16.4f} {s['frac_near_threshold'] * 100:9.1f}%")

    # ---- the controlled pair -------------------------------------------------
    print(f"\n=== КОНТРОЛЬНАЯ ПАРА: одинаковый вход, разный исход ===")
    print(f"  {'нейрон':>18s} {'вес пути':>9s} {'auc вх(ср)':>11s} {'v':>7s} {'до порога':>10s} "
          f"{'dв':>10s} {'dв/шум':>7s} {'auc v':>7s} {'auc спайк':>10s} {'Гц':>6s}")
    pair = [r for r in rows if r["cell_type"] in ("DNbe001", "DNpe056")]
    for r in sorted(pair, key=lambda z: (z["cell_type"], z["side"])):
        print(f"  {r['cell_type'] + ' ' + r['side']:>18s} {r['carrier_weight']:9.4f} "
              f"{r['auc_drive_mean']:11.3f} {r['v_median']:7.4f} {r['dist_to_threshold']:10.4f} "
              f"{r['dv_turn']:+10.5f} {r['dv_over_noise']:7.3f} {r['auc_v']:7.3f} "
              f"{r['auc_fire']:10.3f} {r['fire_hz']:6.2f}")

    # ---- every watched neuron ------------------------------------------------
    print(f"\n=== ВСЕ НАБЛЮДАЕМЫЕ НЕЙРОНЫ ===")
    print(f"  {'нейрон':>18s} {'группа':>16s} {'вход':>8s} {'auc вх(ср)':>11s} {'v':>7s} "
          f"{'до пор':>8s} {'dв/запас':>9s} {'auc v':>7s} {'auc спайк':>10s} {'Гц':>6s}")
    for r in sorted(rows, key=lambda z: (-z["auc_fire"], z["cell_type"])):
        tag = {"keeps": "несёт", "same_input_loses": "тот же вход", "little_input": "мало входа"}[r["set"]]
        print(f"  {r['cell_type'] + ' ' + r['side']:>18s} {tag:>16s} "
              f"{r['carrier_frac'] * 100:7.2f}% {r['auc_drive_mean']:11.3f} "
              f"{r['v_median']:7.4f} {r['dist_to_threshold']:8.4f} "
              f"{r['dv_over_range']:8.1%} {r['auc_v']:7.3f} {r['auc_fire']:10.3f} "
              f"{r['fire_hz']:6.2f}")

    # ---- does the drive reach the threshold? --------------------------------
    print(f"\n=== ДОХОДИТ ЛИ ВХОД ДО ПОРОГА ===")
    dv = np.array([r["dv_turn"] for r in rows])
    dvi = np.array([r["d_in_turn"] for r in rows])
    dist = np.array([r["dist_to_threshold"] for r in rows])
    vmed = np.array([r["v_median"] for r in rows])
    dvss = np.array([r["drive_v_ss"] for r in rows])
    reaches = np.array([r["drive_reaches_threshold"] for r in rows])
    print(f"  медиана v по всем нейронам:        {np.median(vmed):.4f}")
    print(f"  медиана расстояния до порога:      {np.median(dist):.4f}")
    print(f"  медиана сдвига v на повороте:      {np.median(np.abs(dv)):.5f}")
    print(f"  сдвиг v как доля запаса:           "
          f"{np.median(np.abs(dv)) / (thr - base_v) * 100:.1f}%")
    print(f"  сдвиг v как доля шума за шаг:      "
          f"{np.median(np.abs(dv)) / noise * 100:.1f}%")
    print(f"\n  куда бы поставил v постоянный вход от пути (вход x {steady:.1f}):")
    print(f"    медиана этого вклада:            {np.median(dvss):.4f}")
    print(f"    как доля расстояния до порога:   "
          f"{np.median(dvss) / np.median(dist) * 100:.1f}%")
    print(f"    нейронов, где вклад в одиночку доводит v до порога: "
          f"{int(reaches.sum())} из {len(rows)}")

    # correlation chain: does drive predict v predict firing?
    for a, b, la, lb in (("carrier_frac", "dv_over_range", "доля входа", "сдвиг v"),
                         ("d_drive_turn", "dv_turn", "вход", "v"),
                         ("dv_turn", "dfire_turn", "v", "спайки"),
                         ("auc_drive", "auc_fire", "auc входа", "auc спайков"),
                         ("auc_v", "auc_fire", "auc v", "auc спайков"),
                         ("carrier_frac", "auc_fire", "доля входа", "auc спайков"),
                         ("frac_near_threshold", "auc_fire", "доля времени у порога", "auc спайков"),
                         ("fire_hz", "auc_fire", "частота", "auc спайков"),
                         ("v_before_spike", "auc_fire", "v перед спайком", "auc спайков")):
        x = np.array([r[a] for r in rows])
        y = np.array([r[b] for r in rows])
        if x.std() > 0 and y.std() > 0:
            print(f"  корреляция {la} -> {lb}: r = {np.corrcoef(x, y)[0, 1]:+.3f}")

    # ---- outputs ------------------------------------------------------------
    with (OUT / "neuron_threshold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for rs, col, lab in ((keep, "tab:green", "несут направление"),
                         (lose, "tab:red", "тот же вход, теряют"),
                         (little, "0.6", "мало входа")):
        if rs:
            ax.scatter([r["carrier_frac"] * 100 for r in rs],
                       [r["auc_fire"] for r in rs], s=34, color=col, alpha=0.85,
                       label=f"{lab} (n={len(rs)})")
    ax.axhline(0.7, color="0.4", ls="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("доля входа от пути, %")
    ax.set_ylabel("несёт ли нейрон направление (AUC спайков)")
    ax.set_title("вход против исхода")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for rs, col, lab in ((keep, "tab:green", "несут"), (lose, "tab:red", "теряют"),
                         (little, "0.6", "мало входа")):
        if rs:
            ax.scatter([abs(r["dv_over_noise"]) for r in rs],
                       [r["auc_fire"] for r in rs], s=34, color=col, alpha=0.85, label=lab)
    ax.axhline(0.7, color="0.4", ls="--"); ax.axvline(1.0, color="k", ls=":", label="сдвиг = шум")
    ax.set_xscale("log")
    ax.set_xlabel("сдвиг v на повороте / шум за шаг")
    ax.set_ylabel("AUC спайков")
    ax.set_title("доходит ли сдвиг до уровня шума")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist([r["dist_to_threshold"] for r in rows], bins=20, color="tab:blue")
    ax.axvline(dyn["distance_to_threshold"], color="tab:red", ls="--",
               label=f"теория {dyn['distance_to_threshold']:.3f}")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("v нейрона до порога")
    ax.set_ylabel("нейронов")
    ax.set_title("где сидят нейроны относительно порога")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    pair = [r for r in rows if r["cell_type"] in ("DNbe001", "DNpe056") and r["side"] == "L"]
    ax = fig.add_subplot(gs[1, 0])
    # trace around the first left turn for the two neurons
    lw = [w for w in turns if w["kind"] == "LEFT"]
    if pair and lw:
        w0 = lw[0]
        m = (t >= w0["t0"] - 2) & (t <= w0["t1"] + 2)
        for r, col in zip(pair, ("tab:green", "tab:red")):
            j = int(r["col"])
            ax.plot(t[m] - w0["t0"], v_pre[m, j], color=col, lw=1.2,
                    label=f"{r['cell_type']} {r['side']}")
        ax.axhline(thr, color="k", ls="--", lw=1.2, label="порог")
        ax.axhline(base_v, color="0.6", ls=":", label=f"покой {base_v:.3f}")
        ax.axvspan(0, w0["t1"] - w0["t0"], color="tab:purple", alpha=0.12, label="поворот")
        ax.set_ylim(base_v - 0.06, thr + 0.03)
        ax.set_xlabel("время вокруг поворота, с")
        ax.set_ylabel("v")
        ax.set_title("мембрана на повороте:\nодинаковый вход, разный результат")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    xx = np.array([r["d_drive_turn"] for r in rows])
    yy = np.array([r["dv_turn"] for r in rows])
    ax.scatter(xx, yy, s=30, color="tab:blue", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    if xx.std() > 0 and yy.std() > 0:
        p = np.polyfit(xx, yy, 1)
        xs = np.linspace(xx.min(), xx.max(), 10)
        ax.plot(xs, np.polyval(p, xs), color="tab:red", ls="--",
                label=f"r {np.corrcoef(xx, yy)[0, 1]:+.2f}")
    ax.set_xlabel("сдвиг входа на повороте")
    ax.set_ylabel("сдвиг v на повороте")
    ax.set_title("преобразование вход -> мембрана")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    if pair:
        r = pair[0]
        j = int(r["col"])
        m = (t >= 100) & (t <= 300)
        ax.plot(t[m], v_pre[m, j], lw=0.7, color="tab:green", label=f"{r['cell_type']}")
        ax.axhline(thr, color="k", ls="--")
        ax.axhline(base_v, color="0.6", ls=":")
        ax.set_xlabel("время, с"); ax.set_ylabel("v")
        ax.set_title("мембрана на длинном отрезке")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("P05.3 — порог против направленного сигнала. "
                 f"Запас до порога {dyn['distance_to_threshold']:.3f}, "
                 f"шум за шаг {dyn['noise_amp']:.2f}.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "threshold.png", dpi=125)
    plt.close(fig)

    (OUT / "report.json").write_text(json.dumps({
        "dynamics": dyn,
        "n_steps": int(len(t)), "duration_s": float(t[-1] - t[0]),
        "n_neurons": int(v_pre.shape[1]),
        "n_turns": len(turns), "n_right": int(is_R.sum()),
        "summary": S,
        "medians": {
            "v": float(np.median(vmed)),
            "dist_to_threshold": float(np.median(dist)),
            "dv_turn": float(np.median(np.abs(dv))),
            "d_input_turn": float(np.median(np.abs(dvi))),
            "dv_share_of_range": float(np.median(np.abs(dv)) / (thr - base_v)),
            "dv_share_of_noise": float(np.median(np.abs(dv)) / noise),
        },
        "per_neuron": rows,
        "method": "v и спайки читаются на каждом шаге; вход разложен на общий и на вклад "
                  "носителей P05.1 через истинные входящие связи (столбцы матрицы)",
        "verbatim": "веса и порог не менялись, источники взяты замороженными из P05.1",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
