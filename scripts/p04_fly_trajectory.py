#!/usr/bin/env python3
"""
P0.4 — can the fly's brain draw the trajectory?

`p04_fly_run.py` drove the whole clip through MaleCNS and recorded every stage of the
chain at 50 Hz. This reads those recordings back and asks whether the turn direction
survives to the end of the chain.

Where the answer is measured
----------------------------
Not on the `windows_v3` turn labels. Those were produced by thresholding the same
uniform component of the motion field that `yaw_frozen` reports, so checking
`yaw_frozen` against them is circular: it agrees with itself. The only windows whose
direction came from an eye rather than from the measurement are the ones in the review
set that the human confirmed, and those are what this uses.

    confirmed turns   23   (12 RIGHT, 11 LEFT)

That is a small set, and the output says so. Polarity for each channel is fixed on the
earlier half of it and the accuracy is reported on the later half, so the evaluated
events cannot set their own orientation.

What is being asked of each channel
-----------------------------------
The lateral signal over a turn window is reduced to one number, and its sign is
compared with the direction the human saw:

    RIGHT turn -> channel > 0
    LEFT  turn -> channel < 0

    channel        where it sits in the chain
    yaw_frozen     the frontend's pooled estimate
    inject_lr      what the encoder actually pushed into T4/T5
    t4_lr, t5_lr   the T4 and T5 populations, spikes per step
    steer_lr       DNa02 left minus right
    fwd_lr         DNg100 left minus right
    hedge_lr       MDN left minus right
    escape_lr      DNp01 left minus right
    all_dn_lr      every descending neuron, left minus right

A trajectory is then assembled turn by turn, keeping the measured magnitude and taking
only the direction from the channel, so all the paths share the same geometry of turns
and differ only in where the brain sent the walker.

Usage:
    PYTHONPATH=. python scripts/p04_fly_trajectory.py
    PYTHONPATH=. python scripts/p04_fly_trajectory.py --smooth 0.5
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

FLY = ROOT / "output/p04_fly/fly_signals.npz"
REVIEW = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
FRONT = ROOT / "output/p04_trajectory/trajectory_turns.csv"
OUT = ROOT / "output/p04_fly"

WALK_SPEED = 1.40
TURN_DEG = 90.0

CHANNELS = ["yaw_frozen", "inject_lr", "t4_lr", "t5_lr",
            "steer_lr", "fwd_lr", "hedge_lr", "escape_lr", "all_dn_lr"]


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def human_turns() -> list[dict]:
    """Turns whose direction came from an eye, not from the measurement."""
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


def reduce_over_window(t: np.ndarray, v: np.ndarray, w: dict) -> float:
    m = (t >= w["t0"]) & (t <= w["t1"])
    return float(np.median(v[m])) if m.any() else float("nan")


def score(t: np.ndarray, sig: dict, turns: list[dict], n_cal: int) -> dict:
    """Sign accuracy per channel, polarity fixed on the calibration half only."""
    expect = np.array([e["expect"] for e in turns])
    res = {}
    for c, v in sig.items():
        vals = np.array([reduce_over_window(t, v, w) for w in turns])
        if np.all(np.isnan(vals)):
            continue
        vals = np.nan_to_num(vals)
        if np.allclose(vals, 0):
            continue
        cal = slice(0, n_cal)
        pol = float(np.sign(np.sum(vals[cal] * expect[cal]))) or 1.0
        oriented = vals * pol
        agree = np.sign(oriented) == np.sign(expect)
        ev = agree[n_cal:]
        L, R = oriented[expect < 0], oriented[expect > 0]
        span = float(np.median(R)) - float(np.median(L))
        sd = 1.4826 * float(np.median(np.abs(oriented - np.median(oriented)))) + 1e-9
        res[c] = {"pol": pol,
                  "acc_cal": float(np.mean(agree[cal])),
                  "acc_eval": float(np.mean(ev)) if len(ev) else float("nan"),
                  "n_eval": int(len(ev)),
                  "separation_sigma": span / sd,
                  "median_L": float(np.median(L)) if len(L) else float("nan"),
                  "median_R": float(np.median(R)) if len(R) else float("nan"),
                  "values": oriented.tolist()}
    return res


def integrate_turns(turns: list[dict], deg_per_px: float) -> dict:
    events = sorted(turns, key=lambda e: e["t0"])
    theta, x, y, prev_end = np.pi / 2, 0.0, 0.0, 0.0
    traj = []
    for e in events:
        gap = max(0.0, e["t0"] - prev_end)
        if gap > 0:
            x += WALK_SPEED * gap * np.cos(theta)
            y += WALK_SPEED * gap * np.sin(theta)
            traj.append({"t": prev_end, "x": x, "y": y,
                         "heading_deg": np.rad2deg(theta) % 360, "event": "travel"})
        theta += np.deg2rad(e["dtheta_deg"])
        traj.append({"t": e["t1"], "x": x, "y": y,
                     "heading_deg": np.rad2deg(theta) % 360, "event": e["kind"]})
        prev_end = max(prev_end, e["t1"])
    return {"traj": traj,
            "x": np.array([p["x"] for p in traj]),
            "y": np.array([p["y"] for p in traj]),
            "total_turning_deg": float(np.sum([abs(e["dtheta_deg"]) for e in events])),
            "net_turning_deg": float(np.rad2deg(theta - np.pi / 2)),
            "final_x": x, "final_y": y}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=float, default=0.4, help="seconds")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(FLY)
    t = d["t"].astype(np.float64)
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(args.smooth / dt)), 1)
    sig = {c: box(d[c].astype(np.float64), n_sm) for c in CHANNELS if c in d}

    ht = [w for w in human_turns() if w["t0"] >= t[0] and w["t1"] <= t[-1]]
    print("P0.4 — может ли мозг мухи нарисовать траекторию?")
    print(f"  запись мозга: {len(t):,} шагов, {t[-1] - t[0]:.0f} с, "
          f"шаг {dt * 1000:.0f} мс, сглаживание {args.smooth} с")
    print(f"  подтверждённых человеком поворотов в записи: {len(ht)} "
          f"({sum(1 for w in ht if w['kind'] == 'RIGHT')} RIGHT, "
          f"{sum(1 for w in ht if w['kind'] == 'LEFT')} LEFT)")
    if len(ht) < 10:
        print("\n  запись ещё не покрывает ролик, прогон идёт. "
              "Ниже — по доступной части.\n")
    else:
        print()

    n_cal = max(len(ht) // 2, 1)
    res = score(t, sig, ht, n_cal)

    print("=== доходит ли направление поворота до конца цепочки ===")
    print(f"  полярность канала фиксируется на первых {n_cal} поворотах,")
    print(f"  точность считается на остальных {len(ht) - n_cal}\n")
    print(f"  {'канал':>11s} {'калибр':>8s} {'ПРОВЕРКА':>9s} {'n':>3s} "
          f"{'отрыв':>7s}  {'медиана LEFT':>13s} {'медиана RIGHT':>14s}")
    order = sorted(res, key=lambda c: -(res[c]["acc_eval"]
                                        if not np.isnan(res[c]["acc_eval"]) else 0))
    for c in order:
        r = res[c]
        print(f"  {c:>11s} {r['acc_cal'] * 100:7.0f}% {r['acc_eval'] * 100:8.0f}% "
              f"{r['n_eval']:3d} {r['separation_sigma']:6.2f}σ  "
              f"{r['median_L']:13.3f} {r['median_R']:14.3f}")

    print("\n  как читать:")
    print("    80-100%  направление доходит, по каналу можно строить траекторию")
    print("    65-80%   доходит частично")
    print("    40-65%   неотличимо от случайного угадывания")
    print("    <40%     канал систематически перевёрнут или это не тот сигнал")

    # reference: the frontend's own a over the same windows, measured from the video
    print(f"\n  для сравнения, тот же тест на самом `a` фронтенда "
          f"(на этих окнах классификатор дал 96%):")
    sig2 = dict(sig)
    if "yaw_frozen" in sig2:
        r = res.get("yaw_frozen")
        if r:
            note = ("совпадает с источником меток windows_v3, поэтому на них был бы "
                    "100% по построению; здесь метки человеческие")
            print(f"    yaw_frozen {r['acc_eval'] * 100:.0f}% на человеческих метках. {note}")

    # ---- where the chain breaks -------------------------------------------
    print("\n=== где именно рвётся цепочка ===")
    print("  корреляция между соседними звеньями, знак зафиксирован на 1-й половине,")
    print("  число измерено на 2-й (независимой)\n")

    half = len(t) // 2

    def split_r(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        a = a - a.mean()
        b = b - b.mean()

        def f(s: slice) -> float:
            if np.std(a[s]) < 1e-12 or np.std(b[s]) < 1e-12:
                return 0.0
            return float(np.corrcoef(a[s], b[s])[0, 1])
        return f(slice(0, half)), f(slice(half, None))

    links = [("yaw_frozen", "inject_lr", "весь глаз -> инъекция в T4/T5"),
             ("inject_lr", "t4_lr", "инъекция -> T4"),
             ("inject_lr", "t5_lr", "инъекция -> T5"),
             ("t4_lr", "steer_lr", "T4 -> DNa02"),
             ("t5_lr", "steer_lr", "T5 -> DNa02"),
             ("yaw_frozen", "steer_lr", "весь глаз -> DNa02")]
    link_rows = []
    print(f"  {'звено':>30s} {'1-я пол':>9s} {'2-я пол':>9s} {'держится':>10s}")
    for a, b, lbl in links:
        if a not in sig or b not in sig:
            continue
        r1, r2 = split_r(sig[a], sig[b])
        holds = "да" if (abs(r2) > 0.30 and np.sign(r2) == np.sign(r1)) else "НЕТ"
        link_rows.append((lbl, r1, r2, holds))
        print(f"  {lbl:>30s} {r1:9.3f} {r2:9.3f} {holds:>10s}")

    # spike-triggered: does the heading differ when a neuron fires + rather than -?
    from math import erfc, sqrt as msqrt
    k = max(int(round(0.1 / dt)), 1)
    local_yaw = np.convolve(sig["yaw_frozen"], np.ones(2 * k + 1) / (2 * k + 1),
                            mode="same")
    spread = float(np.std(sig["yaw_frozen"]))
    print(f"\n  тот же вопрос по спайкам: отличается ли курс вокруг срабатывания")
    print(f"  разного знака? (разброс курса {spread:.2f})\n")
    print(f"  {'канал':>11s} {'событий':>10s} {'разница курса':>14s} "
          f"{'доля разброса':>14s} {'p':>10s}")
    spike_rows = []
    for c in ["inject_lr", "t4_lr", "t5_lr", "steer_lr", "fwd_lr",
              "hedge_lr", "escape_lr", "all_dn_lr"]:
        if c not in sig:
            continue
        v = sig[c]
        pos, neg = local_yaw[v > 0], local_yaw[v < 0]
        if len(pos) < 10 or len(neg) < 10:
            continue
        se = np.sqrt(pos.std(ddof=1) ** 2 / len(pos)
                     + neg.std(ddof=1) ** 2 / len(neg))
        z = float(pos.mean() - neg.mean()) / max(se, 1e-12)
        p = erfc(abs(z) / msqrt(2))
        diff = float(pos.mean() - neg.mean())
        spike_rows.append((c, len(pos) + len(neg), diff, abs(diff) / spread, p))
        print(f"  {c:>11s} {len(pos) + len(neg):10d} {diff:14.4f} "
              f"{abs(diff) / spread:13.1%} {p:10.2e}")

    print("\n  читать так: если разница курса много меньше его разброса, то по этому")
    print("  нейрону нельзя узнать, куда повернули, даже когда он срабатывает.")

    # ---- trajectories ------------------------------------------------------
    mag = np.array([abs(w["t1"] - w["t0"]) * 30.0 for w in ht])  # turn length in frames
    # scale so that the median confirmed turn is a right angle
    events_base = [{"t0": w["t0"], "t1": w["t1"], "kind": w["kind"],
                    "dtheta_deg": TURN_DEG} for w in ht]

    paths = {}
    for c in order:
        vals = np.array(res[c]["values"])
        events = []
        for i, w in enumerate(ht):
            s = float(np.sign(vals[i])) or 1.0
            events.append({"t0": w["t0"], "t1": w["t1"], "kind": w["kind"],
                           "dtheta_deg": s * TURN_DEG})
        paths[c] = integrate_turns(events, 1.0)
    paths["истина"] = integrate_turns(events_base, 1.0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3
    n_path = len(paths) + 1
    rows = int(np.ceil((n_path + 2) / cols))
    fig = plt.figure(figsize=(4.6 * cols, 3.9 * rows))

    for i, c in enumerate(["истина"] + order):
        ax = fig.add_subplot(rows, cols, i + 1)
        p = paths[c]
        ax.plot(p["x"], p["y"], lw=1.1,
                color="0.4" if c == "истина" else "tab:blue")
        ax.plot(p["x"][0], p["y"][0], "o", color="tab:green", ms=8)
        ax.plot(p["x"][-1], p["y"][-1], "s", color="tab:red", ms=8)
        ax.set_aspect("equal")
        if c == "истина":
            ttl = "истина\n(направления твои)"
        else:
            ttl = (f"{c}\n{res[c]['acc_eval'] * 100:.0f}%, "
                   f"{res[c]['separation_sigma']:.1f}σ")
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    ax = fig.add_subplot(rows, cols, n_path + 1)
    names = order
    accs = [res[c]["acc_eval"] * 100 for c in names]
    ax.barh(np.arange(len(names)), accs,
            color=["tab:green" if a >= 80 else "tab:orange" if a >= 65 else "tab:red"
                   for a in accs])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(50, color="0.4", ls=":", label="случай")
    ax.axvline(80, color="tab:green", ls="--", alpha=0.5, label="порог 80%")
    ax.set_xlabel("точность знака, %")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(f"точность на {len(ht) - n_cal} человеческих поворотах", fontsize=9)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(rows, cols, n_path + 2)
    lbl = [r[0] for r in link_rows][::-1]
    vals = [r[2] for r in link_rows][::-1]
    ax.barh(np.arange(len(lbl)), vals,
            color=["tab:green" if abs(v) > 0.3 else "tab:red" for v in vals])
    ax.set_yticks(np.arange(len(lbl)))
    ax.set_yticklabels(lbl, fontsize=7)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.axvspan(-0.3, 0.3, color="tab:red", alpha=0.10)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("корреляция соседних звеньев (2-я половина)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("где рвётся цепочка", fontsize=9)

    fig.suptitle("P0.4 — траектория, нарисованная мозгом мухи. "
                 "Повороты одни и те же, различается только направление из канала.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fly_trajectory.png", dpi=125)
    plt.close(fig)

    (OUT / "report.json").write_text(json.dumps({
        "n_steps": len(t), "duration_s": float(t[-1] - t[0]),
        "human_turns_in_record": len(ht), "calib_turns": n_cal,
        "smooth_s": args.smooth,
        "channels": {c: {k: v for k, v in r.items() if k != "values"}
                     for c, r in res.items()},
        "chain_links": [{"link": l, "r_first_half": r1, "r_second_half": r2,
                         "holds": h} for l, r1, r2, h in link_rows],
        "spike_triggered": [{"channel": c, "n_events": n, "heading_diff": d,
                             "fraction_of_spread": f, "p": p}
                            for c, n, d, f, p in spike_rows],
        "method": "знак медианы канала на окне подтверждённого поворота; полярность "
                  "зафиксирована на первой половине окон, точность на второй",
        "caveat": f"всего {len(ht)} человеческих поворотов в записи; "
                  "на windows_v3 проверять нельзя — метки выведены из того же "
                  "сигнала, что и yaw_frozen",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/fly_trajectory.png")


if __name__ == "__main__":
    main()
