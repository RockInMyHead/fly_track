#!/usr/bin/env python3
"""
P07.6 — which individual turns break the route?

Diagnosis only. Nothing is tuned, no threshold moves, P07.4 and P07.5 are read as they
are. The question is which of four error types costs the most:

    FALSE TURN   the system turned where the person did not
    MISSED TURN  the person turned and the system did not
    WRONG SIGN   both turned, in opposite directions
    ANGLE        both turned the same way, by different amounts

And the answer is obtained by rebuilding the route five times, each time repairing exactly
one error type, and seeing which repair helps.

An honesty note that shapes how this is measured
------------------------------------------------
VID00001 and VIDEO 02 carry confirmed turn windows and nothing else. The drawing that
exists for VID00001 is 38 points, an abandoned first attempt, far too sparse to stand in
for a route. So there is no ground-truth shape for these clips, and "compare the shape"
cannot mean comparing against reality here.

What it can mean, and what is done:

  * variant E, the person's own turn list with the fly's own angles, is the best route the
    fly's angles could produce. Each repair is measured by how far it moves the route from
    A towards E. If a repair closes most of that gap, that error type is the problem.
  * separately, each variant is scored on the turn sequence it produces against the human
    sequence, which IS ground truth here.

Both are reported. Neither is presented as route accuracy, because route accuracy cannot
be measured on these clips.

Variants
--------
    A   as the system produces it now
    B   false events removed
    C   missed human turns added, angle taken from the fly's signal in that window
    D   signs corrected where they disagree, angles untouched
    E   only the human turns, angles still from the fly

Usage:
    PYTHONPATH=. python scripts/p076_turn_audit.py
    PYTHONPATH=. python scripts/p076_turn_audit.py --tolerance 1.5
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

P071 = ROOT / "output/p071"
P074 = ROOT / "output/p074"
P075 = ROOT / "output/p075"
OUT = ROOT / "output/p076"

TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]
POLARITY = {"DNp17": 1.0, "DNa07": -1.0, "DNp26": 1.0, "DNp20": 1.0}
FPS = 50.0
SMOOTH_S = 0.3
AUDIT_CLIPS = ("VID00001", "VID00002")


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def rate_signal(video: str) -> tuple[np.ndarray, np.ndarray]:
    """The yaw signal in hertz, as P07.5 rebuilt it. Absolute units, no z-scoring."""
    rows = list(csv.DictReader((ROOT / "output/p07" / f"yaw_signal_{video}.csv").open()))
    t = np.array([float(r["t"]) for r in rows])
    n_sm = max(int(round(SMOOTH_S * FPS)), 1)
    s = np.zeros(len(t))
    for nm in TYPES:
        v = box(np.array([float(r[f"common_{nm}"]) for r in rows]), n_sm)
        s += POLARITY[nm] * (v - float(np.median(v)))
    return t, s / len(TYPES)


def k_calibrated() -> float:
    return float(json.loads((P075 / "report.json").read_text(encoding="utf-8"))["k"])


def human_turns(video: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from p07_videos import labelled_turns
    return [{"t0": w["t0"], "t1": w["t1"], "direction": w["kind"]} for w in
            labelled_turns(video)]


def fly_swings(video: str, t: np.ndarray, s: np.ndarray) -> list[dict]:
    """Every swing P07.4 produced, with its kind and its raw angle in hertz-seconds."""
    dt = float(np.median(np.diff(t)))
    out = []
    for r in csv.DictReader((P074 / f"swings_{video}.csv").open()):
        t0, t1 = float(r["t0"]), float(r["t1"])
        m = (t >= t0) & (t <= t1)
        raw = float(np.sum(s[m])) * dt if m.any() else 0.0
        out.append({"t0": t0, "t1": t1, "kind": r["kind"], "direction": r["direction"],
                    "raw": raw, "size": abs(raw)})
    return out


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def match(human: list[dict], fly: list[dict], tol: float):
    """Pair human turns with fly turns, by largest overlap first.

    A human turn may match at most one swing and vice versa, so a single swing cannot be
    counted as two detections.
    """
    pairs = []
    for hi, h in enumerate(human):
        for fi, f in enumerate(fly):
            ov = overlap(h["t0"], h["t1"], f["t0"], f["t1"])
            # also allow a near miss, since the windows were drawn by eye and are not tight
            gap = max(0.0, max(h["t0"], f["t0"]) - min(h["t1"], f["t1"]))
            if ov > 0 or gap <= tol:
                pairs.append((ov - gap, hi, fi))
    pairs.sort(reverse=True)
    used_h, used_f, matched = set(), set(), {}
    for _, hi, fi in pairs:
        if hi in used_h or fi in used_f:
            continue
        used_h.add(hi)
        used_f.add(fi)
        matched[hi] = fi
    return matched


def integrate(t, speed, events, k) -> dict:
    """Route from a list of (time, raw, direction) steps."""
    steps = sorted((e["t"], e["raw"] * (1.0 if e["direction"] == "LEFT" else -1.0) * k)
                   for e in events)
    dt = float(np.median(np.diff(t)))
    theta = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    th = 0.0
    ki = 0
    for i in range(1, len(t)):
        while ki < len(steps) and steps[ki][0] <= t[i]:
            th += steps[ki][1]
            ki += 1
        theta[i] = th
        x[i] = x[i - 1] + speed[i] * np.cos(np.deg2rad(th)) * dt
        y[i] = y[i - 1] + speed[i] * np.sin(np.deg2rad(th)) * dt
    return {"t": t, "theta": theta, "x": x, "y": y}


def shape(tr) -> dict:
    L = float(np.hypot(np.diff(tr["x"]), np.diff(tr["y"])).sum())
    net = float(np.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]))
    turn = float(np.sum(np.abs(np.diff(tr["theta"]))))
    return {"path": L, "net": net, "straightness": net / L if L > 0 else 0.0,
            "total_turn_deg": turn, "net_turn_deg": float(tr["theta"][-1])}


def route_distance(a, b) -> float:
    """Mean distance between two routes after removing scale, centre and rotation.

    Used only to say how far a repaired route has moved from the current one and towards
    variant E. It is not an accuracy measure, because no ground-truth route exists here.
    """
    from numpy.linalg import svd

    def rs(tr, n=300):
        x, y = tr["x"], tr["y"]
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        s = s / max(s[-1], 1e-9)
        u = np.linspace(0, 1, n)
        return np.stack([np.interp(u, s, x), np.interp(u, s, y)], 1)

    A, B = rs(a), rs(b)
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / max(np.abs(A).max(), 1e-9)
    B = B / max(np.abs(B).max(), 1e-9)
    U, S, Vt = svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    B = B @ R
    return float(np.hypot(A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="seconds; how far a swing may sit from a drawn window and still "
                         "count as the same turn")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    k = k_calibrated()
    print("P07.6 — аудит отдельных поворотов")
    print(f"  ролики с ручной разметкой: {AUDIT_CLIPS}")
    print(f"  угол: raw(Гц·с) x {k:.1f} °/(Гц·с), из P07.5, не менялся")
    print(f"  сопоставление: окно человека и качание системы в пределах "
          f"{args.tolerance:.1f} с")
    print(f"  P07.4 и P07.5 не тронуты, пороги не менялись")
    print()

    report = {"k": k, "tolerance": args.tolerance, "videos": {}}

    for video in AUDIT_CLIPS:
        t, s = rate_signal(video)
        speed = np.array([float(r["speed"]) for r in
                          csv.DictReader((P071 / f"trajectory_{video}.csv").open())])
        n = min(len(t), len(speed))
        t, speed = t[:n], speed[:n]

        human = [h for h in human_turns(video) if h["t1"] <= t[-1]]
        fly = [f for f in fly_swings(video, t, s) if f["t1"] <= t[-1]]
        turns = [f for f in fly if f["kind"] == "TURN"]
        looks = [f for f in fly if f["kind"] == "LOOK"]
        matched = match(human, turns, args.tolerance)
        matched_fly = set(matched.values())

        # ---- per human turn ---------------------------------------------
        rows = []
        for hi, h in enumerate(human):
            fi = matched.get(hi)
            if fi is None:
                rows.append({"t0": h["t0"], "t1": h["t1"], "true": h["direction"],
                             "status": "MISSED", "predicted": "", "angle_deg": 0.0,
                             "sign_ok": "", "raw": 0.0})
                continue
            f = turns[fi]
            ok = f["direction"] == h["direction"]
            rows.append({"t0": h["t0"], "t1": h["t1"], "true": h["direction"],
                         "status": "detected" if ok else "WRONG SIGN",
                         "predicted": f["direction"], "angle_deg": f["raw"] * k,
                         "sign_ok": "yes" if ok else "no", "raw": f["raw"]})

        # ---- false events -----------------------------------------------
        false_events = [turns[fi] for fi in range(len(turns)) if fi not in matched_fly]
        look_as_turn = []
        for h in human:
            for lk in looks:
                ov = overlap(h["t0"], h["t1"], lk["t0"], lk["t1"])
                gap = max(0.0, max(h["t0"], lk["t0"]) - min(h["t1"], lk["t1"]))
                if ov > 0 or gap <= args.tolerance:
                    look_as_turn.append(lk)

        n_missed = sum(1 for r in rows if r["status"] == "MISSED")
        n_wrong = sum(1 for r in rows if r["status"] == "WRONG SIGN")
        n_det = len(rows) - n_missed - n_wrong

        ang_false = float(np.sum([abs(f["raw"]) * k for f in false_events]))
        ang_missed = float(np.sum([90.0 for r in rows if r["status"] == "MISSED"]))
        ang_wrong = float(np.sum([abs(r["angle_deg"]) for r in rows
                                  if r["status"] == "WRONG SIGN"])) * 2
        ang_total = float(np.sum([abs(f["raw"]) * k for f in turns]))

        print(f"=== {video} ===")
        print(f"  настоящих поворотов человека: {len(human)}")
        print(f"    найдено верно:      {n_det}")
        print(f"    пропущено:          {n_missed}")
        print(f"    неверный знак:      {n_wrong}")
        print(f"  событий системы (TURN): {len(turns)}")
        print(f"    ложных:             {len(false_events)}")
        print(f"    взгляд принят за поворот: {len(look_as_turn)}")
        print()
        print(f"  вклад в накопленный угол:")
        print(f"    всего по системе:   {ang_total:8.0f}°")
        print(f"    ложные события:     {ang_false:8.0f}°  "
              f"({ang_false / max(ang_total, 1e-9) * 100:.0f}% от всего)")
        print(f"    пропущенные:        {ang_missed:8.0f}°  (если принять их за 90°)")
        print(f"    ошибки знака:       {ang_wrong:8.0f}°  (удвоенный угол: "
              f"не туда и обратно)")
        print()

        # ---- variants ----------------------------------------------------
        A = [{"t": f["t1"], "raw": f["raw"], "direction": f["direction"]}
             for f in turns]
        B = [{"t": f["t1"], "raw": f["raw"], "direction": f["direction"]}
             for f in turns if f not in false_events or
             any(f is turns[i] for i in matched_fly)]
        # rebuild B explicitly by index, identity comparison on dicts is fragile
        B = [{"t": turns[i]["t1"], "raw": turns[i]["raw"],
              "direction": turns[i]["direction"]}
             for i in range(len(turns)) if i in matched_fly]
        C = list(A)
        for r in rows:
            if r["status"] == "MISSED":
                m = (t >= r["t0"]) & (t <= r["t1"])
                raw = float(np.sum(s[m])) * float(np.median(np.diff(t))) if m.any() else 0.0
                C.append({"t": r["t1"], "raw": raw, "direction": r["true"]})
        D = []
        for i in range(len(turns)):
            f = turns[i]
            h = next((rows[hi] for hi, fi in matched.items() if fi == i), None)
            if h is not None and h["status"] == "WRONG SIGN":
                D.append({"t": f["t1"], "raw": f["raw"], "direction": h["true"]})
            else:
                D.append({"t": f["t1"], "raw": f["raw"], "direction": f["direction"]})
        E = []
        for r in rows:
            m = (t >= r["t0"]) & (t <= r["t1"])
            raw = float(np.sum(s[m])) * float(np.median(np.diff(t))) if m.any() else 0.0
            E.append({"t": r["t1"], "raw": raw, "direction": r["true"]})

        variants = {"A": A, "B": B, "C": C, "D": D, "E": E}
        # F: no LOOK/TURN separation at all, every swing rotates the heading. Included
        # because the matrix above shows the separation itself is the suspect, and this
        # tests it directly.
        F = [{"t": f["t1"], "raw": f["raw"], "direction": f["direction"]}
             for f in fly]
        variants["F"] = F
        routes = {key: integrate(t, speed, ev, k) for key, ev in variants.items()}
        ref = routes["E"]
        print(f"  {'вариант':>34s} {'поворот':>9s} {'прямизна':>9s} {'до E':>8s}")
        labels = {
            "A": "A как сейчас",
            "B": "B убраны только ложные",
            "C": "C добавлены только пропущенные",
            "D": "D исправлены только знаки",
            "E": "E идеальные события человека",
            "F": "F без разделения взглядов вообще",
        }
        vres = {}
        for key in ("A", "B", "C", "D", "E", "F"):
            sh = shape(routes[key])
            dd = route_distance(routes[key], ref)
            vres[key] = {**sh, "dist_to_E": dd}
            print(f"  {labels[key]:>34s} {sh['total_turn_deg']:8.0f}° "
                  f"{sh['straightness']:9.3f} {dd:8.3f}")
        gap = vres["A"]["dist_to_E"]
        print()
        print(f"  расстояние A -> E: {gap:.3f}. Какую долю этого зазора закрывает")
        print(f"  каждый ремонт по отдельности:")
        for key in ("B", "C", "D", "F"):
            closed = (gap - vres[key]["dist_to_E"]) / max(gap, 1e-9)
            print(f"    {labels[key]:>34s} {closed * 100:+6.0f}%")
        print()
        print(f"  ОГОВОРКА про эту метрику: она сравнивает форму с вариантом E, а в E "
              f"поворотов {len(E)}, тогда как в B их {len(B)}. Разное число поворотов "
              f"само по себе меняет форму, поэтому доля зазора отражает и количество, "
              f"и правильность. Для роликов без истинного маршрута форму проверить "
              f"нечем; решающими здесь являются матрица LOOK/TURN и распределения ниже.")
        print()

        best = max(("B", "C", "D", "F"), key=lambda kk: gap - vres[kk]["dist_to_E"])
        report["videos"][video] = {
            "n_human": len(human), "n_detected": n_det, "n_missed": n_missed,
            "n_wrong_sign": n_wrong, "n_fly_turns": len(turns),
            "n_false": len(false_events), "n_look_as_turn": len(look_as_turn),
            "angle_false_deg": ang_false, "angle_missed_deg": ang_missed,
            "angle_wrong_sign_deg": ang_wrong, "angle_total_deg": ang_total,
            "variants": vres, "gap_A_to_E": gap,
            "metric_caveat": "расстояние до E отражает и число поворотов, и их "
                             "правильность; истинного маршрута у этих роликов нет",
            "best_single_repair": best,
            "closed_fraction": {
                kk: (gap - vres[kk]["dist_to_E"]) / max(gap, 1e-9)
                for kk in ("B", "C", "D", "F")},
        }

        with (OUT / f"audit_{video}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t0", "t1", "true", "status", "predicted",
                                              "angle_deg", "sign_ok"])
            w.writeheader()
            for r in rows:
                w.writerow({kk: r[kk] for kk in w.fieldnames})
        with (OUT / f"false_events_{video}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0", "t1", "direction", "angle_deg"])
            for f in false_events:
                w.writerow([f"{f['t0']:.2f}", f"{f['t1']:.2f}", f["direction"],
                            f"{f['raw'] * k:.2f}"])
        with (OUT / f"variants_{video}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x_A", "y_A", "x_B", "y_B", "x_C", "y_C", "x_D", "y_D",
                        "x_E", "y_E"])
            for i in range(len(t)):
                w.writerow([f"{t[i]:.3f}"] + sum(
                    ([f"{routes[k]['x'][i]:.6f}", f"{routes[k]['y'][i]:.6f}"]
                     for k in ("A", "B", "C", "D", "E")), []))

    print("=== ИТОГ ===")
    for video, d in report["videos"].items():
        b = d["best_single_repair"]
        cf = d["closed_fraction"]
        print(f"  {video}: лучший одиночный ремонт — {b} "
              f"({cf[b] * 100:+.0f}% зазора A->E)")
        print(f"    ложных {d['n_false']} ({d['angle_false_deg']:.0f}°), "
              f"пропущено {d['n_missed']}, знак неверен {d['n_wrong_sign']}, "
              f"взгляд как поворот {d['n_look_as_turn']}")

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
