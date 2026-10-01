#!/usr/bin/env python3
"""
P07.4 — tell a glance apart from a change of direction.

The problem, and why the first attempt at it failed
---------------------------------------------------
Every event rotates the heading permanently. When the walker merely looks aside, the
camera swings out and comes back and the route should be untouched, but the swing is
recorded as a permanent rotation.

The first attempt looked for pairs among the events `p072` produces. It found almost
none: 0 of 103 on VID00001, 2 of 26 on VID00006. The reason is structural. `p072` merges
firings closer than 0.8 s apart, so a look-out and a look-back, which are separated by
little more than the time it takes to turn the head, arrive as one event. The pairing was
looking for two things that the earlier stage had already glued into one.

Excursions of the accumulated yaw, instead
------------------------------------------
So the search moves to the signal itself. Let

    camera_yaw(t) = cumulative integral of the yaw signal

Zero crossings of the signal are the turning points of camera_yaw: between two of them
the camera swings one way without interruption. Consecutive swings are compared:

    out = |camera_yaw at the first turning point - at the second|
    back = |camera_yaw at the second - at the third|
    net = |camera_yaw at the third - at the first|

    if net is small against out + back   the camera went out and came back: a LOOK
    if net is most of out + back         it went and stayed: a TURN
    between the two                      inconclusive, and reported as such

This is the same test the task describes, no longer defeated by the merge, because it
never depends on the earlier stage having kept the two halves apart.

Angles are integrals, not fixed steps
-------------------------------------
No event is given 90 degrees. Its angle is the accumulated yaw across it times a single
coefficient, fitted so that the median turn comes out at 90 degrees. That coefficient is
the only free number, it is fitted on VID00001 and reused everywhere, and it is stated as
an assumption: turn size is not measured anywhere in this project, and a route is judged
by its shape, which the coefficient does not affect.

Usage:
    PYTHONPATH=. python scripts/p074_looks_vs_turns.py
    PYTHONPATH=. python scripts/p074_looks_vs_turns.py --look-max 0.45 --turn-min 0.70
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
P072 = ROOT / "output/p072"
OUT = ROOT / "output/p074"

SMOOTH_S = 0.3
MIN_SWING = 0.15        # ignore swings smaller than this, they are noise
LOOK_MAX = 0.45         # net/(out+back) below this: the camera came back
TURN_MIN = 0.70         # above this: it went and stayed
MEDIAN_TURN_DEG = 90.0  # the one calibration assumption


def videos() -> list[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from p07_videos import videos as reg
        out = [v for v in ("VID00001", "VID00002") if (P071 / f"trajectory_{v}.csv").exists()]
        for v in reg():
            if v not in out and (P071 / f"trajectory_{v}.csv").exists():
                out.append(v)
        return out
    except Exception:
        return ["VID00001", "VID00002"]


def load(video: str):
    rows = list(csv.DictReader((P071 / f"trajectory_{video}.csv").open()))
    t = np.array([float(r["t"]) for r in rows])
    yaw = np.array([float(r["yaw_gated"]) for r in rows])
    speed = np.array([float(r["speed"]) for r in rows])
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(SMOOTH_S / dt)), 1)
    if n_sm > 1:
        yaw = np.convolve(yaw, np.ones(n_sm) / n_sm, mode="same")
    cam = np.cumsum(yaw) * dt
    return t, yaw, speed, cam


def swings(t: np.ndarray, yaw: np.ndarray, cam: np.ndarray,
           min_swing: float, merge_gap_s: float = 0.20) -> list[dict]:
    """Segments between zero crossings of the signal, i.e. monotone runs of camera_yaw.

    A run of zeros ends a segment rather than extending it. That matters more than it
    sounds: the deadband leaves 92 to 94 percent of frames at exactly zero, so absorbing
    zeros into the current run stretched a segment across every pause and merged several
    separate turns into one. A six second run containing a left turn and a right turn sums
    to nearly zero, is then called a look, and discards both real turns. Measured on the
    labelled clips, that one line accounted for 66 percent of confirmed turns being
    dropped, while the signal itself was correct on 91 of 91 of them.

    A very short zero gap is still bridged, because a turn that clips the deadband for a
    frame or two is one turn and not two. The gap is a stated parameter rather than a
    hidden constant, and it is small: the deadband already removes the noise it guards
    against.
    """
    out = []
    n = len(yaw)
    gap = max(int(round(merge_gap_s / max(float(np.median(np.diff(t))), 1e-9))), 0)
    i = 0
    while i < n:
        if yaw[i] == 0:
            i += 1
            continue
        j = i
        s = np.sign(yaw[i])
        while j + 1 < n:
            if yaw[j + 1] != 0 and np.sign(yaw[j + 1]) == s:
                j += 1
                continue
            # a short run of zeros followed by the same sign is one turn
            k = j + 1
            while k < n and yaw[k] == 0:
                k += 1
            if k < n and (k - (j + 1)) <= gap and np.sign(yaw[k]) == s:
                j = k
                continue
            break
        c0, c1 = cam[i], cam[j]
        size = abs(c1 - c0)
        if size >= min_swing:
            out.append({"t0": float(t[i]), "t1": float(t[j]),
                        "i0": i, "i1": j,
                        "from": float(c0), "to": float(c1),
                        "delta": float(c1 - c0), "size": float(size),
                        "direction": "LEFT" if c1 > c0 else "RIGHT"})
        i = j + 1
    return out


def classify(sw: list[dict], look_max: float, turn_min: float) -> list[dict]:
    """Walk adjacent swings and call each one a look or a turn."""
    for s in sw:
        s["kind"] = "INCONCLUSIVE"
        s["pair"] = None
    k = 0
    while k < len(sw) - 1:
        a, b = sw[k], sw[k + 1]
        # adjacent swings in the same direction are one continuous move, not a pair
        if np.sign(a["delta"]) == np.sign(b["delta"]):
            k += 1
            continue
        net = abs(a["delta"] + b["delta"])
        total = a["size"] + b["size"]
        ratio = net / max(total, 1e-9)
        if ratio <= look_max:
            a["kind"] = b["kind"] = "LOOK"
            a["pair"] = b["t0"]
            b["pair"] = a["t0"]
            k += 2
        elif ratio >= turn_min:
            a["kind"] = b["kind"] = "TURN"
            a["pair"] = b["t0"]
            b["pair"] = a["t0"]
            k += 2
        else:
            # the pair is ambiguous; each half is judged on its own by size alone, and
            # left INCONCLUSIVE so it cannot quietly influence the route
            k += 1
    for s in sw:
        if s["kind"] == "INCONCLUSIVE":
            # a swing with no opposite neighbour is a turn if it is large
            s["kind"] = "TURN" if s["size"] >= 3 * MIN_SWING else "INCONCLUSIVE"
        s["label"] = f"{s['kind']}_{s['direction']}"
    return sw


def integrate(t, speed, sw, scale, use) -> dict:
    steps = sorted([(s["t1"], s["delta"] * scale) for s in sw if s["kind"] in use])
    dt = float(np.median(np.diff(t)))
    theta = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    th = 0.0
    ki = 0
    for k in range(1, len(t)):
        while ki < len(steps) and steps[ki][0] <= t[k]:
            th += steps[ki][1]
            ki += 1
        theta[k] = th
        x[k] = x[k - 1] + speed[k] * np.cos(np.deg2rad(th)) * dt
        y[k] = y[k - 1] + speed[k] * np.sin(np.deg2rad(th)) * dt
    return {"t": t, "theta": theta, "x": x, "y": y}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look-max", type=float, default=LOOK_MAX)
    ap.add_argument("--turn-min", type=float, default=TURN_MIN)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = videos()
    data = {}
    for v in names:
        data[v] = load(v)

    # one coefficient, fitted on the tuning clip
    t0, yaw0, sp0, cam0 = data["VID00001"]
    sw0 = classify(swings(t0, yaw0, cam0, MIN_SWING), args.look_max, args.turn_min)
    turns = [s["size"] for s in sw0 if s["kind"] == "TURN"]
    scale = MEDIAN_TURN_DEG / float(np.median(turns)) if turns else 1.0

    print("P07.4 — взгляд против поворота маршрута")
    print(f"  качание = отрезок между сменами знака сигнала;")
    print(f"  пара называется взглядом, если net/(out+back) <= {args.look_max:.2f},")
    print(f"  поворотом, если >= {args.turn_min:.2f}, иначе неясно")
    print(f"  масштаб угла {scale:.3f} °/ед, подобран на VID00001 "
          f"(медианный поворот = {MEDIAN_TURN_DEG:.0f}°)")
    print()

    report = {"scale": scale, "look_max": args.look_max, "turn_min": args.turn_min,
              "videos": {}}
    for v in names:
        t, yaw, speed, cam = data[v]
        sw = classify(swings(t, yaw, cam, MIN_SWING), args.look_max, args.turn_min)
        n_look = sum(1 for s in sw if s["kind"] == "LOOK")
        n_turn = sum(1 for s in sw if s["kind"] == "TURN")
        n_inc = sum(1 for s in sw if s["kind"] == "INCONCLUSIVE")

        old = integrate(t, speed, sw, scale, {"TURN", "LOOK", "INCONCLUSIVE"})
        new = integrate(t, speed, sw, scale, {"TURN"})

        def st(tr):
            L = float(np.hypot(np.diff(tr["x"]), np.diff(tr["y"])).sum())
            net = float(np.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]))
            return (float(np.sum(np.abs(np.diff(tr["theta"])))),
                    net / L if L > 0 else 0.0, L, net)

        to, so, Lo, no = st(old)
        tn, sn, Ln, nn = st(new)

        print(f"=== {v} ({t[-1]:.0f} с) ===")
        print(f"  качаний {len(sw)}: взглядов {n_look}, поворотов {n_turn}, "
              f"неясных {n_inc}")
        print(f"  суммарный поворот: было {to:7.0f}° -> стало {tn:7.0f}°  "
              f"({to / 360:.1f} -> {tn / 360:.1f} оборота)")
        print(f"  прямизна:          было {so:.3f} -> стало {sn:.3f}")
        report["videos"][v] = {
            "n_swings": len(sw), "n_look": n_look, "n_turn": n_turn,
            "n_inconclusive": n_inc,
            "turn_deg_before": to, "turn_deg_after": tn,
            "straightness_before": so, "straightness_after": sn,
        }

        with (OUT / f"swings_{v}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0", "t1", "kind", "direction", "label", "size",
                        "delta", "angle_deg", "paired_with"])
            for s in sw:
                w.writerow([f"{s['t0']:.3f}", f"{s['t1']:.3f}", s["kind"],
                            s["direction"], s["label"], f"{s['size']:.4f}",
                            f"{s['delta']:.4f}", f"{s['delta'] * scale:.2f}",
                            f"{s['pair']:.3f}" if s["pair"] else ""])
        with (OUT / f"trajectory_{v}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x_before", "y_before", "x_after", "y_after",
                        "theta_before", "theta_after", "camera_yaw"])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.3f}", f"{old['x'][k]:.6f}", f"{old['y'][k]:.6f}",
                            f"{new['x'][k]:.6f}", f"{new['y'][k]:.6f}",
                            f"{old['theta'][k]:.4f}", f"{new['theta'][k]:.4f}",
                            f"{cam[k]:.5f}"])
        print()

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"Wrote {OUT}/")


if __name__ == "__main__":
    main()
