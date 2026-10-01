#!/usr/bin/env python3
"""
P0.2 diagnostic — why did the detector fire on windows that are not turns?

The review returned 35 wrong out of 37, and the user reports most of those windows are
either forward motion or standing still. That combination is impossible if the
measurement is honest: a forward-moving or stationary scene cannot produce a coherent
sideways drift. So either the measurement is reading the wrong thing, or the detector
is firing on something that is large and coherent but is not a yaw.

The measurement so far has used only `dx`. This script adds what was never checked:

    dy                  vertical displacement per cell. A camera pitching down, or
                        walking bob, shows up here and not in dx.
    a = uniform dx      the part a yaw produces
    b = dx slope along azimuth   the part forward motion produces (radial flow)
    c = dy slope along elevation the part forward motion produces vertically
    divergence          b + c, the proper signature of moving forward
    residual            what neither a nor b explains: independent object motion

The classifier is then the physically correct one:

    STATIC   almost nothing moves anywhere
    FWD      the field spreads outward: divergence large, uniform part small
    LEFT/RIGHT  the uniform part dominates the divergence
    MIXED    neither dominates, or the residual is large (people, parallax)

Usage:
    PYTHONPATH=. python scripts/p02_diagnose_candidates.py
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

from fly_vo.content_motion import CROP, DEFAULT_FPS, ENCODE, GRID_H, GRID_W, read_window
from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
CANDIDATES = ROOT / "data/p01r/candidate_turns.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"

# Design matrices for the per-frame least-squares fit.
XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)          # azimuth per cell
YN = np.repeat(np.linspace(-1.0, 1.0, GRID_H), GRID_W)        # elevation per cell
PINV_UNIFORM_SLOPE = np.linalg.pinv(np.stack([np.ones_like(XN), XN], axis=1))
PINV_VERTICAL = np.linalg.pinv(np.stack([np.ones_like(YN), YN], axis=1))

STATIC_ACTIVITY = 0.30     # px/frame: below this the scene is not moving
FWD_DIVERGENCE = 0.60      # px/frame of radial flow that counts as forward motion
DOMINANCE = 1.5            # how much bigger one term must be to be called the cause


def analyse(t0: float, t1: float) -> dict:
    grays = read_window(str(VIDEO), t0, t1, DEFAULT_FPS)
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    A, B, C, AY = [], [], [], []
    adx, ady, adiv, ares = [], [], [], []
    for g in grays:
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, dyf, _d = lk.process(band)
        dx = dxf.reshape(-1)
        dy = dyf.reshape(-1)
        # uniform + slope along azimuth for the horizontal field
        ab = dx @ PINV_UNIFORM_SLOPE.T
        a_val, b_val = float(ab[0]), float(ab[1])
        # uniform + slope along elevation for the vertical field
        cd = dy @ PINV_VERTICAL.T
        c_val = float(cd[1])
        pred = a_val + b_val * XN
        resid = float(np.abs(dx - pred).mean())
        A.append(a_val); B.append(b_val); C.append(c_val)
        AY.append(float(dy.mean()))
        adx.append(float(dx.mean())); ady.append(float(dy.mean()))
        adiv.append(b_val + c_val)
        ares.append(resid)

    med = lambda v: float(np.median(v)) if len(v) else 0.0
    a_m, b_m, c_m, ay_m = med(A), med(B), med(C), med(AY)
    act_x = med([abs(x) for x in adx])          # not used for classification, reported
    local_x = med([float(np.abs(g).mean()) for g in [np.array([0.0])]])  # placeholder
    divergence = med(adiv)
    residual = med(ares)

    # local activity: mean |dx| over cells, which is what the earlier scan reported
    act = []
    for g in grays:
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, dyf, _d = lk.process(band)
        act.append(float(np.abs(dxf).mean()))
    activity = med(act)

    rot_strength = abs(a_m)
    fwd_strength = abs(divergence)

    if activity < STATIC_ACTIVITY and abs(ay_m) < STATIC_ACTIVITY:
        kind = "STATIC"
    elif fwd_strength > FWD_DIVERGENCE and rot_strength * DOMINANCE < fwd_strength:
        kind = "FWD"
    elif rot_strength > FWD_DIVERGENCE and fwd_strength * DOMINANCE < rot_strength:
        kind = "LEFT" if a_m > 0 else "RIGHT"      # a>0 is content right = camera left
    else:
        kind = "MIXED"

    return {
        "t0": t0, "t1": t1, "n_frames": len(grays),
        "a_rotation": a_m, "b_expansion": b_m, "c_vertical_exp": c_m,
        "divergence": divergence, "residual": residual,
        "mean_dy": ay_m, "local_activity_dx": activity,
        "camera_from_a": ("LEFT" if a_m > 0 else "RIGHT") if abs(a_m) > 1e-6 else "NONE",
        "kind": kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p02_diag")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    cands = doc["candidates"]
    verd = {}
    if VERDICTS.exists():
        verd = json.loads(VERDICTS.read_text(encoding="utf-8")).get("verdicts", {})

    print("P0.2 diagnostic — what the field actually contains")
    print(f"  {len(cands)} candidates, {len(verd)} reviewed\n")
    print(f"  {'#':>4s} {'окно':>14s} {'система':>7s} {'человек':>9s} | "
          f"{'a':>7s} {'div':>7s} {'b':>7s} {'c':>7s} {'mean_dy':>8s} {'остаток':>8s} {'движ':>5s} | "
          f"{'вывод':>8s}")

    rows = []
    for i, c in enumerate(cands, 1):
        cid = c.get("id") or f"c{i:03d}"
        v = verd.get(cid, {})
        r = analyse(float(c["t0"]), float(c["t1"]))
        r["id"] = cid
        r["system"] = c["camera_direction"]
        r["human"] = v.get("actual_direction") or ("верно" if v.get("verdict") == "correct" else "?")
        rows.append(r)
        if i <= 40:
            print(f"  {cid:>4s} {c['t0']:6.1f}-{c['t1']:<6.1f} {c['camera_direction']:>7s} "
                  f"{str(r['human']):>9s} | {r['a_rotation']:+7.2f} {r['divergence']:+7.2f} "
                  f"{r['b_expansion']:+7.2f} {r['c_vertical_exp']:+7.2f} "
                  f"{r['mean_dy']:+8.2f} {r['residual']:8.2f} {r['local_activity_dx']:5.2f} | "
                  f"{r['kind']:>8s}")

    # how do the verdicts line up with the physical classification?
    print("\n=== сверка: что видит физика против того, что увидел ты ===")
    pairs = [(r["human"], r["kind"]) for r in rows if r["human"] not in ("?", "None", None)]
    from collections import Counter
    tab = Counter(pairs)
    labels = sorted({h for h, _ in pairs})
    kinds = ["LEFT", "RIGHT", "FWD", "STATIC", "MIXED"]
    print(f"  {'человек':>10s} | " + " ".join(f"{k:>7s}" for k in kinds))
    for h in labels:
        cells = [tab.get((h, k), 0) for k in kinds]
        print(f"  {h:>10s} | " + " ".join(f"{v:7d}" for v in cells))

    agree = sum(1 for h, k in pairs if h == k)
    print(f"\n  совпало: {agree}/{len(pairs)}")

    fwd_or_static = [r for r in rows if r["kind"] in ("FWD", "STATIC")]
    print(f"\n  окон, которые физика считает вперёд или на месте: {len(fwd_or_static)} из {len(rows)}")
    turning = [r for r in rows if r["kind"] in ("LEFT", "RIGHT")]
    print(f"  окон, которые физика считает поворотом: {len(turning)} из {len(rows)}")

    with (out / "diagnostic.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "report.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
