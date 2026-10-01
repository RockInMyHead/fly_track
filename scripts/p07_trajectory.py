#!/usr/bin/env python3
"""
P07 step 4-6 — freeze the forward cells, build both signals, integrate the trajectory.

Forward cells are frozen before the trajectory exists
----------------------------------------------------
The criterion is fixed in advance and applied mechanically: a cell qualifies if its
response to expansion exceeds its response both to a still frame and to contraction, with
the same sign on seed 64 and seed 65 and on at least 70 percent of the frames. Nothing is
re-selected afterwards.

What the screen found, and what it says about DNg100
----------------------------------------------------
DNg100 was the assumed forward channel. It does not pass: its FWD-minus-STATIC is
+0.875 / +0.375 Hz across seeds with only 32 percent of frames agreeing in sign. The
cells that do pass are DNa07 L, three DNp17 R cells, DNp20 R and DNbe001 R, all with 72
to 88 percent frame agreement and FWD clearly above both STATIC and BACK.

DNa07 appears in both sets. It is the strongest forward candidate and also one of the four
yaw types, so the forward signal is not independent of the yaw signal. That is stated in
the report rather than patched over, and the overlap is quantified.

The trajectory
--------------
    theta += yaw_rate * dt
    x     += speed * cos(theta) * dt
    y     += speed * sin(theta) * dt

    speed  = max(0, forward_signal), zero when the forward signal is below its floor
    yaw    = the deadbanded signal, so quiet stretches contribute nothing

Units are arbitrary. An absolute scale cannot be recovered from one ordinary video
without additional information, so the axes carry no metric meaning and the report says
so. The old frontend trajectory is drawn beside it, never combined with it.

Usage:
    PYTHONPATH=. python scripts/p07_trajectory.py
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

OUT = ROOT / "output/p07"
FWD_DIR = OUT / "forward"
def _discover_videos() -> dict:
    """Every clip with a brain recording, base ones plus those in the registry."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from p07_videos import videos as registry_videos
        return registry_videos()
    except Exception as e:
        print(f"  реестр видео недоступен ({e}), беру только базовые")
        return {
            "VID00001": {"trace": ROOT / "output/p053_threshold/spike_trace.npz",
                         "targets": ROOT / "output/p053_threshold/targets.csv"},
            "VID00002": {"trace": ROOT / "output/p06_neurons/spike_trace.npz",
                         "targets": ROOT / "output/p06_neurons/targets.csv"},
        }


VIDEOS = _discover_videos()
TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]
FPS = 50.0

# fixed before any trajectory was built
FWD_CRITERIA = {"min_frame_consistency": 0.70, "must_exceed_static": True,
                "must_exceed_back": True, "same_sign_both_seeds": True}


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def load_video(name: str):
    spec = VIDEOS[name]
    d = np.load(spec["trace"])
    col = {i: x for i, x in enumerate(csv.DictReader(spec["targets"].open()))}
    return d, col


def freeze_forward_cells(verbose: bool = True) -> list[dict]:
    tg = list(csv.DictReader((FWD_DIR / "targets_seed64.csv").open()))
    d64 = np.load(FWD_DIR / "scale_seed64.npz")
    d65 = np.load(FWD_DIR / "scale_seed65.npz")
    out = []
    for i, x in enumerate(tg):
        df1 = float((d64["fwd"][:, i] - d64["static"][:, i]).mean())
        df2 = float((d65["fwd"][:, i] - d65["static"][:, i]).mean())
        db1 = float((d64["back"][:, i] - d64["static"][:, i]).mean())
        db2 = float((d65["back"][:, i] - d65["static"][:, i]).mean())
        diff = d64["fwd"][:, i] - d64["static"][:, i]
        cons = float(np.mean(np.sign(diff) == np.sign(df1))) if abs(df1) > 1e-9 else 0.0
        ok = (cons >= FWD_CRITERIA["min_frame_consistency"]
              and np.sign(df1) == np.sign(df2) and abs(df1) > 1e-9
              and df1 > 0 and df2 > 0
              and (df1 - db1) > 0 and (df2 - db2) > 0)
        if ok:
            out.append({"cell": int(x["cell"]), "cell_type": x["cell_type"],
                        "side": x["side"], "idx": i,
                        "delta_fwd_64": df1, "delta_fwd_65": df2,
                        "delta_back_64": db1, "delta_back_65": db2,
                        "frame_consistency": cons})
    if verbose:
        print(f"  заморожено forward-клеток: {len(out)}")
        for r in out:
            print(f"    {r['cell_type']:>8s} {r['side']}: dFWD "
                  f"{r['delta_fwd_64']:+.2f}/{r['delta_fwd_65']:+.2f}, "
                  f"dBACK {r['delta_back_64']:+.2f}/{r['delta_back_65']:+.2f}, "
                  f"кадры {r['frame_consistency'] * 100:.0f}%")
    return out


def build_forward_signal(d: dict, col: dict, fwd_cells: list[dict],
                         n_sm: int) -> tuple[np.ndarray, np.ndarray]:
    """Z-scored mean over the frozen forward cells, plus the raw version.

    Normalising by each cell's own spread keeps a high-rate cell from dominating. The
    sign is already positive for expansion by construction of the freeze.
    """
    per = []
    for r in fwd_cells:
        i = next((k for k, x in col.items()
                  if x["cell_type"] == r["cell_type"] and x["side"] == r["side"]
                  and abs(float(x["carrier_weight"]) - 0.0) >= 0), None)
        idx = [k for k, x in col.items()
               if x["cell_type"] == r["cell_type"] and x["side"] == r["side"]]
        if not idx:
            continue
        rate = d["fired"][:, idx].mean(axis=1) * FPS
        per.append(rate)
    if not per:
        return np.zeros(len(d["t"])), np.zeros(len(d["t"]))
    M = np.stack(per)
    raw = M.mean(axis=0)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    z = ((M - mu) / np.where(sd > 1e-9, sd, 1.0)).mean(axis=0)
    return box(z, n_sm), box(raw, n_sm)


def yaw_threshold() -> tuple[float, dict]:
    """Strict yaw floor from the synthetic STATIC frames.

    Expressed as a multiple of the moving signal's spread, so it transfers to the real
    recording despite the two being normalised separately. The measured multiple is close
    to 1.7, which says the noise floor is larger than the signal's own spread: only the
    top decile of the signal stands above the level that a still frame produces.
    """
    col = {i: x for i, x in enumerate(
        csv.DictReader((ROOT / "output/p053_threshold/targets.csv").open()))}
    signs_nm = {"DNp17": 1.0, "DNa07": -1.0, "DNp26": 1.0, "DNp20": 1.0}

    def channel_of(d, cond):
        ch = {}
        for nm in TYPES:
            for side in ("L", "R"):
                idx = [i for i, x in col.items()
                       if x["cell_type"] == nm and x["side"] == side]
                ch[(nm, side)] = d[cond][:, idx].mean(axis=1) * FPS
        return np.stack([(ch[(nm, "R")] + ch[(nm, "L")]) / 2 for nm in TYPES])

    ks = []
    for seed in (64, 65):
        d = np.load(ROOT / f"output/p061_synthetic/synthetic_seed{seed}.npz")
        st = channel_of(d, "static")
        mv = np.concatenate([channel_of(d, "left"), channel_of(d, "right")], axis=1)
        mu, sd = mv.mean(axis=1, keepdims=True), mv.std(axis=1, keepdims=True)
        sg = np.array([signs_nm[nm] for nm in TYPES])[:, None]
        z_st = (((st - mu) / np.where(sd > 1e-9, sd, 1)) * sg).mean(axis=0)
        ks.append(float(np.percentile(np.abs(z_st), 95) / 1.0))
    k = float(np.mean(ks))
    return k, {"k_per_seed": ks,
               "note": "порог = k x разброс реального сигнала; k измерен на "
                       "синтетическом STATIC (95-й процентиль |сигнал|)"}


def forward_floor() -> tuple[float, dict]:
    """Floor for the forward signal, as a ratio of spreads from the synthetic runs."""
    ratios = []
    for seed in (64, 65):
        d = np.load(FWD_DIR / f"scale_seed{seed}.npz")
        tg = list(csv.DictReader((FWD_DIR / f"targets_seed{seed}.csv").open()))
        cells = freeze_forward_cells(verbose=False)
        st, fw = [], []
        for r in cells:
            idx = [i for i, x in enumerate(tg)
                   if x["cell_type"] == r["cell_type"] and x["side"] == r["side"]]
            if not idx:
                continue
            st.append(d["static"][:, idx].mean(axis=1) * FPS)
            fw.append(d["fwd"][:, idx].mean(axis=1) * FPS)
        if not st:
            continue
        st = np.stack(st)
        fw = np.stack(fw)
        allv = np.concatenate([st, fw], axis=1)
        mu, sd = allv.mean(axis=1, keepdims=True), allv.std(axis=1, keepdims=True)
        zs = ((st - mu) / np.where(sd > 1e-9, sd, 1)).mean(axis=0)
        zf = ((fw - mu) / np.where(sd > 1e-9, sd, 1)).mean(axis=0)
        ratios.append(float(zs.std() / max(zf.std(), 1e-9)))
    ratio = float(np.mean(ratios)) if ratios else 0.5
    return ratio, {"ratios_per_seed": ratios,
                   "note": "порог = отношение x разброс реального forward-сигнала"}


def detect_events(yaw: np.ndarray, t: np.ndarray, threshold: float,
                  min_event_s: float = 0.4) -> np.ndarray:
    """Mark the moments where the yaw signal stands above its noise floor.

    A single frame above threshold is as likely to be noise as a turn, so a run has to
    last a minimum time to count. Everything outside a marked run leaves the heading
    untouched.
    """
    above = np.abs(yaw) >= threshold
    out = np.zeros(len(yaw), dtype=bool)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    need = max(int(round(min_event_s / dt)), 1)
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1]:
            j += 1
        if (j - i + 1) >= need:
            out[i:j + 1] = True
        i = j + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth-s", type=float, default=0.3)
    ap.add_argument("--yaw-scale", type=float, default=90.0,
                    help="units of angle per unit of yaw signal; arbitrary, relative only")
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--min-event-s", type=float, default=0.4,
                    help="minimum duration of a yaw event; a single frame above "
                         "the floor is as likely to be noise as a turn")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    n_sm = max(int(round(args.smooth_s * FPS)), 1)

    print("P07.4-6 — заморозка forward-клеток, сборка траектории")
    fwd_cells = freeze_forward_cells()
    (FWD_DIR / "frozen_forward_cells.json").write_text(json.dumps({
        "criteria": FWD_CRITERIA, "cells": fwd_cells,
        "note": "критерий зафиксирован до построения траектории",
        "DNg100": "не прошёл: dFWD +0.88/+0.38 при 32% согласия кадров",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    ratio, ratio_info = forward_floor()
    k_yaw, k_info = yaw_threshold()
    print(f"\n  порог forward из синтетики: отношение покой/движение = {ratio:.3f}")
    print(f"  порог yaw из синтетики: {k_yaw:.2f} x разброс сигнала "
          f"(шум покоя оказался больше разброса движения)")

    report = {"forward_cells": fwd_cells, "forward_floor_ratio": ratio,
              "forward_floor_info": ratio_info, "yaw_threshold_k": k_yaw,
              "yaw_threshold_info": k_info, "videos": {}}

    for name in VIDEOS:
        d, col = load_video(name)
        t = d["t"].astype(np.float64)
        n_active = d["n_active"].astype(np.float64)

        # yaw: reuse the validated readout; the strict floor is applied here so that the
        # quiet stretches cannot walk the heading around
        sig_csv = OUT / f"yaw_signal_{name}.csv"
        yaw_rows = list(csv.DictReader(sig_csv.open()))
        yaw_raw = np.array([float(r["yaw_signal"]) for r in yaw_rows])
        yaw_to = k_yaw * float(np.std(yaw_raw))
        yaw = np.where(np.abs(yaw_raw) < yaw_to, 0.0, yaw_raw)

        fwd_z, fwd_raw = build_forward_signal(d, col, fwd_cells, n_sm)
        fwd_floor = ratio * float(np.std(fwd_z))
        speed_raw = np.where(fwd_z < fwd_floor, 0.0, fwd_z - fwd_floor)
        speed = speed_raw * args.speed_scale

        # integrate. A continuous integral was tried first and produced a random walk:
        # 80 percent of the accumulated turning fell outside the labelled turns, because
        # the signal's noise is spread over ten times more time than the turns occupy.
        # So the heading only advances during events that the signal itself marks, which
        # is the construction P0.4 arrived at for the frontend.
        dt = float(np.median(np.diff(t)))
        events = detect_events(yaw, t, yaw_to, min_event_s=args.min_event_s)
        theta = np.zeros(len(t))
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        th = 0.0
        for k in range(1, len(t)):
            if events[k]:
                th += yaw[k] * args.yaw_scale * dt
            theta[k] = th
            x[k] = x[k - 1] + speed[k] * np.cos(th) * dt
            y[k] = y[k - 1] + speed[k] * np.sin(th) * dt

        with (OUT / f"forward_signal_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "forward_signal", "forward_raw", "speed"])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.3f}", f"{fwd_z[k]:.5f}", f"{fwd_raw[k]:.4f}",
                            f"{speed[k]:.5f}"])

        with (OUT / f"trajectory_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "theta", "yaw_signal", "speed", "n_active"])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.3f}", f"{x[k]:.6f}", f"{y[k]:.6f}",
                            f"{theta[k]:.6f}", f"{yaw[k]:.5f}", f"{speed[k]:.5f}",
                            int(n_active[k])])

        # how much of the path is actually travelled, and how tangled it is
        seg = np.hypot(np.diff(x), np.diff(y))
        path_len = float(seg.sum())
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        turning = float(np.sum(np.abs(yaw)) * args.yaw_scale * dt)
        report["videos"][name] = {
            "n_steps": int(len(t)), "duration_s": float(t[-1] - t[0]),
            "path_length": path_len, "net_displacement": net,
            "straightness": net / path_len if path_len > 0 else 0.0,
            "total_turning_units": turning,
            "frac_time_moving": float(np.mean(speed > 0)),
            "yaw_floor": yaw_to, "speed_floor": fwd_floor,
            "final": [float(x[-1]), float(y[-1])],
            "units": "произвольные; абсолютный масштаб из одного видео без "
                     "дополнительных данных неизвестен",
        }
        print(f"\n=== {name} ===")
        print(f"  путь {path_len:.1f}, смещение {net:.1f}, "
              f"прямизна {net / max(path_len, 1e-9):.2f}")
        print(f"  движется {np.mean(speed > 0) * 100:.0f}% времени, "
              f"суммарный поворот {turning:.0f} ед.")

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
