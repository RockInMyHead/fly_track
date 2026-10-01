#!/usr/bin/env python3
"""
P07 step 1 — yaw signal out of MaleCNS, and its validation.

Two ways of combining the eight frozen cells, and why the principal one is not the
obvious one
------------------------------------------------------------------------------
The natural reading of "take the difference between the left and right cells of each
pair" assumes the cells are lateralised: left cell for one direction, right cell for the
other. P06.1 already showed they are not. For DNp17, DNp26 and DNp20 the left and right
cells of a type respond the same way to image motion, and only DNa07 is inverted. The
measured consequence on the labelled turns:

    type      difference R-L      common mode (R+L)/2
    DNp17          -0.588                -3.954
    DNa07          +0.856                +6.162
    DNp26          +0.046                -5.245     <- the difference carries nothing
    DNp20          +0.577                -3.261

The common mode is six to ten times larger for every type and does not vanish for DNp26.
So the yaw signal is built from each cell's own deviation from its baseline, weighted by
the polarity P06.1 measured for that cell. The R-L version is still computed and stored,
as a diagnostic that shows why it was not used.

Polarity comes from P06.1 and is not re-chosen
---------------------------------------------
P06.1 measured each cell's response to rightward minus leftward image motion on clean
frames. A camera turning left moves the image right, so the sign of that measurement is
the sign relating the cell to camera direction. DNp17, DNp26 and DNp20 respond to
rightward image motion and therefore mark LEFT turns; DNa07 responds to leftward motion
and marks RIGHT.

Sign convention
---------------
    yaw > 0   the camera turned LEFT
    yaw < 0   the camera turned RIGHT

Thresholds come from the synthetic STATIC condition
---------------------------------------------------
Not from the labelled turns. P06.1 recorded the same cells while a still frame was shown,
so the spread of the yaw signal with nothing moving is measurable, and the deadband is
set from its percentiles. Using the labelled turns to pick a threshold would make the
validation circular.

Usage:
    PYTHONPATH=. python scripts/p07_fly_readout.py
    PYTHONPATH=. python scripts/p07_fly_readout.py --smooth-s 0.2 --deadband-pct 90
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

def _discover_videos() -> dict:
    """All clips that have a brain recording, including any added through the registry.

    The two base clips are listed in `p07_videos.py`; anything registered there joins
    automatically. Keeping the list in one place means the frozen chain picks up a new
    clip without editing the scripts that make up the chain.
    """
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
SYNTH = ROOT / "output/p061_synthetic"
OUT = ROOT / "output/p07"
TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]
FPS = 50.0


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def load_video(name: str):
    spec = VIDEOS[name]
    d = np.load(spec["trace"])
    t = d["t"].astype(np.float64)
    fired = d["fired"].astype(np.float64)
    col = {i: x for i, x in enumerate(csv.DictReader(spec["targets"].open()))}
    return t, fired, col


def channel_rates(fired: np.ndarray, col: dict) -> dict:
    """Mean firing rate per (type, side), in Hz."""
    out = {}
    for nm in TYPES:
        for side in ("L", "R"):
            idx = [i for i, x in col.items()
                   if x["cell_type"] == nm and x["side"] == side]
            if idx:
                out[(nm, side)] = fired[:, idx].mean(axis=1) * FPS
    return out


def polarity_from_p06() -> dict:
    """Sign relating each cell to camera direction, from P06.1's clean stimulus.

    P06.1 measured response to rightward minus leftward image motion. A camera turning
    left shifts the image right, so a cell that prefers rightward image motion marks
    LEFT turns. Positive here therefore means the cell's activity rises on LEFT turns.
    """
    d64 = np.load(SYNTH / "synthetic_seed64.npz")
    d65 = np.load(SYNTH / "synthetic_seed65.npz")
    col = {i: x for i, x in enumerate(
        csv.DictReader((ROOT / "output/p053_threshold/targets.csv").open()))}
    out = {}
    for nm in TYPES:
        vals = []
        for side in ("L", "R"):
            for i, x in col.items():
                if x["cell_type"] != nm or x["side"] != side:
                    continue
                a = (d64["right"][:, i] - d64["left"][:, i]).mean()
                b = (d65["right"][:, i] - d65["left"][:, i]).mean()
                vals.append(a)
                vals.append(b)
        m = float(np.mean(vals)) if vals else 0.0
        # rightward image motion means the camera went left
        out[nm] = 1.0 if m > 0 else -1.0
        out[(nm, "delta")] = m
    return out


def build_signals(rates: dict, pol: dict, n_smooth: int) -> dict:
    """Both combinations, smoothed."""
    pair, common, polarity = [], [], []
    for nm in TYPES:
        r, l = rates[(nm, "R")], rates[(nm, "L")]
        pair.append(r - l)
        common.append((r + l) / 2)
    pair = np.stack(pair)
    common = np.stack(common)

    # normalise each type by its own spread, so no type dominates by firing rate alone
    def z(M):
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True)
        return (M - mu) / np.where(sd > 1e-9, sd, 1.0)

    signs = np.array([pol[nm] for nm in TYPES])
    # each type's common mode, signed so that positive means a LEFT turn
    yaw_pol = (z(common) * signs[:, None]).mean(axis=0)
    # the literal R-L version, signed the same way for comparability
    yaw_pair = (z(pair) * signs[:, None]).mean(axis=0)
    return {"yaw_polarity": yaw_pol, "yaw_pair": yaw_pair,
            "per_type": {nm: common[i] for i, nm in enumerate(TYPES)}}


def deadband_from_static(pol: dict, pct: float, n_smooth: int) -> tuple[float, dict]:
    """Noise floor of the yaw signal, measured on the synthetic STATIC frames.

    The absolute value of the signal cannot be transferred between the synthetic session
    and the real video, because the per-type normalisation uses each recording's own
    spread. What does transfer is the ratio: how large is the signal's variation when
    nothing moves, against how large it is when something does. That ratio is measured
    here and applied to the real signal's spread.

    `pct` selects which percentile of the STATIC magnitude to use as the floor, so the
    caller can choose between a lenient and a strict band.
    """
    col = {i: x for i, x in enumerate(
        csv.DictReader((ROOT / "output/p053_threshold/targets.csv").open()))}
    signs = np.array([pol[nm] for nm in TYPES])[:, None]
    ratios, medians = [], []
    for seed in (64, 65):
        d = np.load(SYNTH / f"synthetic_seed{seed}.npz")
        ch = {}
        for nm in TYPES:
            for side in ("L", "R"):
                idx = [i for i, x in col.items()
                       if x["cell_type"] == nm and x["side"] == side]
                if idx:
                    ch[(nm, side)] = d["static"][:, idx].mean(axis=1) * FPS
        common_static = np.stack([(ch[(nm, "R")] + ch[(nm, "L")]) / 2 for nm in TYPES])

        def channel(cond):
            c = {}
            for nm in TYPES:
                for side in ("L", "R"):
                    idx = [i for i, x in col.items()
                           if x["cell_type"] == nm and x["side"] == side]
                    if idx:
                        c[(nm, side)] = d[cond][:, idx].mean(axis=1) * FPS
            return np.stack([(c[(nm, "R")] + c[(nm, "L")]) / 2 for nm in TYPES])

        common_move = np.concatenate([channel("left"), channel("right")], axis=1)
        pool = np.concatenate([common_static, common_move], axis=1)
        mu = pool.mean(axis=1, keepdims=True)
        sd = pool.std(axis=1, keepdims=True)
        norm = lambda M: ((M - mu) / np.where(sd > 1e-9, sd, 1.0)) * signs
        z_st = norm(common_static).mean(axis=0)
        z_mv = norm(common_move).mean(axis=0)
        ratios.append(float(z_st.std() / max(z_mv.std(), 1e-9)))
        medians.append(float(np.percentile(np.abs(z_st), pct)))
    ratio = float(np.mean(ratios))
    return ratio, {"ratio_static_over_moving": ratio,
                   "ratios_per_seed": ratios,
                   "static_abs_pct_on_synthetic": float(np.mean(medians)),
                   "note": "порог = отношение x разброс реального сигнала; "
                           "абсолютное значение не переносится между сессиями, "
                           "потому что нормировка идёт по разбросу каждой записи"}


def labelled_turns() -> list[dict]:
    """Human-confirmed turns, each tagged with the clip it belongs to.

    The mapping is explicit rather than inferred from the file name, because a guess
    would silently attribute a clip's turns to the wrong recording.
    """
    sources = [
        (ROOT / "data/p01r/review_set.json",
         ROOT / "data/p01r/turn_verdicts.json", "VID00001"),
        (ROOT / "data/p01r/review_set_v2.json",
         ROOT / "data/p01r/turn_verdicts_v2.json", "VID00001"),
        (ROOT / "data/p01r/review_set_vid2.json",
         ROOT / "data/p01r/turn_verdicts_vid2.json", "VID00002"),
        (ROOT / "data/p01r/review_set_vid5.json",
         ROOT / "data/p01r/turn_verdicts_vid5.json", "VID00005"),
    ]
    out = []
    for setp, verp, video in sources:
        if not setp.exists() or not verp.exists():
            continue
        rs = json.loads(setp.read_text(encoding="utf-8"))["candidates"]
        vd = json.loads(verp.read_text(encoding="utf-8"))["verdicts"]
        for x in rs:
            if x.get("repeat_of"):
                continue
            v = vd.get(x["id"])
            if not v:
                continue
            truth = x["camera_direction"] if v["verdict"] == "correct" \
                else (v.get("actual_direction") or "REJECT")
            if truth in ("LEFT", "RIGHT"):
                out.append({"id": x["id"], "video": video, "t0": x["t0"],
                            "t1": x["t1"], "kind": truth})
    return out


def auc_raw(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    o = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[o] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    return float((rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth-s", type=float, default=0.3)
    ap.add_argument("--deadband-pct", type=float, default=90.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pol = polarity_from_p06()
    print("P07.1 — сигнал поворота из MaleCNS")
    print(f"  полярность из P06.1 (знак связи с направлением камеры):")
    for nm in TYPES:
        print(f"    {nm:>6s}: delta {pol[(nm, 'delta')]:+7.3f} -> "
              f"{'LEFT' if pol[nm] > 0 else 'RIGHT'}")

    deadband_ratio, db_info = deadband_from_static(pol, args.deadband_pct, 1)
    print(f"\n  порог из синтетического STATIC (P06.1):")
    print(f"    разброс сигнала при покое / при движении = "
          f"{deadband_ratio:.3f} (по seed: {[f'{x:.3f}' for x in db_info['ratios_per_seed']]})")
    print(f"    абсолютное значение не переносится между сессиями, поэтому порог")
    print(f"    задаётся этим отношением к разбросу реального сигнала")

    all_turns = labelled_turns()
    rows = []
    summary = {}
    for name in VIDEOS:
        t, fired, col = load_video(name)
        rates = channel_rates(fired, col)
        sig = build_signals(rates, pol, max(int(round(args.smooth_s * FPS)), 1))
        n_sm = max(int(round(args.smooth_s * FPS)), 1)
        yaw = box(sig["yaw_polarity"], n_sm)
        yaw_pair = box(sig["yaw_pair"], n_sm)
        # the floor is a fraction of this recording's own spread, set by the synthetic
        # STATIC ratio. A percentile argument tightens or loosens it.
        deadband = deadband_ratio * float(np.std(yaw)) * (args.deadband_pct / 90.0)
        yaw_db = np.where(np.abs(yaw) < deadband, 0.0, yaw)
        n_active = np.load(VIDEOS[name]["trace"])["n_active"].astype(np.float64)

        with (OUT / f"yaw_signal_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "yaw_signal", "yaw_signal_deadband", "yaw_pair_diagnostic",
                        "n_active"] + [f"common_{nm}" for nm in TYPES])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.3f}", f"{yaw[k]:.5f}", f"{yaw_db[k]:.5f}",
                            f"{yaw_pair[k]:.5f}", int(n_active[k])]
                           + [f"{sig['per_type'][nm][k]:.4f}" for nm in TYPES])

        # validate on labelled turns belonging to this video. When a clip has no labels
        # yet the signal is still written above and only this block is skipped, so a new
        # clip can go through the chain before anyone has reviewed it.
        mine = [x for x in all_turns if x["video"] == name]
        if not mine:
            print(f"\n=== {name} ===")
            print(f"  сигнал записан, разметки для него пока нет — проверка пропущена")
            summary[name] = {"n_turns": 0, "note": "разметки нет"}
            continue
        isR = np.array([x["kind"] == "RIGHT" for x in mine])
        vals, vals_pair, vals_db = [], [], []
        for x in mine:
            m = (t >= x["t0"]) & (t <= x["t1"])
            vals.append(float(np.median(yaw[m])) if m.any() else np.nan)
            vals_pair.append(float(np.median(yaw_pair[m])) if m.any() else np.nan)
            vals_db.append(float(np.median(yaw_db[m])) if m.any() else np.nan)
        vals = np.array(vals)
        vals_pair = np.array(vals_pair)
        vals_db = np.array(vals_db)
        # positive yaw is a LEFT turn, so RIGHT turns should give negative values
        acc = float(np.mean(np.sign(vals) == np.where(isR, -1, 1)))
        acc_pair = float(np.mean(np.sign(vals_pair) == np.where(isR, -1, 1)))
        auc = max(auc_raw(-vals[isR], -vals[~isR]), 1 - auc_raw(-vals[isR], -vals[~isR]))

        # how noisy is the signal between turns, and how much does it just track arousal
        r_arousal = float(np.corrcoef(yaw, n_active)[0, 1]) \
            if yaw.std() > 0 and n_active.std() > 0 else float("nan")
        between = yaw[np.abs(yaw) < np.percentile(np.abs(yaw), 60)]
        summary[name] = {
            "n_turns": len(mine), "n_right": int(isR.sum()),
            "accuracy_polarity": acc, "accuracy_pair": acc_pair,
            "auc": auc, "deadband": deadband,
            "frac_zeroed": float(np.mean(yaw_db == 0)),
            "between_turn_std": float(np.std(between)),
            "corr_with_arousal": r_arousal,
            "median_left": float(np.median(vals[~isR])),
            "median_right": float(np.median(vals[isR])),
        }
        print(f"\n=== {name} ===")
        print(f"  подтверждённых поворотов: {len(mine)} ({int(isR.sum())} RIGHT, "
              f"{int((~isR).sum())} LEFT)")
        print(f"  медиана yaw: LEFT {np.median(vals[~isR]):+.3f}, "
              f"RIGHT {np.median(vals[isR]):+.3f}   (LEFT должен быть > 0)")
        print(f"  точность знака: {acc * 100:.0f}%  (формула R-L дала бы "
              f"{acc_pair * 100:.0f}%)")
        print(f"  AUC: {auc:.3f}")
        print(f"  обнулено порогом: {np.mean(yaw_db == 0) * 100:.0f}% времени")
        print(f"  корреляция с общим возбуждением: {r_arousal:+.3f}")

        for x, v, vp, vd in zip(mine, vals, vals_pair, vals_db):
            rows.append({"video": name, "id": x["id"], "t0": x["t0"], "t1": x["t1"],
                         "label": x["kind"], "yaw": v, "yaw_deadband": vd,
                         "yaw_pair": vp,
                         "correct": bool(np.sign(v) == (-1 if x["kind"] == "RIGHT" else 1))})

    with (OUT / "yaw_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (OUT / "yaw_report.json").write_text(json.dumps({
        "polarity_from": "P06.1, знак связи клетки с направлением камеры",
        "polarity": {nm: pol[nm] for nm in TYPES},
        "polarity_delta": {nm: pol[(nm, "delta")] for nm in TYPES},
        "deadband_from_static_ratio": deadband_ratio,
        "deadband_from_static_info": db_info,
        "deadband_pct": args.deadband_pct,
        "deadband_note": "порог взят из синтетического STATIC (P06.1), не из ручных "
                         "поворотов, иначе проверка была бы круговой",
        "sign_convention": "yaw > 0 = камера повернула ВЛЕВО",
        "why_not_pair_difference": "у DNp17, DNp26 и DNp20 левая и правая клетки "
                                   "реагируют одинаково (P06.1), поэтому R-L теряет "
                                   "сигнал: у DNp26 разница между LEFT и RIGHT окнами "
                                   "всего 0.046 Гц против 5.245 Гц у общей активности",
        "per_video": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
