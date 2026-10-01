#!/usr/bin/env python3
"""
P07.1 — stop integrating yaw noise, integrate only real turns.

The problem being fixed
-----------------------
P07 integrated the yaw signal continuously and produced a ball of string: straightness
0.08 to 0.12, against 0.22 for the frontend route that only integrates at detected
turns. The measurement explained why. Summed over the whole clip,

    |yaw| inside the confirmed turns   ( 9 percent of the time)   1451 units
    |yaw| outside them                 (91 percent of the time)   5821 units

so four fifths of the accumulated turning happened where no turn took place. The signal
does react to turns, but its noise is spread across ten times as much time and wins on
the sum. No amount of smoothing fixes an integral of a signal whose noise exceeds its
signal; the integration has to stop between turns.

Three conditions, all required
------------------------------
    level        |yaw| above the synthetic STATIC floor
    agreement    at least three of the four DN types agree in sign with the combined
                 signal, so a single noisy pair cannot start a turn
    persistence  the sign holds for at least 250 ms

The floor comes only from the synthetic STATIC condition of P06.1, never from the human
labels. Picking it from the labels would make the validation circular, which is exactly
the mistake P05.4 was built to avoid.

Reported in the output
----------------------
    total turning before and after the filter
    the fraction of turning that falls inside the human-confirmed windows, before and
    after, which is the number the whole exercise turns on

Usage:
    PYTHONPATH=. python scripts/p071_turn_filter.py
    PYTHONPATH=. python scripts/p071_turn_filter.py --sensitivity
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
OUT71 = ROOT / "output/p071"
SYNTH = ROOT / "output/p061_synthetic"
TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]
def _applied() -> list[str]:
    """Clips the chain is applied to: the labelled ones plus anything registered whose
    yaw signal and forward signal already exist."""
    out = ["VID00001", "VID00002"]
    try:
        import sys as _s
        _s.path.insert(0, str(ROOT / "scripts"))
        from p07_videos import videos as _reg
        for name in _reg():
            if name in out:
                continue
            if (OUT / f"trace_{name}.npz").exists() or \
               (ROOT / "output/p07" / f"yaw_signal_{name}.csv").exists():
                out.append(name)
    except Exception as e:
        print(f"  реестр недоступен ({e})")
    return out


VIDEOS = _applied()
FPS = 50.0

# polarity measured by P06.1 and not re-chosen
POLARITY = {"DNp17": 1.0, "DNa07": -1.0, "DNp26": 1.0, "DNp20": 1.0}

# fixed before looking at any trajectory
LEVEL_PCT = 95.0          # percentile of |signal| under STATIC
# Agreement is the one condition that can be calibrated from synthetic data without
# touching the video labels, and the measurement is unambiguous. Counting how many of the
# four types agree with the combined signal:
#
#     condition    distribution over 0..4    share with all four
#     LEFT              [0, 0, 6, 7, 12]            48 percent
#     RIGHT             [0, 0, 10, 10, 5]           20 percent
#     STATIC            [0, 2, 2, 21, 0]             0 percent
#
# A still frame never reaches four out of four, at either seed, while moving frames
# sometimes do. Requiring all four therefore excludes the noise case by construction and
# is derived from the synthetic runs rather than from the labelled turns.
MIN_AGREE = 4             # of four types
# The hold time cannot be calibrated this way, because the synthetic clips are fixed
# length. 250 ms sits in the range the task specified.
MIN_HOLD_S = 0.25


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def synthetic_floor(pct: float = LEVEL_PCT) -> tuple[float, dict]:
    """Floor for the yaw signal, from the STATIC condition of P06.1 only.

    Expressed as a multiple of the moving signal's own standard deviation, because the
    absolute values differ between the synthetic session and a real recording: each is
    normalised by its own spread, and that spread depends on how much of the clip is
    moving. The ratio does transfer.
    """
    col = {i: x for i, x in enumerate(
        csv.DictReader((ROOT / "output/p053_threshold/targets.csv").open()))}
    ks = []
    detail = []
    for seed in (64, 65):
        d = np.load(SYNTH / f"synthetic_seed{seed}.npz")

        def channels(cond):
            out = {}
            for nm in TYPES:
                for side in ("L", "R"):
                    idx = [i for i, x in col.items()
                           if x["cell_type"] == nm and x["side"] == side]
                    out[(nm, side)] = d[cond][:, idx].mean(axis=1) * FPS
            return np.stack([(out[(nm, "R")] + out[(nm, "L")]) / 2 for nm in TYPES])

        st = channels("static")
        mv = np.concatenate([channels("left"), channels("right")], axis=1)
        signs = np.array([POLARITY[nm] for nm in TYPES])[:, None]
        # normalise by the moving spread, then average the signed types
        mu, sd = mv.mean(axis=1, keepdims=True), mv.std(axis=1, keepdims=True)
        z_st = (((st - mu) / np.where(sd > 1e-9, sd, 1)) * signs).mean(axis=0)
        z_mv = (((mv - mu) / np.where(sd > 1e-9, sd, 1)) * signs).mean(axis=0)
        k = float(np.percentile(np.abs(z_st), pct)) / float(z_mv.std())
        ks.append(k)
        detail.append({"seed": seed, "k": k,
                       "static_abs_pct": float(np.percentile(np.abs(z_st), pct)),
                       "moving_std": float(z_mv.std())})
    return float(np.mean(ks)), {"per_seed": detail,
                                "note": "порог = k x разброс сигнала реальной записи"}


def load_channels(name: str) -> dict:
    rows = list(csv.DictReader((OUT / f"yaw_signal_{name}.csv").open()))
    t = np.array([float(r["t"]) for r in rows])
    per = {}
    for nm in TYPES:
        per[nm] = np.array([float(r[f"common_{nm}"]) for r in rows])
    return t, per


def build(name: str, n_sm: int) -> dict:
    t, per = load_channels(name)
    # each type normalised by its own spread and signed by its P06.1 polarity, then
    # averaged. This is the same construction P07 validated at 78 percent on the second
    # clip, reused rather than refitted.
    z = {}
    for nm in TYPES:
        v = box(per[nm], n_sm)
        sd = float(np.std(v))
        z[nm] = POLARITY[nm] * (v - float(np.mean(v))) / (sd if sd > 1e-9 else 1.0)
    combined = np.mean([z[nm] for nm in TYPES], axis=0)
    agree = np.sum([np.sign(z[nm]) == np.sign(combined) for nm in TYPES], axis=0)
    return {"t": t, "per": z, "combined": combined, "agree": agree}


def detect(ch: dict, floor: float, min_agree: int, min_hold_s: float) -> np.ndarray:
    """Mark stretches that satisfy all three conditions at once."""
    combined = ch["combined"]
    agree = ch["agree"]
    t = ch["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    need = max(int(round(min_hold_s / dt)), 1)

    above = (np.abs(combined) >= floor) & (agree >= min_agree)
    out = np.zeros(len(combined), dtype=bool)
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1] and \
                np.sign(combined[j + 1]) == np.sign(combined[i]):
            j += 1
        if (j - i + 1) >= need:
            out[i:j + 1] = True
        i = j + 1
    return out


def human_windows(video: str) -> list[tuple[float, float, str]]:
    """Confirmed turns for a clip, from the shared mapping.

    The mapping used to be inferred from the label file's name, which put the first clip's
    second round into the second clip. See `p07_videos.LABEL_SETS`.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from p07_videos import labelled_turns
    return [(w["t0"], w["t1"], w["kind"]) for w in labelled_turns(video)]


def integrate(ch: dict, mask: np.ndarray, yaw_scale: float, speed: np.ndarray,
              mode: str = "amplitude") -> dict:
    """Integrate the gated yaw into a heading.

    `mode` chooses what gets integrated during a detected event:

        amplitude  the signal itself
        sign       only its direction, one unit per step

    The sign form follows from what P07 actually established. The amplitude is not
    trustworthy: the noise floor measured against STATIC is 1.67 times the signal's own
    spread, which is why continuous integration failed. The *sign* inside confirmed turns
    is trustworthy, at 92 to 100 percent depending on the clip. Integrating the sign
    therefore uses the part of the measurement that was validated and discards the part
    that was not, rather than letting a noisy magnitude set how far each turn goes.
    """
    t = ch["t"]
    yaw = ch["combined"]
    dt = float(np.median(np.diff(t)))
    contribution = np.sign(yaw) if mode == "sign" else yaw
    gated = np.where(mask, contribution, 0.0)
    theta = np.zeros(len(t))
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    th = 0.0
    for k in range(1, len(t)):
        th += gated[k] * yaw_scale * dt
        theta[k] = th
        x[k] = x[k - 1] + speed[k] * np.cos(th) * dt
        y[k] = y[k - 1] + speed[k] * np.sin(th) * dt
    return {"t": t, "theta": theta, "x": x, "y": y, "gated": gated}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth-s", type=float, default=0.3)
    ap.add_argument("--yaw-scale", type=float, default=90.0)
    ap.add_argument("--mode", default="amplitude", choices=["amplitude", "sign"],
                    help="what gets integrated during an event")
    ap.add_argument("--sensitivity", action="store_true",
                    help="show how the result varies with the agreement count and hold "
                         "time; reported for context, not used to pick parameters")
    args = ap.parse_args()

    OUT71.mkdir(parents=True, exist_ok=True)
    n_sm = max(int(round(args.smooth_s * FPS)), 1)
    floor_k, floor_info = synthetic_floor()
    print("P07.1 — фильтр реальных поворотов")
    print(f"  порог из синтетического STATIC (P06.1), {LEVEL_PCT:.0f}-й процентиль:")
    for d in floor_info["per_seed"]:
        print(f"    seed {d['seed']}: k = {d['k']:.3f} "
              f"(|сигнал| покоя {d['static_abs_pct']:.2f}, "
              f"разброс движения {d['moving_std']:.2f})")
    print(f"    среднее k = {floor_k:.3f}")
    print(f"  условия поворота: |yaw| >= порог, согласны >= {MIN_AGREE} из 4 пар, "
          f"знак держится >= {MIN_HOLD_S * 1000:.0f} мс")

    report = {"floor_k": floor_k, "floor_pct": LEVEL_PCT,
              "min_agree": MIN_AGREE, "min_hold_s": MIN_HOLD_S,
              "floor_info": floor_info, "videos": {}}

    for name in VIDEOS:
        ch = build(name, n_sm)
        t = ch["t"]
        floor = floor_k * float(np.std(ch["combined"]))
        mask = detect(ch, floor, MIN_AGREE, MIN_HOLD_S)

        speed_rows = list(csv.DictReader((OUT / f"forward_signal_{name}.csv").open()))
        speed = np.array([float(r["speed"]) for r in speed_rows])
        n = min(len(speed), len(t))
        speed = speed[:n]

        wins = human_windows(name)
        in_turn = np.zeros(len(t), dtype=bool)
        for a, b, _ in wins:
            in_turn |= (t >= a) & (t <= b)

        gated = np.where(mask, ch["combined"], 0.0)
        dtm = float(np.median(np.diff(t)))

        # the numbers the exercise turns on
        before = float(np.sum(np.abs(ch["combined"])) * args.yaw_scale * dtm)
        after = float(np.sum(np.abs(gated)) * args.yaw_scale * dtm)
        before_in = float(np.sum(np.abs(ch["combined"])[in_turn]) * args.yaw_scale * dtm)
        after_in = float(np.sum(np.abs(gated)[in_turn]) * args.yaw_scale * dtm)

        tr = integrate(ch, mask, args.yaw_scale, speed[:len(t)], args.mode)
        seg = np.hypot(np.diff(tr["x"]), np.diff(tr["y"]))
        path_len = float(seg.sum())
        net = float(np.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]))

        # sign accuracy inside the human windows, which the filter must not destroy
        ok = tot = 0
        for a, b, kind in wins:
            m = (t >= a) & (t <= b) & mask
            if not m.any():
                continue
            want = 1.0 if kind == "LEFT" else -1.0
            ok += int(np.sign(np.sum(gated[m])) == want)
            tot += 1
        n_events = int(np.sum(np.diff(mask.astype(int)) == 1))

        report["videos"][name] = {
            "n_events": n_events,
            "turns_detected": tot, "turns_total": len(wins),
            "turns_missed": len(wins) - tot,
            "false_events_per_real_turn": n_events / max(tot, 1),
            "frac_time_turning": float(np.mean(mask)),
            "turning_before": before, "turning_after": after,
            "turning_kept": after / before if before > 0 else 0.0,
            "turning_inside_human_before": before_in,
            "turning_inside_human_after": after_in,
            "frac_inside_before": before_in / before if before > 0 else 0.0,
            "frac_inside_after": after_in / after if after > 0 else 0.0,
            "path_length": path_len, "net_displacement": net,
            "straightness": net / path_len if path_len > 0 else 0.0,
            "net_turn_units": float(tr["theta"][-1] - tr["theta"][0]),
            "turns_with_correct_sign": ok, "turns_with_any_event": tot,
            "floor": floor,
        }

        print(f"\n=== {name} ===")
        print(f"  событий поворота: {int(np.sum(np.diff(mask.astype(int)) == 1))}, "
              f"они занимают {np.mean(mask) * 100:.0f}% времени")
        print(f"  суммарный поворот: было {before:.0f}, стало {after:.0f} "
              f"(осталось {after / max(before, 1e-9) * 100:.0f}%)")
        print(f"  доля поворота внутри подтверждённых окон: "
              f"было {before_in / max(before, 1e-9) * 100:.0f}%, "
              f"стало {after_in / max(after, 1e-9) * 100:.0f}%")
        print(f"  настоящих поворотов замечено: {tot} из {len(wins)} "
              f"(пропущено {len(wins) - tot})")
        print(f"  знак верен внутри окон: {ok} из {tot}")
        print(f"  ложных событий на один настоящий поворот: "
              f"{n_events / max(tot, 1):.1f}")
        print(f"  траектория: путь {path_len:.1f}, смещение {net:.1f}, "
              f"прямизна {net / max(path_len, 1e-9):.3f}")

        with (OUT71 / f"trajectory_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "theta", "yaw_gated", "yaw_full", "turning",
                        "agreement", "speed"])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.3f}", f"{tr['x'][k]:.6f}", f"{tr['y'][k]:.6f}",
                            f"{tr['theta'][k]:.6f}", f"{gated[k]:.5f}",
                            f"{ch['combined'][k]:.5f}", int(mask[k]),
                            int(ch["agree"][k]), f"{speed[k]:.5f}"])

        np.savez_compressed(OUT71 / f"turn_mask_{name}.npz",
                            t=t, combined=ch["combined"], gated=tr["gated"],
                            mask=mask, agree=ch["agree"])

    if args.sensitivity:
        print(f"\n=== СВОБОДНЫЙ ПАРАМЕТР И ЕГО ЦЕНА ===")
        print(f"  уровневый порог фиксирован синтетикой (k = {floor_k:.2f}), поэтому")
        print(f"  свободны только согласие и удержание. Их влияние:\n")
        print(f"  {'видео':>9s} {'согл':>5s} {'удерж':>7s} {'замечено':>10s} "
              f"{'знак верен':>11s} {'событий':>8s} {'ложных/наст.':>13s} "
              f"{'прямизна':>9s}")
        sens = []
        for name in VIDEOS:
            ch = build(name, n_sm)
            t = ch["t"]
            fl = floor_k * float(np.std(ch["combined"]))
            speed_rows = list(csv.DictReader((OUT / f"forward_signal_{name}.csv").open()))
            sp = np.array([float(r["speed"]) for r in speed_rows])
            wins = human_windows(name)
            for agree in (3, 4):
                for hold in (0.15, 0.25, 0.40):
                    mask = detect(ch, fl, agree, hold)
                    gated = np.where(mask, ch["combined"], 0.0)
                    det = ok = 0
                    for a, b, kind in wins:
                        mm = (t >= a) & (t <= b) & mask
                        if mm.any():
                            det += 1
                            want = 1.0 if kind == "LEFT" else -1.0
                            ok += int(np.sign(np.sum(gated[mm])) == want)
                    ev = int(np.sum(np.diff(mask.astype(int)) == 1))
                    tr = integrate(ch, mask, args.yaw_scale, sp[:len(t)], args.mode)
                    seg = np.hypot(np.diff(tr["x"]), np.diff(tr["y"]))
                    L = float(seg.sum())
                    net = float(np.hypot(tr["x"][-1] - tr["x"][0],
                                         tr["y"][-1] - tr["y"][0]))
                    st = net / L if L > 0 else 0.0
                    sens.append({"video": name, "min_agree": agree, "hold_s": hold,
                                 "detected": det, "n_turns": len(wins),
                                 "sign_correct": ok, "n_events": ev,
                                 "false_per_real": ev / max(det, 1),
                                 "straightness": st})
                    print(f"  {name:>9s} {agree:>5d} {hold * 1000:6.0f}мс "
                          f"{det:>4d}/{len(wins):<5d} {ok:>4d}/{max(det, 1):<6d} "
                          f"{ev:>8d} {ev / max(det, 1):>13.1f} {st:>9.3f}")
        report["sensitivity"] = sens
        sts = [x["straightness"] for x in sens]
        print(f"\n  прямизна по {len(sts)} конфигурациям: от {min(sts):.3f} до "
              f"{max(sts):.3f} ({max(sts) / max(min(sts), 1e-9):.0f}x разброс)")
        print(f"  и ложных событий на настоящий поворот: от "
              f"{min(x['false_per_real'] for x in sens):.1f} до "
              f"{max(x['false_per_real'] for x in sens):.1f}")
        print(f"  -> форма траектории зависит от свободного параметра сильнее, чем от")
        print(f"     самого сигнала; ни одна настройка не даёт одновременно и мало")
        print(f"     ложных событий, и высокую полноту обнаружения настоящих")

    (OUT71 / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\nWrote {OUT71}/")


if __name__ == "__main__":
    main()
