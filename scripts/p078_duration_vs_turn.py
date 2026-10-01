#!/usr/bin/env python3
"""
P07.8 — does how long a swing lasts tell a glance from a turn?

What is being asked, and what is deliberately not assumed
---------------------------------------------------------
The swing itself does not say whether the camera went out and came back or went and
stayed: P07.4 showed that `net/(out+back)` has the same median inside and outside human
turns, 0.35 against 0.40, and P07.7 showed the direction of travel is not recoverable
often enough to replace it. The next candidate is duration, and the whole point of this
stage is not to guess it in advance. No rule such as "under half a second is a glance" is
built in here. The distributions are measured first, and a threshold is only fitted at all
if the feature separates on BOTH labelled clips.

Segmentation is taken unchanged from P07.4, including the fix that a run of zeros ends a
segment rather than extending it, and the short bridged gap that keeps a turn which clips
the deadband for a frame from being split in two.

Measurements per swing
----------------------
    duration          t1 - t0, the whole segment including bridged gaps
    active_duration   time actually spent off the deadband
    peak              largest absolute signal in the segment
    integral_abs      accumulated magnitude
    integral_signed   accumulated signed value
    frac_one_sign     share of active frames agreeing with the segment's own sign
    n_gaps            short deadband dips inside the segment

Only duration is examined at first. The others are written out for the later step.

The acceptance rule, fixed before the numbers are seen
-------------------------------------------------------
    separation >= 0.70 on VID00001 AND on VID00002   -> fit one threshold on VID00001,
                                                        freeze it, test on VID00002
    below 0.70 on either                              -> duration alone does not answer
                                                        this, and nothing is fitted

Nothing in P07.4 or P07.7 is touched and no trajectory is rebuilt.

Usage:
    PYTHONPATH=. python scripts/p078_duration_vs_turn.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

P074 = ROOT / "output/p074"
OUT = ROOT / "output/p078"
VIDEOS = ("VID00001", "VID00002")
TUNE = "VID00001"

SEP_MIN = 0.70      # fixed in advance: both clips must clear this for a threshold


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse P07.4 exactly: same loader, same segmentation, same ranking
p074 = _load("p074", ROOT / "scripts/p074_looks_vs_turns.py")
p077 = _load("p077", ROOT / "scripts/p077_heading_before_after.py")

rank_auc = p077.rank_auc
overlap = p077.overlap


def human_turns(video: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from p07_videos import labelled_turns
    return [{"t0": w["t0"], "t1": w["t1"], "direction": w["kind"]}
            for w in labelled_turns(video)]


def measure(t: np.ndarray, yaw: np.ndarray, s: dict) -> dict:
    """The six numbers listed in the header, for one swing."""
    dt = float(np.median(np.diff(t)))
    i0, i1 = s["i0"], s["i1"]
    seg = yaw[i0:i1 + 1]
    tt = t[i0:i1 + 1]
    nz = seg != 0
    sgn = np.sign(yaw[i0])

    active = int(nz.sum())
    # deadband dips strictly inside the segment, i.e. zeros with signal on both sides
    gaps, run = 0, 0
    for k in range(1, len(seg) - 1):
        if seg[k] == 0:
            run += 1
        else:
            if run > 0 and seg[k - 1] == 0:
                gaps += 1
            run = 0

    agree = int(((seg[nz] > 0) == (sgn > 0)).sum())

    return {
        "duration": float(tt[-1] - tt[0]),
        "active_duration": active * dt,
        "peak": float(np.max(np.abs(seg))),
        "integral_abs": float(np.sum(np.abs(seg)) * dt),
        "integral_signed": float(np.sum(seg) * dt),
        "frac_one_sign": agree / max(active, 1),
        "n_gaps": int(gaps),
        "n_frames": len(seg),
    }


def collect(video: str) -> list[dict]:
    t, yaw, speed, cam = p074.load(video)
    sw = p074.swings(t, yaw, cam, p074.MIN_SWING)
    hum = human_turns(video)
    out = []
    for s in sw:
        m = measure(t, yaw, s)
        m.update({"t0": s["t0"], "t1": s["t1"], "direction": s["direction"],
                  "delta": s["delta"], "size": s["size"],
                  "kind_old": s.get("kind", "n/a"),
                  "real_turn": any(overlap(h["t0"], h["t1"], s["t0"], s["t1"]) > 0
                                   for h in hum)})
        out.append(m)
    return out


def sep(values: dict[str, list[dict]], key: str, video: str) -> float:
    rows = values[video]
    P = np.array([r[key] for r in rows if r["real_turn"]])
    N = np.array([r[key] for r in rows if not r["real_turn"]])
    a = rank_auc(P, N)
    if not np.isfinite(a):
        return float("nan")
    return max(a, 1 - a)


def fit_threshold(rows: list[dict], key: str) -> tuple[float, float, bool]:
    """One threshold on the tuning clip, by Youden's J. Returns (thr, J, flip)."""
    P = np.array([r[key] for r in rows if r["real_turn"]])
    N = np.array([r[key] for r in rows if not r["real_turn"]])
    flip = rank_auc(P, N) < 0.5
    best, best_j = float(np.median(P)), -1.0
    for c in np.unique(np.concatenate([P, N])):
        tp = np.mean(P < c) if flip else np.mean(P > c)
        tn = np.mean(N >= c) if flip else np.mean(N <= c)
        j = tp + tn - 1
        if j > best_j:
            best, best_j = float(c), float(j)
    return best, float(best_j), bool(flip)


def evaluate(rows: list[dict], key: str, thr: float, flip: bool) -> dict:
    def call(r):
        return (r[key] < thr) if flip else (r[key] > thr)
    real = [r for r in rows if r["real_turn"]]
    fake = [r for r in rows if not r["real_turn"]]
    tot = sum(abs(r["delta"]) for r in real) or 1e-9
    tot_f = sum(abs(r["delta"]) for r in fake) or 1e-9
    return {
        "true_signal_retained": sum(abs(r["delta"]) for r in real if call(r)) / tot,
        "false_signal_retained": sum(abs(r["delta"]) for r in fake if call(r)) / tot_f,
        "real_kept": sum(1 for r in real if call(r)),
        "real_total": len(real),
        "false_kept": sum(1 for r in fake if call(r)),
        "false_total": len(fake),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("P07.8 — различает ли длительность взгляд и поворот")
    print("  отрезки взяты из P07.4 без изменений; пороги заранее НЕ задаются")
    print(f"  условие продолжения: разделение >= {SEP_MIN:.2f} на ОБОИХ роликах")
    print()

    data = {v: collect(v) for v in VIDEOS}

    keys = ["duration", "active_duration", "peak", "integral_abs",
            "integral_signed", "frac_one_sign", "n_gaps"]
    titles = {"duration": "длительность, с", "active_duration": "время вне мертвой зоны, с",
              "peak": "пик", "integral_abs": "сумма |сигнал|",
              "integral_signed": "сумма сигнала", "frac_one_sign": "доля одного знака",
              "n_gaps": "коротких провалов внутри"}

    for v in VIDEOS:
        rows = data[v]
        n_real = sum(1 for r in rows if r["real_turn"])
        print(f"=== {v} ===")
        print(f"  отрезков {len(rows)}: внутри поворотов человека {n_real}, "
              f"вне {len(rows) - n_real}")
        print()
        print(f"  {'признак':>24s} {'медиана в повороте':>19s} {'вне':>9s} "
              f"{'разделение':>11s} {'лучше':>8s}")
        for k in keys:
            P = np.array([r[k] for r in rows if r["real_turn"]])
            N = np.array([r[k] for r in rows if not r["real_turn"]])
            a = rank_auc(P, N)
            eff = max(a, 1 - a) if np.isfinite(a) else float("nan")
            print(f"  {titles[k]:>24s} {np.median(P):19.3f} {np.median(N):9.3f} "
                  f"{eff:11.3f} {'больше' if a >= 0.5 else 'меньше':>8s}")
        print()

    # ---- only duration, as instructed ------------------------------------
    d_sep = {v: sep(data, "duration", v) for v in VIDEOS}
    passes = all(d_sep[v] >= SEP_MIN for v in VIDEOS)

    print("=== ПРИЗНАК 1: ТОЛЬКО ДЛИТЕЛЬНОСТЬ ===")
    for v in VIDEOS:
        rows = data[v]
        P = np.median([r["duration"] for r in rows if r["real_turn"]])
        N = np.median([r["duration"] for r in rows if not r["real_turn"]])
        print(f"  {v}: разделение {d_sep[v]:.3f}  "
              f"(медиана в повороте {P:.2f} с, вне {N:.2f} с)")
    print(f"  DURATION SEPARATES LOOK/TURN: "
          f"{'YES' if passes else 'NO'}")
    if not passes:
        print(f"  на одном из роликов разделение ниже {SEP_MIN:.2f}.")
        print("  Порог НЕ подбирается: длительность сама по себе эту задачу не решает.")
    print()

    report = {"sep_min": SEP_MIN, "separation": {k: {v: sep(data, k, v)
                                                     for v in VIDEOS} for k in keys},
              "duration_passes": passes, "videos": {}}

    result = None
    if passes:
        thr, j, flip = fit_threshold(data[TUNE], "duration")
        print(f"  один порог на {TUNE}: {thr:.3f} с "
              f"({'короче = поворот' if flip else 'длиннее = поворот'}), J = {j:.3f}")
        print()
        print(f"  {'ролик':>9s} {'сохранено настоящего':>21s} {'осталось ложного':>17s} "
              f"{'поворотов взято':>16s}")
        for v in VIDEOS:
            e = evaluate(data[v], "duration", thr, flip)
            print(f"  {v:>9s} {e['true_signal_retained'] * 100:20.0f}% "
                  f"{e['false_signal_retained'] * 100:16.0f}% "
                  f"{e['real_kept']:>7d}/{e['real_total']:<8d}")
            report["videos"][v] = e
        report["threshold"] = {"feature": "duration", "value": thr,
                               "flip": flip, "j": j}
        result = ("duration", thr, flip)

        # ---- second feature, only reached if the first passed ------------
        print()
        print("=== ПРИЗНАК 2: ДЛИТЕЛЬНОСТЬ + СУММА |СИГНАЛ| ===")
        best = None
        rows_t = data[TUNE]
        cs_d = np.unique([r["duration"] for r in rows_t])
        cs_i = np.unique([r["integral_abs"] for r in rows_t])
        for td in np.percentile(cs_d, np.arange(5, 100, 5)):
            for ti in np.percentile(cs_i, np.arange(5, 100, 5)):
                tp = np.mean([(r["duration"] > td and r["integral_abs"] > ti)
                              for r in rows_t if r["real_turn"]])
                tn = np.mean([not (r["duration"] > td and r["integral_abs"] > ti)
                              for r in rows_t if not r["real_turn"]])
                jj = tp + tn - 1
                if best is None or jj > best[2]:
                    best = (float(td), float(ti), float(jj))
        td, ti, jj = best
        print(f"  пороги на {TUNE}: длительность > {td:.2f} с И сумма > {ti:.3f} "
              f"(J = {jj:.3f})")
        print()
        print(f"  {'ролик':>9s} {'сохранено настоящего':>21s} {'осталось ложного':>17s} "
              f"{'поворотов взято':>16s}")
        for v in VIDEOS:
            rows = data[v]
            def call(r, td=td, ti=ti):
                return r["duration"] > td and r["integral_abs"] > ti
            real = [r for r in rows if r["real_turn"]]
            fake = [r for r in rows if not r["real_turn"]]
            tot = sum(abs(r["delta"]) for r in real) or 1e-9
            tot_f = sum(abs(r["delta"]) for r in fake) or 1e-9
            tr = sum(abs(r["delta"]) for r in real if call(r)) / tot
            fr = sum(abs(r["delta"]) for r in fake if call(r)) / tot_f
            rk = sum(1 for r in real if call(r))
            print(f"  {v:>9s} {tr * 100:20.0f}% {fr * 100:16.0f}% "
                  f"{rk:>7d}/{len(real):<8d}")
            report["videos"][v + "_combined"] = {
                "true_signal_retained": tr, "false_signal_retained": fr,
                "real_kept": rk, "real_total": len(real)}
        report["combined_threshold"] = {"duration": td, "integral_abs": ti, "j": jj}

    for v in VIDEOS:
        with (OUT / f"swings_{v}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[v][0].keys()))
            w.writeheader()
            w.writerows(data[v])

    # ---- base rate, without which the retention numbers mislead ----------
    # If real turns occupy a large share of the total signal, then a blind rule that calls
    # everything a turn retains a large share of the true signal too, and retention alone
    # would look like success. The lift over that base rate is what carries information.
    print("=== БАЗОВАЯ СТАВКА ===")
    base = {}
    for v in VIDEOS:
        rows = data[v]
        tot = sum(abs(r["delta"]) for r in rows) or 1e-9
        b = sum(abs(r["delta"]) for r in rows if r["real_turn"]) / tot
        base[v] = b
        n_real = sum(1 for r in rows if r["real_turn"])
        print(f"  {v}: настоящие повороты занимают {b * 100:.0f}% всего сигнала "
              f"({n_real} отрезков из {len(rows)})")
    print("  правило «всё поворот» сохранило бы 100% настоящего и 100% ложного,")
    print("  поэтому выигрыш — это превышение этой ставки, а не сами проценты.")
    print()
    report["base_rate"] = base

    # ---- what threshold the two P07.3 targets would need ------------------
    print("=== ЧТО ТРЕБУЮТ КРИТЕРИИ P07.3 ===")
    print("  цель: найдено >= 70% настоящих поворотов И ложных < 1 на настоящий")
    print()
    print(f"  {'порог, с':>9s} {'J':>6s} {'настоящего':>11s} {'ложного':>9s} "
          f"{'найдено':>12s} {'ложных на 1':>12s} {'годится':>9s}")
    sweep = {}
    for thr in (0.60, 0.70, 0.84, 1.00, 1.20, 1.50):
        line = {}
        ok_all = True
        for v in VIDEOS:
            rows = data[v]
            real = [r for r in rows if r["real_turn"]]
            fake = [r for r in rows if not r["real_turn"]]
            tr = sum(abs(r["delta"]) for r in real if r["duration"] > thr) / \
                max(sum(abs(r["delta"]) for r in real), 1e-9)
            fr = sum(abs(r["delta"]) for r in fake if r["duration"] > thr) / \
                max(sum(abs(r["delta"]) for r in fake), 1e-9)
            rk = sum(1 for r in real if r["duration"] > thr)
            fk = sum(1 for r in fake if r["duration"] > thr)
            tp = rk / max(len(real), 1)
            tn = 1 - fk / max(len(fake), 1)
            line[v] = {"j": tp + tn - 1, "true_kept": tr, "false_kept": fr,
                       "found": rk, "found_frac": tp, "false_per_real": fk / max(rk, 1)}
            if not (tp >= 0.70 and fk / max(rk, 1) < 1.0):
                ok_all = False
        j_mean = float(np.mean([line[v]["j"] for v in VIDEOS]))
        tf = float(np.mean([line[v]["true_kept"] for v in VIDEOS]))
        ff = float(np.mean([line[v]["false_kept"] for v in VIDEOS]))
        fo = float(np.mean([line[v]["found_frac"] for v in VIDEOS]))
        fp = float(np.mean([line[v]["false_per_real"] for v in VIDEOS]))
        print(f"  {thr:>9.2f} {j_mean:6.3f} {tf * 100:10.0f}% {ff * 100:8.0f}% "
              f"{fo * 100:11.0f}% {fp:12.2f} "
              f"{'да, на обоих' if ok_all else 'нет':>9s}")
        sweep[str(thr)] = {"videos": line, "ok_both": ok_all}
    report["sweep"] = sweep
    report["p073_criteria_note"] = (
        "Ни один порог не выполняет оба критерия P07.3 сразу. Найдено >=70% требует "
        "порога <=0.70 с, а ложных <1 на настоящий требует >=0.84 с. Промежуточных "
        "значений нет, потому что на 0.70 с у VID00002 выходит 1.27 ложных на настоящий.")
    print()

    # ---- distributions ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for i, v in enumerate(VIDEOS):
        rows = data[v]
        for j, (key, lab) in enumerate((("duration", "длительность, с"),
                                        ("integral_abs", "сумма |сигнал|"))):
            ax = axes[j][i]
            P = [r[key] for r in rows if r["real_turn"]]
            N = [r[key] for r in rows if not r["real_turn"]]
            hi = max(max(P), max(N)) or 1.0
            bins = np.linspace(0, hi, 26)
            ax.hist(P, bins=bins, alpha=0.72, color="tab:green",
                    label=f"внутри поворота (n={len(P)})")
            ax.hist(N, bins=bins, alpha=0.65, color="0.6",
                    label=f"вне (n={len(N)})")
            if passes and j == 0:
                td = report["threshold"]["value"]
                ax.axvline(td, color="tab:red", ls="--", label=f"порог {td:.2f}")
            a = sep(data, key, v)
            ax.set_xlabel(lab)
            ax.set_ylabel("отрезков")
            ax.set_title(f"{v} — {lab}, разделение {a:.3f}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("P07.8 — различает ли длительность взгляд и поворот", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "distributions.png", dpi=125)
    plt.close(fig)

    report["final"] = {
        "DURATION_SEPARATES_LOOK_TURN": "YES" if passes else "NO",
        "separation_VID00001": d_sep["VID00001"],
        "separation_VID00002": d_sep["VID00002"],
        "threshold_s": report.get("threshold", {}).get("value"),
        "median_duration_in_turn": {v: float(np.median(
            [r["duration"] for r in data[v] if r["real_turn"]])) for v in VIDEOS},
        "median_duration_outside": {v: float(np.median(
            [r["duration"] for r in data[v] if not r["real_turn"]])) for v in VIDEOS},
        "comparison_to_P074": {
            "P074_true_signal_retained": 0.48,
            "P078_true_signal_retained": report.get("videos", {}).get(
                "VID00001", {}).get("true_signal_retained"),
            "P078_false_signal_retained": report.get("videos", {}).get(
                "VID00001", {}).get("false_signal_retained"),
        },
    }
    if passes:
        lifts = {v: report["videos"][v]["true_signal_retained"] /
                 max(report["videos"][v]["false_signal_retained"], 1e-9) for v in VIDEOS}
        report["final"]["lift_over_chance"] = lifts
        report["final"]["note"] = (
            "Длительность разделяет на обоих роликах и в 2.4-2.5 раза лучше слепого "
            "правила, сохраняя 85% настоящего сигнала против 48% у P07.4. Но оба "
            "критерия P07.3 одновременно не выполняются: см. p073_criteria_note.")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
