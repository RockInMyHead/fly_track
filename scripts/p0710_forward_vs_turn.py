#!/usr/bin/env python3
"""
P07.10 — does the forward signal tell a glance from a turn?

Why the test needs a precondition check first
---------------------------------------------
The idea is that forward motion is an independent second source: while glancing, the
walker keeps going at much the same speed, while a real turn slows the walker down or
changes the pace. Independent it is not, and the project already says so in the header of
`p07_trajectory.py`: the frozen forward set contains DNp17 on the right side, and DNp17 is
also one of the four yaw types.

Counted out, the forward set is

    DNp17 R x4, DNa07 L x1, DNp20 R x1, DNbe001 R x1

and the yaw signal is a left-right difference over DNp17, DNa07, DNp26 and DNp20. So four
of the seven forward cells sit inside the yaw set as well, and DNp17 R alone is more than
half the forward weight. The measured correlation on the real clips is +0.77 and +0.76,
which is reported as a failed precondition rather than passed over.

Three tests are therefore run, and they answer different questions.

    1  as specified: forward_drop = during - (before + after)/2, AUC against the turn label
    2  the part of the forward signal with the yaw signal regressed out, so that only what
       is not yaw is left, tested the same way
    3  whether either of them improves on duration alone, compared at equal false alarms

Numbers 1 and 2 are measurements and are reported whatever they say. Number 3 is the one
that decides usefulness, because the task asks for improvement on VID00002 specifically.

Rules fixed before the numbers are seen
----------------------------------------
    thresholds fitted on VID00001 only, VID00002 held out
    useful means on VID00002: more real turns found or fewer false events, with the other
    measure no worse
    thresholds are fitted on VID00001 only, and nothing is tuned for the recorded clips

Nothing in P07.4, P07.7, P07.8 or P07.9 is modified and no trajectory is built.

Usage:
    PYTHONPATH=. python scripts/p07 10_forward_vs_turn.py
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

P07 = ROOT / "output/p07"
OUT = ROOT / "output/p0710"
VIDEOS = ("VID00001", "VID00002")
TUNE = "VID00001"

W_S = 1.0           # window before and after, as specified
AUC_MIN = 0.70      # fixed in advance: both clips must clear this before combining


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p074 = _load("p074", ROOT / "scripts/p074_looks_vs_turns.py")
p077 = _load("p077", ROOT / "scripts/p077_heading_before_after.py")
p078 = _load("p078", ROOT / "scripts/p078_duration_vs_turn.py")

rank_auc = p077.rank_auc
overlap = p077.overlap


def load_forward(video: str):
    rows = list(csv.DictReader((P07 / f"forward_signal_{video}.csv").open()))
    t = np.array([float(r["t"]) for r in rows])
    sig = np.array([float(r["forward_signal"]) for r in rows])
    raw = np.array([float(r["forward_raw"]) for r in rows])
    return t, sig, raw


def load_yaw(video: str):
    rows = list(csv.DictReader((P07 / f"yaw_signal_{video}.csv").open()))
    t = np.array([float(r["t"]) for r in rows])
    return t, np.array([float(r["yaw_signal"]) for r in rows])


def win_mean(t: np.ndarray, sig: np.ndarray, t0: float, t1: float) -> float:
    m = (t >= t0) & (t <= t1)
    return float(sig[m].mean()) if m.sum() > 0 else float("nan")


def build(video: str) -> dict:
    tf, sig, raw = load_forward(video)
    ty, yaw = load_yaw(video)
    n = min(len(tf), len(ty))
    t, sig, raw, yaw = tf[:n], sig[:n], raw[:n], yaw[:n]

    # ---- precondition: how much of forward is yaw ----------------------
    corr_pear = float(np.corrcoef(sig, yaw)[0, 1])
    ra = np.argsort(np.argsort(sig))
    rb = np.argsort(np.argsort(yaw))
    corr_spear = float(np.corrcoef(ra, rb)[0, 1])
    # the part of the forward signal that the yaw signal does not explain
    beta = float(np.dot(sig, yaw) / max(np.dot(yaw, yaw), 1e-12))
    resid = sig - beta * yaw

    t_sw, y_sw, _, cam = p074.load(video)
    sw = p074.swings(t_sw, y_sw, cam, p074.MIN_SWING)
    hum = p078.human_turns(video)

    rows = []
    for s in sw:
        if s["t0"] - W_S < t[0] or s["t1"] + W_S > t[-1]:
            continue
        b = win_mean(t, sig, s["t0"] - W_S, s["t0"])
        d = win_mean(t, sig, s["t0"], s["t1"])
        a = win_mean(t, sig, s["t1"], s["t1"] + W_S)
        if not np.isfinite([b, d, a]).all():
            continue
        br = win_mean(t, resid, s["t0"] - W_S, s["t0"])
        dr = win_mean(t, resid, s["t0"], s["t1"])
        ar = win_mean(t, resid, s["t1"], s["t1"] + W_S)
        rows.append({
            "t0": s["t0"], "t1": s["t1"], "duration": s["t1"] - s["t0"],
            "delta": s["delta"], "direction": s["direction"],
            "forward_before": b, "forward_during": d, "forward_after": a,
            "forward_drop": d - (b + a) / 2.0,
            "forward_recovery": a - d,
            "forward_ratio": d / max((b + a) / 2.0, 1e-9) if (b + a) > 0 else np.nan,
            "resid_before": br, "resid_during": dr, "resid_after": ar,
            "resid_drop": dr - (br + ar) / 2.0,
            "real_turn": any(overlap(h["t0"], h["t1"], s["t0"], s["t1"]) > 0
                             for h in hum),
        })
    return {"rows": rows, "corr_pear": corr_pear, "corr_spear": corr_spear,
            "beta": beta, "t": t, "sig": sig, "yaw": yaw, "resid": resid}


def auc_of(rows: list[dict], key: str) -> tuple[float, float]:
    P = np.array([r[key] for r in rows if r["real_turn"]], float)
    N = np.array([r[key] for r in rows if not r["real_turn"]], float)
    a = rank_auc(P, N)
    if not np.isfinite(a):
        return float("nan"), float("nan")
    return a, max(a, 1 - a)


def pareto(rows: list[dict], call) -> tuple[float, float]:
    real = [r for r in rows if r["real_turn"]]
    fake = [r for r in rows if not r["real_turn"]]
    found = sum(1 for r in real if call(r)) / max(len(real), 1)
    nf = sum(1 for r in fake if call(r))
    nr = sum(1 for r in real if call(r))
    return found, nf / max(nr, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("P07.10 — добавляет ли forward_signal информацию к длительности")
    print(f"  окна: {W_S:.0f} с до, сам отрезок, {W_S:.0f} с после")
    print("  траектория не строится, P07.4/7/8/9 не тронуты")
    print()

    data = {v: build(v) for v in VIDEOS}

    # ---- step 0: the precondition ---------------------------------------
    print("=== ПРЕДУСЛОВИЕ: НЕЗАВИСИМОСТЬ ===")
    print("  forward-набор: DNp17 R x4, DNa07 L, DNp20 R, DNbe001 R")
    print("  yaw-набор:     DNp17 L/R, DNa07 L/R, DNp26 L/R, DNp20 L/R (разность)")
    print("  4 из 7 клеток forward лежат внутри yaw-набора; DNp17 R — больше")
    print("  половины веса forward, и она же входит в разность yaw с минусом.")
    print()
    print(f"  {'ролик':>9s} {'Пирсон':>8s} {'Спирмен':>9s}")
    for v in VIDEOS:
        print(f"  {v:>9s} {data[v]['corr_pear']:8.3f} {data[v]['corr_spear']:9.3f}")
    print()
    indep_ok = all(abs(data[v]["corr_pear"]) < 0.5 for v in VIDEOS)
    print(f"  ПРЕДУСЛОВИЕ НЕЗАВИСИМОСТИ: {'выполнено' if indep_ok else 'НЕ ВЫПОЛНЕНО'}")
    if not indep_ok:
        print("  forward_signal на три четверти повторяет yaw_signal. Это не второй")
        print("  независимый источник, а в основном тот же сигнал поворота.")
    print()

    report = {"window_s": W_S, "auc_min": AUC_MIN,
              "independence": {v: {"pearson": data[v]["corr_pear"],
                                   "spearman": data[v]["corr_spear"],
                                   "beta_yaw_regressed_out": data[v]["beta"]}
                               for v in VIDEOS},
              "independence_ok": bool(indep_ok)}

    # ---- step 5: forward_drop alone --------------------------------------
    print("=== ШАГ 5: ТОЛЬКО forward_drop ===")
    print("  гипотеза: при повороте поступательное движение замедляется,")
    print("  то есть forward_drop отрицателен внутри TURN")
    print()
    print(f"  {'признак':>20s} {'VID00001':>9s} {'VID00002':>9s} {'медиана в TURN':>15s} "
          f"{'вне':>8s}")
    keys = ["forward_drop", "forward_before", "forward_during", "forward_after",
            "forward_recovery"]
    aucs = {}
    for k in keys:
        row, med = {}, {}
        for v in VIDEOS:
            rows = data[v]["rows"]
            a, eff = auc_of(rows, k)
            row[v] = eff
            med[v] = float(np.median([r[k] for r in rows if r["real_turn"]]))
            med[v + "_out"] = float(np.median([r[k] for r in rows if not r["real_turn"]]))
        aucs[k] = row
        print(f"  {k:>20s} {row['VID00001']:9.3f} {row['VID00002']:9.3f} "
              f"{med['VID00001']:15.3f} {med['VID00001_out']:8.3f}")
    print()
    report["auc_forward"] = aucs

    # ---- test 2: with yaw regressed out ----------------------------------
    print("=== ТЕСТ 2: FORWARD С ВЫЧТЕННЫМ YAW ===")
    print("  из forward_signal линейно исключён yaw_signal; остаётся только то,")
    print("  чего в yaw нет. Линейное исключение снимает линейную часть связи.")
    print()
    print(f"  {'признак':>20s} {'VID00001':>9s} {'VID00002':>9s}")
    aucs_r = {}
    for k in ("resid_drop", "resid_during", "resid_after"):
        row = {}
        for v in VIDEOS:
            _, eff = auc_of(data[v]["rows"], k)
            row[v] = eff
        aucs_r[k] = row
        print(f"  {k:>20s} {row['VID00001']:9.3f} {row['VID00002']:9.3f}")
    print()
    report["auc_forward_yaw_removed"] = aucs_r

    # ---- duration, for reference -----------------------------------------
    dur_row = {v: auc_of(data[v]["rows"], "duration")[1] for v in VIDEOS}
    print("=== ДЛИТЕЛЬНОСТЬ ДЛЯ СРАВНЕНИЯ ===")
    print(f"  на этих же окнах: VID00001 {dur_row['VID00001']:.3f}, "
          f"VID00002 {dur_row['VID00002']:.3f}")
    print("  (в P07.8 на одиночных качаниях было 0.782 / 0.776)")
    print()
    report["auc_duration_here"] = dur_row

    fd_pass = all(aucs["forward_drop"][v] >= AUC_MIN for v in VIDEOS)
    print(f"  условие перехода к шагу 6 (AUC >= {AUC_MIN:.2f} на обоих): "
          f"{'выполнено' if fd_pass else 'НЕ выполнено'}")
    print()
    report["forward_drop_passes"] = bool(fd_pass)

    # ---- step 6: does it improve on duration ----------------------------
    print("=== ШАГ 6-8: ПОМОГАЕТ ЛИ К ДЛИТЕЛЬНОСТИ ===")
    print("  порог выбирается ТОЛЬКО на VID00001, затем переносится на VID00002")
    print("  отложенный ролик не используется для выбора ни одной точки")
    print()

    def best_on_tune(rows, keys, flips):
        """Threshold chosen on the tuning clip by Youden's J. Nothing else decides.

        The grid is nested: for a two-feature rule it always contains the point where the
        second feature is switched off, which is exactly the one-feature rule. Without that
        point the combined rule would be forced to use the second feature and would be
        scored on parts of the data it should be free to ignore, and the comparison against
        the one-feature rule would be unfair in the wrong direction.
        """
        ps = np.percentile([r[keys[0]] for r in rows], np.arange(5, 100, 5))
        if len(keys) == 1:
            cands = [("one", (p,), (flips[0],)) for p in ps]
        else:
            qs = list(np.percentile([r[keys[1]] for r in rows], np.arange(5, 100, 5)))
            # q = +inf with the sign that keeps everything is the "second feature off" point
            off = float("inf") if flips[1] else -float("inf")
            cands = [("off", (p, off), (flips[0], flips[1])) for p in ps]
            cands += [("two", (p, q), (flips[0], flips[1])) for p in ps for q in qs]
        best = None
        for kind, vs, fs in cands:
            def call(r, vs=vs, fs=fs):
                ok = True
                for key, v, fl in zip(keys, vs, fs):
                    ok = ok and ((r[key] < v) if fl else (r[key] > v))
                return ok
            tp = np.mean([call(r) for r in rows if r["real_turn"]])
            tn = np.mean([not call(r) for r in rows if not r["real_turn"]])
            j = tp + tn - 1
            if best is None or j > best["j"]:
                best = {"j": float(j), "keys": keys, "values": [float(v) for v in vs],
                        "flips": list(fs), "call": call, "used_second": kind == "two"}
        return best

    def apply(rows, rule):
        real = [r for r in rows if r["real_turn"]]
        fake = [r for r in rows if not r["real_turn"]]
        found = sum(1 for r in real if rule["call"](r))
        nf = sum(1 for r in fake if rule["call"](r))
        tot = sum(abs(r["delta"]) for r in real) or 1e-9
        tot_f = sum(abs(r["delta"]) for r in fake) or 1e-9
        return {"found_frac": found / max(len(real), 1),
                "false_per_real": nf / max(found, 1),
                "true_kept": sum(abs(r["delta"]) for r in real if rule["call"](r)) / tot,
                "false_kept": sum(abs(r["delta"]) for r in fake if rule["call"](r)) / tot_f,
                "found": found, "real_total": len(real), "false": nf, "j_tune": rule["j"]}

    rows_t = data[TUNE]["rows"]
    rules = {
        "только длительность": best_on_tune(rows_t, ["duration"], (False,)),
        "duration + forward_drop": best_on_tune(rows_t, ["duration", "forward_drop"],
                                                (False, True)),
        "duration + resid_drop": best_on_tune(rows_t, ["duration", "resid_drop"],
                                              (False, True)),
    }
    print(f"  {'правило':>24s} {'порог на VID00001':>44s} {'J':>6s} "
          f"{'взял 2-й?':>10s}")
    res = {}
    for name, rule in rules.items():
        thr = ", ".join(("не использован" if not np.isfinite(v)
                         else f"{k}{'<' if f else '>'} {v:.3f}")
                        for k, v, f in zip(rule["keys"], rule["values"], rule["flips"]))
        a = apply(data[TUNE]["rows"], rule)
        b = apply(data["VID00002"]["rows"], rule)
        res[name] = {TUNE: a, "VID00002": b}
        print(f"  {name:>24s} {thr:>44s} {rule['j']:6.3f} "
              f"{('да' if rule['used_second'] else 'нет'):>10s}")
    print()
    print("  развёрнуто, при одном и том же пороге на обоих роликах:")
    print(f"  {'правило':>24s} {'ролик':>9s} {'найдено':>9s} {'ложных на настоящий':>20s} "
          f"{'сохранено настоящего':>21s} {'ложного':>9s}")
    for name, per in res.items():
        for v in VIDEOS:
            s = per[v]
            print(f"  {name:>24s} {v:>9s} {s['found_frac'] * 100:8.0f}% "
                  f"{s['false_per_real']:20.2f} {s['true_kept'] * 100:20.0f}% "
                  f"{s['false_kept'] * 100:8.0f}%")
    print()
    report["step6"] = res
    report["step6_used_second_feature"] = {name: bool(r["used_second"])
                                           for name, r in rules.items()}

    if not any(r["used_second"] for r in rules.values()):
        print("  ни одно правило с добавкой не выбрало порог по второму признаку:")
        print("  на самом VID00001 второй признак оказался бесполезен, и правило")
        print("  сводится к одной длительности. Тогда сравнивать на VID00002 нечего.")
        report["step6_note"] = ("на VID00001 ни одно правило с добавкой не выбрало "
                                "второй признак: улучшения нет уже на ролике подбора")
    print()

    base = res["только длительность"]["VID00002"]
    for name in ("duration + forward_drop", "duration + resid_drop"):
        comb = res[name]["VID00002"]
        better_recall = comb["found_frac"] > base["found_frac"] + 0.03
        better_false = comb["false_per_real"] < base["false_per_real"] - 0.05
        worse_other = (comb["false_per_real"] > base["false_per_real"] + 0.05
                       or comb["found_frac"] < base["found_frac"] - 0.03)
        print(f"  {name} на VID00002: полнота "
              f"{'выше' if better_recall else 'не выше'}, ложных "
              f"{'меньше' if better_false else 'не меньше'}, "
              f"другой показатель {'хуже' if worse_other else 'на уровне'}")
        report.setdefault("step7", {})[name] = {
            "better_recall": bool(better_recall), "fewer_false": bool(better_false),
            "other_worse": bool(worse_other),
            "useful": bool((better_recall or better_false) and not worse_other)}
    print()

    # ---- verdict ---------------------------------------------------------
    useful = [k for k, v in report.get("step7", {}).items() if v["useful"]]
    improves = bool(useful)
    print("=== ИТОГ ===")
    print(f"  FORWARD ADDS INFORMATION: {'YES' if improves else 'NO'}")
    print(f"  AUC duration:            {dur_row['VID00001']:.3f} / {dur_row['VID00002']:.3f}")
    print(f"  AUC forward_drop:        {aucs['forward_drop']['VID00001']:.3f} / "
          f"{aucs['forward_drop']['VID00002']:.3f}")
    print(f"  AUC forward, yaw снят:   {aucs_r['resid_drop']['VID00001']:.3f} / "
          f"{aucs_r['resid_drop']['VID00002']:.3f}")
    b = res["только длительность"]["VID00002"]
    print(f"  duration+forward на VID00002 при пороге с VID00001: "
          f"найдено {res['duration + forward_drop']['VID00002']['found_frac'] * 100:.0f}% "
          f"против {b['found_frac'] * 100:.0f}%, ложных "
          f"{res['duration + forward_drop']['VID00002']['false_per_real']:.2f} "
          f"против {b['false_per_real']:.2f}")
    print()
    if not improves:
        print("  Добавка к длительности не улучшает результат на отложенном ролике")
        print("  одновременно по обоим показателям. Комбинации дальше не перебираются.")
    report["FORWARD_ADDS_INFORMATION"] = "YES" if improves else "NO"
    report["forward_independent_of_yaw"] = bool(indep_ok)
    report["conclusion"] = (
        "forward_signal не является независимым источником: корреляция с yaw_signal "
        f"{data['VID00001']['corr_pear']:+.2f} и {data['VID00002']['corr_pear']:+.2f}, "
        "потому что 4 из 7 клеток forward-набора входят в yaw-набор, а DNp17 R — больше "
        "половины веса forward. " +
        ("Добавка к длительности улучшения на отложенном ролике не дала."
         if not improves else "Добавка улучшила результат на отложенном ролике."))

    for v in VIDEOS:
        with (OUT / f"windows_{v}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[v]["rows"][0].keys()))
            w.writeheader()
            w.writerows(data[v]["rows"])

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, v in enumerate(VIDEOS):
        d = data[v]
        ax = axes[0][i]
        s = 300
        ax.scatter(d["yaw"][::s], d["sig"][::s], s=5, alpha=0.5,
                   c=["tab:green" if r else "0.55"
                      for r in (d["sig"][::s] > 0)][:len(d["yaw"][::s])])
        ax.set_xlabel("yaw_signal")
        ax.set_ylabel("forward_signal")
        ax.set_title(f"{v}: forward против yaw, Пирсон {d['corr_pear']:+.3f}\n"
                     "не независимые источники", fontsize=10)
        ax.grid(alpha=0.3)

        ax = axes[1][i]
        rows = d["rows"]
        P = [r["forward_drop"] for r in rows if r["real_turn"]]
        N = [r["forward_drop"] for r in rows if not r["real_turn"]]
        lo, hi = min(P + N), max(P + N)
        bins = np.linspace(lo, hi, 22)
        ax.hist(P, bins=bins, alpha=0.72, color="tab:green",
                label=f"внутри поворота (n={len(P)})")
        ax.hist(N, bins=bins, alpha=0.65, color="0.6", label=f"вне (n={len(N)})")
        ax.axvline(0, color="k", lw=1, ls=":")
        a = aucs["forward_drop"][v]
        ax.set_xlabel("forward_drop = during - (before+after)/2")
        ax.set_ylabel("отрезков")
        ax.set_title(f"{v}: разделение {a:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("P07.10 — forward_signal против взгляда и поворота", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "forward_vs_turn.png", dpi=125)
    plt.close(fig)

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
