#!/usr/bin/env python3
"""
P07.9 — does "the camera came back" carry anything the duration does not?

The suspicion this stage has to clear first
-------------------------------------------
The idea is to measure how far the accumulated angle ends from where it started, rather
than how large the rotation was. The concern, raised before the experiment was written, is
that this is `net/(out+back)` under a new name, and the first job of the script is to test
that rather than to assume either way.

The algebra is worth stating, because it is stronger evidence than a correlation.

For a single swing the signal keeps one sign, so the accumulated angle is monotone and

    return_error = |end - start| = peak        hence    return_ratio = 1 always

The measure is therefore only defined on a pair of opposite-signed swings, which is exactly
the structure P07.4 built. On such a pair, with `out` and `back` the two magnitudes and
`net = |out - back|`,

    old    net / (out + back) = |out - back| / (out + back)
    new    return_error / peak = |out - back| / out

The numerator is the same quantity. The new measure differs only in what it divides by. A
high correlation is expected, and step 3 of the task stops the experiment if it exceeds
0.95.

A construction not tied to pairs is included as well, because it is the only version of the
idea that could escape that shared numerator. `return_window` measures the return over a
fixed window around each swing, `[t0 - w, t1 + w]`, without regard to whether anything was
paired with it.

Rule fixed before the numbers are seen
---------------------------------------
    correlation with old ratio > 0.95        -> STOP, it is the same feature
    otherwise, feature is useful only if on VID00002 it simultaneously raises the share of
    real turns found, lowers false events, and beats duration alone

Thresholds are fitted on VID00001 only; VID00002 is a held-out check. Nothing in P07.4,
P07.7 or P07.8 is modified and no trajectory is built.

Usage:
    PYTHONPATH=. python scripts/p079_return_to_start.py
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

OUT = ROOT / "output/p079"
VIDEOS = ("VID00001", "VID00002")
TUNE = "VID00001"

CORR_STOP = 0.95    # above this the new feature is the old one wearing a different name
W_S = 1.0           # window for the variant that does not depend on pairing


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


def pairs_of(sw: list[dict]) -> list[dict]:
    """Adjacent opposite-signed swings, the same pairing walk P07.4 performs."""
    out = []
    k = 0
    while k < len(sw) - 1:
        a, b = sw[k], sw[k + 1]
        if np.sign(a["delta"]) == np.sign(b["delta"]):
            k += 1
            continue
        out.append({"a": a, "b": b})
        k += 2          # advance as P07.4 does, so pairings match exactly
    return out


def pair_measures(t: np.ndarray, yaw: np.ndarray, cam: np.ndarray, pr: dict) -> dict:
    a, b = pr["a"], pr["b"]
    out_, back = a["size"], b["size"]
    net = abs(a["delta"] + b["delta"])
    start = a["from"]
    end = b["to"]
    peak = float(np.max(np.abs(cam[a["i0"]:b["i1"] + 1] - start))) if b["i1"] > a["i0"] \
        else max(out_, net)
    peak = max(peak, 1e-9)
    return {
        "t0": a["t0"], "t1": b["t1"],
        "duration": b["t1"] - a["t0"],
        "out": out_, "back": back, "net": net,
        "peak": peak,
        "return_error": abs(end - start),
        "return_ratio": abs(end - start) / peak,
        "old_ratio": net / max(out_ + back, 1e-9),
        "direction": a["direction"],
        "delta_pair": a["delta"] + b["delta"],
    }


def window_measures(t: np.ndarray, yaw: np.ndarray, sw: list[dict]) -> list[dict]:
    """Return measured over a fixed window around each swing, no pairing involved."""
    dt = float(np.median(np.diff(t)))
    n_w = max(int(round(W_S / dt)), 1)
    cam = np.cumsum(yaw) * dt
    out = []
    for s in sw:
        i0, i1 = s["i0"], s["i1"]
        j1 = min(i1 + n_w, len(t) - 1)
        if i0 + n_w > len(t) - 1:
            continue
        start = cam[i0]
        seg = cam[i0:j1 + 1]
        peak = float(np.max(np.abs(seg - start)))
        end = cam[j1]
        out.append({"t0": s["t0"], "t1": s["t1"],
                    "duration": s["t1"] - s["t0"],
                    "peak": peak,
                    "return_ratio_win": abs(end - start) / max(peak, 1e-9),
                    "direction": s["direction"]})
    return out


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spear = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 \
        else float("nan")
    return pear, spear


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("P07.9 — несёт ли «камера вернулась» что-то помимо длительности")
    print("  шаг 3: если корреляция с net/(out+back) > 0.95 — остановиться")
    print()

    data = {}
    for v in VIDEOS:
        t, yaw, speed, cam = p074.load(v)
        sw = p074.swings(t, yaw, cam, p074.MIN_SWING)
        hum = p078.human_turns(v)
        prs = [pair_measures(t, yaw, cam, p) for p in pairs_of(sw)]
        for p in prs:
            p["real_turn"] = any(overlap(h["t0"], h["t1"], p["t0"], p["t1"]) > 0
                                 for h in hum)
        wns = window_measures(t, yaw, sw)
        for w in wns:
            w["real_turn"] = any(overlap(h["t0"], h["t1"], w["t0"], w["t1"]) > 0
                                 for h in hum)
        data[v] = {"swings": sw, "pairs": prs, "windows": wns,
                   "t": t, "yaw": yaw, "cam": cam, "hum": hum}

    # ---- the degeneracy of single swings --------------------------------
    print("=== ОДИНОЧНЫЕ КАЧАНИЯ: ПРИЗНАК ВЫРОЖДЕН ===")
    for v in VIDEOS:
        rr = []
        for s in data[v]["swings"]:
            peak = abs(s["delta"])
            if peak > 1e-9:
                rr.append(abs(s["delta"]) / peak)
        rr = np.array(rr)
        print(f"  {v}: {len(rr)} одиночных качаний, return_ratio "
              f"мин {rr.min():.4f}, медиана {np.median(rr):.4f}, макс {rr.max():.4f}")
    print("  внутри качания сигнал не меняет знак, поэтому накопленный угол монотонен")
    print("  и отношение всегда равно единице. Признак определён только на ПАРАХ,")
    print("  то есть на той же структуре, что построил P07.4.")
    print()

    # ---- step 3: is it the old feature renamed? --------------------------
    print("=== ШАГ 3: ЭТО НОВЫЙ ПРИЗНАК ИЛИ ПЕРЕИМЕНОВАННЫЙ СТАРЫЙ ===")
    print("  мера — ранговая: связь монотонная, но не линейная")
    print("  (делим на out против out+back, числитель один и тот же)")
    print()
    report = {"corr_stop": CORR_STOP, "window_s": W_S, "videos": {}}
    stop = False
    for v in VIDEOS:
        prs = data[v]["pairs"]
        rp = np.array([p["return_ratio"] for p in prs])
        op = np.array([p["old_ratio"] for p in prs])
        pear, spear = corr(rp, op)
        wns = data[v]["windows"]
        rw = np.array([w["return_ratio_win"] for w in wns])
        print(f"  {v}: пар {len(prs)}")
        print(f"    Спирмен {spear:.3f}   Пирсон {pear:.3f}"
              f"{'   <- выше порога' if spear > CORR_STOP else ''}")
        if spear > CORR_STOP or pear > CORR_STOP:
            stop = True
        report["videos"][v] = {"n_pairs": len(prs), "pearson": pear, "spearman": spear,
                               "n_windows": len(wns)}
    print()
    print("  алгебра, независимо от корреляции:")
    print("    старый  net/(out+back) = |out - back| / (out + back)")
    print("    новый   return_error/peak = |out - back| / out")
    print("    числитель ОДИН И ТОТ ЖЕ — |out - back|")
    print()

    report["same_numerator"] = "|out - back| в обоих признаках"
    report["single_swing_degenerate"] = True

    if stop:
        print("  ВЕРДИКТ ШАГА 3: ОСТАНОВКА")
        print("  Ранговая корреляция выше 0.95 на обоих роликах, и числитель у обоих")
        print("  признаков буквально один и тот же. «Камера вернулась» — это уже")
        print("  проверенный net/(out+back) с другим знаменателем. Нового признака нет.")
        report["verdict"] = "STOP: same feature as net/(out+back)"
        report["step3_stopped"] = True
    else:
        report["step3_stopped"] = False
        print("  ВЕРДИКТ ШАГА 3: корреляция ниже 0.95 — признак проверяется дальше")
    print()

    # ---- step 4: AUC, reported either way as evidence --------------------
    # Measuring is not fitting, so these numbers are reported even when step 3 stops: they
    # show whether the feature would have carried anything on its own. No threshold is
    # fitted unless step 3 passed.
    aucs = {}
    print("=== ШАГ 4: AUC (без подбора порогов) ===")
    print(f"  {'признак':>22s} {'VID00001':>10s} {'VID00002':>10s} {'единиц':>12s}")
    for key, src in (("return_ratio", "pairs"), ("duration", "pairs"),
                     ("old_ratio", "pairs"), ("return_ratio_win", "windows")):
        row = {}
        for v in VIDEOS:
            rows = data[v][src]
            P = np.array([r[key] for r in rows if r["real_turn"]])
            N = np.array([r[key] for r in rows if not r["real_turn"]])
            a = rank_auc(P, N)
            row[v] = max(a, 1 - a) if np.isfinite(a) else float("nan")
        aucs[key] = row
        n = len(data[VIDEOS[0]][src])
        print(f"  {key:>22s} {row['VID00001']:10.3f} {row['VID00002']:10.3f} "
              f"{n:>12d}")
    print()
    report["auc"] = aucs
    print("  для сравнения, P07.8 на одиночных качаниях: длительность 0.782 / 0.776")
    print("  здесь длительность на ПАРАХ ниже, потому что пара объединяет два качания")
    print()

    if not stop:
        # ---- step 6-7: does it add to duration on held-out clip ----------
        print("=== ШАГ 6-7: ДОБАВЛЯЕТ ЛИ ЧТО-ТО К ДЛИТЕЛЬНОСТИ ===")
        print("  пороги подбираются только на VID00001, VID00002 — отложенная проверка")
        print()

        def grid(rows_t, keys, pcts=(10, 20, 30, 40, 50, 60, 70, 80, 90)):
            best = None
            for p0 in np.percentile([r[keys[0]] for r in rows_t], pcts):
                combos = [(">", p0)]
                if len(keys) > 1:
                    for p1 in np.percentile([r[keys[1]] for r in rows_t], pcts):
                        combos.append(("&", p1))
                for c in combos:
                    if c[0] == ">":
                        call = lambda r, p0=p0: r[keys[0]] > p0
                    else:
                        call = lambda r, p0=p0, p1=c[1]: (r[keys[0]] > p0
                                                          and r[keys[1]] > p1)
                    tp = np.mean([call(r) for r in rows_t if r["real_turn"]])
                    tn = np.mean([not call(r) for r in rows_t if not r["real_turn"]])
                    j = tp + tn - 1
                    if best is None or j > best[1]:
                        best = (c, j, call)
            return best

        def scored(rows, call):
            real = [r for r in rows if r["real_turn"]]
            fake = [r for r in rows if not r["real_turn"]]
            tot = sum(abs(r["delta_pair"]) for r in real) or 1e-9
            tot_f = sum(abs(r["delta_pair"]) for r in fake) or 1e-9
            rk = sum(1 for r in real if call(r))
            fk = sum(1 for r in fake if call(r))
            return {"found_frac": rk / max(len(real), 1), "false_per_real": fk / max(rk, 1),
                    "true_kept": sum(abs(r["delta_pair"]) for r in real if call(r)) / tot,
                    "false_kept": sum(abs(r["delta_pair"]) for r in fake if call(r)) / tot_f,
                    "found": rk, "real_total": len(real), "false": fk}

        rows_t = data[TUNE]["pairs"]
        base = grid(rows_t, ["duration"])
        both = grid(rows_t, ["duration", "return_ratio"])
        print("  базовая линия — только длительность:")
        res_base = {}
        for v in VIDEOS:
            s = scored(data[v]["pairs"], base[2])
            res_base[v] = s
            print(f"    {v}: найдено {s['found_frac'] * 100:.0f}% настоящих, "
                  f"ложных на настоящий {s['false_per_real']:.2f}, "
                  f"сохранено настоящего {s['true_kept'] * 100:.0f}%, "
                  f"ложного {s['false_kept'] * 100:.0f}%")
        print(f"    пороги на {TUNE}: длительность > {base[0][1]:.3f} (J = {base[1]:.3f})")
        print()
        print("  с добавлением return_ratio:")
        res_both = {}
        for v in VIDEOS:
            s = scored(data[v]["pairs"], both[2])
            res_both[v] = s
            print(f"    {v}: найдено {s['found_frac'] * 100:.0f}% настоящих, "
                  f"ложных на настоящий {s['false_per_real']:.2f}, "
                  f"сохранено настоящего {s['true_kept'] * 100:.0f}%, "
                  f"ложного {s['false_kept'] * 100:.0f}%")
        print(f"    пороги на {TUNE}: длительность > {both[0][1][1]:.3f} И "
              f"return_ratio > {both[0][1]:.3f} (J = {both[1]:.3f})")
        print()

        improved = (res_both["VID00002"]["found_frac"] >
                    res_base["VID00002"]["found_frac"]
                    and res_both["VID00002"]["false_per_real"] <
                    res_base["VID00002"]["false_per_real"])
        print(f"  на VID00002 одновременно выше полнота и меньше ложных: "
              f"{'ДА' if improved else 'НЕТ'}")
        if not improved:
            print("  признак не признаётся полезным: улучшения по обоим пунктам нет")
        report["step7"] = {"base": res_base, "with_return": res_both,
                           "improves_on_holdout": bool(improved)}
        report["thresholds"] = {"duration": float(base[0][1]),
                                "duration_plus_return": [float(x) for x in both[0][1]]}

    # ---- write rows ------------------------------------------------------
    for v in VIDEOS:
        with (OUT / f"pairs_{v}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[v]["pairs"][0].keys()))
            w.writeheader()
            w.writerows(data[v]["pairs"])
        with (OUT / f"windows_{v}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[v]["windows"][0].keys()))
            w.writeheader()
            w.writerows(data[v]["windows"])

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, v in enumerate(VIDEOS):
        prs = data[v]["pairs"]
        rp = np.array([p["return_ratio"] for p in prs])
        op = np.array([p["old_ratio"] for p in prs])
        ax = axes[0][i]
        ax.scatter(op, rp, s=16, alpha=0.7, c=["tab:green" if p["real_turn"]
                                               else "0.6" for p in prs])
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="совпадали бы точно")
        pe, sp = report["videos"][v]["pearson"], report["videos"][v]["spearman"]
        ax.set_xlabel("старый net/(out+back)")
        ax.set_ylabel("новый return_error/peak")
        ax.set_title(f"{v}: {len(prs)} пар, Пирсон {pe:.3f}, Спирмен {sp:.3f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1][i]
        P = [p["return_ratio"] for p in prs if p["real_turn"]]
        N = [p["return_ratio"] for p in prs if not p["real_turn"]]
        hi = max(max(P), max(N)) if P or N else 1.0
        bins = np.linspace(0, hi, 24)
        ax.hist(P, bins=bins, alpha=0.72, color="tab:green",
                label=f"внутри поворота (n={len(P)})")
        ax.hist(N, bins=bins, alpha=0.65, color="0.6", label=f"вне (n={len(N)})")
        ax.set_xlabel("return_error / peak")
        ax.set_ylabel("пар")
        ax.set_title(f"{v}: зелёные = настоящий поворот, серые = нет", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("P07.9 — «камера вернулась» против net/(out+back)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "return_vs_old.png", dpi=125)
    plt.close(fig)

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
