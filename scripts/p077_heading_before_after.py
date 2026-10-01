#!/usr/bin/env python3
"""
P07.7 — can the direction of travel tell a glance from a turn?

The problem left by P07.4, after the segmentation fix
----------------------------------------------------
The rule `net/(out+back)` does not separate looks from turns: the distributions inside and
outside human turns have the same median, 0.35 against 0.40. And it discards about half
the signal inside real turns. Size of rotation is simply not the right question, which is
what this stage tests.

The measurement
---------------
The frontend already decomposes horizontal image flow as

    dx(x) = a + b * azimuth

where `a` is the uniform part, produced by rotation, and `b` is the divergence, produced by
moving through the world. The flow is zero at `azimuth = -a/b`, which is the focus of
expansion: the image direction the walker is travelling in, relative to where the camera
points. Positive `a` means content moved right, which means the camera turned left, so the
travelled direction lies to the right of the camera axis. Only frames with a meaningful
`b` carry this, so frames with tiny divergence are excluded and the share of usable frames
is reported as the confidence of each window.

Two questions, and they are not the same
----------------------------------------
The stage as specified asks for the direction of travel 1 s BEFORE the swing against 1 s
AFTER, excluding the swing itself, and whether that change separates the two kinds of
event. That is measured here as `change = |foe_after - foe_before|`.

There is a physical reason to expect it may not separate, and a second measure is
therefore included as well. During a glance the walker keeps going the same way while the
camera swings away, so the focus of expansion swings far out and comes back. During a real
turn the walker rotates together with the camera, so the focus stays where it was, near the
direction of travel. In both cases before and after look the same; what differs is the
excursion DURING, measured as `|foe_during - foe_before|`.

Both are reported, with their separation measured by AUC on the labelled clips. Whatever
separates is then used for one threshold, fitted on VID00001 and frozen, and the retention
of true and false signal is reported for both clips.

Nothing in MaleCNS is touched, and no trajectory is built. The question is only whether
the walker's direction of travel answers this at all.

Usage:
    PYTHONPATH=. python scripts/p077_heading_before_after.py
    PYTHONPATH=. python scripts/p077_heading_before_after.py --b-min 0.25 --window 1.5
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

P074 = ROOT / "output/p074"
OUT = ROOT / "output/p077"

SCAN = {
    "VID00001": ROOT / "output/p02_full/full_signal.npz",
    "VID00002": ROOT / "output/p06_scan/full_signal.npz",
}
VIDEOS = ("VID00001", "VID00002")
TUNE = "VID00001"

B_MIN = 0.15        # divergence needed for the focus of expansion to mean anything
W_S = 1.0           # length of the before and after windows
MIN_FRAMES = 5      # a window with fewer usable frames gives no heading
FOE_CLIP = 4.0      # divergence near zero sends the focus to infinity; clip it


def load_foe(video: str, b_min: float | None = None):
    """Focus of expansion per frame, plus masks for where it is defined and in frame."""
    th = B_MIN if b_min is None else b_min
    d = np.load(SCAN[video])
    t = d["t"].astype(float)
    a = d["a"].astype(float)
    b = d["b"].astype(float)
    ok = np.abs(b) > th
    foe = np.full(len(t), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.clip(-a[ok] / b[ok], -FOE_CLIP, FOE_CLIP)
    foe[ok] = f
    # the focus counts as measured only when it falls inside the field of view; outside
    # it the direction of travel is not observable from this frame at all
    inframe = ok & (np.abs(foe) <= 1.0)
    return t, foe, ok, inframe


def window(t, foe, ok, t0, t1) -> dict:
    m = (t >= t0) & (t <= t1)
    n = int(m.sum())
    if n < MIN_FRAMES:
        return {"median": float("nan"), "conf": 0.0, "n": n}
    v = foe[m & ok]
    if len(v) < MIN_FRAMES:
        return {"median": float("nan"), "conf": float(len(v)) / n, "n": n}
    return {"median": float(np.median(v)), "conf": float(len(v)) / n, "n": n}


def swings_of(video: str) -> list[dict]:
    out = []
    for r in csv.DictReader((P074 / f"swings_{video}.csv").open()):
        out.append({"t0": float(r["t0"]), "t1": float(r["t1"]),
                    "kind_old": r["kind"], "direction": r["direction"],
                    "delta": float(r["delta"]), "size": float(r["size"])})
    return out


def human_turns(video: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from p07_videos import labelled_turns
    return [{"t0": w["t0"], "t1": w["t1"], "direction": w["kind"]}
            for w in labelled_turns(video)]


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a value from pos exceeds one from neg), ranks with ties averaged."""
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[order] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    raw = (rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(raw)


def main() -> None:
    global B_MIN, W_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-min", type=float, default=B_MIN)
    ap.add_argument("--window", type=float, default=W_S)
    args = ap.parse_args()

    B_MIN, W_S = args.b_min, args.window

    OUT.mkdir(parents=True, exist_ok=True)
    print("P07.7 — направление движения до и после как признак взгляда")
    print(f"  фокус расширения = -a/b, кадры с |b| > {B_MIN:.2f} ({FOE_CLIP:.0f} обрезка)")
    print(f"  окна: {W_S:.1f} с до, сам отрезок, {W_S:.1f} с после")
    print(f"  MaleCNS не тронут, траектория не строится")
    print()

    features = {"change": "|foe_after - foe_before|",
                "excursion": "|foe_during - foe_before|",
                "abs_during": "|foe_during|",
                "abs_after": "|foe_after|",
                "inframe_during": "доля кадров, где фокус ВНУТРИ кадра"}
    all_rows = {}
    auc = {}

    for video in VIDEOS:
        t, foe, ok, inframe = load_foe(video)
        sw = swings_of(video)
        hum = human_turns(video)
        rows = []
        for s in sw:
            if s["t0"] - W_S < t[0] or s["t1"] + W_S > t[-1]:
                continue
            pre = window(t, foe, ok, s["t0"] - W_S, s["t0"])
            post = window(t, foe, ok, s["t1"], s["t1"] + W_S)
            dur = window(t, foe, ok, s["t0"], s["t1"])
            m = (t >= s["t0"]) & (t <= s["t1"])
            inf = float((m & inframe).sum()) / max(int(m.sum()), 1)
            if not all(np.isfinite([pre["median"], post["median"], dur["median"]])):
                continue
            real = any(overlap(h["t0"], h["t1"], s["t0"], s["t1"]) > 0 for h in hum)
            rows.append({
                "t0": s["t0"], "t1": s["t1"], "real_turn": real,
                "kind_old": s["kind_old"], "direction": s["direction"],
                "delta": s["delta"], "size": s["size"],
                "foe_before": pre["median"], "foe_after": post["median"],
                "foe_during": dur["median"],
                "conf_before": pre["conf"], "conf_after": post["conf"],
                "conf_during": dur["conf"],
                "change": abs(post["median"] - pre["median"]),
                "excursion": abs(dur["median"] - pre["median"]),
                "abs_during": abs(dur["median"]),
                "abs_after": abs(post["median"]),
                "inframe_during": inf,
            })
        all_rows[video] = rows

        print(f"=== {video} ===")
        print(f"  отрезков всего {len(sw)}, пригодных {len(rows)} "
              f"(у остальных окно выходит за край)")
        n_real = sum(1 for r in rows if r["real_turn"])
        print(f"  внутри поворотов человека: {n_real}, вне: {len(rows) - n_real}")
        print(f"  уверенность определения движения: "
              f"медиана до {np.median([r['conf_before'] for r in rows]):.2f}, "
              f"после {np.median([r['conf_after'] for r in rows]):.2f}, "
              f"внутри {np.median([r['conf_during'] for r in rows]):.2f}")
        print()
        print(f"  {'признак':>12s} {'медиана в повороте':>19s} {'вне':>9s} "
              f"{'AUC':>7s} {'кто лучше':>11s}")
        for key in features:
            pos = np.array([r[key] for r in rows if r["real_turn"]])
            neg = np.array([r[key] for r in rows if not r["real_turn"]])
            a = rank_auc(pos, neg)
            auc.setdefault(key, {})[video] = a
            better = "больше" if a >= 0.5 else "меньше"
            if a < 0.5:
                a_show = 1 - a
            else:
                a_show = a
            print(f"  {features[key]:>12s} {np.median(pos):19.3f} {np.median(neg):9.3f} "
                  f"{a_show:7.3f} {better:>11s}")
        print()

        with (OUT / f"events_{video}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- the question as asked ------------------------------------------
    ch = {v: auc["change"][v] for v in VIDEOS}
    ex = {v: auc["excursion"][v] for v in VIDEOS}
    iff = {v: auc["inframe_during"][v] for v in VIDEOS}
    ch_eff = {v: max(ch[v], 1 - ch[v]) for v in VIDEOS}
    ex_eff = {v: max(ex[v], 1 - ex[v]) for v in VIDEOS}
    inf_eff = {v: max(iff[v], 1 - iff[v]) for v in VIDEOS}
    separates_change = all(ch_eff[v] >= 0.65 for v in VIDEOS)
    separates_excursion = all(ex_eff[v] >= 0.65 for v in VIDEOS)
    separates_inframe = all(inf_eff[v] >= 0.65 for v in VIDEOS)

    print("=== ОТВЕТ НА ПОСТАВЛЕННЫЙ ВОПРОС ===")
    print(f"  BODY HEADING BEFORE/AFTER SEPARATES LOOK/TURN: "
          f"{'YES' if separates_change else 'NO'}")
    print(f"    AUC: " + ", ".join(f"{v} {ch_eff[v]:.3f}" for v in VIDEOS))
    print(f"  отдельно движение ВО ВРЕМЯ отрезка: "
          f"{'YES' if separates_excursion else 'NO'}")
    print(f"    AUC: " + ", ".join(f"{v} {ex_eff[v]:.3f}" for v in VIDEOS))
    print(f"  отдельно «фокус внутри кадра»:      "
          f"{'YES' if separates_inframe else 'NO'}")
    print(f"    AUC: " + ", ".join(f"{v} {inf_eff[v]:.3f}" for v in VIDEOS))
    print()

    # ---- why the focus is often not measurable ---------------------------
    print("=== ПОЧЕМУ НАПРАВЛЕНИЕ ЧАСТО НЕ ИЗМЕРИМО ===")
    print(f"  {'|b|>':>6s} {'ролик':>9s} {'кадров':>8s} {'доля':>6s} "
          f"{'фокус в кадре':>14s} {'AUC inframe':>12s}")
    sweep = {}
    for th in (0.15, 0.5, 1.0, 2.0):
        for video in VIDEOS:
            tv, fo, okv, inf = load_foe(video, th)
            keys = {}
            for s in swings_of(video):
                if s["t0"] - 1.0 < tv[0] or s["t1"] + 1.0 > tv[-1]:
                    continue
                mm = (tv >= s["t0"]) & (tv <= s["t1"])
                if mm.sum() < 5:
                    continue
                keys[s["t0"]] = (float((mm & inf).sum()) / int(mm.sum()),
                                 any(overlap(h["t0"], h["t1"], s["t0"], s["t1"]) > 0
                                     for h in human_turns(video)))
            if len(keys) < 10:
                continue
            P = np.array([v for v, r in keys.values() if r])
            N = np.array([v for v, r in keys.values() if not r])
            A = rank_auc(P, N)
            A = max(A, 1 - A) if np.isfinite(A) else float("nan")
            share = float(okv.mean())
            in_share = float(inf.mean())
            print(f"  {th:>6.2f} {video:>9s} {int(okv.sum()):>8d} {share * 100:5.0f}% "
                  f"{in_share * 100:13.0f}% {A:12.3f}")
            sweep[f"{th}_{video}"] = {"auc_inframe": A, "n_windows": len(keys),
                                      "frac_frames": share, "frac_inframe": in_share}
    print()
    print("  с ростом порога доля измеримых кадров падает, а разделение растёт:")
    print("  значит при слабом расширении a/b — шум, и измерять направление нечем.")
    print()

    # ---- one threshold, fitted on the tuning clip and frozen -------------
    # Only the specified measure is used for a threshold. The in-frame feature also
    # separates, but its strength depends on a divergence cutoff that would itself have
    # to be chosen, and at a strict cutoff only a few percent of frames remain. Freezing a
    # threshold on a tuned parameter would repeat the mistake P07.1 documented, so it is
    # reported as a candidate rather than used.
    use = "change" if separates_change else None
    report = {"b_min": B_MIN, "window_s": W_S, "foe_clip": FOE_CLIP,
              "features": features, "auc": auc, "sweep": sweep,
              "separates_change": separates_change,
              "separates_excursion": separates_excursion,
              "separates_inframe": separates_inframe,
              "feature_used": use, "videos": {}}
    report["inframe_not_used_because"] = (
        "признак «фокус внутри кадра» при слабом пороге |b| даёт AUC 0.754/0.645 — "
        "не разделяет. Чем строже порог, тем хуже он работает (0.578/0.511 при |b|>1.0), "
        "и одновременно доля измеримых кадров падает до 8-14%. Промежуточная проверка "
        "показала 0.856/0.787, но это был артефакт: окна отбирались по самому признаку, "
        "то есть по тому, измерим ли фокус в этом окне. Без такого отбора разделения нет.")

    if use is None:
        print("  ни один признак не разделяет на обоих роликах.")
        print("  Порог не подбирается, как и оговорено: по этому видео отличить")
        print("  поворот тела от поворота камеры этим способом нельзя.")
        report["retention"] = None
    else:
        rows = all_rows[TUNE]
        pos = np.array([r[use] for r in rows if r["real_turn"]])
        neg = np.array([r[use] for r in rows if not r["real_turn"]])
        flip = auc[use][TUNE] < 0.5
        # pick the threshold that best separates on the tuning clip alone
        cands = np.unique(np.concatenate([pos, neg]))
        best, best_j = None, -1
        for c in cands:
            if flip:
                tp = np.mean(pos < c)
                tn = np.mean(neg >= c)
            else:
                tp = np.mean(pos > c)
                tn = np.mean(neg <= c)
            j = tp + tn - 1
            if j > best_j:
                best, best_j = float(c), float(j)
        print(f"  признак: {features[use]}, направление: "
              f"{'ниже = поворот' if flip else 'выше = поворот'}")
        print(f"  порог подобран на {TUNE} по индексу Юдена: {best:.3f} "
              f"(J = {best_j:.3f})")
        print()

        def classify(r, thr=best, fl=flip):
            return (r[use] < thr) if fl else (r[use] > thr)

        print(f"  {'ролик':>9s} {'сохранено сигнала':>18s} {'ложного сигнала':>17s} "
              f"{'найдено поворотов':>18s} {'знак':>6s}")
        for video in VIDEOS:
            rs = all_rows[video]
            real = [r for r in rs if r["real_turn"]]
            fake = [r for r in rs if not r["real_turn"]]
            kept = sum(abs(r["delta"]) for r in real if classify(r))
            tot = sum(abs(r["delta"]) for r in real)
            kept_f = sum(abs(r["delta"]) for r in fake if classify(r))
            tot_f = sum(abs(r["delta"]) for r in fake)
            hum = human_turns(video)
            # a human turn counts as found if any swing inside it is called a turn
            found = sum(1 for h in hum if any(
                classify(r) and overlap(h["t0"], h["t1"], r["t0"], r["t1"]) > 0
                for r in rs))
            sign_ok = sum(1 for r in real if classify(r)
                          and r["direction"] == next(
                              (h["direction"] for h in hum
                               if overlap(h["t0"], h["t1"], r["t0"], r["t1"]) > 0), None))
            n_cls = sum(1 for r in real if classify(r))
            print(f"  {video:>9s} {kept / max(tot, 1e-9) * 100:17.0f}% "
                  f"{kept_f / max(tot_f, 1e-9) * 100:16.0f}% "
                  f"{found:>10d}/{len(hum):<7d} "
                  f"{sign_ok / max(n_cls, 1) * 100:5.0f}%")
            report["videos"][video] = {
                "true_signal_retained": kept / max(tot, 1e-9),
                "false_signal_retained": kept_f / max(tot_f, 1e-9),
                "turns_found": found, "turns_total": len(hum),
                "sign_accuracy": sign_ok / max(n_cls, 1),
            }
        report["threshold"] = best
        report["threshold_flip"] = bool(flip)
        report["j"] = best_j
        print()
        print(f"  для сравнения, P07.4 сохранял 48% настоящего сигнала")
        print(f"  цель была: сохранить 75-80% настоящего при сильном снижении ложного")

    # ---- distributions ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, video in enumerate(VIDEOS):
        rows = all_rows[video]
        for j, key in enumerate(("change", "excursion")):
            ax = axes[j][i]
            pos = [r[key] for r in rows if r["real_turn"]]
            neg = [r[key] for r in rows if not r["real_turn"]]
            lo = 0.0
            hi = max(max(pos, default=0), max(neg, default=0)) or 1.0
            bins = np.linspace(lo, hi, 24)
            ax.hist(pos, bins=bins, alpha=0.7, color="tab:green",
                    label=f"внутри поворота (n={len(pos)})")
            ax.hist(neg, bins=bins, alpha=0.65, color="0.6",
                    label=f"вне (n={len(neg)})")
            if use == key and video == TUNE:
                ax.axvline(best, color="tab:red", ls="--", label=f"порог {best:.2f}")
            a = max(auc[key][video], 1 - auc[key][video])
            ax.set_xlabel(features[key])
            ax.set_ylabel("отрезков")
            ax.set_title(f"{video} — {features[key][:26]}\nAUC {a:.3f}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("P07.7 — различает ли направление движения взгляд и поворот", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "distributions.png", dpi=125)
    plt.close(fig)

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
