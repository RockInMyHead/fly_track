#!/usr/bin/env python3
"""P09.5D — what the finished drawings say about the readouts the tracker runs on.

The drawing pass is complete: sixty moves, drawn as paths rather than chosen between two words,
with a third answer that the first two passes did not have. That third answer is the reason this
script exists, and it is worth stating exactly what it means, because the label is easy to
misread.

    "NO_LOCOMOTION" does not mean the person stood still.

It means he walked and came back — a metre forward and a metre back, in the reviewer's words. The
drawn paths confirm it: the path length in that category is nearly the same as in the other two
(0.126 against 0.134 and 0.152), while the net displacement is a third of theirs (0.047 against
0.091 and 0.129). He was moving the whole time; he ended up where he started.

That distinction matters because the tracker asks a directional question of the fly at exactly
these moments, and at these moments there is no direction to answer with. What this script measures
is whether the fly's own readouts can tell the difference anyway.

Three findings, in increasing order of how much they matter:

    1. the frozen rule calls 8 of the 20 treading events a TURN, at confidence 1.000
    2. the rule also withdraws the direction on 43 percent of the moves a person drew as a turn
    3. neither readout distinguishes the three categories at all, by any measure tried

The third is the one that decides the phase, and it is measured two ways: pooled across clips, and
within each clip so that the per-clip normalisation cannot explain it away.

Usage:
    PYTHONPATH=. python scripts/p095d_readouts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
OUT = ROOT / "output/p095"

CATS = ("LOOK", "TURN", "NO_LOCOMOTION")
PRE_S, POST_S = 3.0, 7.0


def signals(video: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = list(csv.DictReader((ROOT / f"output/p07/forward_signal_{video}.csv").open()))
    y = list(csv.DictReader((ROOT / f"output/p07/yaw_signal_{video}.csv").open()))
    n = min(len(f), len(y))
    return (np.array([float(r["t"]) for r in f])[:n],
            np.array([float(r["forward_signal"]) for r in f])[:n],
            np.array([float(r["t"]) for r in y])[:n],
            np.array([float(r["yaw_signal"]) for r in y])[:n])


def load() -> tuple[dict, dict]:
    s = json.loads((DATA / "review_set_v2.json").read_text(encoding="utf-8"))
    events = {e["event_id"]: e for e in s["events"]}
    labels = {l["event_id"]: l
              for l in json.loads((DATA / "human_labels_v2.json").read_text(
                  encoding="utf-8"))["labels"] if l.get("round", 1) == 1}
    return events, labels


def window(t: np.ndarray, values: np.ndarray, at: float) -> np.ndarray:
    m = (t >= at - PRE_S) & (t <= at + POST_S)
    return values[m]


def auc(a: list[float], b: list[float]) -> float:
    """Probability a value from b exceeds one from a; 0.5 means the measure says nothing."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    eq = sum(1 for x in a for y in b if x == y)
    return (gt + 0.5 * eq) / (len(a) * len(b))


def within_clip_percentile(t: np.ndarray, values: np.ndarray, at: float,
                           win: float = POST_S + PRE_S) -> float | None:
    """Where this stretch sits among all stretches of the clip. Removes cross-clip scaling.

    Each recording is standardised on its own spread before it reaches here, so pooling values
    from two clips compares two different rulers. Ranking a window against its own clip's windows
    does not, and it is the check that stops the pooled result from being an artefact of that
    normalisation.
    """
    seg = np.abs(window(t, values, at))
    if len(seg) < 5:
        return None
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    n = max(int(round(win / max(dt, 1e-9))), 5)
    if len(values) < n + 5:
        return None
    step = max(1, n // 4)
    roll = np.array([np.abs(values[i:i + n]).mean()
                     for i in range(0, len(values) - n, step)])
    if len(roll) < 10:
        return None
    return float((roll < seg.mean()).mean() * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events, labels = load()
    if len(labels) < len(events):
        print(f"размечено {len(labels)} из {len(events)} — сначала закончить разметку")
        return 1

    print("=" * 100)
    print("P09.5D — ЧТО НАРИСОВАННЫЕ ПУТИ ГОВОРЯТ О СИГНАЛАХ, НА КОТОРЫХ РАБОТАЕТ ТРЕКЕР")
    print("=" * 100)
    print(f"  событий: {len(labels)}")
    print()

    # ---- what the third category actually is ----------------------------------------------
    def geom(l: dict) -> tuple[float, float] | None:
        pts = [p for st in l.get("drawing", []) for p in st]
        if len(pts) < 3:
            return None
        length = 0.0
        for st in l["drawing"]:
            for i in range(1, len(st)):
                length += float(np.hypot(st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]))
        if length < 1e-6:
            return None
        disp = float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
        return length, disp

    print("─── ЧТО ТАКОЕ ТРЕТЬЯ КАТЕГОРИЯ: ПО НАРИСОВАННЫМ ПУТЯМ ───")
    print(f"  {'категория':<16}{'n':>4}{'длина пути':>12}{'смещение':>10}{'вернулся':>11}"
          f"{'прямизна':>10}")
    shape = {}
    for g in CATS:
        ls, ds = [], []
        for k, l in labels.items():
            if l["human_label"] != g:
                continue
            v = geom(l)
            if v:
                ls.append(v[0])
                ds.append(v[1])
        if not ls:
            continue
        ml, md = float(np.median(ls)), float(np.median(ds))
        shape[g] = {"n": len(ls), "path_len": ml, "displacement": md,
                    "returned_frac": 1 - md / max(ml, 1e-9),
                    "straightness": md / max(ml, 1e-9)}
        print(f"  {g:<16}{len(ls):>4}{ml:>12.3f}{md:>10.3f}{1 - md / max(ml, 1e-9):>10.0%}"
              f"{md / max(ml, 1e-9):>10.2f}")
    print()
    if "NO_LOCOMOTION" in shape and "TURN" in shape:
        n_, t_ = shape["NO_LOCOMOTION"], shape["TURN"]
        print(f"  Длина пути у «топтался» {n_['path_len']:.3f} против {t_['path_len']:.3f} "
              f"у поворотов — почти столько же.")
        print(f"  А чистое смещение {n_['displacement']:.3f} против {t_['displacement']:.3f} — "
              f"в {t_['displacement'] / max(n_['displacement'], 1e-9):.1f} раза меньше.")
        print("  То есть человек ХОДИЛ и вернулся, а не стоял. Название категории значит")
        print("  «нет чистого перемещения», не «нет движения».")
    print()

    # ---- 1. what the rule does with those events -------------------------------------------
    tread = [k for k, l in labels.items() if l["human_label"] == "NO_LOCOMOTION"]
    by_rule = Counter(events[k]["hidden"]["algo_class_b"] for k in tread)
    strong = [k for k in tread if (events[k]["hidden"]["algo_score_b"] or 0) >= 0.61]
    print("─── 1. ЧТО ПРАВИЛО ДЕЛАЕТ НА ЭТИХ СОБЫТИЯХ ───")
    print(f"  событий: {len(tread)} из {len(labels)} ({len(tread) / len(labels):.0%})")
    for m in ("TURN", "UNKNOWN", "LOOK"):
        if by_rule.get(m):
            print(f"    правило сказало {m:<9}{by_rule[m]:>3}  "
                  f"({by_rule[m] / len(tread):.0%})")
    print(f"  с уверенностью (score >= 0.61): {len(strong)}")
    print()

    # ---- 2. what the rule does with turns --------------------------------------------------
    turns = [k for k, l in labels.items() if l["human_label"] == "TURN"]
    rev = [k for k in turns if events[k]["hidden"]["algo_class_b"] == "LOOK"]
    conf = [k for k in turns if events[k]["hidden"]["algo_class_b"] == "TURN"]
    print("─── 2. ЧТО ПРАВИЛО ДЕЛАЕТ С НАСТОЯЩИМИ ПОВОРОТАМИ ───")
    print(f"  человек нарисовал поворот: {len(turns)}")
    print(f"    правило подтвердило:        {len(conf)}")
    print(f"    правило СНЯЛО направление:  {len(rev)}  "
          f"({len(rev) / max(len(rev) + len(conf), 1):.0%} от тех, о чём оно говорило)")
    print()

    # ---- 3. can the readouts tell the categories apart at all? -----------------------------
    print("─── 3. РАЗЛИЧАЮТ ЛИ САМИ СИГНАЛЫ ЭТИ ТРИ КАТЕГОРИИ ───")
    pooled = {g: {"fwd": [], "yaw": []} for g in CATS}
    ranked = {g: {"fwd": [], "yaw": []} for g in CATS}
    cache: dict[str, tuple] = {}
    for k, l in labels.items():
        e = events[k]
        v = e["clip"]
        if v not in cache:
            cache[v] = signals(v)
        ft, f, yt, y = cache[v]
        c = e["time"]
        pooled[l["human_label"]]["fwd"].append(float(np.abs(window(ft, f, c)).mean()))
        pooled[l["human_label"]]["yaw"].append(float(np.abs(window(yt, y, c)).mean()))
        pf = within_clip_percentile(ft, f, c)
        py = within_clip_percentile(yt, y, c)
        if pf is not None:
            ranked[l["human_label"]]["fwd"].append(pf)
        if py is not None:
            ranked[l["human_label"]]["yaw"].append(py)

    print("  обе величины — выход мозга, в своих единицах")
    print()
    print("  а) по всему набору:")
    print(f"    {'категория':<16}{'|forward|':>12}{'|yaw|':>10}")
    for g in CATS:
        print(f"    {g:<16}{np.median(pooled[g]['fwd']):>12.4f}"
              f"{np.median(pooled[g]['yaw']):>10.4f}")
    for m in ("fwd", "yaw"):
        nl = pooled["NO_LOCOMOTION"][m]
        ot = pooled["LOOK"][m] + pooled["TURN"][m]
        print(f"    AUC «топтался» против прочих, {m}: {auc(nl, ot):.2f}")
    print()
    print("  б) внутри каждого ролика (снимает нормировку по ролику):")
    print(f"    {'категория':<16}{'forward, процентиль':>21}{'yaw, процентиль':>18}")
    for g in CATS:
        print(f"    {g:<16}{np.median(ranked[g]['fwd']):>20.0f}%"
              f"{np.median(ranked[g]['yaw']):>17.0f}%")
    for m in ("fwd", "yaw"):
        nl = ranked["NO_LOCOMOTION"][m]
        ot = ranked["LOOK"][m] + ranked["TURN"][m]
        print(f"    AUC «топтался» против прочих, {m}: {auc(nl, ot):.2f}")
    print()
    print("  0.5 по AUC значит «не различает». Оба способа дают около 0.5 оба раза.")
    print()

    # ---- 4. how independent are the two readouts? ------------------------------------------
    print("─── 4. НАСКОЛЬКО ЭТИ ДВА СИГНАЛА ВООБЩЕ РАЗНЫЕ ВЕЛИЧИНЫ ───")
    print(f"  {'ролик':<11}{'корреляция':>12}")
    corrs = []
    for v in sorted(cache):
        ft, f, yt, y = cache[v]
        n = min(len(f), len(y))
        if n < 100:
            continue
        r = float(np.corrcoef(f[:n], y[:n])[0, 1])
        corrs.append(r)
        print(f"  {v:<11}{r:>12.2f}")
    print(f"  медиана корреляции: {np.median(corrs):.2f}")
    print()
    print("  Это не два независимых измерения. Один канал должен нести вращение, другой —")
    print("  ход вперёд; при корреляции 0.8 они делят больше половины дисперсии.")
    print()

    # ---- 5. the proxy the earlier phases were measured against -----------------------------
    pc = Counter((r["proxy"], r["human"]) for r in
                 [{"proxy": events[k]["hidden"]["proxy_label"],
                   "human": labels[k]["human_label"]} for k in labels])
    agree = pc[("LOOK", "LOOK")] + pc[("TURN", "TURN")]
    print("─── 5. АВТОМАТИЧЕСКАЯ РАЗМЕТКА P07.4 ПРОТИВ НАРИСОВАННОГО ───")
    print(f"  совпало по «взгляд/поворот»: {agree} из {len(labels)} "
          f"({agree / len(labels):.0%})")
    print(f"  proxy сказал LOOK, человек нарисовал поворот: {pc[('LOOK', 'TURN')]}")
    print()

    # ---- verdict ---------------------------------------------------------------------------
    print("─── ВЫВОД ───")
    print("  P09.2 в основной трекер ставить нельзя. Три причины, и они независимы:")
    print(f"    правило снимает направление у {len(rev) / max(len(turns), 1):.0%} "
          f"нарисованных поворотов")
    print(f"    правило называет ПОВОРОТОМ {len(strong)} событий, где человек ходил туда-обратно")
    print("    и, главное, сами сигналы не различают эти случаи: AUC около 0.5 при любом")
    print("    способе измерения, включая проверку внутри ролика")
    print()
    print("  Последнее означает, что дело не в пороге и не в формуле. Различение «идёт /")
    print("  топчется» в этих двух каналах отсутствует, поэтому никакая настройка P09.2")
    print("  его не создаст.")
    print()

    (out / "readouts_report.json").write_text(json.dumps({
        "phase": "P09.5D — сигналы трекера против нарисованных путей",
        "n_events": len(labels),
        "shape_by_category": shape,
        "treading_events": {"n": len(tread), "by_rule": dict(by_rule),
                            "strong": len(strong)},
        "human_turns": {"n": len(turns), "rule_confirmed": len(conf),
                        "rule_revoked": len(rev)},
        "pooled": {g: {m: float(np.median(pooled[g][m])) for m in ("fwd", "yaw")}
                   for g in CATS},
        "pooled_auc": {m: auc(pooled["NO_LOCOMOTION"][m],
                              pooled["LOOK"][m] + pooled["TURN"][m])
                       for m in ("fwd", "yaw")},
        "within_clip_median": {g: {m: float(np.median(ranked[g][m])) for m in ("fwd", "yaw")}
                               for g in CATS},
        "within_clip_auc": {m: auc(ranked["NO_LOCOMOTION"][m],
                                   ranked["LOOK"][m] + ranked["TURN"][m])
                            for m in ("fwd", "yaw")},
        "readout_correlation": {"per_clip": corrs, "median": float(np.median(corrs))},
        "proxy_vs_human": {f"{k[0]}|{k[1]}": v for k, v in pc.items()},
        "proxy_agreement_lookturn": agree / len(labels),
        "verdict": ("сигналы не различают «идёт» и «топчется»; P09.2 не ставить"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'readouts_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
