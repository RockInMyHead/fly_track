#!/usr/bin/env python3
"""P10 — the table and the verdict.

Reads output/p10/levels_report.json, prints the table the phase was asked for, decides which of
the three outcomes it supports, and draws one figure.

The three outcomes are about *where* the difference disappears, not about how large it is:

    VIDEO_FAIL        the raw video does not contain it either. Then no amount of work on the fly
                      will recover it, and the honest tracker is one that keeps several routes
                      alive until the next moment that is informative.
    ENCODER_LOSS      the video contains it but the drive handed to the brain no longer does. The
                      loss is in the visual front end, and that is where to work.
    BRAIN_READOUT     the early visual cells still contain it and the descending cells or the two
                      current readouts no longer do. The loss is inside the brain or in how it is
                      being read, and replacing the front end would change nothing.

Which outcome applies is decided per task, because the two tasks can fail at different places.

Usage:
    PYTHONPATH=. python scripts/p10_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p10"

# Distance from the retina, in order. The verdict walks this list.
PIPELINE = ("video_full", "video_band", "input", "early", "dn", "yaw", "forward")
SHORT = {
    "video_full": "сырое видео",
    "video_band": "видео, центр",
    "input": "вход в мозг",
    "early": "T4/T5/LC4/LPLC2",
    "dn": "68 нисходящих",
    "yaw": "yaw",
    "forward": "forward",
}
TASKS = ("SAME_vs_DIFFERENT", "MOVING_vs_NO_NET")
TASK_RU = {
    "SAME_vs_DIFFERENT": "SAME / DIFFERENT",
    "MOVING_vs_NO_NET": "MOVE / NO_MOVE",
}
ALPHA = 0.05


def cell(rep: dict | None, task: str) -> tuple[float, float, float, float]:
    if not rep or task not in rep:
        return (float("nan"),) * 4
    r = rep[task]
    return (r["auc_aggregate"], r["p_aggregate"], r["auc_best_channel"], r["p_best_channel"])


def significant(rep: dict | None, task: str) -> bool:
    a, pa, b, pb = cell(rep, task)
    return (not np.isnan(pa) and pa < ALPHA) or (not np.isnan(pb) and pb < ALPHA)


def verdict(levels: dict, task: str) -> tuple[str, str, str]:
    """Which of the named outcomes the measurements support, plus a note about the video row.

    The order matters and is deliberate. The most specific finding wins: if the drive handed to
    the brain or the first cells of the brain separate the two things and the descending cells do
    not, then the loss has been located inside the brain and saying anything more general would
    throw that away. Only when nothing downstream works does the video row decide the answer.

    The note is kept separate because the coarse-video result and the brain result are two
    different statements and merging them loses the useful one. If some level downstream of the
    retina separates the categories, then the information was in the film — the brain read it. A
    failure of the twelve pooled scalars is a failure of those scalars, and reporting it as
    "the video does not contain this" would be the expensive mistake.
    """
    vid = any(significant(levels.get(l), task) for l in ("video_full", "video_band"))
    inp = significant(levels.get("input"), task)
    ear = significant(levels.get("early"), task)
    dn = significant(levels.get("dn"), task)
    rd = any(significant(levels.get(l), task) for l in ("yaw", "forward"))
    downstream = inp or ear or dn or rd

    if not downstream and not vid:
        v, why = ("VIDEO_FAIL", "не найдено нигде: ни в геометрии видео, ни дальше по цепи")
    elif (inp or ear) and not dn:
        v, why = ("BRAIN_READOUT_LOSS",
                  "во входе и в ранних зрительных клетках есть, в 68 нисходящих — уже нет")
    elif vid and not inp and not ear and not dn:
        v, why = ("ENCODER_LOSS", "в геометрии видео есть, во входе в мозг и дальше — нет")
    elif dn and not rd:
        v, why = ("READOUT_LOSS",
                  "в нисходящих клетках есть, но текущее чтение yaw/forward его не берёт")
    elif not vid:
        v, why = ("VIDEO_FEATURES_COARSE",
                  "в мозге есть, а в двенадцати сводных признаках видео — нет")
    else:
        v, why = ("CARRY_THROUGH", "различение проходит через все уровни без потерь")

    note = ("" if vid else
            "сводные геометрические признаки видео (12 величин) этого не берут, "
            "хотя различение есть дальше по цепи — значит, оно есть в кадре")
    return v, why, note


# each of the three outcomes the phase named, and where the extra cases sit
MAPPING = {
    "VIDEO_FAIL": "A",
    "ENCODER_LOSS": "B",
    "BRAIN_READOUT_LOSS": "C",
    "READOUT_LOSS": "C",
    "CARRY_THROUGH": "—",
    "VIDEO_FEATURES_COARSE": "не A",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    rep = json.loads((out / "levels_report.json").read_text(encoding="utf-8"))
    levels = rep["levels"]

    print("=" * 104)
    print("P10 — ГДЕ ТЕРЯЕТСЯ ИНФОРМАЦИЯ О ДВИЖЕНИИ ТЕЛА")
    print("=" * 104)
    print(f"  событий {rep['n_events']}, нуль: перемешивание меток ВНУТРИ ролика, "
          f"{rep['n_perm']} повторов")
    print()

    print("  ТАБЛИЦА. В клетке: AUC лучшей модели / p против нуля. "
          "Жирным — то, что выше нуля (p<0.05).")
    print()
    print(f"  {'уровень':<22}{'каналов':>8}   {TASK_RU['SAME_vs_DIFFERENT']:>26}"
          f"   {TASK_RU['MOVING_vs_NO_NET']:>26}")
    print("  " + "-" * 92)
    order = [l for l in PIPELINE if l in levels and "error" not in levels[l]]
    for l in order:
        r = levels[l]
        nch = r.get("n_channels", 0)
        line = f"  {SHORT[l]:<22}{nch:>8}  "
        for t in TASKS:
            a, pa, b, pb = cell(r, t)
            use = a if (not np.isnan(pa) and (np.isnan(pb) or pa <= pb)) else b
            p = min(x for x in (pa, pb) if not np.isnan(x)) if not (
                np.isnan(pa) and np.isnan(pb)) else float("nan")
            mark = "*" if p < ALPHA else " "
            line += f"{use:>10.3f} (p={p:.3f}){mark}   "
        print(line)
    print()
    print("  *  выше нуля с p<0.05")
    print()

    print("  ЧТО ИМЕННО ВЫБИРАЛОСЬ КАК ЛУЧШИЙ КАНАЛ (устойчивость важна не меньше AUC):")
    for l in order:
        for t in TASKS:
            r = levels[l].get(t)
            if not r:
                continue
            ch = [c["channel"] for c in r.get("chosen_columns", [])]
            if not ch:
                continue
            same = len(set(ch)) == 1
            note = "один и тот же канал во всех фолдах" if same else f"{len(set(ch))} разных каналов"
            print(f"    {SHORT[l]:<20}{TASK_RU[t]:<20}{', '.join(ch[:5])}   [{note}]")
    print()

    print("  ВЫВОД")
    print("  " + "-" * 92)
    print("  A = VIDEO_FAIL, B = ENCODER_LOSS, C = BRAIN/READOUT_LOSS — три исхода, названные")
    print("  в задании; случай, который в них не укладывается, помечен явно.")
    print()
    verd = {}
    for t in TASKS:
        v, why, note = verdict(levels, t)
        verd[t] = {"verdict": v, "why": why, "letter": MAPPING.get(v, "?"), "note": note}
        print(f"    {TASK_RU[t]:<20} {MAPPING.get(v,'?'):<6}{v}")
        print(f"    {'':<20} {why}")
        if note:
            print(f"    {'':<20} примечание: {note}")
        print()

    print("  ОТДЕЛЬНО ПРО «ТОПТАЛСЯ»")
    print("  " + "-" * 92)
    geom = no_net_geometry()
    if geom:
        print(f"    {'категория':<22}{'n':>4}{'длина пути':>12}{'смещение':>10}"
              f"{'вернулся':>11}{'прямизна':>10}")
        for c, g in geom.items():
            print(f"    {c:<22}{g['n']:>4}{g['path_len']:>12.3f}{g['displacement']:>10.3f}"
                  f"{g['returned_frac']:>10.0%}{g['straightness']:>10.2f}")
        n_ = geom.get("NO_NET_DISPLACEMENT")
        t_ = geom.get("DIFFERENT_DIRECTION")
        if n_ and t_:
            print()
            print(f"    У «топтался» длина пути {n_['path_len']:.3f} против {t_['path_len']:.3f} "
                  f"у поворотов — почти столько же,")
            print(f"    а чистое смещение {n_['displacement']:.3f} против "
                  f"{t_['displacement']:.3f} — в "
                  f"{t_['displacement']/max(n_['displacement'],1e-9):.1f} раза меньше.")
            print("    Человек шёл. Отсутствует чистое перемещение, а не движение.")
    print()
    r = levels.get("dn", {}).get("MOVING_vs_NO_NET")
    if r:
        print(f"    68 нисходящих клеток на задаче MOVE/NO_MOVE: AUC {r['auc_aggregate']:.3f}, "
              f"p={r['p_aggregate']:.3f}")
        print(f"    нуль: медиана {r['null_aggregate_median']:.3f}, "
              f"p95 {r['null_aggregate_p95']:.3f}")
    print()

    print("  ПОПРАВКА К ОПИСАНИЮ МЕТОДА (найдена в P11.A, числа не меняются)")
    print("  " + "-" * 92)
    print("    В таблице выше уровень «68 нисходящих» — это логистика по восьми статистикам")
    print("    всех 68 каналов, то есть 544 колонки. Описание этого уровня как сжатия каналов")
    print("    до семи скаляров было неверным: настоящее сжатие до семи скаляров даёт 0.451.")
    print("    Сами числа посчитаны корректно; неверна была подпись под одним из уровней.")
    print("    Прямое сопоставление «68 клеток против yaw» нельзя опирать на 0.767 против")
    print("    0.617, потому что там сравниваются 544 колонки с 56. Выровненные сравнения —")
    print("    в output/p11a/. Выводы P10 об уровнях остаются: нуль для каждого уровня")
    print("    строился на той же модели, что и сам уровень.")
    print()

    (out / "p10_verdict.json").write_text(json.dumps({
        "phase": rep["phase"], "n_events": rep["n_events"], "n_perm": rep["n_perm"],
        "verdicts": verd, "task_names": TASK_RU, "correction": CORRECTION,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.figure:
        make_figure(levels, out)
    print(f"  записано: {out/'p10_verdict.json'}")
    return 0


CORRECTION = {
    "what_was_wrong": (
        "P10 описывал число 0.767 как уровень «68 нисходящих клеток», а метод — как сжатие "
        "каналов до семи физических скаляров. Код этого не делал: строка 0.767 получена "
        "логистической регрессией по to_matrix(rows), то есть по восьми статистикам всех 68 "
        "каналов — 544 колонки. Настоящее сжатие до семи скаляров (медиана по каналам) даёт "
        "AUC 0.451."
    ),
    "what_stays": (
        "Все приведённые числа верны: они посчитаны именно тем кодом, который описан в "
        "levels_report.json. Неверным было словесное описание одного из уровней, а не "
        "измерения."
    ),
    "why_it_matters": (
        "Сравнение 544 колонок против 56 колонок у yaw часть разрыва объясняет размером "
        "модели, а не клетками. Выводы P10 об уровнях (где различение есть, а где нет) "
        "остаются в силе, потому что нуль для каждого уровня строился на той же модели, что и "
        "сам уровень. Но прямое сопоставление «68 клеток против yaw» на этом числе опираться "
        "не может."
    ),
    "who_fixed_it": "P11.A — выровненные по размеру модели сравнения в output/p11a/",
    "corrected_reading": {
        "yaw, 7 чисел x 8 статистик (56 колонок)": 0.617,
        "те же 18 yaw-клеток x 8 статистик (144 колонки)": 0.737,
        "случайные 7 клеток из 68 (56 колонок), медиана": 0.628,
        "все 68 клеток x 8 статистик (544 колонки)": 0.767,
    },
}


def no_net_geometry() -> dict:
    """Path length against net displacement, straight from the drawings.

    The whole point of the third category is that it is not "stood still". These numbers are the
    evidence: the path is about as long as in the other two categories, while the net displacement
    is half. A category called STATIC would be contradicted by its own data.
    """
    from collections import defaultdict

    p = ROOT / "data/p095/human_labels_v2.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))["labels"]
    rename = {"LOOK": "SAME_DIRECTION", "TURN": "DIFFERENT_DIRECTION",
              "NO_LOCOMOTION": "NO_NET_DISPLACEMENT"}
    acc = defaultdict(lambda: {"len": [], "disp": []})
    for l in raw:
        if l.get("round", 1) != 1:
            continue
        strokes = l.get("drawing") or []
        pts = [pt for st in strokes for pt in st]
        if len(pts) < 3:
            continue
        length = 0.0
        for st in strokes:
            for i in range(1, len(st)):
                length += float(np.hypot(st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]))
        if length < 1e-6:
            continue
        disp = float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
        key = rename.get(l["human_label"], l["human_label"])
        acc[key]["len"].append(length)
        acc[key]["disp"].append(disp)
    out = {}
    for k, v in acc.items():
        ml = float(np.median(v["len"]))
        md = float(np.median(v["disp"]))
        out[k] = {"n": len(v["len"]), "path_len": ml, "displacement": md,
                  "returned_frac": 1.0 - md / max(ml, 1e-9),
                  "straightness": md / max(ml, 1e-9)}
    return out


def make_figure(levels: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [l for l in PIPELINE if l in levels and "error" not in levels[l]]
    ymax = 1.0
    fig = plt.figure(figsize=(16, 10.5), dpi=110)
    fig.patch.set_facecolor("#0f1117")
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.975, top=0.89, bottom=0.02,
                            hspace=0.20, wspace=0.22, height_ratios=(1.0, 0.74))

    for k, t in enumerate(TASKS):
        ax = fig.add_subplot(grid[0, k])
        ax.set_facecolor("#0f1117")
        a = [cell(levels.get(l), t)[0] for l in order]
        p = [cell(levels.get(l), t)[1] for l in order]
        b = [cell(levels.get(l), t)[2] for l in order]
        pb = [cell(levels.get(l), t)[3] for l in order]
        cols = ["#ffd60a" if (pp < 0.05 or pbb < 0.05) else "#5a6580"
                for pp, pbb in zip(p, pb)]
        x = np.arange(len(order))
        ax.bar(x - 0.2, a, 0.4, color=cols, label="агрегат уровня")
        ax.bar(x + 0.2, b, 0.4, color=cols, alpha=0.45,
               label="лучший одиночный канал")
        ax.axhline(0.5, color="#ff9f1c", lw=1.8, ls="--")
        ax.text(len(order) - 0.5, 0.515, "0.5 = случай", color="#ff9f1c", fontsize=9, ha="right")
        for i in range(len(order)):
            for v, pp, off in ((a[i], p[i], -0.2), (b[i], pb[i], 0.2)):
                if np.isnan(v):
                    continue
                ax.text(i + off, v + 0.015, f"{v:.2f}", ha="center", color="white", fontsize=8.5)
                if pp < 0.05:
                    ax.text(i + off, v + 0.055, "*", ha="center", color="#3ddc84", fontsize=13)
        ax.set_xticks(x)
        short = {"video_full": "видео\n(кадр)", "video_band": "видео\n(центр)",
                 "input": "вход в\nмозг", "early": "ранние\nклетки",
                 "dn": "68\nнисходящих", "yaw": "yaw", "forward": "forward"}
        ax.set_xticklabels([short[l] for l in order], color="#c8cede", fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.tick_params(colors="#c8cede", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#39415a")
        ax.set_title(TASK_RU[t], color="white", fontsize=13.5, loc="left")
        ax.grid(alpha=0.12, color="#5a6580", axis="y")
        ax.legend(facecolor="#171b24", edgecolor="#39415a", labelcolor="#e8ecf5", fontsize=8.5)
        ax.axhspan(0.4975, 0.5025, color="#ff9f1c", alpha=0.12)

    ax = fig.add_subplot(grid[1, :])
    ax.set_facecolor("#0f1117")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#39415a")
    lines = [(f"СИНЯЯ ЛИНИЯ — 0.5: результат не отличается от случайного. "
              f"* — выше нуля при p<0.05.", "#8f9ab5", 11.5)]
    for t in TASKS:
        v, why, note = verdict(levels, t)
        col = {"VIDEO_FAIL": "#ff9f1c", "VIDEO_FEATURES_COARSE": "#ffd60a",
               "ENCODER_LOSS": "#e05c5c", "BRAIN_READOUT_LOSS": "#e05c5c",
               "READOUT_LOSS": "#b07de0", "CARRY_THROUGH": "#3ddc84"}.get(v, "#c8cede")
        lines.append(("", "#000", 6))
        lines.append((f"{TASK_RU[t]}:  {MAPPING.get(v,'?')}   {v}", col, 15))
        lines.append((f"    {why}", "#c8cede", 11.5))
        if note:
            lines.append((f"    {note}", "#8f9ab5", 10.5))
    lines.append(("", "#000", 6))
    lines.append(("Нуль построен перемешиванием меток ВНУТРИ каждой записи, поэтому сцепление",
                  "#8f9ab5", 11))
    lines.append(("«ролик—метка» (VID00001 дал 5 из 6 как SAME_DIRECTION, VID00005 и VID00009 "
                  "сбалансированы)", "#8f9ab5", 11))
    lines.append(("присутствует и в нуле тоже. Проверка: обучение на 4 роликах, проверка на пятом.",
                  "#8f9ab5", 11))
    y = 1.0
    for txt, col, size in lines:
        if txt:
            ax.text(0, y, txt, color=col, fontsize=size,
                    weight="bold" if size >= 15 else "normal")
        y -= 0.085 if size >= 15 else (0.072 if size >= 11 else 0.052)

    fig.suptitle("P10 — на каком уровне исчезает различение «двигаюсь / топчусь» и "
                 "«туда же / в другую сторону»", color="white", fontsize=15)
    dest = out / "p10_levels.png"
    fig.savefig(dest, dpi=110, facecolor="#0f1117")
    print(f"  рисунок: {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
