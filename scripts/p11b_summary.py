#!/usr/bin/env python3
"""P11.B — итог фазы, собранный из двух прогонов в один документ.

Written after the control run, because the control changed what the analysis numbers mean and
leaving the two files to contradict each other would be worse than either alone.

Usage:
    PYTHONPATH=. python scripts/p11b_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p11b"


def main() -> int:
    A = json.loads((OUT / "analyze_report.json").read_text(encoding="utf-8"))
    C = json.loads((OUT / "control_report.json").read_text(encoding="utf-8"))
    P10 = {"forward": 0.318, "video_full": 0.568, "dn": 0.560, "early": 0.695,
           "input": 0.705}

    ladder = {
        "forward": {"this_phase": C["forward"]["obs"], "p10": P10["forward"],
                    "p_this_phase": C["forward"]["p"],
                    "note": "P10 измерил ниже случая; эта процедура тоже ниже случая"},
        "video_geometry": {"this_phase": C["video"]["obs"], "p10": P10["video_full"],
                           "p_this_phase": C["video"]["p"]},
        "dn68": {"this_phase": C["dn68"]["obs"], "p10": P10["dn"],
                 "p_this_phase": C["dn68"]["p"]},
        "early20": {"this_phase": C["early"]["obs"], "p10": P10["early"],
                    "p_this_phase": C["early"]["p"]},
        "candidates": {"this_phase": C["candidates"]["obs"],
                       "p_this_phase": C["candidates"]["p"],
                       "p_balanced": C["candidates"]["p_balanced"],
                       "n_groups": C["candidates"]["n_groups"]},
    }

    doc = {
        "phase": "P11.B — MOVE / NO_NET после ранних зрительных клеток",
        "rules_followed": A["rules"],
        "orientation_note": (
            "connectome хранит indices как ЦЕЛИ, а не источники; обход вниз идёт по строкам W. "
            "Проверено на известных цепях: в T4a входят Mi1/Tm3, из T4a выходят LPT31/LLPC1."
        ),
        "depth_note": A["depth_note"],

        "result": {
            "candidates_vs_early": C["gaps"]["candidates_minus_early_pooled"],
            "candidates_vs_early_per_fold": C["gaps"]["candidates_minus_early_per_fold"],
            "candidates_vs_dn68": (C["candidates"]["obs"] - C["dn68"]["obs"]),
            "subsample_68_groups_median": C["subsample68"]["median"],
            "verdict": "NO_NEW_OUTPUT",
            "plain": (
                "Кандидаты значимы (0.749, p=0.010), но не лучше клеток, от которых они "
                "отходят: ранние популяции дают 0.734. 38 304 клетки покупают +0.015. "
                "Нового выхода для MOVE/NO_NET не найдено."
            ),
        },

        "what_changed_about_p10": {
            "plain": (
                "Разрыв «ранние 0.695 → 68 нисходящих 0.560», на котором была построена "
                "посылка P11.B, оказался в основном свойством оценки, а не потерей в мозге. "
                "При одинаковом чтении это 0.734 против 0.675, и 68 нисходящих тоже значимы "
                "(p=0.033). Логистика P10 по 544 колонкам при 40 событиях недооценивала "
                "уровень нисходящих клеток."
            ),
            "consequence": (
                "Информация о чистом перемещении не исчезает между ранними клетками и "
                "нисходящими — она слабеет. Формулировка «здесь нужен другой выход мозга» "
                "была слишком сильной; правильнее — «здесь нужен лучший способ читать то, "
                "что уже есть»."
            ),
        },

        "controls": {
            "negative_extreme": {
                "input": "forward трекера (P10: 0.318, ниже случая)",
                "this_phase": C["forward"]["obs"],
                "reading": ("процедура не поднимает заведомо пустой вход выше случая, "
                            "значит она не фабрикует сигнал"),
            },
            "size_effect": {
                "question": "не объясняется ли преимущество кандидатов числом групп (159 против 68)",
                "answer": C["subsample68"],
                "reading": ("нет: случайные 68 групп кандидатов дают ту же медиану. "
                            "Но и 159 групп дают ту же — значит достигнуто насыщение"),
            },
            "per_fold_vs_pooled": {
                "reading": ("среднее по фолдам выше склейки (0.828 против 0.749) и это "
                            "завышение: один из оцениваемых роликов содержит 4 события и "
                            "решает свой фолд одним исходом. Опираться следует на склейку"),
            },
        },

        "limitation_that_cannot_be_removed": {
            "plain": (
                "Для задачи MOVE/NO_NET оцениваемы только 3 ролика из 5: у VID00001 и "
                "VID00006 нет ни одного события NO_NET_DISPLACEMENT, поэтому AUC внутри "
                "такого ролика не определена. Сбалансированных (>=3 события каждого класса) "
                "всего два — VID00005 и VID00009."
            ),
            "why_it_matters": (
                "треть из 60 событий лежит в роликах, которые не могут быть проверены "
                "отдельно. Все числа этой фазы держатся на трёх роликах, а надёжно — на двух"
            ),
        },

        "ladder": ladder,
        "table_top_types": A["table"][:12],
        "n_pairs_screened_per_fold": A["n_pairs_screened_per_fold"],

        "next_step": (
            "как и предполагалось во втором варианте: не новые типы, а временная динамика и "
            "совместная активность — информация в ранней зрительной системе есть, но "
            "одиночные популяции в фиксированных окнах не дают больше, чем сами ранние клетки"
        ),
    }

    (OUT / "p11b_final.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # the analysis file's own verdict said FOUND, which is true of significance and misleading as
    # a conclusion; record the corrected reading there too so the two files agree
    A["verdict"] = "SIGNIFICANT_BUT_NOT_BETTER_THAN_EARLY"
    A["verdict_note"] = (
        "кандидаты значимы (p=0.010), но не превосходят ранние клетки (0.734). "
        "Нового выхода не найдено. См. p11b_final.json и control_report.json."
    )
    (OUT / "analyze_report.json").write_text(
        json.dumps(A, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print("P11.B — итог")
    print(f"  кандидаты {C['candidates']['obs']:.3f}  против  ранних {C['early']['obs']:.3f}"
          f"   разрыв {C['gaps']['candidates_minus_early_pooled']:+.3f}")
    print(f"  контроль forward {C['forward']['obs']:.3f} (P10: 0.318) — ниже случая, "
          f"процедура не фабрикует")
    print(f"  урезанные до 68 групп: {C['subsample68']['median']:.3f}")
    print(f"  записано: {OUT/'p11b_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
