#!/usr/bin/env python3
"""P14.1 — freeze the CAMERA5 analysis before any of its labels exist.

WHY A FREEZE AND NOT A PROTOCOL IN PROSE
----------------------------------------
The set is growing from three recordings to five, and the question is whether a result that failed
at three comes back at five. A result that arrives after the analysis was chosen is worth much less
than one the analysis predicted, because everything is adjustable after the fact: which levels
count, which threshold separates, whether the aggregate or the per-recording table decides. So the
specification is written here, hashed, and stored before the labels are drawn. The hashes cover the
code that computes the numbers, so an edit after the fact is visible as a mismatch rather than
arguable.

WHAT IS FROZEN
--------------
The selection rule and its outcome; the levels and their primary status; the features; the
leave-one-recording-out split; the standardisation; the permutation null inside each recording; the
task definitions; the ambiguous band of the drawing rule; the per-recording requirement; and the
decision rules that turn numbers into a verdict.

The decision rules are stated as thresholds on purpose. Without them, "about 0.5" and "came back"
are matters of taste, and a taste formed after seeing the result is not evidence.

Usage:
    PYTHONPATH=. python scripts/p14_freeze_camera5.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/p14/FROZEN_CAMERA5.json"

# Code that computes the numbers. A change to any of these changes what the frozen spec means, so
# their digest is part of the freeze rather than a note asking people to remember.
CODE = (
    "scripts/p10_levels.py",            # levels, split, null, per-recording table
    "scripts/p10_video_features.py",    # the video level's features
    "scripts/p10_record.py",            # input and early visual recording
    "scripts/p09_real_holdout.py",      # 68 descending cells recording
    "scripts/p14_camera_labels.py",     # label building and clip namespacing
    "scripts/p14_select_camera5.py",    # the selection rule
    "scripts/p11a_route_change.py",     # route_change, measured but not to be changed
    "scripts/p095_build_set.py",        # the window the reviewer sees
    "scripts/p095b_build.py",
    "scripts/p095c_server.py",          # the drawing page
    "fly_vo/visual_encoder.py",
    "fly_vo/content_motion.py",
    "fly_vo/config.py",
    "fly_vo/brain_clock.py",
    "fly_vo/malecns_engine.py",
)

TASKS = {
    "SAME_vs_DIFFERENT": {
        "definition": ("класс 1 — SAME_DIRECTION, класс 0 — DIFFERENT_DIRECTION; события других "
                       "категорий в эту задачу не входят"),
        "code": "np.where(cat == 'SAME_DIRECTION', 1, 0); keep = isin(cat, [SAME, DIFFERENT])",
    },
    "MOVE_vs_NO_NET": {
        "definition": ("класс 0 — NO_NET_DISPLACEMENT, класс 1 — всё остальное "
                       "(SAME_DIRECTION, DIFFERENT_DIRECTION). Это «было ли чистое продвижение», "
                       "а не «шёл против стоял»"),
        "code": "np.where(cat == 'NO_NET_DISPLACEMENT', 0, 1); keep = все события",
    },
}

THRESHOLDS = {
    "levels_primary": ["input", "early", "dn"],
    "levels_context": ["video_full", "video_band", "yaw", "forward"],
    "aggregate_auc_min": 0.65,
    "aggregate_auc_floor_for_absence": 0.60,
    "p_perm_max": 0.05,
    "per_recording_auc_min": 0.50,
    "per_recording_min_count": 4,
    "n_recordings": 5,
    "move_no_net_auc_min": 0.60,
}


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--stage", default="before_labels",
                    help="before_labels | data_ready — что зафиксировано на этот момент")
    ap.add_argument("--verify", action="store_true",
                    help="сверить текущий код с замороженными хэшами и ничего не писать")
    ap.add_argument("--revision", default=None,
                    help="причина переиздания заморозки; записывается в журнал редакций")
    a = ap.parse_args()

    if a.verify:
        if not OUT.exists():
            print(f"нет заморозки: {OUT}")
            return 1
        doc = json.loads(OUT.read_text(encoding="utf-8"))
        frozen = doc["code_hashes"]
        print("=" * 100)
        print("ПРОВЕРКА ЗАМОРОЗКИ CAMERA5: код против хэшей")
        print("=" * 100)
        print(f"  заморожено: {doc['frozen_at']} (этап {doc.get('stage')})")
        bad = []
        for rel, h in frozen.items():
            p = ROOT / rel
            if not p.exists():
                print(f"  {rel:<38} НЕТ ФАЙЛА")
                bad.append((rel, h, None))
                continue
            now = sha16(p)
            if now != h:
                print(f"  {rel:<38} ИЗМЕНИЛСЯ (было {h}, стало {now})")
                bad.append((rel, h, now))
        print()
        if bad:
            print(f"  РАСХОЖДЕНИЙ: {len(bad)} из {len(frozen)}")
            print("  Спецификация и код разошлись. Результат после такой правки уже нельзя")
            print("  предъявлять как предсказанный: сначала надо объяснить, что именно изменилось.")
            return 1
        print(f"  все {len(frozen)} файлов совпадают — код соответствует замороженной спецификации")
        return 0

    sel_path = ROOT / "data/p14/SELECTION_CAMERA5.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))

    hashes = {}
    missing = []
    for rel in CODE:
        p = ROOT / rel
        if p.exists():
            hashes[rel] = sha16(p)
        else:
            missing.append(rel)

    # Re-freezing is allowed only while no result exists, and only with the reason written down.
    # Re-hashing silently would defeat the point of hashing: the record would show a clean freeze
    # with no trace of what changed between it and the one before.
    revisions = []
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
            revisions = old.get("revisions", [])
            if old.get("code_hashes") != hashes:
                if not a.revision:
                    print("ВНИМАНИЕ: хэши изменились, а --revision не указан.")
                    print("  Хэши не перезаписываю: сначала объясните причину.")
                    return 1
                revisions.append({
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "reason": a.revision,
                    "changed": sorted([rel for rel, h in hashes.items()
                                       if old["code_hashes"].get(rel) != h]),
                    "stage": a.stage,
                })
        except Exception:
            revisions = []

    doc = {
        "phase": "P14.1 — CAMERA5: замороженный анализ до разметки",
        "stage": a.stage,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "why": ("набор растёт с трёх записей до пяти, и вопрос в том, вернётся ли результат, "
                "который на трёх не подтвердился. Спецификация записана и захэширована ДО появления "
                "меток: иначе любой её пункт можно подкрутить после результата"),

        "selection": {
            "rule_source": str(sel_path.relative_to(ROOT)),
            "rule": sel["rule"],
            "chosen": sel["chosen"],
            "eligible": sel["eligible"],
            "camera_set_after": sel["camera_set_after"],
            "note": ("выбор сделан механически по происхождению; содержимое кусков не учитывалось "
                     "ни в каком виде"),
        },

        "levels": {
            "primary": THRESHOLDS["levels_primary"],
            "context": THRESHOLDS["levels_context"],
            "why_primary_three": ("вопрос про вход и ранние клетки — есть ли сигнал до мотора; "
                                  "68 нисходящих — теряется ли он к выходу"),
        },
        "features": {
            "video": ("12 фиксированных геометрических величин оптического потока, полный кадр и "
                      "центральная полоса; scripts/p10_video_features.py"),
            "brain": ("вход — сумма управляющего сигнала по 12 группам T4a/b, T5a/b, LC4, LPLC2; "
                      "ранние — скорости 20 популяций; скрипт scripts/p10_record.py"),
            "dn": "68 нисходящих клеток, скрипт scripts/p09_real_holdout.py",
            "statistics_per_channel": ["signed_before", "signed_after", "mag_before", "mag_after",
                                       "act_change", "dir_change", "reversal", "abs_diff"],
        },
        "split": {
            "scheme": "leave-one-recording-out",
            "code": "for c in unique(clips): test = (clips == c); train = (clips != c)",
            "why": "тестовый ролик целиком отложен, поэтому различие сцен не может попасть в обучение",
        },
        "normalisation": {
            "scheme": "стандартизация по обучающим фолдам: mu, sd считаются на train, применяются к test",
            "code": "mu, sd = Xtr.mean(0), Xtr.std(0); (Xte - mu) / sd",
            "regularisation": "логистика, l2 = 1/max(C,1e-6), C = 1.0",
        },
        "null": {
            "scheme": "перестановка меток ВНУТРИ каждого ролика",
            "code": "for c in unique(clips): yp[clips == c] = rng.permutation(y[clips == c])",
            "why": ("общая перестановка по пяти записям смешала бы различия камер и сцен, и именно "
                    "эти различия могли бы выглядеть как сигнал"),
            "n_perm": 300,
            "p_formula": "(число нулей >= наблюдения + 1) / (число нулей + 1)",
        },
        "tasks": TASKS,

        "ambiguous_band": {
            "rule": ("метка выводится из формы рисунка: нетто-поворот <= 35° — SAME_DIRECTION, "
                     ">= 70° — DIFFERENT_DIRECTION"),
            "band": [35.0, 70.0],
            "treatment": ("события, у которых выведенный нетто-угол попадает строго между 35° и 70°, "
                          "из основной задачи SAME/DIFFERENT ИСКЛЮЧАЮТСЯ: они не считаются "
                          "уверенной меткой ни в одну сторону"),
            "how": "по полю derived_turn_net_deg в файле меток; события без угла (нажатие кнопки) не исключаются",
            "measured_prevalence": {"round_v2": 4, "round_vid10": 4},
        },

        "pathway_under_test": {
            "built_from_drawing": "первичный путь: только события label_source == derived_from_drawing",
            "all_labels": "вторично: все события, включая кнопочные",
            "why": ("кнопочная страница спрашивала про момент, который на видео не был отмечен, "
                    "поэтому кнопочные метки менее надёжны; первичный путь на них не опирается"),
        },

        "per_recording_required": {
            "table": "уровень | каждый ролик | агрегат",
            "reason": ("агрегат не отличает эффект, держащийся между записями, от эффекта, который "
                       "создаёт одна запись. При пяти записях это и есть главный вопрос"),
        },

        "decision_rules": {
            "read_this_before_the_result": True,
            "SAME_vs_DIFFERENT": {
                "ESTABLISHED": (f"агрегат >= {THRESHOLDS['aggregate_auc_min']} И p < "
                                f"{THRESHOLDS['p_perm_max']} И не меньше "
                                f"{THRESHOLDS['per_recording_min_count']} из "
                                f"{THRESHOLDS['n_recordings']} роликов с собственной AUC >= "
                                f"{THRESHOLDS['per_recording_auc_min']}"),
                "NOT_ESTABLISHED": "всё остальное",
                "if_not_established": ("зафиксировать SAME/DIFFERENT NOT ESTABLISHED и прекратить "
                                       "поиск на этих данных: ни новых клеток, ни новых порогов, "
                                       "ни спасения route_change"),
                "if_established": ("не объявлять победу по агрегату: проверить одинаковость знака "
                                   "между роликами, что результат не тянет один кусок, и что "
                                   "перестановочный контроль пройден"),
            },
            "MOVE_vs_NO_NET": {
                "HOLDS": (f"агрегат >= {THRESHOLDS['move_no_net_auc_min']} И p < "
                          f"{THRESHOLDS['p_perm_max']} И не меньше "
                          f"{THRESHOLDS['per_recording_min_count']} из "
                          f"{THRESHOLDS['n_recordings']} роликов >= "
                          f"{THRESHOLDS['per_recording_auc_min']}"),
                "if_holds": ("становится главным кандидатом на воспроизводимый дополнительный "
                             "сигнал мозга: отвечает не «куда повернул», а «было ли чистое "
                             "продвижение»"),
                "note": "порог ниже, чем у SAME/DIFFERENT: на трёх кусках он уже давал 0.620",
            },
        },

        "route_change": {
            "status": "НЕ ПОДТВЕРЖДЁН",
            "rule": ("не переобучать: клетки, признаки, веса и пороги не меняются. Можно измерить "
                     "на пяти кусках тем же замороженным способом, и результат не меняет статус, "
                     "пока новые данные сами не заставят"),
            "measured_at_three": {"auc": 0.460, "p": 0.645},
            "not_to_do": ["искать новые клетки под него", "менять признаки", "менять пороги",
                          "реанимировать заранее"],
        },

        "code_hashes": hashes,
        "code_missing": missing,
        "verification": ("PYTHONPATH=. python scripts/p14_freeze_camera5.py --verify — сверяет "
                         "текущий код с этими хэшами и падает с ненулевым кодом при расхождении. "
                         "Запускать перед финальным прогоном: файлы в рабочей папке наблюдались "
                         "откатывающимися к более ранним версиям, и без этой сверки откат файла "
                         "прошёл бы незамеченным."),
        "hash_meaning": ("sha256 первых 16 знаков файлов, реализующих перечисленные правила. "
                         "Несовпадение после этой заморозки означает, что спецификация и код "
                         "разошлись"),
        "uniqueness_requirement": {
            "clip_names": "vXX_eNNN — идентификатор куска плюс номер события",
            "why": ("нумерация eNNN у раундов совпадает, а признаки кэшируются по имени клипа; "
                    "без префикса пересчёт затирает признаки предыдущего раунда. Это уже "
                    "случилось один раз"),
        },
        "what_must_not_change_before_the_result": [
            "список уровней и их первичность",
            "признаки",
            "схема разбиения",
            "способ нормировки",
            "перестановочный нуль внутри ролика",
            "правила задач и пороги решения",
            "правило неоднозначной полосы",
        ],
    }

    # Recorded after the specification is built, and only with a stated reason, so a re-issue leaves
    # a trace instead of a clean-looking freeze that silently replaced the previous one.
    doc["revisions"] = revisions
    doc["revision_rule"] = ("переиздать заморозку можно, пока результат не посчитан, и только с "
                            "записанной причиной. Каждая редакция попадает в журнал: молчаливое "
                            "перехэширование уничтожило бы смысл хэша")

    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 100)
    print("P14.1 — CAMERA5 ЗАМОРОЖЕН ДО РАЗМЕТКИ")
    print("=" * 100)
    print(f"  этап: {a.stage}")
    print(f"  выбранные куски: {sel['chosen']}  → камерный набор {sel['camera_set_after']}")
    print(f"  первичные уровни: {THRESHOLDS['levels_primary']}")
    print(f"  правила: ESTABLISHED при агрегате >= {THRESHOLDS['aggregate_auc_min']}, p < "
          f"{THRESHOLDS['p_perm_max']}, и >= {THRESHOLDS['per_recording_min_count']} из "
          f"{THRESHOLDS['n_recordings']} роликов >= {THRESHOLDS['per_recording_auc_min']}")
    print(f"  неоднозначная полоса {THRESHOLDS.get('band', [35.0, 70.0])} исключается")
    print(f"  хэшей кода: {len(hashes)}" + (f", нет файлов: {missing}" if missing else ""))
    print(f"  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
