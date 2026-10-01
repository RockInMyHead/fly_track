#!/usr/bin/env python3
"""P13.1D — freeze the test of the brain's net_displacement as a second feature for the 12-28 band.

WHY THIS PHASE EXISTS
---------------------
P13.1C showed that the video measure cannot separate standing from walking between 12 and 28 of its
own units. Ten of forty windows in that band were a person walking, spread from 19.8 to 26.0, and
standing windows sat at the same values — 19.77 standing against 19.79 walking, 21.46 against 21.73.
No threshold divides them, so the invented path that lives in that band cannot be removed by moving
the threshold. A different signal is needed, and the project already has one.

WHAT IS BEING TESTED
--------------------
`net_displacement` — the P12 channel built from the twenty early visual populations, fitted on the
five recordings used throughout P09 and frozen in data/p12/FROZEN_BEFORE_VID00010.json. Its question
is "was there net displacement over this window", with classes MOVE and NO_NET.

It is applied here exactly as frozen: the reader fitted on all five old recordings, its three
(unit, feature) pairs, its weights, its bias. Nothing is refitted on the labels this phase collects.

THE CAVEAT THAT BELONGS IN ADVANCE
----------------------------------
This channel answers a different question from the one the gate asks. NO_NET_DISPLACEMENT includes a
person who walked and came back — that is what the class was named for, and it is why this channel
was ruled out as a gate in the first place. So:

    a good result here is informative, because a window where the person never moved should read as
    NO_NET under any sensible definition;

    a poor result is not conclusive evidence against the channel, because "no net displacement over
    seven seconds" and "not moving at this instant" are not the same statement.

Both readings are recorded now, before the numbers exist, so that whichever way it falls the
interpretation is not invented afterwards.

THE DECISION RULE, FIXED IN ADVANCE
-----------------------------------
Measured on the windows already reviewed in P13.1B and P13.1C, whose labels exist and will not be
recollected:

    primary     AUC of the channel's score for standing against walking, inside the 12-28 band.
                That band is the whole point: elsewhere the video measure already works.

    usable      AUC >= 0.70 in that band. Then it can serve as a second feature.
    weak        0.60 <= AUC < 0.70. Then it carries something but not enough alone.
    unusable    AUC < 0.60. Then the band needs a third signal or a combination.

    secondary   for the best single threshold on the channel's score, the pair (FALSE STOP,
                MISSED STOP) over all reviewed windows, to be compared with the video gate's
                (0 of 40, 32 of 49).

The AUC threshold is a shape choice made now, not fitted later. No window is excluded after seeing
its prediction, and no reweighting of the three pairs is allowed.

THE READER AND THE WINDOW
-------------------------
    reader      data/p12/FROZEN_BEFORE_VID00010.json → readers_for_new_recording.net_displacement,
                fitted on VID00001, 00002, 00005, 00006, 00009. The nine chunks under test are
                VID00007, 00008, 00011-00017, none of which is in that set, so no reviewed window is
                in-sample for the reader.
    window      the same one P12 used: three seconds before the moment to half a second before, the
                second around it, and half a second to seven seconds after. Measured on the chunk's
                own recording, which covers the whole clip, so the window is complete even when the
                five-second clip is not.

Usage:
    PYTHONPATH=. python scripts/p13_1d_freeze.py
    PYTHONPATH=. python scripts/p13_1d_freeze.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/p13/FROZEN_NET_DISPLACEMENT_TEST.json"

CODE = (
    "scripts/p13_1d_build.py",
    "scripts/p13_1d_score.py",
    "scripts/p10_record.py",                 # produced the recordings this reads
    "data/p12/FROZEN_BEFORE_VID00010.json",  # the frozen channel
    "data/p13/p13_1b_key.json",              # reviewed windows round 1
    "output/p13/p13_1b_answers.json",
    "data/p13/p13_1c_key.json",              # round 2
    "output/p13/p13_1c_answers.json",
    "data/p13/FROZEN_THRESHOLD_28.json",     # the result that motivates this
)

PRIOR = {
    "accuracy_at_12": {"false_stop": "0 из 40", "missed_stop": "32 из 49"},
    "accuracy_at_28": {"false_stop": "10 из 49", "verdict": "отвергнут"},
    "band_12_28": {"n": 40, "stands": 29, "walks": 10, "unclear": 1,
                   "reading": "классы перемешаны: 19.77 стоит против 19.79 идёт"},
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--note", default=None,
                    help="почему заморозка пересоздана; пишется в файл и остаётся видимым")
    a = ap.parse_args()

    if a.verify:
        if not DEST.exists():
            print(f"нет файла заморозки {DEST}")
            return 1
        d = json.loads(DEST.read_text(encoding="utf-8"))
        bad = []
        for rel, m in d["manifest"].items():
            p = ROOT / rel
            if not p.exists():
                bad.append(f"{rel} — пропал")
            elif sha256(p) != m["sha256"]:
                bad.append(f"{rel} — изменён (было {m['sha256'][:16]})")
        if bad:
            print("ЗАМОРОЗКА НАРУШЕНА:")
            for b in bad:
                print(f"  {b}")
            return 1
        print(f"заморозка цела: {len(d['manifest'])} файлов совпадают")
        print(f"  заморожено {d['frozen_at']}, порог успеха AUC {d['decision_rule']['usable_auc']}")
        return 0

    manifest, missing = {}, []
    for rel in CODE:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        manifest[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    if missing:
        print(f"СТОП: нет файлов {missing}")
        return 1

    doc = {
        "phase": "P13.1D — net_displacement как второй признак для полосы 12–28",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "refrozen_note": a.note,
        "refrozen_rule": (
            "пересоздавать заморозку можно только пока результат не посчитан; после первого "
            "взгляда на числа это уже подгонка"
        ),
        "question": (
            "разделяет ли канал net_displacement из P12 те окна полосы 12–28, где мера по видео "
            "не разделяет: человек стоит против человек идёт"
        ),
        "reader": {
            "source": "data/p12/FROZEN_BEFORE_VID00010.json → readers_for_new_recording"
                      ".net_displacement",
            "fitted_on": ["VID00001", "VID00002", "VID00005", "VID00006", "VID00009"],
            "tested_on": ["VID00007", "VID00008", "VID00011", "VID00012", "VID00013",
                          "VID00014", "VID00015", "VID00016", "VID00017"],
            "no_in_sample_windows": (
                "ни один проверяемый кусок не входит в обучающую пятёрку, поэтому ни одно окно не "
                "является для читателя своим"
            ),
            "pairs": "T4b_L/integral, T5b_L/integral, T4b_R/integral — как заморожено",
            "nothing_refitted": "веса, признаки и порог берутся из заморозки; на этих метках "
                                "читатель не переобучается",
        },
        "window": {
            "before": [-3.0, -0.5],
            "during": [-0.5, 0.5],
            "after": [0.5, 7.0],
            "note": "то же окно, что в P12: три секунды до, секунда вокруг, семь после",
        },
        "labels": {
            "source": "метки человека из P13.1B и P13.1C, 120 окон",
            "reused_not_recollected": (
                "метки уже собраны вслепую и заново не собираются: повторный сбор после "
                "знакомства с результатом был бы подгонкой"
            ),
            "classes": {"STANDS": "человек стоит", "WALKS": "человек идёт",
                        "UNCLEAR": "в разбор не идёт"},
        },
        "decision_rule": {
            "primary": "AUC канала для «стоит» против «идёт» внутри полосы 12–28",
            "usable_auc": 0.70,
            "weak_auc": 0.60,
            "secondary": "для лучшего порога канала — пара (ложный STOP, пропущено стояния) по "
                         "всем окнам, в сравнении с (0 из 40, 32 из 49) у видео-гейта",
            "fixed_in_advance": (
                "пороги AUC выбраны сейчас, до чисел; после просмотра ни одно окно не "
                "исключается и веса трёх пар не пересматриваются"
            ),
        },
        "caveat_recorded_in_advance": {
            "the_channel_answers_a_different_question": (
                "NO_NET_DISPLACEMENT включает человека, который шёл и вернулся — именно поэтому "
                "канал был отклонён как гейт ранее"
            ),
            "good_result_reading": (
                "окно, где человек вообще не двигался, должен читаться как NO_NET при любом "
                "разумном определении, поэтому хороший результат информативен"
            ),
            "poor_result_reading": (
                "плохой результат НЕ доказывает, что канал плох: «нет чистого перемещения за семь "
                "секунд» и «не двигается прямо сейчас» — разные утверждения"
            ),
        },
        "prior_evidence": PRIOR,
        "manifest": manifest,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 96)
    print("P13.1D — ТЕСТ net_displacement ЗАМОРОЖЕН")
    print("=" * 96)
    print(f"  вопрос: {doc['question']}")
    print()
    print(f"  читатель: {doc['reader']['pairs']}")
    print(f"  обучен на: {', '.join(doc['reader']['fitted_on'])}")
    print(f"  проверяем на: {len(doc['reader']['tested_on'])} кусках, ни одного своего окна")
    print()
    print(f"  метки: {doc['labels']['source']}")
    print(f"  правило: AUC полосы 12–28 ≥ {doc['decision_rule']['usable_auc']} — годится; "
          f"≥ {doc['decision_rule']['weak_auc']} — слабо; ниже — не годится")
    print()
    print("  ОГОВОРКА, ЗАПИСАННАЯ ЗАРАНЕЕ:")
    print("    канал отвечает на другой вопрос, чем гейт. Хороший результат информативен;")
    print("    плохой — не доказывает, что канал плох.")
    print()
    for rel, m in manifest.items():
        print(f"  {m['sha256'][:16]}  {m['bytes']:>8}  {rel}")
    print()
    print(f"  записано: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
