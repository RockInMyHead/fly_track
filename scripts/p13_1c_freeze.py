#!/usr/bin/env python3
"""P13.1C — freeze the threshold change to 28, and the sample that will test it.

WHY THIS PHASE IS NEEDED, AND WHY IT IS NOT THE SAME TEST AGAIN
--------------------------------------------------------------
P13.1B found that the stop-gate never freezes real walking: 0 of 30 frozen windows were a person
walking. It also found, in the same sixty labels, that the gate misses a great deal — 12 of the 30
windows it let through were standing, and the separation between standing and walking in that data
begins near a motion of 30, far above the threshold of 12.

Raising the threshold is therefore a candidate improvement. But choosing it from the labels that
judge it is the same circularity this whole exercise exists to avoid: the quantity that defines the
class cannot pick the threshold on the same set. P13.1B was run to escape that circle; running its
result back into the threshold would step back inside it.

So the threshold is fixed here, in advance, at 28, together with the sample that will test it. What
is frozen is not only the number but the whole shape of the test, because the shape is what a
reviewer's answers depend on.

WHAT IS FROZEN
--------------
    threshold   28, to be kept whatever the answers say. No re-picking 27 or 31 afterwards.
    bands       forty clips from [12, 28), ten from below 12, ten from 28 and above. The middle
                band is where the change acts: those moments are MOVE under 12 and STOP under 28,
                and they are the only place where the two thresholds differ. The outer ten are
                controls — the safe behaviour at the bottom, and the behaviour that must not
                change at the top.
    sample      new episodes only, at least fifteen seconds from any window already reviewed, so
                that no clip is a second look at a moment already judged.
    indicator   unchanged: the frame-difference measure of scripts/p13_stillness.py.
    code        hashes, so a later edit cannot pass unnoticed.

Usage:
    PYTHONPATH=. python scripts/p13_1c_freeze.py
    PYTHONPATH=. python scripts/p13_1c_freeze.py --verify
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
sys.path.insert(0, str(ROOT))
DEST = ROOT / "data/p13/FROZEN_THRESHOLD_28.json"

CODE = (
    "scripts/p13_1c_build.py",
    "scripts/p13_1b_build.py",        # cut(), motion_series() reused from here
    "scripts/p13_stillness.py",       # the indicator
    "scripts/p13_1b_server.py",       # the review page, unchanged
    "scripts/p13_1c_score.py",
    "data/p13/FROZEN_STILLNESS_GATE.json",
    "data/p13/p13_1b_key.json",       # the windows already reviewed, to stay away from
)

# P13.1B, the evidence that motivates the change
PRIOR = {
    "false_stop_at_12": {"stands": 30, "walks": 0,
                         "reading": "0 из 30: гейт не выбрасывает ходьбу"},
    "missed_standing_at_12": {"walks": 18, "stands": 12,
                              "reading": "12 из 30 окон класса MOVE оказались стоянием"},
    "boundary_in_that_sample": {"below_28_all_standing": True,
                                "first_walking_motion": 33.2,
                                "reading": "ниже движения 28 стоят 36 из 36 окон"},
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
        print(f"  порог {d['threshold']}, заморожено {d['frozen_at']}")
        return 0

    manifest = {}
    missing = []
    for rel in CODE:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        manifest[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    if missing:
        print(f"СТОП: нет файлов {missing}")
        print("  заморозка должна ссылаться на существующий код, иначе проверять нечего")
        return 1

    doc = {
        "phase": "P13.1C — проверка порога 28",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "refrozen_note": a.note,
        "refrozen_rule": (
            "пересоздавать заморозку можно только пока ни один клип не размечен; после первого "
            "ответа человека это уже подгонка"
        ),
        "what_changes": "STOP-порог 12 -> 28; ничего больше",
        "what_does_not_change": (
            "P13.1 с порогом 12 остаётся доказанным безопасным вариантом. P13.1C проверяет "
            "только полноту: ловит ли более высокий порог больше стояния, не начиная при этом "
            "выбрасывать ходьбу"
        ),
        "why_not_chosen_from_p13_1b": (
            "28 выбран по меткам P13.1B, а те же метки его проверять не могут: величина, "
            "задающая класс, не может на том же множестве выбирать порог. Поэтому порог "
            "зафиксирован заранее и проверяется на новых эпизодах"
        ),
        "threshold": 28.0,
        "threshold_is_final": (
            "после разметки запрещено выбирать между 27, 28 и 31 или как-либо иначе подбирать "
            "порог по ответам; тестируется ровно 28"
        ),
        "bands": {
            "middle": {"range": [12.0, 28.0], "n": 40,
                       "why": "здесь порог меняет поведение: при 12 это MOVE, при 28 STOP"},
            "low": {"range": [0.0, 12.0], "n": 10,
                    "why": "контроль старого безопасного STOP"},
            "high": {"range": [28.0, 1e9], "n": 10,
                     "why": "контроль MOVE, который не должен измениться"},
        },
        "sample_rules": {
            "clean_run_s": 2.5,
            "separation_from_reviewed_s": 15.0,
            "separation_within_new_s": 15.0,
            "moment": "центр чистого участка",
            "clip_s": 5.0,
            "moment_position": "ровно в середине клипа, у всех одинаково",
            "new_episodes_only": (
                "окна, уже показанные человеку в P13.1B, исключены: ни один клип не должен быть "
                "вторым взглядом на уже оценённый момент"
            ),
        },
        "blindness": {
            "shown": ["id", "file", "момент внутри клипа"],
            "not_shown": ["video_motion", "диапазон", "STOP/MOVE", "порог", "ролик",
                          "скорость трекера", "длина чистого участка"],
            "same_length": "все клипы ровно 5 с, момент у всех в одном месте",
        },
        "main_metric": "FALSE STOP при пороге 28 = движение < 28 И человек сказал ИДЁТ",
        "also_reported": {
            "middle_band": "сколько СТОИТ, ИДЁТ, НЕЯСНО среди сорока окон 12–28",
            "comparison": "порог 12 против 28 по FALSE STOP и MISSED STOP",
        },
        "verdict_rule": {
            "replace_12_with_28_if": (
                "у 28 снова практически нет FALSE STOP и он заметно уменьшает MISSED STOP"
            ),
            "keep_12_if": "появляются реальные случаи ходьбы ниже 28",
        },
        "limitation": (
            "набор из тех же девяти отложенных роликов проверяет новые эпизоды, но не новые "
            "условия записи; окончательное подтверждение потребует следующего нового видео"
        ),
        "prior_evidence": PRIOR,
        "manifest": manifest,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 96)
    print("P13.1C — ПОРОГ 28 ЗАМОРОЖЕН")
    print("=" * 96)
    print(f"  меняется: {doc['what_changes']}")
    print(f"  порог {doc['threshold']}, окончательно; подбор после разметки запрещён")
    print()
    b = doc["bands"]
    for k, name in (("middle", "12–28"), ("low", "<12"), ("high", "≥28")):
        print(f"  {name:<7} {b[k]['n']:>3} клипов   {b[k]['why']}")
    print(f"  всего {sum(b[k]['n'] for k in b)}")
    print()
    print(f"  чистый участок ≥{doc['sample_rules']['clean_run_s']} с, "
          f"разнос ≥{doc['sample_rules']['separation_from_reviewed_s']:.0f} с")
    print(f"  главная метрика: {doc['main_metric']}")
    print()
    for rel, m in manifest.items():
        print(f"  {m['sha256'][:16]}  {m['bytes']:>8}  {rel}")
    print()
    print(f"  записано: {DEST}")
    print("  Дальше: scripts/p13_1c_build.py — нарезать набор. Разметка только после этого.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
