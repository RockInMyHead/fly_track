#!/usr/bin/env python3
"""P13.1C — compare the frozen threshold of 28 against the proven 12, on the new labels.

The threshold was fixed before the clips were cut, and this script does not revisit it. It reports
the two errors that matter for each threshold and the same two for the band that motivated the
change, then applies the rule that was written down in advance.

    FALSE STOP    frozen, but the person was walking. Real path thrown away. The dangerous one.
    MISSED STOP   let through, but the person was standing. Invented path kept.

For a threshold of 12 and one of 28 the two are computed from the same answers: a clip belongs to
the STOP side when its motion is below the threshold, and to the MOVE side otherwise. Nothing is
refitted, and no neighbouring threshold is examined — 27 and 31 are not on the table.

Usage:
    PYTHONPATH=. python scripts/p13_1c_score.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
DATA = ROOT / "data/p13"
FREEZE = DATA / "FROZEN_THRESHOLD_28.json"
KEY = DATA / "p13_1c_key.json"
ANSWERS = OUT / "p13_1c_answers.json"
DEST = OUT / "p13_1c_score.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()

    fr = json.loads(FREEZE.read_text(encoding="utf-8"))
    key = {i["id"]: i for i in json.loads(KEY.read_text(encoding="utf-8"))["items"]}
    ans = {}
    if ANSWERS.exists():
        ans = json.loads(ANSWERS.read_text(encoding="utf-8")).get("answers", {})

    print("=" * 96)
    print("P13.1C — ПОРОГ 28 ПРОТИВ ПОРОГА 12")
    print("=" * 96)
    print(f"  порог заморожен {fr['frozen_at']}: {fr['threshold']}")
    print(f"  размечено {len(ans)} из {len(key)}")
    print()

    missing = [i for i in key if i not in ans]
    if missing:
        print(f"  СТОП: не размечено {len(missing)} клипов: {missing[:6]}")
        return 1

    def tally(th: float) -> dict:
        out = {"false_stop": 0, "true_stop": 0, "true_move": 0, "missed_stop": 0, "unclear": 0}
        for cid, k in key.items():
            a = ans[cid]
            frozen = k["motion"] < th
            if a == "UNCLEAR":
                out["unclear"] += 1
                continue
            if frozen and a == "WALKS":
                out["false_stop"] += 1
            elif frozen and a == "STANDS":
                out["true_stop"] += 1
            elif not frozen and a == "WALKS":
                out["true_move"] += 1
            else:
                out["missed_stop"] += 1
        n_stop = out["false_stop"] + out["true_stop"]
        n_move = out["true_move"] + out["missed_stop"]
        out["n_stop_side"] = n_stop
        out["n_move_side"] = n_move
        out["false_stop_rate"] = out["false_stop"] / n_stop if n_stop else None
        out["missed_stop_rate"] = out["missed_stop"] / n_move if n_move else None
        return out

    r12 = tally(12.0)
    r28 = tally(28.0)

    print(f"  {'':<18}{'порог 12':>12}{'порог 28':>12}")
    print("  " + "-" * 42)
    for label, k in (("заморожено, стоя", "true_stop"),
                     ("ЛОЖНЫЙ STOP", "false_stop"),
                     ("пропущено, идёт", "true_move"),
                     ("ПРОПУЩЕНО СТОЯНИЯ", "missed_stop"),
                     ("неясно", "unclear")):
        print(f"  {label:<18}{r12[k]:>12}{r28[k]:>12}")
    print()
    print(f"  FALSE STOP:  при 12 — {100*(r12['false_stop_rate'] or 0):.0f}% "
          f"({r12['false_stop']} из {r12['n_stop_side']} замороженных)")
    print(f"               при 28 — {100*(r28['false_stop_rate'] or 0):.0f}% "
          f"({r28['false_stop']} из {r28['n_stop_side']} замороженных)")
    print()
    print(f"  ПРОПУЩЕНО СТОЯНИЯ: при 12 — {r12['missed_stop']} из {r12['n_move_side']}, "
          f"при 28 — {r28['missed_stop']} из {r28['n_move_side']}")
    print()

    # the band the change acts on
    mid = [c for c in key if key[c]["band"] == "middle"]
    cnt = {"STANDS": 0, "WALKS": 0, "UNCLEAR": 0}
    for c in mid:
        cnt[ans[c]] += 1
    print(f"  ПОЛОСА 12–28, где порог меняет поведение ({len(mid)} клипов):")
    print(f"    человек СТОИТ : {cnt['STANDS']}")
    print(f"    человек ИДЁТ  : {cnt['WALKS']}   <- это и был бы ложный STOP при пороге 28")
    print(f"    неясно        : {cnt['UNCLEAR']}")
    print()

    prev = {
        "at_12_from_p13_1b": {"false_stop": 0, "n_stop": 30,
                              "missed_stop": 12, "n_move": 30},
    }
    print("  ДЛЯ СРАВНЕНИЯ, P13.1B (другой набор, порог 12):")
    print(f"    ложных STOP 0 из 30, пропущено стояния 12 из 30")
    print()

    verdict = ("REPLACE_12_WITH_28"
               if (r28["false_stop"] == 0 or (r28["false_stop_rate"] or 1) <= 0.05)
               and r28["missed_stop"] < r12["missed_stop"]
               else "KEEP_12")
    print("  ВЕРДИКТ ПО ПРАВИЛУ, НАПИСАННОМУ ЗАРАНЕЕ:")
    if verdict == "REPLACE_12_WITH_28":
        print(f"    у 28 практически нет ложных STOP ({r28['false_stop']}) и пропущенного "
              f"стояния меньше ({r28['missed_stop']} против {r12['missed_stop']})")
        print("    → заменить 12 на 28 в следующей версии трекера")
        print("    оговорка: это новые эпизоды на тех же девяти записях; окончательное")
        print("    подтверждение — на следующем новом видео")
    else:
        print(f"    при 28 появляются случаи ходьбы ниже порога: ложных STOP "
              f"{r28['false_stop']} из {r28['n_stop_side']}")
        print("    → 28 не внедрять, оставить 12 до новых данных")

    DEST.write_text(json.dumps({
        "phase": "P13.1C — порог 28 против 12",
        "threshold_frozen": fr["threshold"],
        "n_answered": len(ans),
        "at_12": r12, "at_28": r28,
        "middle_band": {**cnt, "n": len(mid)},
        "prior_p13_1b": prev,
        "verdict": verdict,
        "note": "порог не подбирался: 27 и 31 не рассматривались",
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n  записано: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
