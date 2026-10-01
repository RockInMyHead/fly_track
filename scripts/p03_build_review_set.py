#!/usr/bin/env python3
"""
Build a *balanced* review set, where agreement cannot be earned by default.

Why the first set could not test anything
-----------------------------------------
The candidate list contained only yaw moments: it had been selected as the top 3% of
frames by sideways signal strength, so every item was labelled LEFT or RIGHT by
construction. A reviewer who agrees with all of them has demonstrated nothing,
because the correct answer was always "there is a turn here". Confirming 37/37 is
compatible both with "the measurement is right" and with "the reviewer approved
everything".

What makes a test informative
-----------------------------
The set must contain windows where the right answer is *not* a turn, in roughly equal
numbers, and they must be shuffled together so the class cannot be guessed from order:

    LEFT    the camera rotated left
    RIGHT   the camera rotated right
    FWD     the camera moved forward, no rotation
    STATIC  nothing moved
    MIXED   neither dominates, or the field is not explainable by ego-motion

A reviewer who is actually watching must reject the FWD and STATIC items for the test
to pass. If they approve those too, the approval carries no information — and that is
exactly what needs to be measured.

Windows are taken from `windows_labelled.json`, which was produced by a sequential scan
in true content time, and are sorted by confidence so the clearest example of each class
is offered first.

Usage:
    PYTHONPATH=. python scripts/p03_build_review_set.py
    PYTHONPATH=. python scripts/p03_build_review_set.py --per-class 12 --seed 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "data/p01r/windows_labelled.json"
OUT = ROOT / "data/p01r/review_set.json"

# Minimum length for a window to be worth judging: below this the reviewer cannot see
# enough of anything, in either direction.
MIN_LEN_S = 1.0
MIN_CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12,
                    help="how many windows to offer per class")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    doc = json.loads(WINDOWS.read_text(encoding="utf-8"))
    windows = doc["windows"]

    by_class: dict[str, list[dict]] = {}
    for w in windows:
        if w["duration_s"] < MIN_LEN_S:
            continue
        by_class.setdefault(w["kind"], []).append(w)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for kind in ("LEFT", "RIGHT", "FWD", "STATIC", "MIXED"):
        pool = by_class.get(kind, [])
        # prefer confident, clear examples, then take a spread across the clip
        pool = sorted(pool, key=lambda w: (MIN_CONF_RANK.get(w["confidence"], 3),
                                           -abs(w.get("a", 0.0))))
        if len(pool) > args.per_class:
            # spread over the whole clip rather than taking the first N
            step = len(pool) / args.per_class
            pool = [pool[int(i * step)] for i in range(args.per_class)]
        for w in pool:
            item = dict(w)
            item["id"] = f"{kind[:1].lower()}{len(picked) + 1:03d}"
            item["expected"] = {
                "LEFT": "camera rotated LEFT",
                "RIGHT": "camera rotated RIGHT",
                "FWD": "moved forward, no rotation",
                "STATIC": "nothing moved",
                "MIXED": "unclear, or mixed motions",
            }[kind]
            picked.append(item)

    rng.shuffle(picked)
    for i, it in enumerate(picked, 1):
        it["id"] = f"r{i:03d}"

    from collections import Counter
    cnt = Counter(it["kind"] for it in picked)
    print("сбалансированный набор для проверки")
    print(f"  окон: {len(picked)}  {dict(cnt)}")
    print(f"  каждой категории примерно поровну, порядок перемешан (seed {args.seed})")
    print(f"\n  {'id':>5s} {'класс':>7s} {'окно, с':>14s} {'длит':>6s} {'a':>7s} "
          f"{'div':>6s} {'увер':>7s}")
    for it in picked:
        print(f"  {it['id']:>5s} {it['kind']:>7s} {it['t0']:7.1f}-{it['t1']:<7.1f} "
              f"{it['duration_s']:6.2f} {it['a']:+7.2f} {it['divergence']:+6.2f} "
              f"{it['confidence']:>7s}")

    Path(args.out).write_text(json.dumps({
        "purpose": "Сбалансированный набор: четверть окон не является поворотами. "
                   "Согласие со всеми пунктами ничего не доказывает, поэтому отказ "
                   "от FWD/STATIC так же важен, как подтверждение поворотов.",
        "source": str(WINDOWS),
        "per_class": args.per_class, "seed": args.seed,
        "counts": dict(cnt), "n": len(picked),
        "candidates": picked,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
