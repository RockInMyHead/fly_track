#!/usr/bin/env python3
"""P09.5B — the same sixty moves, asked about in different words.

The first pass asked a person to decide whether the camera's rotation was a glance or a turn,
and defined those words on the page. The definitions were the problem. They described the
mechanism — head turned, body did not — which is exactly the distinction the machine cannot make
and which is hard to see in a first-person recording where your own body is not in frame. The
labels that came back disagreed with the frozen rule far more than any measurement so far, and one
of the disagreements was on the very event the rule was built for. That is either a real failure
of the rule or a failure of the question, and the two cannot be told apart from the labels.

So this asks the same sixty moves again, with one change: the question is about the outcome, not
the mechanism.

    «После отмеченного момента человек продолжил идти в прежнем направлении
     или окончательно пошёл в новом?»

    В ТУ ЖЕ СТОРОНУ   -> LOOK
    В ДРУГУЮ СТОРОНУ  -> TURN
    НЕ МОГУ ПОНЯТЬ    -> UNCLEAR

The words glance and turn do not appear on the page at all. They are internal names, recorded in
the file, because the whole point of the exercise is that those two words drag a person toward
thinking about heads and bodies instead of watching where the walker ended up.

What is held fixed
------------------
The sixty events, their clips and the algorithmic data are the same, so the two passes are
comparable event by event. What changes is the wording, the order — reshuffled with a different
seed, so the second pass cannot be answered from memory of the first — and the default window,
which goes back to something short enough to hold one response: three seconds before the move and
seven after. The wider window is one button away and was already cut into the files.

The first pass is not discarded
--------------------------------
It is kept as `human_labels_v1.json` and is a first attempt at a hard question, which is worth
having. What it is not is the final test of the rule, because it may have answered a different
question than the one intended. That is what this pass is for.

Usage:
    PYTHONPATH=. python scripts/p095b_build.py
    PYTHONPATH=. python scripts/p095b_server.py        # or open the standalone page
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
OUT = ROOT / "output/p095"

SEED = 1751               # different from the first pass, so the order is not remembered
N_REPEATS = 10
CLIP_LEAD = 5.0           # the files were cut starting five seconds before the move
NARROW = (2.0, 12.0)      # 3 s before the centre .. 7 s after
WIDE = (0.0, 30.0)        # the whole file: 5 s before .. 25 s after


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    # Input and output are arguments so a second clip can be prepared without overwriting the round
    # that the earlier phases were measured on. The paths used to be constants, and asking for one
    # output while the script wrote another is how a set gets replaced by accident.
    ap.add_argument("--in", dest="src", default=None, help="набор первого прохода")
    ap.add_argument("--out", dest="dest", default=None, help="куда писать набор v2")
    args = ap.parse_args()
    src = (Path(args.src) if str(args.src).startswith("/") else ROOT / args.src) \
        if args.src else DATA / "review_set.json"
    dest = (Path(args.dest) if str(args.dest).startswith("/") else ROOT / args.dest) \
        if args.dest else DATA / "review_set_v2.json"

    s1 = json.loads(src.read_text(encoding="utf-8"))
    events = s1["events"]
    print("=" * 96)
    print("P09.5B — ПОВТОРНАЯ РАЗМЕТКА С ИСПРАВЛЕННЫМ ВОПРОСОМ")
    print("=" * 96)
    print(f"  вход: {src}")
    print(f"  событий: {len(events)} (те же, что в первом проходе)")
    print(f"  порядок: перемешан заново, seed {args.seed} (в первом проходе был 950)")
    print(f"  окно по умолчанию: 3 с до, 7 с после   ·   шире: 5 с до, 25 с после")
    print()

    # ---- keep the first pass, never overwrite it -------------------------------------------
    src = DATA / "human_labels.json"
    if src.exists():
        shutil.copy(src, DATA / "human_labels_v1.json")
        n1 = len(json.loads(src.read_text(encoding="utf-8")).get("labels", []))
        print(f"  первый проход сохранён: human_labels_v1.json ({n1} записей)")
    if (DATA / "human_labels_v1.json").exists():
        n1 = len(json.loads((DATA / "human_labels_v1.json").read_text(encoding="utf-8"))
                 .get("labels", []))
        print(f"  human_labels_v1.json на месте: {n1} записей")
    # the second pass writes its own file; nothing here touches the first
    print(f"  второй проход будет писать в human_labels_v2.json")
    print()

    # ---- reshuffle and re-window -----------------------------------------------------------
    rng = random.Random(args.seed)
    order = list(range(len(events)))
    rng.shuffle(order)
    out = []
    for n, i in enumerate(order, 1):
        e = dict(events[i])
        eid = f"r{n:03d}"                       # a new id, so v1 and v2 can never be confused
        e["event_id"] = eid
        e["clip_file"] = f"clips/{events[i]['event_id']}.mp4"   # the file keeps its old name
        e["narrow"] = [NARROW[0], NARROW[1]]
        e["wide"] = [WIDE[0], WIDE[1]]
        e["source_event_id"] = events[i]["event_id"]
        out.append(e)
    repeats = rng.sample([e["event_id"] for e in out], N_REPEATS)

    payload = {
        "phase": "P09.5B — повторная слепая разметка, исправленный вопрос",
        "seed": args.seed, "n_repeats": N_REPEATS,
        "question": "После отмеченного момента человек продолжил идти в прежнем направлении "
                    "или окончательно пошёл в новом?",
        "buttons": {"В ТУ ЖЕ СТОРОНУ": "LOOK", "В ДРУГУЮ СТОРОНУ": "TURN",
                    "НЕ МОГУ ПОНЯТЬ": "UNCLEAR"},
        "words_glance_turn_hidden": True,
        "window_first": {"before_s": 3.0, "after_s": 7.0},
        "window_wide": {"before_s": CLIP_LEAD, "after_s": 25.0},
        "why_again": (
            "первый проход определял слова «взгляд» и «поворот» через механизм — голова "
            "повернулась, тело нет. Это ровно то различие, которое машина не умеет делать и "
            "которое трудно увидеть в видео от первого лица. Новый вопрос про результат "
            "движения, а не про механизм."),
        "v1_kept_as": "human_labels_v1.json",
        "v2_file": "human_labels_v2.json",
        "events": out,
        "repeat_round": repeats,
        "frozen": str(DATA / "FROZEN_P094.json"),
    }
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- blindness: the same structural guarantee as before --------------------------------
    public = ("event_id", "clip_file", "narrow", "wide", "source_event_id")
    leaked = []
    for e in out:
        extra = set(e) - set(public) - {"hidden"}
        leaked += [k for k in extra if k in ("algo_class_b", "algo_score_b", "algo_direction",
                                             "proxy_label", "pair_status")]
    print(f"  порядок: первые пять {[e['event_id'] for e in out[:5]]}")
    print(f"  повторы: {repeats}")
    print(f"  алгоритмических полей в публичной части: {set(leaked) or 'ни одного'}")
    print(f"  записано: {dest}")
    print()
    print("  страница: PYTHONPATH=. python scripts/p095b_server.py")
    print("  или автономно: PYTHONPATH=. python scripts/p095b_standalone.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
