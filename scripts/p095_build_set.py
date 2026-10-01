#!/usr/bin/env python3
"""P09.5 — build the first blind set of moves for a person to judge.

Everything so far has been one automatic rule checked against another. P09.4 came as close as
that can get: it showed the frozen filter withdraws 8 percent of the moves its own labelling calls
turns, once the ratio is measured on the signal the pair is built from. But its labels are P07.4's,
and P07.4 calls a glance exactly the pair that cancels — so its LOOK label and the pair structure
are very nearly one variable (all 516 glances are paired; 250 of 268 turns are not). Agreement
with it is largely reproduction, not accuracy. This script sets up the first measurement that is
not circular.

What a person is shown, and what they are not
---------------------------------------------
A clip around the move, and three buttons. Not the algorithmic class, not the yaw, not the score,
not the direction, not the clip's identity beyond a number, and nothing about which class the
event was drawn from. The order is shuffled across clips so that consecutive events do not give
the pattern away, and the same event may come back later without being marked as a repeat.

The selection is 20 from each of the three algorithmic classes, weighted towards VID00005 and
VID00009, which took no part in building the rule. The classes are used only to balance the
sample — the person never sees them — and the split is recorded here so the sampling is
reproducible and can be argued about.

The clip window is fixed for everyone: three seconds before the move and five after. A wider
window is available behind a button, but the first thing everyone sees is the same length, so
that a difference between events cannot come from the video being longer for some of them.

What is frozen while this runs
------------------------------
Everything. See `data/p095/FROZEN_P094.json`. The rule, its threshold, the signals, the pair
window, the policy and the graph are not touched until the labels are in, because otherwise the
rule and the truth would move at the same time and the comparison would mean nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p092_look_turn as LT  # noqa: E402

OUT = ROOT / "output/p095"
DATA = ROOT / "data/p095"
MEDIA = ROOT / "webapp/media"
CLIPS = ROOT / "output/p095/clips"
P094 = ROOT / "output/p094"

# Sampling. The three classes are balanced at 20 each, and within a class the clips are weighted
# towards the two that were held out. The exact numbers are here rather than computed so that the
# sample can be re-drawn identically and checked.
PER_CLASS = 20
FOCUS = {"VID00005": 8, "VID00009": 8, "VID00001": 2, "VID00002": 1, "VID00006": 1}
UNKNOWN_SPLIT = {"VID00005": 7, "VID00009": 7, "VID00001": 2, "VID00002": 2, "VID00006": 2}
SEED = 950
N_REPEATS = 10

PRE_S, POST_S = 3.0, 5.0          # the window everyone sees first
WIDE_PRE_S, WIDE_POST_S = 5.0, 8.0  # behind the "show wider" button; the file covers this


def strong_events() -> list[dict]:
    """The events the filter is ever invoked on, with the middle of the move for the clip.

    The clip is centred on the *move*, not on the moment the tracker would have decided, because
    the question asked of the person is about the move. For a swing that is the swing's own span;
    for an event with no swing it is the stretch where the signal was strong.
    """
    out = []
    for name in ("pair_found.csv", "pair_not_found.csv"):
        p = P094 / name
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            t0, t1 = float(r["t0"]), float(r["t1"])
            out.append({
                "clip": r["clip"], "kind": "SWING", "t_decision": float(r["t"]),
                "move_start": t0, "move_end": t1, "move_center": 0.5 * (t0 + t1),
                "algo_class_b": (r["class_gated"] or "UNKNOWN"),
                "algo_score_b": float(r["score_gated"]) if r["score_gated"] else None,
                "algo_class_a": (r["class_deadband"] or "UNKNOWN"),
                "algo_score_a": float(r["score_deadband"]) if r["score_deadband"] else None,
                "algo_direction": r["yaw_direction"],
                "algo_yaw_integral": float(r["yaw_integral"]),
                "pair_status": r["pair_status"], "proxy_label": r["truth"],
            })
    # events with no swing: the strong stretches outside every labelled oscillation
    p = P094 / "outside_oscillations.csv"
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8")):
            a, b = float(r["region_start"]), float(r["region_end"])
            out.append({
                "clip": r["clip"], "kind": "NO_SWING", "t_decision": b,
                "move_start": a, "move_end": b, "move_center": 0.5 * (a + b),
                "algo_class_b": "UNKNOWN", "algo_score_b": None,
                "algo_class_a": "UNKNOWN", "algo_score_a": None,
                "algo_direction": "", "algo_yaw_integral": None,
                "pair_status": "NO_SWING", "proxy_label": "UNKNOWN",
            })
    return out


def draw(events: list[dict], rng: random.Random) -> list[dict]:
    """Twenty from each algorithmic class, weighted towards the held-out clips."""
    by_class = {"LOOK": [], "TURN": [], "UNKNOWN": []}
    for e in events:
        by_class[e["algo_class_b"]].append(e)
    chosen = []
    for cls in ("LOOK", "TURN", "UNKNOWN"):
        pool = by_class[cls]
        split = UNKNOWN_SPLIT if cls == "UNKNOWN" else FOCUS
        for clip, want in split.items():
            avail = [e for e in pool if e["clip"] == clip]
            rng.shuffle(avail)
            take = avail[:want]
            if len(take) < want:
                print(f"  внимание: {cls} в {clip} только {len(take)} из {want}")
            chosen += take
        # top up from the whole pool if a clip was short, so each class reaches its 20
        if len([e for e in chosen if e["algo_class_b"] == cls]) < PER_CLASS:
            have = {id(e) for e in chosen}
            rest = [e for e in pool if id(e) not in have]
            rng.shuffle(rest)
            need = PER_CLASS - len([e for e in chosen if e["algo_class_b"] == cls])
            chosen += rest[:need]
    return chosen


def cut(clip: str, center: float, dest: Path, pre: float, post: float) -> bool:
    """Cut one file per event, wide enough for both windows.

    One file rather than two: the page plays the narrow window out of it and the wider button
    starts the same file earlier, so a person cannot be shown a different recording for the same
    move depending on which button they pressed.
    """
    src = MEDIA / f"{clip}_fixed.mp4"
    if not src.exists() or dest.exists():
        return dest.exists()
    start = max(center - pre, 0.0)
    dur = post + pre
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{dur:.3f}", "-vf", "scale=854:-2", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "30", "-an", "-movflags", "+faststart", str(dest)]
    return subprocess.run(cmd).returncode == 0


def main() -> int:
    global P094, FOCUS, UNKNOWN_SPLIT, SEED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--clips-dir", default=str(CLIPS))
    ap.add_argument("--no-cut", action="store_true")
    ap.add_argument("--p094", default=None, help="каталог с событиями (по умолчанию output/p094)")
    ap.add_argument("--focus-clip", default=None,
                    help="собрать набор только по этому клипу, по PER_CLASS событий на класс")
    ap.add_argument("--out-json", default=None, help="куда писать набор")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # The candidate-event builder wrote one directory per earlier phase, and a set for a new clip
    # has to read that clip's own events rather than the five-clip directory. Drawing from the
    # wrong directory would silently produce a bank whose timestamps belong to other recordings —
    # the same class of error as reading the wrong video.
    if args.p094:
        P094 = Path(args.p094) if str(args.p094).startswith("/") else ROOT / args.p094
    SEED = args.seed
    if args.focus_clip:
        FOCUS = {args.focus_clip: PER_CLASS}
        UNKNOWN_SPLIT = {args.focus_clip: PER_CLASS}
    data_dir, clips_dir = Path(args.out), Path(args.clips_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("P09.5 — НАБОР ДЛЯ СЛЕПОЙ РУЧНОЙ РАЗМЕТКИ LOOK / TURN")
    print("=" * 100)
    print(f"  события берутся из {P094}")
    if args.focus_clip:
        print(f"  набор только по клипу {args.focus_clip}, по {PER_CLASS} на класс")
    events = strong_events()
    print(f"  событий с названным направлением всего: {len(events)}")

    rng = random.Random(SEED)
    chosen = draw(events, rng)
    print(f"  отобрано: {len(chosen)} (по {PER_CLASS} на класс)")
    for cls in ("LOOK", "TURN", "UNKNOWN"):
        sel = [e for e in chosen if e["algo_class_b"] == cls]
        per = {}
        for e in sel:
            per[e["clip"]] = per.get(e["clip"], 0) + 1
        print(f"    {cls:<8}{len(sel):>4}   " +
              "  ".join(f"{k}:{v}" for k, v in sorted(per.items())))
    focus = sum(1 for e in chosen if e["clip"] in ("VID00005", "VID00009"))
    print(f"  из них на VID00005+VID00009 (не участвовали в построении): {focus} "
          f"({focus/len(chosen):.0%})")
    print()

    # ---- cut the clips -------------------------------------------------------------------
    order = list(range(len(chosen)))
    rng.shuffle(order)
    review = []
    for n, i in enumerate(order, 1):
        e = chosen[i]
        eid = f"e{n:03d}"
        dest = clips_dir / f"{eid}.mp4"
        ok = True if args.no_cut else cut(e["clip"], e["move_center"], dest,
                                         WIDE_PRE_S, WIDE_POST_S)
        review.append({
            "event_id": eid, "clip": e["clip"], "time": round(e["move_center"], 3),
            "clip_file": f"clips/{eid}.mp4",
            "clip_offset_s": round(WIDE_PRE_S, 3),           # where the file starts, rel. centre
            "narrow": [round(WIDE_PRE_S - PRE_S, 3), round(WIDE_PRE_S + POST_S, 3)],
            "wide": [0.0, round(WIDE_PRE_S + WIDE_POST_S, 3)],
            "cut_ok": bool(ok),
            # hidden from the reviewer; joined only after all answers are in
            "hidden": {
                "algo_class_b": e["algo_class_b"], "algo_score_b": e["algo_score_b"],
                "algo_class_a": e["algo_class_a"], "algo_score_a": e["algo_score_a"],
                "algo_direction": e["algo_direction"],
                "algo_yaw_integral": e["algo_yaw_integral"],
                "proxy_label": e["proxy_label"], "pair_status": e["pair_status"],
                "kind": e["kind"], "t_decision": e["t_decision"],
                "move_start": e["move_start"], "move_end": e["move_end"],
            },
        })
    missing = [r["event_id"] for r in review if not r["cut_ok"]]
    print(f"  нарезано клипов: {len(review) - len(missing)} из {len(review)}")
    if missing:
        print(f"  НЕ нарезаны: {missing[:10]}")
    print()

    # ---- the repeat round -----------------------------------------------------------------
    # Drawn from the same sixty and placed in a second pass, after the first is finished, so that
    # the count "N of 60" stays honest and the two viewings are genuinely separated in time.
    rep = rng.sample([r["event_id"] for r in review], N_REPEATS)
    print(f"  на повторный показ во втором проходе отобрано: {len(rep)}")
    print("  (человеку не сообщается, что это повторы)")
    print()

    payload = {
        "phase": "P09.5 — первая слепая ручная разметка LOOK/TURN",
        "seed": SEED, "per_class": PER_CLASS, "focus_split": FOCUS,
        "unknown_split": UNKNOWN_SPLIT, "n_repeats": N_REPEATS,
        "window_first": {"before_s": PRE_S, "after_s": POST_S},
        "window_wide": {"before_s": WIDE_PRE_S, "after_s": WIDE_POST_S},
        "clips": sorted({r["clip"] for r in review}),
        "focus_clips": ["VID00005", "VID00009"],
        "no_algorithmic_info_shown": True,
        "events": review,
        "repeat_round": rep,
        "frozen": str(data_dir / "FROZEN_P094.json"),
    }
    # The destination used to be written out literally as `data_dir / "review_set.json"`, so a
    # caller asking for another file silently got the round-one set overwritten instead. A script
    # that accepts an output path must honour it, or the parameter is worse than absent: it reads
    # like protection that is not there.
    out_json = (Path(args.out_json) if str(args.out_json).startswith("/")
                else ROOT / args.out_json) if args.out_json else data_dir / "review_set.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out_json}")
    print("разметку запускает scripts/p095_review_server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
