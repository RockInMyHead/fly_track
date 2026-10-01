#!/usr/bin/env python3
"""P10 step 0 — freeze the sixty drawn moves and the names they will be called by.

The review that produced these labels was run twice, and the second run is the one that counts:
the reviewer drew the path the person took instead of picking a word, and the three answers he
could give were named after the outcome of the movement rather than after a hypothesis about
heads and cameras.

    LOOK            -> SAME_DIRECTION        the path ended pointing where it started
    TURN            -> DIFFERENT_DIRECTION   the path ended pointing elsewhere
    NO_LOCOMOTION   -> NO_NET_DISPLACEMENT   the path went out and came back

The third name is the important one and it is deliberately not "STATIC". The person was moving
the whole time — a metre forward, a metre back, in his own words — so nothing about these moments
is still. What is absent is not motion but *net* motion: he returned to roughly where he began.
Calling that "static" would invite the mistake this whole phase exists to avoid, namely reading a
missing direction as a direction.

Frozen alongside the labels:

    P08.6   the route algorithm, unchanged, still the tracker's front end
    P09.2   NOT used as a filter anywhere. It was measured against these labels in P09.5D and
            withdrawn; nothing in P10 reintroduces it.

Usage:
    PYTHONPATH=. python scripts/p10_freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RENAME = {
    "LOOK": "SAME_DIRECTION",
    "TURN": "DIFFERENT_DIRECTION",
    "NO_LOCOMOTION": "NO_NET_DISPLACEMENT",
}
# Old names must not survive into the P10 files — they carry the head/camera reading that the
# reviewers were explicitly asked to stop using.
ORDER = ("SAME_DIRECTION", "DIFFERENT_DIRECTION", "NO_NET_DISPLACEMENT")

WHY = (
    "P10 measures one thing: at which level of the pipeline — raw video, brain input, early visual "
    "cells, descending cells, or the two current readouts — the difference between these three "
    "kinds of movement is still present. Nothing is repaired in this phase. The levels are all "
    "measured with the same classifier and the same held-out-clip split, so the answer is which "
    "level to work on next, not a new percentage on the current signal."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DATA / "p10"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    review_path = DATA / "p095/review_set_v2.json"
    labels_path = DATA / "p095/human_labels_v2.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    raw = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]

    events = {e["event_id"]: e for e in review["events"]}
    first = [l for l in raw if l.get("round", 1) == 1]
    labels = []
    for l in first:
        e = events[l["event_id"]]
        labels.append({
            "event_id": l["event_id"],
            "clip": e["clip"],
            "time": float(e["time"]),
            "clip_file": e["clip_file"],
            "clip_offset_s": float(e["clip_offset_s"]),
            "narrow": [float(x) for x in e["narrow"]],
            "category": RENAME[l["human_label"]],
            "old_name": l["human_label"],
            "drawn_from_drawing": l.get("label_source") == "derived_from_drawing",
            "net_deg": l.get("derived_turn_net_deg"),
            "total_deg": l.get("derived_turn_total_deg"),
        })
    labels.sort(key=lambda x: x["event_id"])

    counts = Counter(l["category"] for l in labels)
    missing = [c for c in ORDER if c not in counts]
    if missing:
        print(f"СТОП: нет событий категории {missing}")
        return 1

    doc = {
        "phase": "P10 — где теряется информация о движении тела",
        "what_this_is": WHY,
        "n_events": len(labels),
        "categories": list(ORDER),
        "category_counts": {c: counts[c] for c in ORDER},
        "old_to_new": RENAME,
        "why_not_static": (
            "NO_NET_DISPLACEMENT means the drawn path left and came back: path length is close to "
            "the other two categories while net displacement is about half. The person walked; he "
            "did not stand. The name says no net displacement, never no motion."
        ),
        "labels": labels,
        "sources": {
            "review_set": {"path": str(review_path.relative_to(ROOT)), "sha256_16": sha(review_path)},
            "human_labels_v2": {"path": str(labels_path.relative_to(ROOT)), "sha256_16": sha(labels_path)},
        },
        "frozen": {
            "route_algorithm": "P08.6 (multi-hypothesis at a fork, unchanged)",
            "look_turn_filter": "P09.2 — НЕ ИСПОЛЬЗУЕТСЯ. Снят в P09.5D; в P10 не возвращается.",
            "visual_encoder": "не менялся",
            "graph": "не менялась",
            "cells": "не менялись",
            "thresholds": "не менялись",
        },
        "rule_of_the_phase": (
            "только диагностика: ни одна клетка, вес, порог, encoder или граф не меняются, пока не "
            "найден уровень, на котором различие исчезает"
        ),
    }
    dest = out / "FROZEN_P10.json"
    dest.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 92)
    print("P10 — ЗАМОРОЖЕНО")
    print("=" * 92)
    print(f"  событий: {len(labels)}")
    for c in ORDER:
        print(f"    {c:<22} {counts[c]:>3}")
    print()
    print("  переименование (старое имя больше не используется):")
    for old, new in RENAME.items():
        print(f"    {old:<16} -> {new}")
    print()
    print(f"  P09.2: не используется")
    print(f"  записано: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
