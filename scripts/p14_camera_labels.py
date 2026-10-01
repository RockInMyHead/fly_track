#!/usr/bin/env python3
"""P14 — the camera set's labels, with clip names that cannot collide and the frozen rules applied.

WHAT IT BUILDS
--------------
`data/p14/LABELS_CAMERA5.json` — the events of the canonical camera recordings in the P10 format.
Five recordings once the two new ones have been drawn; three until then. Each round of drawing
numbers its clips from `e001`, and the video-feature cache is keyed by clip name, so the names are
prefixed with the recording: `v03_eNNN`, `v10_eNNN`. Without that, computing features for a new
round would overwrite the previous round's under the same names — which has already happened once in
this project, to a review set.

WHAT THE FROZEN RULES CHANGE
----------------------------
Two rules from `FROZEN_CAMERA5.json` are applied here rather than left to the analysis:

    the ambiguous band  the drawing rule reads a net turn of <= 35 degrees as SAME and >= 70 as
                        DIFFERENT, so anything strictly between is not a confident label of either
                        kind and is dropped from the SAME/DIFFERENT task

    the pathway         events whose label was derived from a drawing are marked, and events that
                        came from a button press are marked separately, because the button page
                        asked about a moment the video did not mark

An event dropped by the band rule is counted and listed, not silently removed.

Usage:
    PYTHONPATH=. python scripts/p14_camera_labels.py
    PYTHONPATH=. python scripts/p14_camera_labels.py --only VID00003
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RENAME = {"LOOK": "SAME_DIRECTION", "TURN": "DIFFERENT_DIRECTION",
          "NO_LOCOMOTION": "NO_NET_DISPLACEMENT"}
BAND = (35.0, 70.0)

# Where each camera recording's labels come from. The first two were made in the round whose labels
# `FROZEN_P10.json` holds, and are taken from there rather than rebuilt, so the published mixture
# stays reproducible. The rest are drawing rounds of their own.
FROM_FROZEN = ("VID00005", "VID00009")
PREFIX = {"VID00005": "v05", "VID00009": "v09", "VID00010": "v10",
          "VID00003": "v03", "VID00004": "v04", "VID00007": "v07", "VID00008": "v08",
          "VID00011": "v11", "VID00012": "v12", "VID00014": "v14"}
ORDER = ["VID00005", "VID00009", "VID00010", "VID00003", "VID00004"]


def resolve_round(clip: str) -> dict | None:
    """Find a clip's drawing round by looking, instead of by a table of written paths.

    The table is what broke: it said `review_set_vid03_v2.json` while the file built for that clip
    was `review_set_vid3_v2.json`, and a wrong path then read as "labels not drawn yet". Both
    spellings occur in this project because different rounds were named at different times, so the
    only reliable answer is to look for either and refuse to guess if both exist.
    """
    n = clip.replace("VID", "").lstrip("0") or "0"
    padded = n.zfill(2)

    def find(kind: str, suffix: str) -> Path | None:
        cands = []
        for tag in {n, padded}:
            cands += sorted((ROOT / "data/p095").glob(f"{kind}_vid{tag}{suffix}"))
        cands = sorted(set(cands))
        if len(cands) > 1:
            raise SystemExit(f"{clip}: {kind} найден в двух вариантах: {cands}")
        return cands[0] if cands else None

    set_p = find("review_set", "_v2.json")
    labels_p = find("human_labels", ".json")
    clips_dir = None
    for tag in {n, padded}:
        d = ROOT / f"output/p095_vid{tag}/clips"
        if d.is_dir():
            clips_dir = d
            break
    if set_p is None and labels_p is None and clips_dir is None:
        return None
    return {"set": str(set_p.relative_to(ROOT)) if set_p else None,
            "labels": str(labels_p.relative_to(ROOT)) if labels_p else None,
            "clips": str(clips_dir.relative_to(ROOT)) if clips_dir else None,
            "prefix": PREFIX.get(clip, f"v{n}")}


def from_drawing_round(clip: str, spec: dict) -> tuple[list[dict], list[dict]]:
    """Convert one drawing round. Returns (kept events, dropped by the band rule).

    A missing file raises rather than returning nothing. The first version returned an empty list,
    and a clip whose set path was spelled `vid03` while the file was named `vid3` therefore looked
    like a clip whose labels had simply not been drawn yet — a wrong name reading as "not yet",
    which is the same mistake as every other one in this phase.
    """
    set_p = ROOT / spec["set"]
    lab_p = ROOT / spec["labels"]
    for p, what in ((set_p, "набор событий"), (lab_p, "метки")):
        if not p.exists():
            raise FileNotFoundError(f"{clip}: нет файла ({what}): {p}")
    s = json.loads(set_p.read_text(encoding="utf-8"))
    by_id = {e["event_id"]: e for e in s["events"]}
    labels = json.loads(lab_p.read_text(encoding="utf-8"))["labels"]
    first = [l for l in labels if l.get("round", 1) == 1]

    clips_dir = ROOT / spec["clips"]
    ns_dir = clips_dir.parent / "clips_ns"
    ns_dir.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    for l in first:
        ev = by_id.get(l["event_id"])
        if ev is None:
            continue
        src = clips_dir / Path(ev["clip_file"]).name
        if not src.exists():
            continue
        name = f"{spec['prefix']}_{Path(ev['clip_file']).stem}.mp4"
        link = ns_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(src, link)

        cat = RENAME.get(l["human_label"])
        ang = l.get("derived_turn_net_deg")
        rec = {
            "event_id": f"{spec['prefix']}_{l['event_id']}",
            "clip": clip,
            "time": float(ev["time"]),
            "clip_file": str((clips_dir.parent / "clips_ns" / name).relative_to(ROOT / "output")),
            "clip_offset_s": float(ev["clip_offset_s"]),
            "narrow": [float(x) for x in ev["narrow"]],
            "category": cat,
            "old_name": l["human_label"],
            "drawing": l.get("label_source") == "derived_from_drawing",
            "net_deg": None if ang is None else float(ang),
        }
        if cat in ("SAME_DIRECTION", "DIFFERENT_DIRECTION") and ang is not None \
                and BAND[0] < float(ang) < BAND[1]:
            rec["dropped_reason"] = f"нетто-угол {float(ang):.1f}° в полосе {BAND}"
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped


def from_frozen(clip: str, frozen: dict) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for l in frozen["labels"]:
        if l["clip"] != clip:
            continue
        ang = l.get("net_deg")
        rec = {
            "event_id": l["event_id"], "clip": clip, "time": float(l["time"]),
            "clip_file": l["clip_file"], "clip_offset_s": float(l["clip_offset_s"]),
            "narrow": [float(x) for x in l["narrow"]], "category": l["category"],
            "old_name": l.get("old_name", ""), "drawing": bool(l.get("drawn_from_drawing")),
            "net_deg": None if ang is None else float(ang),
        }
        if rec["category"] in ("SAME_DIRECTION", "DIFFERENT_DIRECTION") \
                and ang is not None and BAND[0] < float(ang) < BAND[1]:
            rec["dropped_reason"] = f"нетто-угол {float(ang):.1f}° в полосе {BAND}"
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", default="data/p10/FROZEN_P10.json")
    ap.add_argument("--out", default="data/p14/LABELS_CAMERA5.json")
    ap.add_argument("--only", default=None, help="собрать только один кусок (для отладки)")
    a = ap.parse_args()

    frozen = json.loads((ROOT / a.frozen).read_text(encoding="utf-8"))
    clips = [a.only] if a.only else ORDER

    print("=" * 100)
    print("P14 — МЕТКИ КАМЕРНОГО НАБОРА (правила из FROZEN_CAMERA5.json)")
    print("=" * 100)
    print(f"  неоднозначная полоса: строго между {BAND[0]:.0f}° и {BAND[1]:.0f}° — "
          f"из SAME/DIFFERENT исключается")

    events, all_dropped, per_clip = [], [], {}
    for clip in clips:
        if clip in FROM_FROZEN:
            kept, dropped = from_frozen(clip, frozen)
            src = "FROZEN_P10"
            state = f"{len(kept)} событий" if kept else "нет меток"
        else:
            spec = resolve_round(clip)
            if spec is None or not spec["labels"]:
                per_clip[clip] = {"source": "раунд ещё не размечен", "kept": 0,
                                  "dropped_by_band": 0, "from_drawing": 0}
                print(f"  {clip:<10} {'раунд ещё не размечен':<40} ждёт рисования")
                continue
            kept, dropped = from_drawing_round(clip, spec)
            src = spec["labels"]
            state = f"{len(kept)} событий"
        per_clip[clip] = {"source": src, "kept": len(kept), "dropped_by_band": len(dropped),
                          "from_drawing": sum(1 for e in kept if e["drawing"])}
        events += kept
        all_dropped += dropped
        print(f"  {clip:<10} {src:<40} {state}"
              + (f", исключено полосой {len(dropped)}" if dropped else ""))

    events.sort(key=lambda e: (e["clip"], e["time"]))
    cat = Counter(e["category"] for e in events)
    sd = Counter(e["category"] for e in events
                 if e["category"] in ("SAME_DIRECTION", "DIFFERENT_DIRECTION"))
    print()
    print(f"  всего событий: {len(events)}")
    print(f"  по категориям: {dict(cat)}")
    print(f"  SAME/DIFFERENT: {sum(sd.values())} ({dict(sd)})")
    print(f"  из них выведено из рисунка: {sum(1 for e in events if e['drawing'])}")
    if all_dropped:
        print(f"  исключено полосой: {len(all_dropped)}")
        for e in all_dropped:
            print(f"    {e['event_id']} {e['clip']} {e['old_name']} {e['dropped_reason']}")

    ready = [c for c in ORDER if per_clip.get(c, {}).get("kept")]
    doc = {
        "phase": "P14 — метки камерного набора",
        "rules_applied": {
            "ambiguous_band": f"строго между {BAND[0]}° и {BAND[1]}° исключено из SAME/DIFFERENT",
            "drawing_pathway": "поле drawing отмечает метки, выведенные из нарисованного пути",
        },
        "clips": ready,
        "n_clips": len(ready),
        "counts": {"per_category": dict(cat), "same_different": dict(sd), "total": len(events),
                   "per_clip": per_clip},
        "dropped_by_band": all_dropped,
        "labels": events,
    }
    out = ROOT / a.out
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  кусков готово: {len(ready)} — {ready}")
    print(f"  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
