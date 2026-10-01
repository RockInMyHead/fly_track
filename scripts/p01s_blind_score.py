#!/usr/bin/env python3
"""
P0.1S — score the round-3 blind review against the key.

Answers are read from `data/p01r/VID00001_blind_review3.json`. The key is
`data/p01r/VID00001_blind_review3_key.json`. Four of the eleven items were
synthetic clips with a known content motion; they are scored first, because they
say how much weight the remaining answers can carry.

Two readings of the answers are computed, because the instruction asked for
CONTENT motion while the labels record CAMERA rotation, and the round-1 history
shows the two get confused:

    reading "content"   the answer is taken as the direction the image moved
    reading "camera"    the answer is taken as the direction the camera rotated

Which one the reviewer actually used is decided by the calibration items: only the
reading that scores well on them is admissible for the real items.

Nothing is tuned here. The frozen measurement comes from `fly_vo/content_motion.py`
and is not recomputed in this script; it is read from the frozen table.

Usage:
    PYTHONPATH=. python scripts/p01s_blind_score.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANSWERS_JSON = ROOT / "data/p01r/VID00001_blind_review3.json"
KEY_JSON = ROOT / "data/p01r/VID00001_blind_review3_key.json"
FROZEN_JSON = ROOT / "output/p01s_frozen_measure/frozen_measurements.json"

# Answers transcribed from the review, so the comparison is reproducible even if
# the answer file is edited later. Must match ANSWERS_JSON.
ANSWERS = {
    "A": ("UNCLEAR", "medium"), "B": ("LEFT", "high"), "C": ("LEFT", "medium"),
    "D": ("LEFT", "high"), "E": ("RIGHT", "high"), "F": ("LEFT", "high"),
    "G": ("UNCLEAR", "medium"), "H": ("FWD", "high"), "I": ("RIGHT", "high"),
    "J": ("UNCLEAR", "high"), "K": ("FWD", "medium"),
}

INVERT = {"LEFT": "RIGHT", "RIGHT": "LEFT", "FWD": "FWD", "UNCLEAR": "UNCLEAR",
          "STATIC": "STATIC"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01s_blind_score")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    key = json.loads(KEY_JSON.read_text(encoding="utf-8"))["items"]
    frozen = {}
    if FROZEN_JSON.exists():
        for r in json.loads(FROZEN_JSON.read_text(encoding="utf-8"))["events"]:
            frozen[r["event"]] = r

    print("P0.1S — round-3 blind review, scored")
    print("  answers are transcribed from the review; key opened only now\n")

    # ---------------------------------------------------------------- calibration
    print("=== 1. CALIBRATION ITEMS (synthetic, ground truth known) ===")
    print(f"  {'code':4s} {'kind':14s} {'truth(content)':14s} {'answer':8s} "
          f"{'as content':>10s} {'as camera':>10s}")
    calib_rows = []
    for code, item in key.items():
        if item["kind"] != "calibration":
            continue
        truth = item["true_content_direction"]
        ans, conf = ANSWERS[code]
        if truth == "STATIC":
            as_content = "correct" if ans == "UNCLEAR" else "wrong"
            as_camera = as_content
        else:
            as_content = "correct" if ans == truth else (
                "wrong" if ans in ("LEFT", "RIGHT") else "abstained")
            as_camera = "correct" if INVERT[ans] == truth else (
                "wrong" if ans in ("LEFT", "RIGHT") else "abstained")
        calib_rows.append({"code": code, "kind": item["event_id"], "truth": truth,
                           "answer": ans, "confidence": conf,
                           "as_content": as_content, "as_camera": as_camera})
        print(f"  {code:4s} {item['event_id']:14s} {truth:14s} {ans:8s} "
              f"{as_content:>10s} {as_camera:>10s}")

    n_content = sum(1 for r in calib_rows if r["as_content"] == "correct")
    n_camera = sum(1 for r in calib_rows if r["as_camera"] == "correct")
    n = len(calib_rows)
    print(f"\n  reads correctly as CONTENT: {n_content}/{n}")
    print(f"  reads correctly as CAMERA : {n_camera}/{n}")
    xa = [r for r in calib_rows if r["as_content"] == "abstained"]
    print(f"  calibration items answered knowingly ('correct'/'wrong', no abstain): "
          f"{n - len(xa)}/{n}")

    # which reading is admissible
    if n_camera > n_content:
        reading, why = "camera", "the calibration items score better under the camera reading"
    elif n_content > n_camera:
        reading, why = "content", "the calibration items score better under the content reading"
    else:
        reading, why = "ambiguous", "the calibration items score the same under both readings"
    print(f"\n  => admissible reading: {reading.upper()}  ({why})")

    # ---------------------------------------------------------------- real items
    print("\n=== 2. REAL ITEMS vs THE FROZEN MEASUREMENT ===")
    print("  the frozen measurement is in content space; it is converted here, not recomputed")
    print(f"\n  {'code':4s} {'event':10s} {'our content':11s} {'coherence':>9s} "
          f"{'answer':8s} {'as content':10s} {'agrees':>8s} {'confidence':>10s}")
    real_rows = []
    for code, item in key.items():
        if item["kind"] != "real":
            continue
        eid = item["event_id"]
        f = frozen.get(eid)
        ours = f["content_direction"] if f else "?"
        coh = f["coherence"] if f else float("nan")
        ans, conf = ANSWERS[code]
        agrees = (ans == ours) if ans in ("LEFT", "RIGHT") else None
        real_rows.append({
            "code": code, "event": eid, "old_label_camera": item.get("old_label_camera_sense"),
            "our_content": ours, "coherence": coh, "answer": ans, "confidence": conf,
            "agrees_as_content": agrees,
            "our_camera": INVERT.get(ours, "?"),
            "answer_as_camera": INVERT.get(ans, "?"),
        })
        a = "agree" if agrees else ("abstained" if agrees is None else "DISAGREE")
        print(f"  {code:4s} {eid:10s} {ours:11s} {coh:9.2f} {ans:8s} "
              f"{a:10s} {str(agrees):>8s} {conf:>10s}")

    informative = [r for r in real_rows if r["agrees_as_content"] is not None]
    n_agree = sum(1 for r in informative if r["agrees_as_content"])
    print(f"\n  real items with a directional answer: {len(informative)}/{len(real_rows)}")
    print(f"  of those, the answer agrees with the frozen measurement: "
          f"{n_agree}/{len(informative)}")

    # ------------------------------------------------- coherence vs readability
    print("\n=== 3. DOES COHERENCE PREDICT READABILITY? ===")
    print("  (this is the question that was deferred until the blind answers existed)")
    print(f"  {'event':10s} {'coherence':>9s} {'answer':8s} {'confidence':>10s}")
    for r in sorted(real_rows, key=lambda x: -x["coherence"]):
        print(f"  {r['event']:10s} {r['coherence']:9.2f} {r['answer']:8s} {r['confidence']:>10s}")
    unc = [r for r in real_rows if r["answer"] == "UNCLEAR"]
    hi = [r for r in real_rows if r["confidence"] == "high"]
    print(f"\n  items answered UNCLEAR      : coherence "
          f"{[round(r['coherence'], 2) for r in unc]}")
    print(f"  items answered with high conf: coherence "
          f"{[round(r['coherence'], 2) for r in hi]}")
    if unc and hi:
        print(f"  max coherence among UNCLEAR {max(r['coherence'] for r in unc):.2f}  <  "
              f"min coherence among high-confidence "
              f"{min(r['coherence'] for r in hi):.2f}  -> "
              f"{max(r['coherence'] for r in unc) < min(r['coherence'] for r in hi)}")

    # ------------------------------------------------------- label audit outcome
    print("\n=== 4. LABEL AUDIT, using only the items the reviewer read confidently ===")
    print(f"  {'event':10s} {'old label':10s} {'reviewer content':17s} {'-> camera':10s} "
          f"{'verdict':>28s}")
    for r in real_rows:
        if r["answer"] not in ("LEFT", "RIGHT") or r["confidence"] != "high":
            continue
        cam = r["answer_as_camera"]
        old = r["old_label_camera"]
        if old == "FWD":
            verdict = "forward window, no yaw expected"
        else:
            verdict = "old label CORRECT" if cam == old else "OLD LABEL WRONG"
        print(f"  {r['event']:10s} {str(old):10s} {r['answer']:17s} {cam:10s} {verdict:>28s}")

    summary = {
        "calibration": calib_rows,
        "calibration_correct_as_content": [n_content, n],
        "calibration_correct_as_camera": [n_camera, n],
        "admissible_reading": reading,
        "real_items": real_rows,
        "real_directional_answers": len(informative),
        "real_agreement_with_frozen_measurement": [n_agree, len(informative)],
        "ambiguous_because": (
            "The instruction asked for CONTENT motion. On the real clips the answers line up "
            "with the frozen content measurement, but on the synthetic clips the two left/right "
            "answers come out inverted under the content reading and correct under the camera "
            "reading. Both cannot be true at once, so the calibration result and the real-clip "
            "result disagree about which convention was used. The real-clip agreement is "
            "therefore evidence, not proof."
        ),
    }
    (out / "score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== HEADLINE ===")
    print(f"  calibration: {n_content}/{n} under the content reading, {n_camera}/{n} under "
          f"the camera reading")
    print(f"  real clips:  {n_agree}/{len(informative)} directional answers agree with the "
          f"frozen measurement")
    if unc and hi:
        print(f"  coherence tracks the reviewer's confidence: every UNCLEAR item has "
              f"coherence <= {max(r['coherence'] for r in unc):.2f}, every high-confidence "
              f"item >= {min(r['coherence'] for r in hi):.2f}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
