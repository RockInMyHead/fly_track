#!/usr/bin/env python3
"""
P0.1S — relabel audit: what the event labels actually mean, and what is confirmed.

Chain of evidence that forced this file
---------------------------------------
1. `left_65` was labelled `LEFT_YAW` and used for months as the primary reference
   for a left turn.
2. Four independent measurements of the *content* motion in that window — the 1-D
   Reichardt panorama, the 2-D Reichardt grid, the Lucas-Kanade patch field, and
   the integrated rotation coefficient — all report content moving LEFT, with the
   highest confidence of any window in the set (field coherence 0.72, 76% of cells
   agreeing, z = -24).
3. For a camera yaw, image content moves opposite to the camera. Content LEFT
   therefore means the camera turned RIGHT.
4. The user watched the window and confirmed: the camera turns RIGHT.

So the measurement was right and the label was wrong: `left_65` is a RIGHT turn,
not a left one. Any number produced against the old label — including the frozen
readout's "left_65 correct" — is void for that event.

What this file does
-------------------
It records the correction with its evidence, and states plainly which labels are
now settled and which are not. It does NOT rewrite the remaining labels: only the
one that was checked by eye is changed. The rest are marked REVIEW and must be
re-decided from the footage, not from our measurement.

Convention, stated once and used everywhere from here on
-------------------------------------------------------
    LABEL      = the direction the CAMERA rotates.  LEFT / RIGHT / FWD / UNCLEAR
    MEASUREMENT = the direction the CONTENT moves.  positive dx = content right

    content LEFT  <=> camera RIGHT
    content RIGHT <=> camera LEFT
    forward motion moves content outward from the focus of expansion, which is not
    a yaw at all, so FWD must never be inferred from the sign of a pooled dx.

Usage:
    PYTHONPATH=. python scripts/p01s_relabel_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
OUT_JSON = ROOT / "data/p01r/VID00001_labels_v2.json"

# Content-motion measurement per window, mean dx in px/frame (positive = content
# right). From scripts/p01s_instrument_check.py, Lucas-Kanade patch field.
MEASURED_CONTENT_DX = {
    "left_65":   -1.923,
    "left_81":   +0.461,
    "left_108":  +0.353,
    "right_67":  -0.479,
    "right_210": -0.426,
    "right_216": +0.281,
    "fwd_140":   -0.318,
}
# Field coherence |mean dx| / mean|dx|: 1 = every cell agrees, 0 = split in half.
# This sets how much the sign of a pooled mean can be trusted at all.
COHERENCE = {
    "left_65": 0.72, "left_81": 0.11, "left_108": 0.25,
    "right_67": 0.30, "right_210": 0.22, "right_216": 0.13, "fwd_140": 0.27,
}
# Below this the pooled sign is a small residue of large opposed motions.
COHERENCE_TRUSTWORTHY = 0.40

CONFIRMED_BY_EYE = {
    "left_65": {
        "camera": "RIGHT",
        "evidence": "watched the window directly; camera turns right",
        "content_expected": "LEFT",
        "measured_content": "LEFT",
        "agrees_with_measurement": True,
    }
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=str(OUT_JSON))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)

    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists — pass --force to regenerate (it holds recorded labels)")

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    print("P0.1S — relabel audit")
    print(f"  convention: LABEL = camera rotation, MEASUREMENT = content motion")
    print(f"  content LEFT <=> camera RIGHT;  content RIGHT <=> camera LEFT\n")

    events = []
    for e in doc["events"]:
        eid = e["id"]
        old = e["label"]
        old_cam = {"LEFT_YAW": "LEFT", "RIGHT_YAW": "RIGHT", "FWD": "FWD",
                   "MIXED": "UNCLEAR", "REJECT": "UNCLEAR"}.get(old, "UNCLEAR")
        rec = {
            "id": eid, "anchor_s": e["anchor_s"], "window": [e["start"], e["end"]],
            "old_label": old, "old_label_camera_sense": old_cam,
            "measured_content_dx": MEASURED_CONTENT_DX.get(eid),
            "coherence": COHERENCE.get(eid),
        }
        if eid in CONFIRMED_BY_EYE:
            c = CONFIRMED_BY_EYE[eid]
            rec.update({
                "new_label": c["camera"],
                "status": "CONFIRMED_BY_EYE",
                "evidence": c["evidence"],
                "old_label_was": "WRONG",
                "flipped_from_old": c["camera"] != old_cam,
            })
        elif eid in MEASURED_CONTENT_DX:
            rec["measured_content_direction"] = (
                "RIGHT" if MEASURED_CONTENT_DX[eid] > 0 else "LEFT"
            )
            rec["measured_camera_direction"] = (
                "LEFT" if MEASURED_CONTENT_DX[eid] > 0 else "RIGHT"
            )
            rec["measurement_trustworthy"] = bool(
                COHERENCE.get(eid, 0.0) >= COHERENCE_TRUSTWORTHY
            )
            rec["agrees_with_old_label"] = (
                rec["measured_camera_direction"] == old_cam
            )
            rec["new_label"] = None
            rec["status"] = "NEEDS_BLIND_REVIEW"
        else:
            rec["new_label"] = None
            rec["status"] = "NOT_SCORED"
        events.append(rec)

    print(f"  {'event':10s} {'old label':10s} {'old=camera':10s} {'our content':11s} "
          f"{'-> camera':10s} {'coherence':>9s} {'status':>20s}")
    for r in events:
        content = r.get("measured_content_direction") or "-"
        cam = r.get("measured_camera_direction") or "-"
        coh = f"{r['coherence']:.2f}" if r["coherence"] is not None else "-"
        print(f"  {r['id']:10s} {r['old_label']:10s} {r['old_label_camera_sense']:10s} "
              f"{content:11s} {cam:10s} {coh:>9s} {r['status']:>20s}")

    # ---- how do the old labels fare under each convention? ----
    scored = [r for r in events if r.get("measured_camera_direction")]
    cam_ok = sum(1 for r in scored if r["agrees_with_old_label"])
    content_ok = sum(
        1 for r in scored
        if r["measured_content_direction"] == r["old_label_camera_sense"]
    )
    print(f"\n  old labels vs our measurement, camera-convention reading : "
          f"{cam_ok}/{len(scored)}")
    print(f"  old labels vs our measurement, content-convention reading: "
          f"{content_ok}/{len(scored)}")

    # ---- which windows can carry any weight at all? ----
    # Include the eye-confirmed window here too: excluding it (as the first
    # version of this script did) made the output claim "no trustworthy window"
    # while the confirmed one had by far the strongest field in the set.
    coherence_all = {
        r["id"]: r["coherence"] for r in events if r["coherence"] is not None
    }
    strong = [k for k, v in coherence_all.items() if v >= COHERENCE_TRUSTWORTHY]
    weak = [k for k, v in coherence_all.items() if v < COHERENCE_TRUSTWORTHY]
    print(f"\n  every window with a coherence figure ({len(coherence_all)}):")
    for k, v in sorted(coherence_all.items(), key=lambda kv: -kv[1]):
        mark = "trustworthy" if v >= COHERENCE_TRUSTWORTHY else "residue of opposed motions"
        print(f"    {k:10s} {v:.2f}   {mark}")
    print(f"\n  trustworthy (coherence >= {COHERENCE_TRUSTWORTHY}): {strong or 'none'}")
    print(f"  untrustworthy: {weak or 'none'}")

    flipped = [r["id"] for r in events if r.get("flipped_from_old")]
    summary = {
        "convention": {
            "label": "direction the CAMERA rotates",
            "measurement": "direction the CONTENT moves, positive dx = content right",
            "mapping": "content LEFT <=> camera RIGHT; content RIGHT <=> camera LEFT",
            "warning": "forward motion is not a yaw and must not be inferred from a "
                       "pooled dx sign",
        },
        "confirmed_corrections": CONFIRMED_BY_EYE,
        "events": events,
        "old_labels_camera_reading_agreement": [cam_ok, len(scored)],
        "old_labels_content_reading_agreement": [content_ok, len(scored)],
        "trustworthy_windows": strong,
        "untrustworthy_windows": weak,
        "coherence_trustworthy_threshold": COHERENCE_TRUSTWORTHY,
        "flipped_from_old": flipped,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== HEADLINE ===")
    print(f"  {len(flipped)} confirmed correction(s): "
          + ", ".join(f"{k}: old label wrong, now {v['camera']}" for k, v in CONFIRMED_BY_EYE.items()))
    print(f"  {len(scored) - len(strong)} of {len(scored)} windows have a pooled sign that "
          f"cannot be trusted on its own, so the old '3/6' figure was never a valid score.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
