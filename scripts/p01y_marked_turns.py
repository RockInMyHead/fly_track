#!/usr/bin/env python3
"""
Score a hand-marked turn list against the frozen content-motion measurement.

Input is the JSON produced by the annotator (`download json` or
`data/p01r/trajectory_annotations.json`):

    {"turns":[{"id","t0","t1","direction","confidence","notes"}, ...],
     "trajectory":[...]}

For every marked turn it runs the frozen measurement from `fly_vo/content_motion.py`
on exactly that window and reports what the image did, converted to camera terms.

It also runs three structural checks on the list itself, because a list of turns
can be self-defeating regardless of what the measurement says:

  1. OVERLAP: two marked turns that share time. Overlapping windows are not
     independent evidence.
  2. CONTRADICTION: two overlapping turns marked with opposite directions. Either
     the camera genuinely reversed inside the overlap, or one mark is wrong. In
     either case the pair cannot both be used as a clean reference.
  3. SHORT: turns below a usable length, where the measurement has too few frames.

Nothing is tuned. The label side is the human's; the measurement side is frozen.

Usage:
    PYTHONPATH=. python scripts/p01y_marked_turns.py --turns ~/Downloads/turns.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.content_motion import measure_window

VIDEO = ROOT / "data/p01r/VID00001.AVI"
DEFAULT_TURNS = ROOT / "data/p01r/trajectory_annotations.json"
SAVE_JSON = ROOT / "data/p01r/trajectory_annotations.json"
CAM_TO_CONTENT = {"LEFT": "RIGHT", "RIGHT": "LEFT", "FWD": "FWD", "UNCLEAR": None}
MIN_LEN_S = 1.0
MIN_FRAMES = 30


def load_turns(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        doc = {"turns": doc, "trajectory": []}
    doc.setdefault("turns", [])
    doc.setdefault("trajectory", [])
    return doc


def structural_checks(turns: list[dict]) -> list[dict]:
    issues = []
    s = sorted(turns, key=lambda t: t["t0"])
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = s[i], s[j]
            if b["t0"] >= a["t1"]:
                break
            lo, hi = max(a["t0"], b["t0"]), min(a["t1"], b["t1"])
            overlap = hi - lo
            same = a["direction"] == b["direction"]
            issues.append({
                "kind": "overlap_same" if same else "overlap_contradiction",
                "a": a["id"], "b": b["id"], "overlap_s": round(overlap, 3),
                "a_dir": a["direction"], "b_dir": b["direction"],
                "a_window": [a["t0"], a["t1"]], "b_window": [b["t0"], b["t1"]],
            })
    for t in turns:
        if t["t1"] - t["t0"] < MIN_LEN_S:
            issues.append({"kind": "short", "a": t["id"], "length_s": t["t1"] - t["t0"]})
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default=str(DEFAULT_TURNS))
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--save-to", default=str(SAVE_JSON))
    ap.add_argument("-o", "--output", default="output/p01y_marked")
    args = ap.parse_args()

    src = Path(args.turns).expanduser()
    if not src.exists():
        raise SystemExit(f"turn file not found: {src}")
    doc = load_turns(src)
    turns = doc["turns"]
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print(f"marked turns — {src}")
    print(f"  {len(turns)} turns, {len(doc.get('trajectory') or [])} trajectory points\n")

    # ------------------------------------------------------------------ structure
    issues = structural_checks(turns)
    print("=== structural checks on the list ===")
    if not issues:
        print("  no overlaps, no contradictions, no short turns")
    for it in issues:
        if it["kind"] == "overlap_contradiction":
            print(f"  CONTRADICTION  {it['a']} ({it['a_dir']}) vs {it['b']} ({it['b_dir']})  "
                  f"share {it['overlap_s']:.2f} s  "
                  f"[{it['a_window'][0]:.2f}-{it['a_window'][1]:.2f}] vs "
                  f"[{it['b_window'][0]:.2f}-{it['b_window'][1]:.2f}]")
        elif it["kind"] == "overlap_same":
            print(f"  overlap        {it['a']} and {it['b']} both {it['a_dir']}  "
                  f"share {it['overlap_s']:.2f} s (not independent of each other)")
        elif it["kind"] == "short":
            print(f"  short turn     {it['a']}  {it['length_s']:.2f} s")

    # ------------------------------------------------------------------ measurement
    print("\n=== frozen measurement on each marked window ===")
    print(f"  {'id':9s} {'camera':8s} {'conf':7s} {'t0':>8s} {'t1':>8s} "
          f"{'our content':>11s} {'-> camera':>10s} {'coh':>5s} {'rel':>5s} "
          f"{'agrees':>7s} {'N':>4s}")
    rows = []
    for t in sorted(turns, key=lambda x: x["t0"]):
        m = measure_window(args.video, float(t["t0"]), float(t["t1"]), t["id"])
        cam_measured = m.camera_direction if m.content_dx != 0 else "NONE"
        if t["direction"] in ("LEFT", "RIGHT"):
            agrees = cam_measured == t["direction"]
        else:
            agrees = None
        rows.append({
            "id": t["id"], "marked_camera": t["direction"], "confidence": t.get("confidence", ""),
            "t0": float(t["t0"]), "t1": float(t["t1"]),
            "content_dx": m.content_dx, "mean_dx": m.mean_dx,
            "content_direction": m.content_direction, "measured_camera": cam_measured,
            "coherence": m.coherence, "reliable_fraction": m.reliable_fraction,
            "local_activity": m.local_activity, "n_frames": m.n_frames,
            "agrees": agrees, "notes": t.get("notes", ""),
        })
        ag = "—" if agrees is None else ("yes" if agrees else "NO")
        print(f"  {t['id']:9s} {t['direction']:8s} {t.get('confidence',''):7s} "
              f"{t['t0']:8.2f} {t['t1']:8.2f} {m.content_direction:>11s} {cam_measured:>10s} "
              f"{m.coherence:5.2f} {m.reliable_fraction:5.2f} {ag:>7s} {m.n_frames:4d}")

    # ------------------------------------------------------------------ summary
    directional = [r for r in rows if r["agrees"] is not None]
    n_ok = sum(1 for r in directional if r["agrees"])
    by_coh = sorted(rows, key=lambda r: -r["coherence"])
    print(f"\n=== summary ===")
    print(f"  turns marked left/right : {len(directional)}")
    print(f"  measurement agrees      : {n_ok}/{len(directional)}")
    if directional:
        print(f"  agreement among the higher-coherence half (top {len(directional)//2 or 1}):")
        top = [r for r in by_coh if r["agrees"] is not None][: max(len(directional) // 2, 1)]
        print(f"    {sum(1 for r in top if r['agrees'])}/{len(top)}  "
              f"(coherence {min(r['coherence'] for r in top):.2f}..{max(r['coherence'] for r in top):.2f})")
    disagree = [r for r in directional if not r["agrees"]]
    if disagree:
        print(f"\n  disagreements (marked vs measured camera direction):")
        for r in disagree:
            print(f"    {r['id']:9s} t={r['t0']:.2f}-{r['t1']:.2f}  marked {r['marked_camera']:5s}  "
                  f"measured {r['measured_camera']:5s}  coherence {r['coherence']:.2f}")

    # which pairs can actually serve as independent evidence
    clean = [r for r in rows if r["coherence"] >= 0.40]
    print(f"\n  turns at coherence >= 0.40 (readable field): {len(clean)}")
    for r in clean:
        print(f"    {r['id']:9s} t={r['t0']:.2f}-{r['t1']:.2f}  marked {r['marked_camera']:5s}  "
              f"measured {r['measured_camera']:5s}  coherence {r['coherence']:.2f}")

    # ------------------------------------------------------------------ write
    if args.save_to:
        SAVE_JSON.write_text(json.dumps({**doc, "scored_at": "p01y", "scored": rows},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  annotations copied into the project: {SAVE_JSON}")

    with (out / "turns_scored.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "report.json").write_text(json.dumps(
        {"source": str(src), "n_turns": len(turns), "structural_issues": issues,
         "turns": rows, "agreement": [n_ok, len(directional)]}, indent=2), encoding="utf-8")
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
