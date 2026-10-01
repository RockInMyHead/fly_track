#!/usr/bin/env python3
"""
P0.3 — classifier v2, validated against the human verdicts.

What the review found
---------------------
The first balanced review gave 44/60. The confusion matrix showed two distinct
failures, and they need different fixes.

  MIXED scored 0/12. All twelve were called STATIC by the reviewer, and their
  numbers were near zero (|a| < 0.3, |div| < 0.2) with low confidence. So the
  class "neither term dominates" was being used for "nothing is happening", when
  the right answer was STATIC. The still branch existed but its gate was wrong.

  One LEFT scored wrong: window r044 at t=17.8 s. Per-frame inspection shows why.
  Comparing it against controls:

      true STATIC    |dx| = 0.27 px/frame, sign flips 55/120
      true RIGHT     |dx| = 4.92 px/frame, sign flips  6/26,  LK vs phase 96%
      r044           |dx| = 2.70 px/frame, sign flips 12/40,  LK vs phase 55%

  r044 has ten times the motion of a still scene, so something moved, but the sign
  flips constantly and two independent estimators disagree. It is not a rotation:
  a rotation holds one sign. The old classifier took the median over the window and
  happened to land on +2.5, which looked like a left turn.

The fix, in one line: a rotation must hold its sign. Medians are not enough, because
the median of an oscillating signal is whatever the oscillation happened to average to.

Classifier v2
-------------
    STATIC   |dx| and |dy| both small                     nothing moved
    ROTATION |a| large, sign held, divergence small       camera rotated
    FWD      divergence large and compact, |a| small      camera moved ahead
    MIXED    motion present but incoherent, or neither term dominates

"Sign held" is measured per frame: the fraction of frames whose instantaneous `a`
matches the window's sign. A window with heavy sign flipping is not a rotation no
matter how large its median.

Usage:
    PYTHONPATH=. python scripts/p03_classify_v2.py
    PYTHONPATH=. python scripts/p03_classify_v2.py --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIGNAL = ROOT / "output/p02_full/full_signal.npz"
REVIEW_SET = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
OUT = ROOT / "data/p01r/windows_v2.json"
FPS = 30.0

# Thresholds. Chosen from the controls above, then checked against the verdicts;
# stated here rather than tuned per window.
STATIC_ACTIVITY = 0.60      # px/frame; true-static control measured 0.27
TURN_MIN_A = 1.00           # |uniform component| for a rotation
FWD_MIN_DIV = 0.45          # |divergence| for forward motion
DOMINANCE = 1.30            # how much larger the winning term must be
SIGN_HELD_MIN = 0.75        # fraction of frames that must agree on the sign
COHERENT_PER_FRAME = 0.60   # per-frame |a| that counts as "this frame moved"
SMOOTH_S = 0.5
MIN_SEG_S = 0.9


def load_signal():
    d = np.load(SIGNAL)
    return {k: d[k] for k in d.files}


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def window_stats(sig: dict, i0: int, i1: int) -> dict:
    a = sig["a"][i0:i1]
    div = sig["b"][i0:i1] + sig["c"][i0:i1]
    ax = sig["act_x"][i0:i1]
    ay = sig["act_y"][i0:i1]
    res = sig["residual"][i0:i1]
    med_a = float(np.median(a))
    # sign that the window is claiming, and how often frames actually agree with it
    sgn = np.sign(med_a)
    strong = np.abs(a) >= COHERENT_PER_FRAME
    if strong.any():
        held = float(np.mean(np.sign(a[strong]) == sgn))
    else:
        held = 0.0
    return {
        "a": med_a,
        "a_abs": float(np.median(np.abs(a))),
        "div": float(np.median(div)),
        "activity": float(np.median(ax)),
        "activity_vertical": float(np.median(ay)),
        "residual": float(np.median(res)),
        "sign_held": held,
        "moved_frames": int(strong.sum()),
        "n_frames": int(len(a)),
    }


def classify(w: dict) -> tuple[str, str, dict]:
    """Return (kind, confidence, why)."""
    why = {}
    still = (w["activity"] < STATIC_ACTIVITY and w["activity_vertical"] < STATIC_ACTIVITY)
    why["still_by_activity"] = still
    if still:
        return "STATIC", "high", why

    rot = abs(w["a"])
    fwd = abs(w["div"])
    coherent = w["sign_held"] >= SIGN_HELD_MIN
    why.update({"rotation_strength": rot, "forward_strength": fwd,
                "sign_held": w["sign_held"], "coherent": coherent})

    if rot >= TURN_MIN_A and coherent and fwd < DOMINANCE * rot:
        kind = "LEFT" if w["a"] > 0 else "RIGHT"
        conf = ("high" if rot >= 2.5 and w["sign_held"] >= 0.9 else
                "medium" if rot >= 1.5 else "low")
        return kind, conf, why

    if fwd >= FWD_MIN_DIV and rot < DOMINANCE * fwd:
        conf = "high" if fwd >= 0.9 else "medium"
        return "FWD", conf, why

    # motion is present but the sign does not hold, or neither term wins
    return "MIXED", "low", why


def scan_windows(sig: dict) -> list[dict]:
    n_sm = max(int(round(SMOOTH_S * FPS)), 1)
    a_s = box(sig["a"], n_sm)
    div_s = box(sig["b"] + sig["c"], n_sm)
    act_s = box(sig["act_x"], n_sm)
    kinds = []
    for i in range(len(a_s)):
        w = window_stats(
            {"a": a_s[i:i + 1], "b": div_s[i:i + 1], "c": np.zeros(1),
             "act_x": act_s[i:i + 1], "act_y": sig["act_y"][i:i + 1],
             "residual": sig["residual"][i:i + 1]}, 0, 1)
        kinds.append(w)
    # segment by stable label, then compute real stats per segment
    labels = [classify(w)[0] for w in kinds]
    segs = []
    i = 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if (j - i + 1) / FPS >= MIN_SEG_S:
            st = window_stats(sig, i, j + 1)
            kind, conf, why = classify(st)
            segs.append({
                "kind": kind, "confidence": conf,
                "t0": round(float(sig["t"][i]), 2), "t1": round(float(sig["t"][j]), 2),
                "duration_s": round((j - i + 1) / FPS, 2),
                "a": round(st["a"], 3), "divergence": round(st["div"], 3),
                "activity": round(st["activity"], 3),
                "activity_vertical": round(st["activity_vertical"], 3),
                "residual": round(st["residual"], 3),
                "sign_held": round(st["sign_held"], 3),
                "moved_frames": st["moved_frames"], "n_frames": st["n_frames"],
                "camera_direction": {"LEFT": "LEFT", "RIGHT": "RIGHT", "FWD": "FWD",
                                     "STATIC": "STATIC"}.get(kind, "UNCLEAR"),
                "why": {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in why.items()},
            })
        i = j + 1
    return segs


def validate(segs: list[dict]) -> dict:
    rs = json.loads(REVIEW_SET.read_text(encoding="utf-8"))["candidates"]
    v = json.loads(VERDICTS.read_text(encoding="utf-8"))["verdicts"]
    if not v:
        return {}

    def kind_at(t: float) -> str:
        hits = [s for s in segs if s["t0"] <= t <= s["t1"]]
        if not hits:
            return "MIXED"
        return max(hits, key=lambda s: s["duration_s"])["kind"]

    rows = []
    for c in rs:
        x = v.get(c["id"])
        if not x:
            continue
        mid = (c["t0"] + c["t1"]) / 2
        new = kind_at(mid)
        if x["verdict"] == "correct":
            truth = c["kind"]
        else:
            truth = {"LEFT": "LEFT", "RIGHT": "RIGHT", "FWD": "FWD",
                     "STATIC": "STATIC"}.get(x.get("actual_direction") or "UNCLEAR", "MIXED")
        rows.append({"id": c["id"], "old": c["kind"], "new": new, "truth": truth})
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", default=True)
    args = ap.parse_args()

    sig = load_signal()
    segs = scan_windows(sig)
    print("классификатор v2")
    print(f"  пороги: покой |dx|,|dy| < {STATIC_ACTIVITY}; поворот |a| >= {TURN_MIN_A} "
          f"и знак держится >= {SIGN_HELD_MIN:.0%}; вперёд div >= {FWD_MIN_DIV}")
    print(f"  окон: {len(segs)}   {dict(Counter(s['kind'] for s in segs))}")

    res = validate(segs)
    if res:
        rows = res["rows"]
        kinds = ["LEFT", "RIGHT", "FWD", "STATIC", "MIXED"]
        print(f"\n=== ПРОВЕРКА НА 60 ТВОИХ ВЕРДИКТАХ ===")
        for tag in ("old", "new"):
            tab = defaultdict(int)
            for r in rows:
                tab[(r[tag], r["truth"])] += 1
            ok = sum(tab.get((k, k), 0) for k in kinds)
            print(f"\n  {tag} классификатор, точность {ok}/{len(rows)} = {ok/len(rows):.0%}")
            print(f"  {'':>8s} | " + " ".join(f"{k:>7s}" for k in kinds))
            for k in kinds:
                cells = [tab.get((k, t), 0) for t in kinds]
                row_ok = tab.get((k, k), 0)
                n = sum(cells)
                pct = f"{row_ok/n:.0%}" if n else "  -"
                print(f"  {k:>8s} | " + " ".join(f"{v:7d}" for v in cells) + f"  {pct:>5s}")
            print(f"  по классам: " + "  ".join(
                f"{k} {sum(1 for r in rows if r[tag]==k and r['truth']==k)}/"
                f"{sum(1 for r in rows if r[tag]==k)}" for k in kinds))
        print(f"\n  изменения по конкретным окнам:")
        for r in rows:
            if r["old"] != r["new"]:
                mark = "исправилось" if r["new"] == r["truth"] else (
                       "СЛОМАЛОСЬ" if r["old"] == r["truth"] else "")
                print(f"    {r['id']}  ты: {r['truth']:>6s}  v1: {r['old']:>6s} -> "
                      f"v2: {r['new']:>6s}   {mark}")

    OUT.write_text(json.dumps({
        "version": 2,
        "changed": "поворот требует, чтобы знак держался; покой определяется по величине "
                   "локального движения, а не по отсутствию доминирования",
        "thresholds": {"static_activity": STATIC_ACTIVITY, "turn_min_a": TURN_MIN_A,
                       "fwd_min_div": FWD_MIN_DIV, "dominance": DOMINANCE,
                       "sign_held_min": SIGN_HELD_MIN, "smooth_s": SMOOTH_S},
        "validated_against": str(VERDICTS),
        "n_windows": len(segs), "counts": dict(Counter(s["kind"] for s in segs)),
        "windows": segs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
