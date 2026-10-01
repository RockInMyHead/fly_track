#!/usr/bin/env python3
"""P13.1D — the same channel seen per recording, for transfer rather than for a headline.

`p13_1d_score.py` pools all reviewed windows and reports one AUC. That answers "does the channel
carry this distinction" but not "does it carry it in the same way everywhere", and the second
question is the one that decides whether a channel can be used: a pooled AUC of 0.8 made of one
recording at 0.95 and four at 0.55 is not a usable second feature.

This script reads the same frozen scores and does not recompute, refit or rethreshold anything. It
exists apart from the frozen scorer so that adding a breakdown cannot alter the number the freeze
names: the frozen file's hash stays valid, and this one can be extended without touching it.

Reported per recording, for the 12-28 band and for the whole sample:

    AUC           standing against walking, within that recording alone
    direction     whether standing scores above walking there, so disagreement is visible
    false stop    frozen windows where the person was walking — the error that costs real path
    missed stop   unfrozen windows where the person was standing — invented path kept

A recording with fewer than three windows of either class is reported as not measurable rather than
given a number that a single window decided.

Usage:
    PYTHONPATH=. python scripts/p13_1d_pervideo.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
SCORES = OUT / "p13_1d_scores.json"
DEST = OUT / "p13_1d_pervideo.json"
THRESHOLD = 28.0          # the frozen threshold under test


def auc(pos: list[float], neg: list[float]) -> float | None:
    if len(pos) < 3 or len(neg) < 3:
        return None
    p, n = np.asarray(pos), np.asarray(neg)
    gt = float((p[:, None] > n[None, :]).sum())
    eq = float((p[:, None] == n[None, :]).sum())
    return (gt + 0.5 * eq) / (len(p) * len(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()

    d = json.loads(SCORES.read_text(encoding="utf-8"))
    w = [x for x in d["windows"] if x["answer"] in ("STANDS", "WALKS")]
    by: dict[str, list[dict]] = defaultdict(list)
    for x in w:
        by[x["video"]].append(x)

    print("=" * 104)
    print("P13.1D — ПО РОЛИКАМ")
    print("=" * 104)
    print(f"  окон всего: {len(w)} на {len(by)} роликах")
    print(f"  признак: {d['sign_convention']}")
    print(f"  порог, под который считается ложный/пропущенный STOP: {THRESHOLD}")
    print()

    out = {"threshold": THRESHOLD, "n_windows": len(w), "per_video": {}}
    for band_name, lo, hi in (("полоса 12–28", 12.0, 28.0), ("все окна", 0.0, 1e9)):
        print(f"─── {band_name} ───")
        print(f"  {'ролик':<10}{'n':>4}{'стоит':>7}{'идёт':>6}{'AUC':>8}"
              f"{'направл.':>11}{'лож.STOP':>10}{'проп.стоп':>11}")
        print("  " + "-" * 68)
        rows = []
        for v in sorted(by):
            sub = [x for x in by[v] if lo <= x["motion"] < hi]
            stands = [x["score"] for x in sub if x["answer"] == "STANDS"]
            walks = [x["score"] for x in sub if x["answer"] == "WALKS"]
            a_ = auc(stands, walks)
            fs = sum(1 for x in sub if x["score"] >= 0 and x["answer"] == "WALKS")
            ms = sum(1 for x in sub if x["score"] < 0 and x["answer"] == "STANDS")
            dirc = "—" if a_ is None else ("верно" if a_ >= 0.5 else "ОБРАТНОЕ")
            print(f"  {v:<10}{len(sub):>4}{len(stands):>7}{len(walks):>6}"
                  f"{(a_ if a_ is not None else float('nan')):>8.3f}{dirc:>11}{fs:>10}{ms:>11}")
            rows.append({"video": v, "n": len(sub), "stands": len(stands),
                         "walks": len(walks), "auc": a_, "direction_correct":
                         (None if a_ is None else a_ >= 0.5),
                         "false_stop": fs, "missed_stop": ms})
        vals = [r["auc"] for r in rows if r["auc"] is not None]
        if vals:
            print(f"  {'медиана':<10}{'':>4}{'':>7}{'':>6}{np.median(vals):>8.3f}"
                  f"{'':>11}{sum(r['false_stop'] for r in rows):>10}"
                  f"{sum(r['missed_stop'] for r in rows):>11}")
            agree = sum(1 for v in vals if v >= 0.5)
            print(f"  направление верно в {agree} из {len(vals)} роликов, где посчиталось")
        print()
        out["per_video"][band_name] = rows

    # how the channel would do if it alone gated, per recording
    print("─── ЕСЛИ ГЕЙТИТЬ ОДНИМ ЭТИМ КАНАЛОМ ───")
    print(f"  {'ролик':<10}{'лож.STOP':>10}{'проп.стоп':>11}{'из замороженных':>18}")
    tf = tm = tn = 0
    for v in sorted(by):
        sub = by[v]
        fs = sum(1 for x in sub if x["score"] >= 0 and x["answer"] == "WALKS")
        ms = sum(1 for x in sub if x["score"] < 0 and x["answer"] == "STANDS")
        nf = sum(1 for x in sub if x["score"] >= 0)
        tf += fs
        tm += ms
        tn += nf
        print(f"  {v:<10}{fs:>10}{ms:>11}{nf:>18}")
    print(f"  {'ИТОГО':<10}{tf:>10}{tm:>11}{tn:>18}")
    print(f"  для сравнения видео-гейт при пороге 12: ложных STOP 0 из 40, "
          f"пропущено стояния 32 из 49")
    print()
    out["if_gated_alone"] = {"false_stop": tf, "missed_stop": tm, "n_frozen": tn,
                             "video_gate": {"false_stop": 0, "n_frozen": 40,
                                            "missed_stop": 32, "n_unfrozen": 49}}

    DEST.write_text(json.dumps({
        "phase": "P13.1D — по роликам",
        "note": "дополнение к замороженному подсчёту; сам подсчёт не изменялся",
        "threshold_for_errors": THRESHOLD,
        **out,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  записано: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
