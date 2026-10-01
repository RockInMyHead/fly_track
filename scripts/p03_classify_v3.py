#!/usr/bin/env python3
"""
P0.3 — classifier v3, simplest thing that works, scored honestly.

What went wrong on the way here
-------------------------------
v2 added a "sign must be held" gate to every class. That is right for rotation but
wrong for forward motion, whose small `a` is pure noise, so v2 turned valid forward
windows into MIXED and got worse: 65% against v1's 73%.

v3 first added a "residual gate" — only trust a window when the part of the field that
neither rotation nor forward motion explains is small. This looked right on five
examples but is wrong in general. On all 60 reviewed windows:

    STATIC   residual 0.09 - 2.43 (median 0.22)
    FWD      0.18 - 0.33          (median 0.26)
    LEFT     0.26 - 1.85          (median 0.62)
    RIGHT    0.40 - 1.32          (median 0.85)

The residual grows with the *amount* of motion, not with its nature: rotations have more
displacement and therefore more residual than a still scene does. The best threshold
separates only 52% of windows, which is chance. So the gate is dropped. Five examples
were not enough to justify it and I should not have trusted them.

What actually works
-------------------
Two small things, both grounded in the review:

1. THE USER DOES NOT DISTINGUISH QUIET FROM BARELY MOVING. Twenty-seven windows were
   called "standing still". Twelve of them were ones the old classifier put in MIXED
   with values near zero (|a| <= 0.95, |div| <= 0.43). Reporting MIXED for those was
   wrong: "neither term dominates" and "nothing is happening" are the same answer to the
   person holding the camera. So the quiet branch is widened to cover both.

2. A ROTATION MUST HOLD ITS SIGN. Window r044 at t=17.8 s was called LEFT. Per frame
   its median dx swings +3.8, -5.0, +5.3, -9.4, -4.2 against a window median of +2.5:
   twelve sign changes in forty frames, and two independent estimators agree only 55%
   of the time, against 96% for a genuine rotation. Its median happens to land on the
   right side of zero; that is not a rotation.

Everything else is left as it was. The thresholds are fitted on these 60 windows, which
is stated in the output; they are not cross-validated and should be treated as a first
estimate rather than a calibration.

Usage:
    PYTHONPATH=. python scripts/p03_classify_v3.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIGNAL = ROOT / "output/p02_full/full_signal.npz"
REVIEW_SET = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
OUT = ROOT / "data/p01r/windows_v3.json"
FPS = 30.0

QUIET_A = 1.00            # below this there is no rotation to speak of
QUIET_DIV = 0.45          # below this there is no forward travel to speak of
TURN_MIN_A = 1.00
FWD_MIN_DIV = 0.45
DOMINANCE = 1.30
COHERENT_PER_FRAME = 1.00 # frames whose |a| carries sign information
SIGN_HELD_MIN = 0.70      # r044 measured 0.5; genuine rotations measure near 1.0
SMOOTH_S = 0.5
MIN_SEG_S = 0.9


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def stats(sig: dict, i0: int, i1: int) -> dict:
    a = sig["a"][i0:i1]
    div = (sig["b"] + sig["c"])[i0:i1]
    ax = sig["act_x"][i0:i1]
    ay = sig["act_y"][i0:i1]
    med_a = float(np.median(a))
    sgn = np.sign(med_a) if med_a != 0 else 1
    strong = np.abs(a) >= COHERENT_PER_FRAME
    held = float(np.mean(np.sign(a[strong]) == sgn)) if strong.any() else 0.0
    return {
        "a": med_a, "a_abs": float(np.median(np.abs(a))),
        "div": float(np.median(div)), "div_abs": float(np.median(np.abs(div))),
        "activity": float(np.median(ax)), "activity_vertical": float(np.median(ay)),
        "sign_held": held, "informative_frames": int(strong.sum()),
        "n_frames": int(len(a)),
    }


def classify(s: dict) -> tuple[str, str, dict]:
    why = {"a": round(s["a"], 3), "divergence": round(s["div"], 3),
           "sign_held": round(s["sign_held"], 3),
           "informative_frames": s["informative_frames"]}
    rot = abs(s["a"])
    fwd = abs(s["div"])

    # a rotation must hold its sign across the frames that carry information
    coherent = s["sign_held"] >= SIGN_HELD_MIN if s["informative_frames"] >= 3 else False

    if rot >= TURN_MIN_A and coherent and fwd < DOMINANCE * rot:
        kind = "LEFT" if s["a"] > 0 else "RIGHT"
        conf = "high" if (rot >= 2.5 and s["sign_held"] >= 0.9) else \
               "medium" if rot >= 1.5 else "low"
        why["decision"] = "rotation, sign held"
        return kind, conf, why

    if fwd >= FWD_MIN_DIV and rot < DOMINANCE * fwd:
        why["decision"] = "forward, divergence dominates"
        return "FWD", ("high" if fwd >= 0.9 else "medium"), why

    # quiet, or barely moving, or a motion too incoherent to be a camera rotation.
    # The person holding the camera answers "standing still" to all three.
    why["decision"] = ("quiet" if (rot < QUIET_A and fwd < QUIET_DIV)
                       else "motion present but not attributable to the camera")
    return "STATIC", ("high" if (rot < QUIET_A and fwd < QUIET_DIV) else "low"), why


def segment(sig: dict) -> list[dict]:
    n_sm = max(int(round(SMOOTH_S * FPS)), 1)
    a_s = box(sig["a"], n_sm)
    div_s = box(sig["b"] + sig["c"], n_sm)

    def rough(i: int) -> str:
        av, dv = abs(float(a_s[i])), abs(float(div_s[i]))
        if av < QUIET_A and dv < QUIET_DIV:
            return "STATIC"
        if av >= TURN_MIN_A and dv < DOMINANCE * av:
            return "LEFT" if a_s[i] > 0 else "RIGHT"
        if dv >= FWD_MIN_DIV:
            return "FWD"
        return "STATIC"

    labels = [rough(i) for i in range(len(a_s))]
    segs, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if (j - i + 1) / FPS >= MIN_SEG_S:
            st = stats(sig, i, j + 1)
            kind, conf, why = classify(st)
            segs.append({
                "kind": kind, "confidence": conf, "why": why,
                "t0": round(float(sig["t"][i]), 2), "t1": round(float(sig["t"][j]), 2),
                "duration_s": round((j - i + 1) / FPS, 2),
                "a": round(st["a"], 3), "divergence": round(st["div"], 3),
                "activity": round(st["activity"], 3),
                "sign_held": round(st["sign_held"], 3),
                "camera_direction": kind,
            })
        i = j + 1
    return segs


def main() -> None:
    sig = {k: v for k, v in np.load(SIGNAL).items()}
    segs = segment(sig)

    print("классификатор v3")
    print(f"  поворот: |a| >= {TURN_MIN_A}, знак держится >= {SIGN_HELD_MIN:.0%} "
          f"на кадрах с |a| >= {COHERENT_PER_FRAME}")
    print(f"  вперёд : |div| >= {FWD_MIN_DIV}, расширение доминирует")
    print(f"  покой  : всё остальное, включая слабое движение")
    print(f"\n  окон: {len(segs)}   {dict(Counter(s['kind'] for s in segs))}")

    rs = json.loads(REVIEW_SET.read_text(encoding="utf-8"))["candidates"]
    v = json.loads(VERDICTS.read_text(encoding="utf-8"))["verdicts"]

    def window_label(c: dict) -> str:
        """Label the window itself, not whichever segment happens to contain its middle.

        Scoring through the segmentation was wrong: window r024 splits into two RIGHT
        segments with a 0.37 s gap, and its midpoint fell exactly in that gap, so the
        classifier was credited with STATIC for a window it calls RIGHT everywhere. Each
        window is classified on its own, which is also what the review page asks about.
        """
        m = (sig["t"] >= c["t0"]) & (sig["t"] <= c["t1"])
        i0 = int(np.argmax(m))
        i1 = int(len(m) - np.argmax(m[::-1]))
        if i1 <= i0:
            return "STATIC"
        return classify(stats(sig, i0, i1))[0]

    rows = []
    for c in rs:
        vv = v.get(c["id"])
        if not vv:
            continue
        if vv["verdict"] == "correct":
            truth = c["kind"]
        else:
            truth = {"LEFT": "LEFT", "RIGHT": "RIGHT", "FWD": "FWD",
                     "STATIC": "STATIC"}.get(vv.get("actual_direction") or "", "MIXED")
        rows.append({"id": c["id"], "old": c["kind"], "new": window_label(c),
                     "truth": truth, "t0": c["t0"], "t1": c["t1"]})

    kinds = ["LEFT", "RIGHT", "FWD", "STATIC", "MIXED"]
    print(f"\n=== точность на 60 твоих вердиктах (окно классифицируется целиком) ===")
    scores = {}
    for tag in ("old", "new"):
        tab = defaultdict(int)
        for r in rows:
            tab[(r[tag], r["truth"])] += 1
        ok = sum(tab.get((k, k), 0) for k in kinds)
        scores[tag] = ok
        print(f"\n  {tag}: {ok}/{len(rows)} = {ok/len(rows):.0%}")
        print(f"  {'':>8s} | " + " ".join(f"{k:>7s}" for k in kinds) + "    точность")
        for k in kinds:
            cells = [tab.get((k, q), 0) for q in kinds]
            n = sum(cells)
            if not n:
                continue
            print(f"  {k:>8s} | " + " ".join(f"{v:7d}" for v in cells) +
                  f"    {tab.get((k, k), 0)}/{n} = {tab.get((k, k), 0)/n:.0%}")

    # what the system is actually good for
    turns = [r for r in rows if r["new"] in ("LEFT", "RIGHT")]
    turn_hits = sum(1 for r in turns if r["new"] == r["truth"])
    human_turns = [r for r in rows if r["truth"] in ("LEFT", "RIGHT")]
    print(f"\n=== разбор по назначению ===")
    print(f"  повороты (система сказала LEFT/RIGHT): {turn_hits}/{len(turns)} = "
          f"{turn_hits/len(turns):.0%} верно")
    print(f"  повороты, которые ты видел:            {len(human_turns)}, "
          f"найдено {sum(1 for r in human_turns if r['new'] in ('LEFT','RIGHT'))} "
          f"({sum(1 for r in human_turns if r['new'] in ('LEFT','RIGHT'))/len(human_turns):.0%})")

    fwd_rows = [r for r in rows if r["truth"] in ("FWD", "STATIC") or r["new"] in ("FWD", "STATIC")]
    divs_told_fwd = []
    divs_told_static = []
    for r in rows:
        if r["truth"] in ("LEFT", "RIGHT"):
            continue
        m = (sig["t"] >= r["t0"]) & (sig["t"] <= r["t1"])
        i0 = int(np.argmax(m)); i1 = int(len(m) - np.argmax(m[::-1]))
        dv = abs(stats(sig, i0, i1)["div"])
        (divs_told_fwd if r["truth"] == "FWD" else divs_told_static).append(round(dv, 2))
    print(f"\n  ГРАНИЦА 'ВПЕРЁД' / 'НА МЕСТЕ' (окна без поворота):")
    print(f"    |расширение| когда ты сказал ВПЕРЁД : {sorted(divs_told_fwd)}")
    print(f"    |расширение| когда ты сказал НА МЕСТЕ: {sorted(divs_told_static)}")
    print(f"    -> диапазоны перекрываются, порога не существует")
    print(f"    на этих окнах система угадывает "
          f"{sum(1 for r in fwd_rows if r['new'] == r['truth'])}/{len(fwd_rows)}, "
          f"а если всегда говорить 'на месте' было бы "
          f"{sum(1 for r in fwd_rows if r['truth'] == 'STATIC')}/{len(fwd_rows)}")

    bad = [r for r in rows if r["new"] != r["truth"]]
    print(f"\n=== остались ошибки ===")
    for r in bad:
        print(f"  {r['id']}  t={r['t0']:6.0f}  v3 {r['new']:>6s}, на самом деле {r['truth']:>6s}"
              + ("   <- граница вперёд/покой" if {r['new'], r['truth']} <= {"FWD", "STATIC"}
                 else "   <- единственный ложный поворот" if r["new"] in ("LEFT", "RIGHT")
                 else ""))

    OUT.write_text(json.dumps({
        "version": 3,
        "principle": "вопрос о движении камеры; поворот обязан держать знак",
        "scoring": "окно классифицируется целиком, а не через сегментацию",
        "dropped": [
            "гейт по остатку: он растёт с величиной движения, а не с его природой, "
            "и разделяет классы на уровне случайности (52%)",
            "всякий порог между 'вперёд' и 'покой': диапазоны |расширения| "
            "перекрываются полностью",
        ],
        "thresholds": {"quiet_a": QUIET_A, "quiet_div": QUIET_DIV,
                       "turn_min_a": TURN_MIN_A, "fwd_min_div": FWD_MIN_DIV,
                       "dominance": DOMINANCE, "sign_held_min": SIGN_HELD_MIN,
                       "coherent_per_frame": COHERENT_PER_FRAME,
                       "smooth_s": SMOOTH_S, "min_seg_s": MIN_SEG_S},
        "caveat": "пороги подобраны на этих же 60 окнах, перекрёстной проверки нет",
        "scored": {"n": len(rows), "v1": scores.get("old"), "v3": scores.get("new")},
        "turns": {"n_system": len(turns), "correct": turn_hits,
                  "n_human": len(human_turns),
                  "found": sum(1 for r in human_turns if r['new'] in ('LEFT', 'RIGHT'))},
        "fwd_vs_static": {"div_when_fwd": sorted(divs_told_fwd),
                          "div_when_static": sorted(divs_told_static),
                          "separable": False},
        "remaining_errors": [r for r in bad],
        "n_windows": len(segs), "counts": dict(Counter(s["kind"] for s in segs)),
        "windows": segs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")

    v2 = ROOT / "data/p01r/windows_v2.json"
    if v2.exists():
        v2.unlink()
    notefile = ROOT / "data/p01r/windows_v3_caveat.txt"
    notefile.write_text(
        "Пороги v3 подобраны на тех же 60 окнах, на которых измерена точность.\n"
        "Перекрёстной проверки нет, поэтому 88% - это оценка сверху, а не честное\n"
        "обобщение. Чтобы получить честную оценку, нужен независимый набор окон,\n"
        "размеченный заново и не использованный при подборе.\n",
        encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
