#!/usr/bin/env python3
"""P13.1D — does the brain's net_displacement separate the band the video cannot?

Reads output/p13/p13_1d_scores.json and applies the rule frozen in
data/p13/FROZEN_NET_DISPLACEMENT_TEST.json. The AUC thresholds and the sign convention were fixed
before any of these numbers existed; nothing here revisits them.

Reported:

    overall      AUC of the channel's score for standing against walking, all windows
    band 12-28   the same inside the band where the video measure fails. This is the question.
    below 12     the control where the video measure is already reliable
    above 28     the other control

    secondary    for the best threshold on the score, the pair (false stop, missed stop) over all
                 windows, against the video gate's (0 of 40, 32 of 49). The best threshold is
                 reported for description; the band AUC is what the verdict uses.

The sign is fixed by the frozen convention: a positive logit means NO_NET, so a higher score means
standing. AUC is computed for standing against walking directly and not flipped afterwards.

Usage:
    PYTHONPATH=. python scripts/p13_1d_score.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
DATA = ROOT / "data/p13"
FREEZE = DATA / "FROZEN_NET_DISPLACEMENT_TEST.json"
SCORES = OUT / "p13_1d_scores.json"
DEST = OUT / "p13_1d_result.json"


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability a standing window scores above a walking one. 0.5 means no separation."""
    if len(pos) < 3 or len(neg) < 3:
        return None
    p, n = np.asarray(pos), np.asarray(neg)
    gt = float((p[:, None] > n[None, :]).sum())
    eq = float((p[:, None] == n[None, :]).sum())
    return (gt + 0.5 * eq) / (len(p) * len(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    fr = json.loads(FREEZE.read_text(encoding="utf-8"))
    s = json.loads(SCORES.read_text(encoding="utf-8"))
    w = s["windows"]

    print("=" * 100)
    print("P13.1D — net_displacement против полосы, которую видео не разделяет")
    print("=" * 100)
    print(f"  заморожено {fr['frozen_at']}")
    print(f"  пороги правила: годится ≥{fr['decision_rule']['usable_auc']}, "
          f"слабо ≥{fr['decision_rule']['weak_auc']}")
    print(f"  окон посчитано: {len(w)}")
    print()

    bands = {
        "все окна": lambda m: True,
        "полоса 12–28": lambda m: 12 <= m < 28,
        "ниже 12": lambda m: m < 12,
        "28 и выше": lambda m: m >= 28,
    }
    res = {}
    print(f"  {'выборка':<16}{'n':>4}{'стоит':>7}{'идёт':>6}{'AUC':>8}   чтение")
    print("  " + "-" * 72)
    for name, f in bands.items():
        sub = [x for x in w if f(x["motion"]) and x["answer"] in ("STANDS", "WALKS")]
        stands = [x["score"] for x in sub if x["answer"] == "STANDS"]
        walks = [x["score"] for x in sub if x["answer"] == "WALKS"]
        a_ = auc(stands, walks)
        res[name] = {"n": len(sub), "stands": len(stands), "walks": len(walks), "auc": a_}
        if a_ is None:
            read = "мало данных"
        elif a_ >= fr["decision_rule"]["usable_auc"]:
            read = "разделяет"
        elif a_ >= fr["decision_rule"]["weak_auc"]:
            read = "слабо"
        elif a_ >= 0.5:
            read = "почти случайно"
        else:
            read = "обратное направление"
        print(f"  {name:<16}{len(sub):>4}{len(stands):>7}{len(walks):>6}"
              f"{(a_ if a_ is not None else float('nan')):>8.3f}   {read}")
    print()

    band = res["полоса 12–28"]["auc"]
    allw = res["все окна"]["auc"]
    print(f"  ГЛАВНОЕ, ПОЛОСА 12–28: AUC = {band:.3f}" if band is not None
          else "  ГЛАВНОЕ: полоса не посчиталась")
    print()

    # the secondary figure: how the channel would gate on its own
    sub = [x for x in w if x["answer"] in ("STANDS", "WALKS")]
    best = None
    for th in np.arange(min(x["score"] for x in sub), max(x["score"] for x in sub), 0.01):
        frozen = [x for x in sub if x["score"] >= th]      # high score means standing
        let = [x for x in sub if x["score"] < th]
        fs = sum(1 for x in frozen if x["answer"] == "WALKS")
        ms = sum(1 for x in let if x["answer"] == "STANDS")
        if best is None or ms < best[1] or (ms == best[1] and fs < best[2]):
            best = (float(th), ms, fs, len(frozen))
    if best:
        th, ms, fs, nf = best
        res["best_threshold"] = {"threshold": th, "missed_stop": ms, "false_stop": fs,
                                 "n_frozen": nf}
        print("  ЕСЛИ ГЕЙТИТЬ ТОЛЬКО ЭТИМ КАНАЛОМ, лучший порог:")
        print(f"    порог {th:+.2f}: заморожено {nf}, из них ложных STOP {fs}, "
              f"пропущено стояния {ms}")
        print(f"    для сравнения видео-гейт: ложных STOP 0 из 40, пропущено стояния 32 из 49")
    print()

    v = ("USABLE" if band is not None and band >= fr["decision_rule"]["usable_auc"]
         else "WEAK" if band is not None and band >= fr["decision_rule"]["weak_auc"]
         else "UNUSABLE")
    print("  ВЕРДИКТ ПО ПРАВИЛУ, ЗАПИСАННОМУ ЗАРАНЕЕ:")
    if v == "USABLE":
        print(f"    AUC полосы {band:.3f} ≥ {fr['decision_rule']['usable_auc']} — канал годится "
              f"как второй признак")
        print("    оговорка: он отвечает на вопрос о чистом перемещении, а не о движении в момент")
    elif v == "WEAK":
        print(f"    AUC полосы {band:.3f}: канал несёт что-то, но сам полосу не закрывает")
        print("    нужна комбинация с видео-мерой или третий признак")
    else:
        print(f"    AUC полосы {band:.3f} < {fr['decision_rule']['weak_auc']} — полоса этим "
              f"каналом не закрывается")
        print("    ВАЖНО: это не доказывает, что канал плох. Он отвечает на другой вопрос:")
        print("    «нет чистого перемещения за семь секунд» ≠ «не двигается прямо сейчас»")
    print()

    DEST.write_text(json.dumps({
        "phase": "P13.1D — результат", "frozen_at": fr["frozen_at"],
        "n_scored": len(w), "bands": res,
        "band_12_28_auc": band, "overall_auc": allw,
        "video_gate_reference": {"false_stop": "0 из 40", "missed_stop": "32 из 49"},
        "verdict": v,
        "caveat": fr["caveat_recorded_in_advance"],
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  записано: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
