#!/usr/bin/env python3
"""
P06.3 — the frozen cells on the second video, and whether the two clips agree.

This is the test that the second clip makes possible. The cells were chosen on the first
clip, so on their own their performance there proves little. The question here is
agreement: does a cell that preferred RIGHT turns on the first route still prefer RIGHT
turns on a different route, through different corridors, past different objects?

    consistent across clips  -> the cell reads something about the direction of motion
    flips between clips      -> it was keyed to a feature of one particular scene

The mirror control of P05.5 could not settle this, because flipping an image reverses
both the motion and any left-right asymmetry of the scene. Two different routes do
separate them: a scene feature that happens to look "leftward" in one corridor has no
reason to recur in another, whereas the direction of motion means the same thing
everywhere.

The same decision rule is applied, unchanged
-------------------------------------------
Per cell, the oriented AUC between firing rate on confirmed RIGHT and confirmed LEFT
turns. A cell passes at AUC >= 0.80. Then, as in `data/p05_frozen_targets.json`:

    found the readout   >=4 of 8 primary pass, and <=1 of 3 controls
    close the branch    <=2 primary pass
    inconclusive        3 primary pass
    invalid test        >1 control passes

Two differences from the first clip's test, both stated up front
---------------------------------------------------------------
The labels here are new and were assigned without seeing this recording, so unlike
P05.4 there is no selection on this data to invalidate them.

The samples are smaller. The standard error accompanies every AUC and each pass is
marked as clear or not; at these sizes an AUC of 0.85 is not on its own decisive.

Usage:
    PYTHONPATH=. python scripts/p063_second_video_test.py
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

TRACE2 = ROOT / "output/p06_neurons/spike_trace.npz"
TARGETS2 = ROOT / "output/p06_neurons/targets.csv"
BANK2 = ROOT / "data/p01r/review_set_vid2.json"
VERDICTS2 = ROOT / "data/p01r/turn_verdicts_vid2.json"

TRACE1 = ROOT / "output/p053_threshold/spike_trace.npz"
TARGETS1 = ROOT / "output/p053_threshold/targets.csv"
BANK1A = ROOT / "data/p01r/review_set.json"
VERDICTS1A = ROOT / "data/p01r/turn_verdicts.json"
BANK1B = ROOT / "data/p01r/review_set_v2.json"
VERDICTS1B = ROOT / "data/p01r/turn_verdicts_v2.json"

FROZEN = ROOT / "data/p05_frozen_targets.json"
OUT = ROOT / "output/p06_test"


def auc_raw(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    o = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[o] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    return float((rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def hanley_se(auc: float, n1: int, n2: int) -> float:
    if n1 < 2 or n2 < 2:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc)
         + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2)
    return float(np.sqrt(max(v, 0.0)))


def turns(bank: Path, verdicts: Path) -> list[dict]:
    if not bank.exists() or not verdicts.exists():
        return []
    rs = json.loads(bank.read_text(encoding="utf-8"))["candidates"]
    vd = json.loads(verdicts.read_text(encoding="utf-8"))["verdicts"]
    out = []
    for x in rs:
        if x.get("repeat_of"):
            continue
        v = vd.get(x["id"])
        if not v:
            continue
        truth = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        if truth in ("LEFT", "RIGHT"):
            out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "kind": truth})
    return sorted(out, key=lambda e: e["t0"])


def rates(trace: Path, targets: Path, items: list[dict]) -> tuple[np.ndarray, dict]:
    if not trace.exists() or not items:
        return np.zeros((0, 0)), {}
    d = np.load(trace)
    t = d["t"].astype(np.float64)
    fired = d["fired"].astype(np.float64)
    dt = float(np.median(np.diff(t)))
    M = np.zeros((len(items), fired.shape[1]))
    for k, w in enumerate(items):
        m = (t >= w["t0"]) & (t <= w["t1"])
        if m.any():
            M[k] = fired[m].mean(axis=0) / dt
        else:
            M[k] = np.nan
    col = {i: x for i, x in enumerate(csv.DictReader(targets.open()))}
    return M, col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-threshold", type=float, default=0.80)
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    targets = frozen["primary"] + frozen["controls_expected_to_fail"]

    w2 = turns(BANK2, VERDICTS2)
    w1 = turns(BANK1A, VERDICTS1A) + turns(BANK1B, VERDICTS1B)
    M2, col2 = rates(TRACE2, TARGETS2, w2)
    M1, col1 = rates(TRACE1, TARGETS1, w1)

    print("P06.3 — замороженные клетки на втором видео")
    print(f"  видео 1: {len(w1)} подтверждённых поворотов (оба раунда)")
    print(f"  видео 2: {len(w2)} подтверждённых поворотов")
    if not w2:
        print("\n  ВТОРОЕ ВИДЕО ЕЩЁ НЕ РАЗМЕЧЕНО")
        print("  запустить: REVIEW_TAG=vid2 REVIEW_VIDEO=VID00002.AVI ./webapp/run.sh")
        print("  затем:     PYTHONPATH=. python scripts/p063_second_video_test.py")
        return

    ok1 = M1.shape[0] > 0
    isR2 = np.array([w["kind"] == "RIGHT" for w in w2])
    isR1 = np.array([w["kind"] == "RIGHT" for w in w1]) if ok1 else None

    def score(M: np.ndarray, isR, col, cell: str, side: str) -> dict:
        j = next((k for k, x in col.items()
                  if x["cell_type"] == cell and x["side"] == side), None)
        if j is None or M.shape[0] == 0:
            return {}
        v = M[:, j]
        good = np.isfinite(v)
        r = auc_raw(v[good & isR], v[good & ~isR])
        n1 = int((good & isR).sum())
        n2 = int((good & ~isR).sum())
        se = hanley_se(max(r, 1 - r), n1, n2)
        return {"raw": r, "auc": max(r, 1 - r), "se": se, "n_right": n1, "n_left": n2,
                "rate_R": float(np.mean(v[good & isR])),
                "rate_L": float(np.mean(v[good & ~isR]))}

    print(f"\n=== ТЕСТ НА ВТОРОМ ВИДЕО (порог {args.pass_threshold}) ===")
    print(f"  {isR2.sum()} RIGHT, {(~isR2).sum()} LEFT\n")
    rows, prim, ctrl = [], [], []
    for group, cells in (("основная", frozen["primary"]),
                         ("контроль", frozen["controls_expected_to_fail"])):
        for tgt in cells:
            s2 = score(M2, isR2, col2, tgt["cell_type"], tgt["side"])
            if not s2:
                continue
            s1 = score(M1, isR1, col1, tgt["cell_type"], tgt["side"]) if ok1 else {}
            passed = s2["auc"] >= args.pass_threshold
            clear = np.isfinite(s2["se"]) and (s2["auc"] - 1.96 * s2["se"] > 0.5)
            agree = (np.sign(s2["raw"] - 0.5) == np.sign(s1["raw"] - 0.5)) if s1 else None
            rows.append({"cell_type": tgt["cell_type"], "side": tgt["side"],
                         "group": group, **s2,
                         "passed": passed, "clear": clear,
                         "auc_video1": s1.get("auc"), "raw_video1": s1.get("raw"),
                         "agrees": agree})
            (prim if group == "основная" else ctrl).append(passed)
            tag = "ПРОШЛА" if passed else "нет"
            if not clear and passed:
                tag += " (шумно)"
            ag = ("совпало" if agree else "РАСХОДИТСЯ") if agree is not None else "—"
            v1 = f"{s1['auc']:.3f}" if s1 else "—"
            print(f"  {tgt['cell_type'] + ' ' + tgt['side']:>12s} {group:>10s} "
                  f"AUC2 {s2['auc']:.3f}±{s2['se']:.2f}  AUC1 {v1:>5s}  "
                  f"знак {ag:>11s}  {tag}")

    n_p, n_c = int(np.sum(prim)), int(np.sum(ctrl))
    print(f"\n=== РЕШЕНИЕ ПО ПРАВИЛУ ИЗ data/p05_frozen_targets.json ===")
    print(f"  основные прошли: {n_p} из {len(prim)}")
    print(f"  контроли прошли: {n_c} из {len(ctrl)}")
    if n_c > 1:
        decision = ("ТЕСТ НЕДЕЙСТВИТЕЛЕН: прошли контроли, значит метки на втором "
                    "видео согласованы с чем-то общим, а не с направлением")
    elif n_p >= max(4, len(prim) // 2):
        decision = ("ВЫХОД ПОДТВЕРЖДЁН НА ВТОРОМ ВИДЕО: замороженные клетки "
                    "различают повороты на маршруте, которого не видели")
    elif n_p <= 2:
        decision = "ВЕТКУ ЗАКРЫВАЕМ: на втором видео сигнал не воспроизводится"
    else:
        decision = f"НЕОПРЕДЕЛЁННО: прошли {n_p} из {len(prim)}"
    print(f"\n  {decision}")

    agree_rows = [r for r in rows if r["agrees"] is not None]
    n_agree = sum(1 for r in agree_rows if r["agrees"])
    print(f"\n=== СОГЛАСИЕ ЗНАКА МЕЖДУ ВИДЕО ===")
    print(f"  совпало направление предпочтения: {n_agree} из {len(agree_rows)}")
    if agree_rows:
        pr = [r for r in agree_rows if r["group"] == "основная"]
        ct = [r for r in agree_rows if r["group"] == "контроль"]
        print(f"  основные: {sum(1 for r in pr if r['agrees'])} из {len(pr)}")
        print(f"  контроли: {sum(1 for r in ct if r['agrees'])} из {len(ct)}")
        print(f"\n  это и есть главный вопрос второго видео: разные маршруты, разные")
        print(f"  сцены, но то же направление движения")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "second_video.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (bool(v) if isinstance(v, np.bool_) else
                            float(v) if isinstance(v, np.floating) else
                            int(v) if isinstance(v, np.integer) else
                            None if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in r.items()})

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o

    (OUT / "report.json").write_text(json.dumps(clean({
        "n_turns_video1": len(w1), "n_turns_video2": len(w2),
        "n_right_video2": int(isR2.sum()), "n_left_video2": int((~isR2).sum()),
        "pass_threshold": args.pass_threshold,
        "n_primary_passed": n_p, "n_controls_passed": n_c,
        "decision": decision,
        "sign_agreement": {"n_agree": n_agree, "n_total": len(agree_rows)},
        "per_cell": rows,
        "rule": "то же правило, что в data/p05_frozen_targets.json",
        "why_this_clip": "замороженные клетки выбраны на первом ролике, здесь другой "
                         "маршрут и другие сцены; совпадение знака означает, что "
                         "предпочтение не привязано к конкретной сцене",
        "caveat": "выборка меньше, чем на первом видео; стандартная ошибка указана "
                  "рядом с каждым значением",
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
