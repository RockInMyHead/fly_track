#!/usr/bin/env python3
"""Freeze the V3 stand rule on VID00020, before any test clip is labelled.

Fits p = sigmoid(b + w_fly*fly + w_bob*bob) to the human stand labels of the train clip,
smooths over SMOOTH_S (fixed in advance, not tuned), picks the threshold with the best
balanced accuracy on the train clip, and writes everything — including how the test will
be judged — to data/final_tracker_v3/FROZEN_STAND.json together with code hashes.

An existing freeze is not overwritten without --revision "reason".

Usage:
    PYTHONPATH=. .venv/bin/python scripts/stand_freeze.py
    PYTHONPATH=. .venv/bin/python scripts/stand_freeze.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_tracker as V1  # noqa: E402
import final_tracker_v2 as V2  # noqa: E402
import final_tracker_v3 as V3  # noqa: E402
from p08_graph import Graph  # noqa: E402

TRAIN_CLIP = "VID00020"
SMOOTH_S = 5.0
L2 = 1e-3
HASHED = ["scripts/final_tracker_v3.py", "scripts/stand_freeze.py", "scripts/final_tracker_v2.py",
          "scripts/final_tracker.py"]


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def labels(clip: str) -> tuple[list, float]:
    doc = json.loads((ROOT / "data/final_tracker" / f"{clip}_stand_labels.json").read_text(encoding="utf-8"))
    stand = [tuple(map(float, iv)) for iv in doc["stand"]]
    rev = float(doc.get("reviewed_until") or 0) or max(b for _, b in stand)
    return stand, rev


def channels(clip: str):
    path = V1.resolve_video(clip)
    g = Graph.load(ROOT / "data/p08/graph.json")
    freeze = json.loads(V1.FREEZE.read_text(encoding="utf-8"))
    s = json.loads((V2.OUT_BASE / clip / "report.json").read_text(encoding="utf-8"))["start"]
    checks = V2.load_checks(clip, path, g, freeze, s["edge"], s["from"])
    return V1.Channels(clip, path, checks["_t"], freeze), path, checks["_t"]


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-A @ w))
        grad = A.T @ (p - y) + L2 * np.r_[0, w[1:]]
        H = (A * (p * (1 - p))[:, None]).T @ A + L2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def balanced(flag: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * (flag[y].mean() + (~flag[~y]).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=None, help="причина перезаписи заморозки")
    ap.add_argument("--verify", action="store_true", help="сверить хэши кода с заморозкой")
    a = ap.parse_args()
    out = V3.FROZEN_STAND

    if a.verify:
        doc = json.loads(out.read_text(encoding="utf-8"))
        bad = [p for p, h in doc["code_sha256"].items() if sha(p) != h]
        print("сверка:", "все хэши совпали" if not bad else f"РАСХОЖДЕНИЕ: {bad}")
        return 1 if bad else 0
    if out.exists() and not a.revision:
        print(f"ОТКАЗ: {out} уже есть. Перезапись только с --revision \"причина\"", file=sys.stderr)
        return 1

    stand, rev = labels(TRAIN_CLIP)
    ch, path, t = channels(TRAIN_CLIP)
    grid = np.arange(float(t[0]), min(float(t[-1]), rev) + 1e-9, V3.GRID_S)
    F = V3.stand_features(ch, path, grid)
    y = np.array([any(lo <= x <= hi for lo, hi in stand) for x in grid])
    w = fit_logistic(F, y.astype(float))
    rule = {"bias": float(w[0]), "w_fly": float(w[1]), "w_bob": float(w[2]),
            "smooth_s": SMOOTH_S, "threshold": 0.5,
            "step_band_hz": list(V3.STEP_BAND), "bob_window_s": V3.BOB_WIN_S, "grid_s": V3.GRID_S}
    p = V3.stand_probability(F, rule)
    cands = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [balanced(p > c, y) for c in cands]
    rule["threshold"] = float(cands[int(np.argmax(scores))])
    flag = p > rule["threshold"]

    doc = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_clip": TRAIN_CLIP,
        "train_labels": f"data/final_tracker/{TRAIN_CLIP}_stand_labels.json",
        "rule": rule,
        "train_in_sample": {
            "stand_s": float(y.sum() * V3.GRID_S), "walk_s": float((~y).sum() * V3.GRID_S),
            "recall_stand": round(float(flag[y].mean()), 3),
            "walk_kept": round(float((~flag[~y]).mean()), 3),
            "balanced": round(float(balanced(flag, y)), 3),
            "note": "на том же ролике, где подбирались веса и порог — оптимистично",
        },
        "test_protocol": {
            "clip": "любой ролик, кроме VID00020, размеченный ПОСЛЕ frozen_at",
            "labels": "data/final_tracker/{clip}_stand_labels.json, отрезки «стоит» + reviewed_until",
            "metric": "сбалансированная точность флага «стоит» V3 против V2 на одних секундах",
            "pass": "V3 balanced > V2 balanced И V3 balanced > 0.5 с p < 0.05 "
                    "(сдвиг разметки по кругу)",
            "no_retune": "веса, сглаживание и порог не меняются после просмотра теста",
        },
        "code_sha256": {p_: sha(p_) for p_ in HASHED},
        "revisions": [],
    }
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        doc["revisions"] = old.get("revisions", []) + [{"at": doc["frozen_at"], "why": a.revision,
                                                        "previous_rule": old.get("rule")}]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    tr = doc["train_in_sample"]
    print(f"заморожено {out}")
    print(f"  p = sigmoid({rule['bias']:+.3f} {rule['w_fly']:+.3f}*fly {rule['w_bob']:+.3f}*bob), "
          f"сглаживание {SMOOTH_S} с, порог {rule['threshold']}")
    print(f"  на VID00020 (в выборке): стояния поймано {tr['recall_stand']:.0%}, "
          f"ходьба не тронута {tr['walk_kept']:.0%}, сбалансированно {tr['balanced']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
