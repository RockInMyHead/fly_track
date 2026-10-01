#!/usr/bin/env python3
"""P13.1D — apply the frozen net_displacement reader to every reviewed window.

The reader is taken from data/p12/FROZEN_BEFORE_VID00010.json and used as it stands: three
(unit, feature) pairs, their weights, their bias. It was fitted on five recordings, none of which is
among the nine under test, so no window here is in-sample for it.

Each window is measured on its own chunk's recording with the same span P12 used — three seconds
before the moment to half a second before, the second around it, half a second to seven seconds
after — and reduced to the three features the reader expects.

Usage:
    PYTHONPATH=. python scripts/p13_1d_build.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.p11a_route_change import FEATS, features_at  # noqa: E402

OUT = ROOT / "output/p13"
DATA = ROOT / "data/p13"
P12 = ROOT / "data/p12/FROZEN_BEFORE_VID00010.json"
DEST = OUT / "p13_1d_scores.json"


def reader() -> dict:
    d = json.loads(P12.read_text(encoding="utf-8"))
    return d["readers_for_new_recording"]["net_displacement"]


_BRAIN: dict[str, tuple] = {}


def brain(video: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if video not in _BRAIN:
        d = np.load(ROOT / f"output/p10/brain_{video}.npz")
        _BRAIN[video] = (d["t"].astype(float), d["vis_rate"].astype(float),
                         [str(g) for g in d["vis_groups"]])
    return _BRAIN[video]


def score_at(r: dict, video: str, at: float) -> float | None:
    """The channel's logit at one moment, using the frozen pairs and weights.

    The weights are not three numbers but twenty-four: the reader is a logistic model over all
    eight statistics of each of its three cells, the layout `fit_readout` builds in P12, where the
    column corresponding to cell j and statistic k is `j * n_features + k`. The `feat` field names
    the single best statistic per cell and is descriptive — using the three of them against a
    twenty-four-long weight vector does not align, which is how this was caught.
    """
    t, X, names = brain(video)
    f = features_at(t, X, at)                 # [n_groups, 8], statistics in FEATS order
    if f is None:
        return None
    vec = []
    for unit in r["cells"]:
        if unit not in names:
            return None
        vec.extend(f[names.index(unit), :].tolist())
    if len(vec) != len(r["w"]):
        return None
    return float(np.dot(np.asarray(vec), np.asarray(r["w"])) + r["b"])


def windows() -> list[dict]:
    """Every reviewed window from both rounds, with its motion and its human label."""
    out = []
    k1 = {i["id"]: i for i in json.loads((DATA / "p13_1b_key.json").read_text(
        encoding="utf-8"))["items"]}
    a1 = json.loads((ROOT / "output/p13/p13_1b_answers.json").read_text(encoding="utf-8"))["answers"]
    for cid, k in k1.items():
        if cid in a1:
            out.append({"id": cid, "round": "1B", "video": k["video"],
                        "motion": round(12.0 + k["margin"], 2), "answer": a1[cid],
                        "center": k["center"]})
    k2 = {i["id"]: i for i in json.loads((DATA / "p13_1c_key.json").read_text(
        encoding="utf-8"))["items"]}
    a2 = json.loads((ROOT / "output/p13/p13_1c_answers.json").read_text(encoding="utf-8"))["answers"]
    for cid, k in k2.items():
        if cid in a2:
            out.append({"id": cid, "round": "1C", "video": k["video"],
                        "motion": k["motion"], "answer": a2[cid], "center": k["center"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    r = reader()
    print("=" * 92)
    print("P13.1D — net_displacement на 120 размеченных окнах")
    print("=" * 92)
    print(f"  читатель: {', '.join(f'{c}/{f}' for c, f in zip(r['cells'], r['feat']))}")
    print(f"  порог решения: logit > 0 означает NO_NET (человек не переместился)")
    print(f"  то есть «стоит» предсказывается положительным логитом")
    print()

    w = windows()
    print(f"  окон: {len(w)} (1B {sum(1 for x in w if x['round']=='1B')}, "
          f"1C {sum(1 for x in w if x['round']=='1C')})")
    have = {v for v in {x["video"] for x in w} if (ROOT / f"output/p10/brain_{v}.npz").exists()}
    print(f"  записей ранних клеток есть для: {sorted(have)}")
    print()

    ok, skipped = [], 0
    for x in w:
        if x["video"] not in have:
            skipped += 1
            continue
        s = score_at(r, x["video"], x["center"])
        if s is None:
            skipped += 1
            continue
        ok.append({**x, "score": round(s, 4)})
    print(f"  посчитано: {len(ok)}, пропущено (нет записи): {skipped}")
    print()

    DEST.write_text(json.dumps({
        "phase": "P13.1D — net_displacement на размеченных окнах",
        "reader": {k: r[k] for k in ("cells", "feat", "w", "b", "k", "n_groups")},
        "sign_convention": "logit > 0 → NO_NET; «стоит» предсказывается положительным логитом",
        "n_scored": len(ok), "n_skipped": skipped,
        "windows": ok,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  записано: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
