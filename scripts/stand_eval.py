#!/usr/bin/env python3
"""Stand / walk: how well each available signal matches the human labels.

The labels come from /final (data/final_tracker/{clip}_stand_labels.json): every
interval in "stand" is standing, everything else up to "reviewed_until" is walking.
Time base is seconds of the clip, which equals tracker time at zero offset.

Signals, one value per second:
    tracker_stop   is_stop from the V2 trajectory (binary, pixel motion < STOP)
    pixel_motion   p13_stillness.frame_motion; low = standing
    fly_no_net     net_displacement channel; +confidence for NO_NET, -confidence for MOVE
    step_bob       share of vertical-shift power at 1.4-2.6 Hz over 6 s; low = standing
    turn_dx        mean |horizontal shift| over 6 s; reported only as a control

AUC 0.5 is chance. Nothing here changes the tracker.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/stand_eval.py --clip VID00020
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STEP_BAND = (1.4, 2.6)
WIN_S = 6.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a standing second > score of a walking second), ties count half."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def shifts(video: Path, cache: Path) -> tuple[np.ndarray, np.ndarray, float]:
    if cache.exists():
        d = np.load(cache)
        return d["dx"], d["dy"], float(d["fps"])
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, win, dx, dy = None, None, [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (sx, sy), _ = cv2.phaseCorrelate(prev, g, win)
            dx.append(sx)
            dy.append(sy)
        prev = g
    dx, dy = np.array(dx), np.array(dy)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, dx=dx, dy=dy, fps=fps)
    return dx, dy, fps


def windowed(dx: np.ndarray, dy: np.ndarray, fps: float, centers: np.ndarray):
    t = np.arange(len(dy)) / fps
    bob, turn = [], []
    for c in centers:
        m = (t >= c - WIN_S / 2) & (t < c + WIN_S / 2)
        s = dy[m] - dy[m].mean() if m.sum() else np.array([])
        if len(s) < fps * 2:
            bob.append(np.nan)
            turn.append(np.nan)
            continue
        p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        f = np.fft.rfftfreq(len(s), 1 / fps)
        tot = p[(f > 0.3) & (f < 6)].sum()
        bob.append(p[(f >= STEP_BAND[0]) & (f <= STEP_BAND[1])].sum() / tot if tot > 0 else np.nan)
        turn.append(float(np.abs(dx[m]).mean()))
    return np.array(bob), np.array(turn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="VID00020")
    ap.add_argument("--version", default="v2", choices=["v1", "v2", "v3", "v4", "v5"])
    a = ap.parse_args()
    clip = a.clip

    lab_path = ROOT / "data/final_tracker" / f"{clip}_stand_labels.json"
    lab = json.loads(lab_path.read_text(encoding="utf-8"))
    stand = [tuple(map(float, iv)) for iv in lab.get("stand", [])]
    sub = "final_tracker" if a.version == "v1" else f"final_tracker_{a.version}"
    run_dir = ROOT / "output" / sub / clip
    rows = list(csv.DictReader((run_dir / "trajectory.csv").open(encoding="utf-8")))
    reviewed = float(lab.get("reviewed_until") or 0)
    if reviewed <= 0:
        reviewed = max((b for _, b in stand), default=0.0)
        print(f"  reviewed_until не задан — беру конец последнего отрезка: {reviewed:.1f} с")

    rows = [r for r in rows if float(r["time"]) <= reviewed]
    t = np.array([float(r["time"]) for r in rows])
    y = np.array([any(lo <= x <= hi for lo, hi in stand) for x in t])
    if y.sum() == 0 or (~y).sum() == 0:
        print("  нужны и стоящие, и идущие секунды в размеченной части")
        return 1

    from p13_stillness import frame_motion
    video = ROOT / "data/p01r" / f"{clip}.AVI"
    mot, dt = frame_motion(video)
    pix = np.array([mot[min(int(x / dt), len(mot) - 1)] for x in t])
    dx, dy, fps = shifts(video, ROOT / "output/final_tracker_v2" / clip / "shift_xy.npz")
    bob, turn = windowed(dx, dy, fps, t)
    stop = np.array([r["is_stop"] in ("1", 1) for r in rows])
    fly = np.array([(1 if r["net_displacement"] == "NO_NET" else -1)
                    * float(r["net_displacement_confidence"] or 0) for r in rows])

    n_s, n_w = int(y.sum()), int((~y).sum())
    print(f"{clip}: размечено {reviewed:.0f} с — стоит {n_s} с ({len(stand)} отрезков), идёт {n_w} с")
    print("  AUC «отличает стоит от идёт»: 0.5 = наугад, 1.0 = идеально\n")

    res = {}
    def report(name, score_stand_high, note):
        ok = np.isfinite(score_stand_high)
        v = auc(score_stand_high[ok & y], score_stand_high[ok & ~y])
        res[name] = round(v, 3)
        print(f"  {name:14s} AUC {v:.3f}   {note}")

    report("pixel_motion", -pix, "меньше движения кадра = стоит")
    report("fly_no_net", fly, "муха: NO_NET с уверенностью")
    report("step_bob", -bob, "меньше ритма шагов = стоит")
    report("turn_dx", -turn, "контроль: меньше поворотов = стоит")

    tp = int((stop & y).sum())
    fp = int((stop & ~y).sum())
    rec = tp / n_s
    spec = 1 - fp / n_w
    res["tracker_stop"] = {"recall_stand": round(rec, 3), "walk_kept": round(spec, 3),
                           "balanced": round((rec + spec) / 2, 3)}
    bal = (rec + spec) / 2
    null = []
    for s in range(15, len(y) - 15):
        yy = np.roll(y, s)
        null.append(0.5 * ((stop & yy).sum() / yy.sum() + (~stop & ~yy).sum() / (~yy).sum()))
    p_bal = (np.sum(np.array(null) >= bal) + 1) / (len(null) + 1)
    res["tracker_stop"]["p_shift"] = round(float(p_bal), 4)
    print(f"\n  tracker_stop {a.version}: поймал стояния {rec:.0%}, ходьбу не тронул {spec:.0%}, "
          f"сбалансированно {bal:.0%} (50% = наугад), p(сдвиг разметки)={p_bal:.3f}")
    if n_s < 30 or len(stand) < 4:
        print(f"\n  мало разметки ({len(stand)} отрезков, {n_s} с стояния) — цифры шумные, нужно ≥ 4–5 отрезков")

    out = run_dir / "stand_eval.json"
    out.write_text(json.dumps({"clip": clip, "reviewed_until": reviewed, "stand_s": n_s,
                               "walk_s": n_w, "segments": len(stand), "results": res},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  записано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
