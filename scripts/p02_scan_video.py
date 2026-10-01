#!/usr/bin/env python3
"""
P0.2 — scan the whole clip and list every candidate turn, with a direction.

The point of this script is to invert the workflow. Until now a human watched the
footage and wrote down where a turn was; here the system proposes and the human only
confirms or rejects. That is far cheaper per turn, and it produces many more labels,
which is what the statistics have been missing.

Stage 1 (fast, frontend only)
    Run the frozen content-motion measurement over the whole clip at native frame
    rate and cache the per-frame signal. No brain, no injection: this is the same
    estimator that was validated against hand labels on readable windows (7/8
    agreement), so it is a reasonable thing to propose turns with.

Stage 2 (candidate detection)
    Find stretches where the measured sideways motion is large, one-signed, coherent
    and sustained. Each candidate gets a direction, a confidence from its coherence,
    and the evidence behind it.

The output is a list of proposals. Nothing here is a label: a proposal becomes a
label only when a human confirms it, which is what the review app is for.

Only stage 1 costs real time (about 6 minutes for the whole clip); its result is
cached so re-running stage 2 is instant.

Usage:
    PYTHONPATH=. python scripts/p02_scan_video.py                # scan + detect
    PYTHONPATH=. python scripts/p02_scan_video.py --only detect  # reuse the cache
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.content_motion import CROP, ENCODE, GRID_H, GRID_W, read_window
from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
CACHE = ROOT / "output/p02_scan/frontend_signal.npz"
OUT_JSON = ROOT / "data/p01r/candidate_turns.json"
FPS = 30.0

# Detection parameters. Chosen so that a real turn shows up and idle wobble does not;
# stated here rather than tuned per result.
SMOOTH_S = 0.8          # smoothing of the pooled signal before peak picking
MIN_TURN_S = 0.8        # a candidate must last at least this long
MAX_TURN_S = 12.0       # longer than this and it is a cruise, not a turn
PEAK_FRAC = 0.5         # extend a candidate while the signal stays above half its peak
MIN_COHERENCE = 0.25    # below this the field is not readable; candidate is flagged
MIN_LOCAL_MOTION = 0.35  # px/frame; below this the scene is essentially still


def scan(video: Path) -> dict:
    """Per-frame frozen measurement over the whole clip, at native frame rate."""
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    med, mean, act, cov = [], [], [], []
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, _dy, diag = lk.process(band)
        flat = dxf.reshape(-1)
        med.append(float(np.median(flat)))
        mean.append(float(flat.mean()))
        act.append(float(np.abs(flat).mean()))
        cov.append(float(diag.coverage))
        i += 1
        if i % 1500 == 0:
            el = time.time() - t0
            print(f"    {i}/{n_total or '?'} frames  ({el:.0f}s elapsed, "
                  f"{i / max(el, 1e-9):.0f} fps)")
    cap.release()
    return {
        "median": np.array(med, dtype=np.float32),
        "mean": np.array(mean, dtype=np.float32),
        "activity": np.array(act, dtype=np.float32),
        "coverage": np.array(cov, dtype=np.float32),
        "t": np.arange(len(med), dtype=np.float32) / FPS,
    }


def box_smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n, dtype=np.float64) / n
    return np.convolve(x, k, mode="same")


def detect(sig: dict) -> list[dict]:
    """Peak-picking on the smoothed pooled signal, with coherence as the gate."""
    t = sig["t"]
    raw = sig["median"].astype(np.float64)
    act = sig["activity"].astype(np.float64)
    cov = sig["coverage"].astype(np.float64)
    n_sm = max(int(round(SMOOTH_S * FPS)), 1)
    sm = box_smooth(raw, n_sm)
    sm_act = box_smooth(act, n_sm)

    # local coherence on the smoothed series: does the pooled sign actually hold?
    sign = np.sign(sm)
    coh = box_smooth(np.abs(sign), n_sm)          # fraction of frames agreeing
    magnitude = np.abs(sm)
    # strength combines "how big" with "how consistent" and with the local activity
    strength = magnitude * coh * np.minimum(1.0, sm_act / max(MIN_LOCAL_MOTION, 1e-9))

    thr = float(np.percentile(strength, 97.0))
    thr = max(thr, 0.05)
    above = strength >= thr
    cands = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1]:
            j += 1
        # extend while the raw signal keeps the same sign and stays above half the peak
        pk = float(np.max(np.abs(sm[i:j + 1])))
        s0 = np.sign(sm[i])
        a, b = i, j
        while a > 0 and np.sign(sm[a - 1]) == s0 and abs(sm[a - 1]) > PEAK_FRAC * pk:
            a -= 1
        while b + 1 < len(sm) and np.sign(sm[b + 1]) == s0 and abs(sm[b + 1]) > PEAK_FRAC * pk:
            b += 1
        dur = (b - a + 1) / FPS
        i = j + 1
        if dur < MIN_TURN_S or dur > MAX_TURN_S:
            continue
        # pooled values over the extended window
        seg_sm = sm[a:b + 1]
        seg_act = sm_act[a:b + 1]
        med_v = float(np.median(seg_sm))
        coh_v = abs(med_v) / max(float(np.median(seg_act)), 1e-9)
        frac_agree = float(np.mean(np.sign(seg_sm) == np.sign(med_v)))
        direction_camera = "LEFT" if med_v > 0 else "RIGHT"   # content right = camera left
        conf = ("high" if coh_v >= 0.50 and frac_agree >= 0.80 else
                "medium" if coh_v >= 0.30 and frac_agree >= 0.70 else "low")
        cands.append({
            "t0": round(float(t[a]), 2), "t1": round(float(t[b]), 2),
            "duration_s": round(dur, 2),
            "content_dx": round(med_v, 4),
            "camera_direction": direction_camera,
            "coherence": round(coh_v, 3),
            "frac_frames_agree": round(frac_agree, 3),
            "local_activity": round(float(np.median(seg_act)), 3),
            "coverage": round(float(np.median(cov[a:b + 1])), 3),
            "peak_strength": round(float(pk), 4),
            "readable": bool(coh_v >= MIN_COHERENCE),
            "confidence": conf,
        })
    cands.sort(key=lambda c: c["t0"])
    # Merge duplicates: the peak picker can fire twice inside one physical turn, which
    # would ask the reviewer the same question twice and inflate the count.
    merged: list[dict] = []
    for c in cands:
        if merged:
            p = merged[-1]
            overlap = min(p["t1"], c["t1"]) - max(p["t0"], c["t0"])
            shorter = min(p["t1"] - p["t0"], c["t1"] - c["t0"])
            if overlap > 0.6 * shorter and p["camera_direction"] == c["camera_direction"]:
                keep = p if p["coherence"] >= c["coherence"] else c
                keep["t0"] = round(min(p["t0"], c["t0"]), 2)
                keep["t1"] = round(max(p["t1"], c["t1"]), 2)
                keep["duration_s"] = round(keep["t1"] - keep["t0"], 2)
                merged[-1] = keep
                continue
        merged.append(dict(c))
    for n, c in enumerate(merged, 1):
        c["id"] = f"c{n:03d}"
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["scan", "detect", "all"], default="all")
    ap.add_argument("--video", default=str(VIDEO))
    args = ap.parse_args()

    CACHE.parent.mkdir(parents=True, exist_ok=True)

    if args.only in ("scan", "all") or not CACHE.exists():
        print("stage 1 — frozen frontend over the whole clip")
        sig = scan(Path(args.video))
        np.savez_compressed(CACHE, **sig)
        print(f"  {len(sig['t'])} frames -> {CACHE}")
    else:
        d = np.load(CACHE)
        sig = {k: d[k] for k in d.files}
        print(f"stage 1 — reusing cache: {len(sig['t'])} frames")

    cands = detect(sig)
    dur_total = float(sig["t"][-1]) if len(sig["t"]) else 0.0

    print(f"\nstage 2 — candidates over {dur_total:.0f} s of video")
    print(f"  {'#':>3s} {'t0':>7s} {'t1':>7s} {'len':>6s} {'camera':>7s} {'content_dx':>11s} "
          f"{'coher':>6s} {'agree':>6s} {'act':>5s} {'conf':>7s}")
    for n, c in enumerate(cands, 1):
        print(f"  {n:3d} {c['t0']:7.2f} {c['t1']:7.2f} {c['duration_s']:6.2f} "
              f"{c['camera_direction']:>7s} {c['content_dx']:11.4f} {c['coherence']:6.2f} "
              f"{c['frac_frames_agree']:6.2f} {c['local_activity']:5.2f} "
              f"{c['confidence']:>7s}")

    n_l = sum(1 for c in cands if c["camera_direction"] == "LEFT")
    n_r = sum(1 for c in cands if c["camera_direction"] == "RIGHT")
    n_hi = sum(1 for c in cands if c["confidence"] == "high")
    n_read = sum(1 for c in cands if c["readable"])
    print(f"\n  total {len(cands)}: {n_l} camera-LEFT, {n_r} camera-RIGHT")
    print(f"  confidence high {n_hi}, readable (coherence >= {MIN_COHERENCE}) {n_read}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "source": str(args.video), "frames": int(len(sig["t"])),
        "duration_s": round(dur_total, 2),
        "detector": {
            "smooth_s": SMOOTH_S, "min_turn_s": MIN_TURN_S, "max_turn_s": MAX_TURN_S,
            "peak_frac": PEAK_FRAC, "min_coherence": MIN_COHERENCE,
            "min_local_motion": MIN_LOCAL_MOTION,
            "threshold": "97th percentile of the smoothed strength",
            "sign": "content moving right means the camera turned left",
        },
        "note": "These are PROPOSALS, not labels. A proposal becomes a label only after "
                "a human confirms or rejects it.",
        "n_candidates": len(cands),
        "n_left": n_l, "n_right": n_r, "n_high_confidence": n_hi, "n_readable": n_read,
        "candidates": cands,
    }, indent=2), encoding="utf-8")

    with (ROOT / "output/p02_scan/candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cands[0].keys()) if cands else ["t0"])
        w.writeheader()
        w.writerows(cands)
    print(f"\nWrote {OUT_JSON} and output/p02_scan/candidates.csv")


if __name__ == "__main__":
    main()
