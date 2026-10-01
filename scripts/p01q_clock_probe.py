#!/usr/bin/env python3
"""
P0.1Q clock probe — is the 30 fps -> 50 Hz video clock destroying the signal?

`iter_video_at_brain_hz` samples a 30 fps video at 50 Hz by seeking to
millisecond positions. 40% of the resulting samples are exact duplicates of the
previous frame. On a duplicate sample the Reichardt correlator returns exactly
zero (`x*x - x*x`), so two fifths of every event contributes no motion at all,
and the *physical* time span of a fixed temporal lag (dt=2, 3 brain steps) is not
constant: it depends on where the duplicates fall, which differs per event.

This probe measures how much of the frontend result is an artefact of that clock.

  A. dilution      — same 50 Hz stream, statistics conditioned on samples where
                     the video actually advanced.
  B. native clock  — the frontend driven at the video's own 30 fps with exact
                     frame stepping, no seeking, no duplicates.

Nothing is tuned; the same frozen detectors are used throughout.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/p01q_clock_probe.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS = {
    "left_65":   (63.5, "LEFT"),
    "left_81":   (78.5, "LEFT"),
    "left_108":  (106.0, "LEFT"),
    "right_67":  (66.0, "RIGHT"),
    "right_210": (209.0, "RIGHT"),
    "right_216": (214.0, "RIGHT"),
    "fwd_140":   (137.5, "FWD"),
}
SIGN = {"LEFT": -1, "RIGHT": +1, "FWD": 0}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DT_BRAIN = 0.02
ENCODE = (96, 72)
FPS = 30.0


def _front(mode: str = "panorama") -> DirectionalMotionFrontend:
    return DirectionalMotionFrontend(
        FlowConfig(n_azimuth_bins=128, ema_tau_s=0.5, brain_dt=DT_BRAIN, crop_mode="center_band",
                   field_mode=mode, grid_w=16, grid_h=8)
    )


def _prep(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def stats(values: np.ndarray, want: int) -> dict:
    """values: [n_samples, n_units] of raw signed field values."""
    if values.size == 0 or want == 0:
        return {}
    signs = np.sign(values)
    correct = signs == want
    pooled = values.mean(axis=1)
    best = cur = 0
    for v in np.sign(pooled) == want:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {
        "n_frames": int(values.shape[0]),
        "unit_frac_correct_median": float(np.median(correct.mean(axis=0))),
        "units_above_70pct": float(np.mean(correct.mean(axis=0) >= 0.7)),
        "pooled_frac_correct": float(np.mean(np.sign(pooled) == want)),
        "pooled_signed_mean": float(pooled.mean()),
        "pooled_sign_correct": bool(np.sign(pooled.mean()) == want),
    }


def run_brain_hz(start: float, end: float) -> tuple[list[dict], int]:
    front = _front()
    rows = []
    last = None
    for t, frame in iter_video_at_brain_hz(str(VIDEO), start, end, DT_BRAIN):
        img = _prep(frame)
        dup = last is not None and np.array_equal(img, last)
        last = img
        d = front.process(img)
        rows.append({
            "t": t,
            "yaw": float(d.yaw_frozen),
            "field": np.asarray(d.signed_flow_field, dtype=np.float64),
            "duplicate": bool(dup),
        })
    return rows, int(round(PRE_ROLL_S / DT_BRAIN))


def run_native_fps(start: float, end: float, lags: tuple[int, ...] = (2, 3)) -> list[dict]:
    """Read consecutive frames from disk — exact frame stepping, no ms seeking."""
    cfg = FlowConfig(n_azimuth_bins=128, ema_tau_s=0.5, brain_dt=1.0 / FPS,
                     crop_mode="center_band", temporal_lags=lags)
    front = DirectionalMotionFrontend(cfg)
    cap = cv2.VideoCapture(str(VIDEO))
    f0 = int(round(start * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    rows = []
    for k in range(int(round((end - start) * FPS))):
        ok, frame = cap.read()
        if not ok:
            break
        d = front.process(_prep(frame))
        rows.append({
            "t": start + k / FPS,
            "yaw": float(d.yaw_frozen),
            "field": np.asarray(d.signed_flow_field, dtype=np.float64),
        })
    cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01q_clock")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1Q clock probe — 30 fps video driven at 50 Hz\n")
    print(f"  {'event':10s} {'label':5s} {'dup%':>6s} {'archived':>9s} "
          f"{'50Hz':>9s} {'50Hz adv':>9s} {'native30':>9s}")
    rows = []
    for eid, (start, lab) in EVENTS.items():
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        want = SIGN[lab]

        rows50, n_pre = run_brain_hz(t0, t1)
        dup_flags = [r["duplicate"] for r in rows50]
        dup_rate = float(np.mean(dup_flags))

        ev = rows50[n_pre:]
        keep = [f for f, d in zip(ev, dup_flags[n_pre:]) if not d]
        yaw50 = np.array([r["yaw"] for r in ev])
        yaw_adv = np.array([r["yaw"] for r in keep])
        field50 = np.stack([r["field"] for r in ev], axis=0)

        nat = run_native_fps(t0, t1)
        nat_ev = nat[int(round(PRE_ROLL_S * FPS)):]
        yaw_nat = np.array([r["yaw"] for r in nat_ev])

        arch = None
        bench = ROOT / "output/p01r_event_benchmark/report.json"
        if bench.exists():
            for e in json.loads(bench.read_text(encoding="utf-8"))["events"]:
                if e["id"] == eid:
                    arch = float((e.get("stats") or {}).get("yaw_frozen_median", 0.0))

        rec = {
            "event": eid, "label": lab, "dup_rate": dup_rate,
            "archived_median": arch,
            "yaw50_median": float(np.median(yaw50)),
            "yaw50_advanced_median": float(np.median(yaw_adv)) if len(yaw_adv) else None,
            "yaw_native30_median": float(np.median(yaw_nat)),
            "field50_units": stats(field50, want),
        }
        rows.append(rec)
        print(f"  {eid:10s} {lab:5s} {dup_rate:6.0%} "
              f"{arch if arch is not None else float('nan'):+9.4f} "
              f"{np.median(yaw50):+9.4f} "
              f"{(np.median(yaw_adv) if len(yaw_adv) else float('nan')):+9.4f} "
              f"{np.median(yaw_nat):+9.4f}")

    print(f"\n  sign accuracy (yaw<0 = LEFT):")
    for key, name in (("archived_median", "archived 50 Hz seek"),
                      ("yaw50_median", "current 50 Hz seek"),
                      ("yaw50_advanced_median", "50 Hz, advanced frames only"),
                      ("yaw_native30_median", "native 30 fps")):
        ok = tot = 0
        for r in rows:
            v = r.get(key)
            if r["label"] == "FWD" or v is None:
                continue
            want = -1 if r["label"] == "LEFT" else +1
            tot += 1
            ok += int(np.sign(v) == want)
        print(f"    {name:30s} {ok}/{tot}")

    print(f"\n  per-cell stability on the 50 Hz stream (event frames only):")
    print(f"    {'event':10s} {'per-cell med':>13s} {'cells>=70%':>11s} "
          f"{'pooled':>7s} {'signed mean':>12s}")
    for r in rows:
        s = r["field50_units"]
        if not s:
            continue
        print(f"    {r['event']:10s} {s['unit_frac_correct_median']:13.0%} "
              f"{s['units_above_70pct']:11.0%} {s['pooled_frac_correct']:7.0%} "
              f"{s['pooled_signed_mean']:+12.4f}")

    with (out / "clock_probe.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "label", "dup_rate", "archived_median", "yaw50_median",
                    "yaw50_advanced_median", "yaw_native30_median"])
        for r in rows:
            w.writerow([r["event"], r["label"], f"{r['dup_rate']:.4f}",
                        r["archived_median"], f"{r['yaw50_median']:.6f}",
                        f"{r['yaw50_advanced_median']:.6f}" if r["yaw50_advanced_median"] is not None else "",
                        f"{r['yaw_native30_median']:.6f}"])

    dup = float(np.mean([r["dup_rate"] for r in rows]))

    def acc(key: str) -> int:
        n = 0
        for r in rows:
            v = r.get(key)
            if r["label"] == "FWD" or v is None:
                continue
            n += int(np.sign(v) == (-1 if r["label"] == "LEFT" else +1))
        return n

    cur, adv, nat = acc("yaw50_median"), acc("yaw50_advanced_median"), acc("yaw_native30_median")
    if max(adv, nat) > cur + 1:
        verdict = (
            f"CLOCK ARTEFACT: {dup:.0%} of samples were duplicate video frames. The current 50 Hz "
            f"seek clock gives {cur}/6, dropping duplicates gives {adv}/6 and native 30 fps "
            f"gives {nat}/6 — the measurement chain was the dominant defect."
        )
    else:
        verdict = (
            f"NOT THE CAUSE: {dup:.0%} of samples are duplicate video frames, so two fifths of every "
            f"event contributes exactly zero motion and the physical span of a fixed lag varies per "
            f"event. But removing them changes nothing ({cur}/6 current vs {adv}/6 advanced-only vs "
            f"{nat}/6 at native 30 fps). The clock dilutes the signal and is worth fixing for "
            "cleanliness, but it is not what produces the wrong signs."
        )
    print(f"\n=== VERDICT ===\n{verdict}")
    (out / "report.json").write_text(
        json.dumps({"rows": rows, "mean_dup_rate": dup, "verdict": verdict}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
