#!/usr/bin/env python3
"""
P0.1R+ REAL-EVENT BENCHMARK — frozen frontend on blind-labeled VID00001 events.

Usage:
    PYTHONPATH=. python scripts/p01r_event_gate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.optic_flow import FROZEN_YAW_DETECTORS, DirectionalMotionFrontend, FlowConfig
from fly_vo.p01r_gate import SNR_THRESHOLD, YAW_NEAR_ZERO, evaluate_event_benchmark
from scripts.p01r_calibrate import VISUAL_COLS, _resize

EMA_TAU = 0.5
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
OUT_DIR = ROOT / "output/p01r_event_benchmark"


def run_visual_range(
    video: Path,
    t_start: float,
    t_end: float,
    crop_mode: str = "center_band",
    ema_tau: float = EMA_TAU,
) -> list[dict]:
    cfg = FlowConfig(crop_mode=crop_mode, ema_tau_s=ema_tau, brain_dt=0.02)
    front = DirectionalMotionFrontend(cfg)
    rows: list[dict] = []
    for t, frame in iter_video_at_brain_hz(str(video), t_start, t_end, cfg.brain_dt):
        img = _resize(frame).astype("float32") / 255.0
        d = front.process(img)
        row = {"time": round(t, 4), "crop_mode": crop_mode}
        row.update({k: round(v, 8) if isinstance(v, float) else v for k, v in d.as_dict().items()})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(EVENTS_JSON))
    parser.add_argument("-o", "--output", default=str(OUT_DIR))
    args = parser.parse_args()

    doc = json.loads(Path(args.events).read_text(encoding="utf-8"))
    if not doc.get("blind_review_done"):
        raise SystemExit("blind_review_done=false — complete frame review first")

    video = Path(doc["video_path"])
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    events = doc["events"]
    t_min = min(e["start"] for e in events) - 5.0
    t_max = max(e["end"] for e in events) + 5.0

    print(f"P0.1R+ event benchmark: {video.name}")
    print(f"Frozen: {FROZEN_YAW_DETECTORS}  tau={EMA_TAU}  crop=center_band")
    print(f"Processing t={t_min:.1f}–{t_max:.1f}s ({len(events)} events)...")

    rows = run_visual_range(video, max(0, t_min), t_max)

    result = evaluate_event_benchmark(doc, rows)
    report = {
        "video": str(video),
        "frozen_readout": [{"dx": dx, "dt": lag, "weight": 1 / 3} for dx, lag in FROZEN_YAW_DETECTORS],
        "ema_tau": EMA_TAU,
        "snr_threshold": SNR_THRESHOLD,
        "yaw_near_zero": YAW_NEAR_ZERO,
        **result,
    }

    # events.csv — one row per event with key metrics
    csv_rows = []
    for ev in result["events"]:
        s = ev.get("stats") or {}
        csv_rows.append({
            "id": ev["id"],
            "start": ev["start"],
            "end": ev["end"],
            "label": ev["label"],
            "yaw_frozen_median": s.get("yaw_frozen_median"),
            "yaw_frozen_mad": s.get("yaw_frozen_mad"),
            "sign_ok": ev.get("sign_ok"),
            "R_dx4_dt2_yaw": s.get("R_dx4_dt2_yaw_median"),
            "R_dx8_dt3_yaw": s.get("R_dx8_dt3_yaw_median"),
            "R_dx4_dt3_yaw": s.get("R_dx4_dt3_yaw_median"),
            "coherence_L": s.get("coherence_L_median"),
            "coherence_R": s.get("coherence_R_median"),
            "expansion": s.get("expansion_median"),
        })
    events_csv = out / "events.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        w.writeheader()
        w.writerows(csv_rows)

    # full time series for audited events
    with (out / "visual_range.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VISUAL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    g = result["gate"]
    print(f"\n=== Gate metrics ===")
    print(f"  LEFT:  n={g['n_left']}  sign_acc={g['left_sign_accuracy']:.0%}  SNR vs FWD={g['snr_left_vs_fwd']:.2f}")
    print(f"  RIGHT: n={g['n_right']}  sign_acc={g['right_sign_accuracy']:.0%}  SNR vs FWD={g['snr_right_vs_fwd']:.2f}")
    print(f"  FWD:   n={g['n_fwd']}  |median|={g['fwd_abs_median']:.4f}")
    print(f"  Ordering: L={g['median_ordering']['left']:+.3f} < F={g['median_ordering']['fwd']:+.3f} < R={g['median_ordering']['right']:+.3f}  ok={g['ordering_ok']}")
    print(f"  macro-F1={g['macro_f1']:.3f}")

    print(f"\n=== Per-event (scored) ===")
    for ev in result["events"]:
        if ev["label"] in ("MIXED", "REJECT"):
            continue
        s = ev.get("stats") or {}
        mark = "✓" if ev.get("sign_ok") else "✗"
        print(
            f"  {mark} {ev['id']:14s} {ev['label']:10s} "
            f"t={ev['start']:.1f}-{ev['end']:.1f}  yaw={s.get('yaw_frozen_median', 0):+.4f}"
        )

    print(f"\n{'PASS' if g['pass'] else 'FAIL'} P0.1R+ EVENT BENCHMARK")
    if g["pass"]:
        print("→ P0.1R+ CLOSED. Proceed to P0.2.")
    print(f"Wrote {out}/")
    raise SystemExit(0 if g["pass"] else 1)


if __name__ == "__main__":
    main()
