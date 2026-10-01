#!/usr/bin/env python3
"""
Permanent P0.1R regression fixture.

Run after ANY change to fly_vo/optic_flow.py or fly_vo/visual_encoder.py:

    PYTHONPATH=. python scripts/p01r_regression.py

Checks (all frozen — no retune):
  1. Synthetic P0.1 macro-F1 ≥ 0.90
  2. Calibration 51-70 LEFT PASS (if video present)
  3. Controlled FWD→LEFT→FWD→RIGHT→FWD (if blind_review_done + video present)
  4. Negative control 100-130 reported (PASS not required)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.p01r_gate import evaluate_event_benchmark
from fly_vo.segment_profiles import VideoRole, profile_for_video
from scripts.p01r_calibrate import run_visual_only
from scripts.p01r_finalize import SNR_THRESHOLD, evaluate_frozen

EMA_TAU = 0.5
DEFAULT_DOWNLOADS = Path("/Users/artem/Downloads")
OUT = ROOT / "output/p01r_regression"


def _run_synthetic() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/visual_observability_test.py")],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    ok = r.returncode == 0 and "PASS P0.1 GATE" in r.stdout
    return ok, r.stdout.splitlines()[-3] if r.stdout else r.stderr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"frozen_tau": EMA_TAU, "snr_threshold": SNR_THRESHOLD, "checks": {}}

    # 1. Synthetic
    syn_ok, syn_line = _run_synthetic()
    report["checks"]["synthetic_p01"] = {"pass": syn_ok, "tail": syn_line}
    print(f"[{'PASS' if syn_ok else 'FAIL'}] synthetic P0.1 — {syn_line}")

    # 2. Calibration 51-70
    cal_video = DEFAULT_DOWNLOADS / "VID00001_51-70.mp4"
    cal_ok = False
    if cal_video.exists():
        profile = profile_for_video(cal_video)
        rows = run_visual_only(cal_video, "center_band", OUT / "visual_51-70.csv", EMA_TAU)
        ev = evaluate_frozen(rows, profile)
        cal_ok = ev.get("pass") is True
        report["checks"]["calibration_51-70"] = ev
        print(f"[{'PASS' if cal_ok else 'FAIL'}] calibration 51-70 LEFT  SNR={ev.get('snr_robust')}")
    else:
        report["checks"]["calibration_51-70"] = {"pass": None, "reason": "video missing"}
        print("[SKIP] calibration 51-70 — video not found")

    # 3. VID00001 event benchmark
    evt_ok = None
    evt_json = ROOT / "data/p01r/VID00001_events.json"
    if evt_json.exists():
        evt_doc = json.loads(evt_json.read_text())
        evt_video = Path(evt_doc["video_path"])
        if not evt_video.is_absolute():
            evt_video = ROOT / evt_video
        if evt_doc.get("blind_review_done") and evt_video.exists():
            from scripts.p01r_event_gate import run_visual_range

            events = evt_doc["events"]
            t0 = max(0, min(e["start"] for e in events) - 5)
            t1 = max(e["end"] for e in events) + 5
            rows = run_visual_range(evt_video, t0, t1, ema_tau=EMA_TAU)
            ev = evaluate_event_benchmark(evt_doc, rows)
            evt_ok = ev.get("gate", {}).get("pass", False)
            report["checks"]["event_benchmark"] = ev["gate"]
            g = ev["gate"]
            print(
                f"[{'PASS' if evt_ok else 'FAIL'}] event benchmark  "
                f"L={g.get('left_sign_accuracy', 0):.0%} R={g.get('right_sign_accuracy', 0):.0%} "
                f"F1={g.get('macro_f1', 0):.2f}"
            )
        else:
            report["checks"]["event_benchmark"] = {"pass": None, "reason": "video or blind_review"}
            print("[PENDING] event benchmark")
    else:
        report["checks"]["event_benchmark"] = {"pass": None, "reason": "no events json"}
        print("[PENDING] event benchmark — no VID00001_events.json")

    # 4. Negative control (informational)
    neg_video = DEFAULT_DOWNLOADS / "VID00001_100-130.mp4"
    if neg_video.exists():
        profile = profile_for_video(neg_video)
        rows = run_visual_only(neg_video, "center_band", OUT / "visual_100-130.csv", EMA_TAU)
        ev = evaluate_frozen(rows, profile)
        report["checks"]["negative_control_100-130"] = {
            "pass_required": False,
            "snr": ev.get("snr_robust"),
            "sign_ok": ev.get("sign_ok"),
            "yaw_turn": ev.get("yaw_turn_median"),
        }
        print(f"[INFO] negative control 100-130  SNR={ev.get('snr_robust')} sign={ev.get('sign_ok')}")
    else:
        report["checks"]["negative_control_100-130"] = {"pass_required": False, "reason": "missing"}

    # Overall: synthetic + cal required; controlled required when configured
    required = [syn_ok]
    if cal_video.exists():
        required.append(cal_ok)
    if evt_ok is not None:
        required.append(evt_ok)

    overall = all(required)
    report["pass"] = overall
    (OUT / "regression_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'PASS' if overall else 'FAIL'} P0.1R REGRESSION")
    print(f"Report: {OUT}/regression_report.json")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
