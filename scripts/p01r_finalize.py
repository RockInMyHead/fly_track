#!/usr/bin/env python3
"""
P0.1R+ FINALIZATION — frozen Reichardt readout + multi-clip gate.

Gate structure:
  51-70          calibration LEFT     — must PASS
  holdout A/B    clean FWD→YAW→FWD     — must PASS (when profiles exist)
  100-130        complex-path control  — reported, PASS not required
  synthetic P0.1 — macro-F1 ≥ 0.90     — must PASS

Usage:
    PYTHONPATH=. python scripts/p01r_finalize.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.optic_flow import FROZEN_YAW_DETECTORS, SPATIAL_DX, TEMPORAL_LAGS
from fly_vo.p01r_gate import SNR_THRESHOLD, YAW_NEAR_ZERO, robust_snr
from fly_vo.segment_profiles import (
    BLIND_REVIEW_REJECTED,
    VideoRole,
    profile_for_video,
    profiles_by_role,
)
from scripts.p01r_calibrate import run_visual_only
DEFAULT_DOWNLOADS = Path("/Users/artem/Downloads")


def _segment_rows(rows: list[dict], profile) -> dict[str, list[dict]]:
    p0, p1 = profile.pre
    t0, t1 = profile.turn
    q0, q1 = profile.post
    return {
        "forward_pre": [r for r in rows if p0 <= r["time"] < p1],
        "turn": [r for r in rows if t0 <= r["time"] < t1],
        "forward_post": [r for r in rows if q0 <= r["time"] < q1],
        "segments": {"pre": profile.pre, "turn": profile.turn, "post": profile.post},
    }


def detector_snr_table(rows: list[dict], profile) -> dict[str, float]:
    seg = _segment_rows(rows, profile)
    pre = seg["forward_pre"]
    turn = seg["turn"]
    if not pre or not turn:
        return {}
    out: dict[str, float] = {}
    keys = ["yaw_signal", "yaw_frozen"] + [
        f"R_dx{dx}_dt{lag}_yaw" for dx, lag in FROZEN_YAW_DETECTORS
    ]
    for dx in SPATIAL_DX:
        for lag in TEMPORAL_LAGS:
            keys.append(f"R_dx{dx}_dt{lag}_yaw")
    for key in keys:
        pre_v = np.array([float(r.get(key) or 0) for r in pre])
        turn_v = np.array([float(r.get(key) or 0) for r in turn])
        out[key] = round(robust_snr(pre_v, turn_v), 3)
    return out


def evaluate_frozen(rows: list[dict], profile) -> dict:
    seg = _segment_rows(rows, profile)
    pre = seg["forward_pre"]
    turn = seg["turn"]
    post = seg["forward_post"]
    if not pre or not turn:
        return {"pass": False, "reason": "empty segment", "role": profile.role.value}

    yaw_pre = np.array([float(r["yaw_signal"]) for r in pre])
    yaw_turn = np.array([float(r["yaw_signal"]) for r in turn])
    yaw_post = np.array([float(r["yaw_signal"]) for r in post]) if post else np.array([])

    snr = robust_snr(yaw_pre, yaw_turn)
    med_pre = float(np.median(yaw_pre))
    med_turn = float(np.median(yaw_turn))
    med_post = float(np.median(yaw_post)) if len(yaw_post) else None

    sign_ok = med_turn * profile.turn_sign > 0
    pre_near_zero = abs(med_pre) < YAW_NEAR_ZERO
    post_ok = (
        abs(med_post) < abs(med_turn) * 0.35 and abs(med_post) < YAW_NEAR_ZERO
        if med_post is not None
        else True
    )
    turn_separated = snr >= SNR_THRESHOLD

    gate_pass = sign_ok and turn_separated and pre_near_zero and post_ok
    if profile.role == VideoRole.NEGATIVE_CONTROL:
        gate_pass = None  # informational only

    return {
        "role": profile.role.value,
        "expected_turn": profile.expected_turn,
        "segments": seg["segments"],
        "snr_robust": round(snr, 3),
        "yaw_pre_median": med_pre,
        "yaw_turn_median": med_turn,
        "yaw_post_median": med_post,
        "sign_ok": sign_ok,
        "pre_near_zero": pre_near_zero,
        "post_decay_ok": post_ok,
        "turn_separated": turn_separated,
        "pass": gate_pass,
        "notes": profile.notes,
    }


def save_turn_frames(video: Path, out_dir: Path, times: list[float]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for t in times:
        for _vt, frame in iter_video_at_brain_hz(str(video), t, t + 0.02, 0.02):
            path = out_dir / f"frame_t{t:05.2f}s.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(str(path))
            break
    return saved


def eval_clip(video: Path, out: Path, ema_tau: float) -> dict | None:
    profile = profile_for_video(video)
    if profile is None:
        return None
    rows = run_visual_only(video, "center_band", out / f"visual_{video.stem}.csv", ema_tau)
    return {
        "video": video.name,
        "profile_key": video.stem,
        "detector_snr": detector_snr_table(rows, profile),
        "eval": evaluate_frozen(rows, profile),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="output/p01r_finalize")
    parser.add_argument("--ema-tau", type=float, default=0.5)
    parser.add_argument("--downloads", default=str(DEFAULT_DOWNLOADS))
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dl = Path(args.downloads)

    report: dict = {
        "gate_version": "P0.1R+ multi-clip",
        "frozen_readout": [{"dx": dx, "dt": lag, "weight": 1 / 3} for dx, lag in FROZEN_YAW_DETECTORS],
        "crop_mode": "center_band",
        "ema_tau": args.ema_tau,
        "snr_threshold": SNR_THRESHOLD,
        "acceptance": {
            "sign": "yaw_turn * turn_sign > 0",
            "snr": f"robust SNR >= {SNR_THRESHOLD}",
            "pre": "|yaw_pre| < 0.15",
            "post": "|yaw_post| < 0.15 and |yaw_post| < 0.35 * |yaw_turn|",
            "no_retune": "weights/gain/tau/thresholds frozen",
        },
        "blind_review_rejected": BLIND_REVIEW_REJECTED,
    }

    # --- Evaluate all profiled clips ---
    clip_results: dict[str, dict] = {}
    for key, profile in profiles_by_role(VideoRole.CALIBRATION).items():
        video = dl / f"{key}.mp4"
        if not video.exists():
            raise SystemExit(f"Missing calibration video: {video}")
        print(f"\n=== {profile.role.value.upper()}: {key} ===")
        res = eval_clip(video, out, args.ema_tau)
        if res:
            clip_results[key] = res
            ev = res["eval"]
            print(f"  segments: {ev['segments']}  expected={ev['expected_turn']}")
            print(
                f"  SNR={ev['snr_robust']:.2f}  sign={ev['sign_ok']}  "
                f"pre≈0={ev['pre_near_zero']}  post={ev['post_decay_ok']}  PASS={ev['pass']}"
            )
            print(
                f"  yaw: pre={ev['yaw_pre_median']:+.4f}  "
                f"turn={ev['yaw_turn_median']:+.4f}  post={ev['yaw_post_median']}"
            )

    for key, profile in profiles_by_role(VideoRole.HOLDOUT).items():
        video = dl / f"{key}.mp4"
        if video.exists():
            print(f"\n=== HOLDOUT: {key} ===")
            res = eval_clip(video, out, args.ema_tau)
            if res:
                clip_results[key] = res
                ev = res["eval"]
                print(f"  PASS={ev['pass']}")

    for key, profile in profiles_by_role(VideoRole.NEGATIVE_CONTROL).items():
        video = dl / f"{key}.mp4"
        if video.exists():
            print(f"\n=== NEGATIVE CONTROL: {key} ===")
            res = eval_clip(video, out, args.ema_tau)
            if res:
                clip_results[key] = res
                ev = res["eval"]
                print(f"  (informational — gate PASS not required)")
                print(
                    f"  SNR={ev['snr_robust']:.2f}  sign={ev['sign_ok']}  "
                    f"pre≈0={ev['pre_near_zero']}  post={ev['post_decay_ok']}"
                )
                print(
                    f"  yaw: pre={ev['yaw_pre_median']:+.4f}  "
                    f"turn={ev['yaw_turn_median']:+.4f}  post={ev['yaw_post_median']}"
                )

    report["clips"] = clip_results

    # --- Synthetic P0.1 ---
    print("\n=== Synthetic P0.1 regression ===")
    syn = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "visual_observability_test.py")],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    report["synthetic_exit"] = syn.returncode
    report["synthetic_tail"] = syn.stdout.splitlines()[-8:]
    print("\n".join(report["synthetic_tail"]))

    # --- Gate verdict ---
    cal = clip_results.get("VID00001_51-70", {}).get("eval", {})
    holdouts = [
        v["eval"]
        for k, v in clip_results.items()
        if v["eval"].get("role") == VideoRole.HOLDOUT.value
    ]
    neg = clip_results.get("VID00001_100-130", {}).get("eval", {})
    syn_pass = syn.returncode == 0 and "PASS P0.1 GATE" in syn.stdout

    cal_pass = cal.get("pass") is True
    holdout_pass = all(h.get("pass") for h in holdouts) if holdouts else False
    holdout_pending = not holdouts

    report["gate"] = {
        "calibration_51-70": cal_pass,
        "holdouts": {h.get("expected_turn", "?"): h.get("pass") for h in holdouts},
        "holdout_pending": holdout_pending,
        "negative_control_100-130": {
            "evaluated": neg != {},
            "pass": neg.get("pass"),
            "required": False,
        },
        "synthetic": syn_pass,
        "pass": cal_pass and syn_pass and holdout_pass and not holdout_pending,
    }

    (out / "p01r_finalize_report.json").write_text(
        json.dumps(report, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o),
        encoding="utf-8",
    )

    print(f"\n{'PASS' if report['gate']['pass'] else 'FAIL'} P0.1R+ GATE")
    print(f"  51-70 calibration: {'PASS' if cal_pass else 'FAIL'}")
    print(f"  synthetic:         {'PASS' if syn_pass else 'FAIL'}")
    if holdout_pending:
        print(f"  holdouts:          use scripts/p01r_event_gate.py (VID00001 event benchmark)")
    else:
        for h in holdouts:
            print(f"  holdout {h.get('expected_turn')}: {'PASS' if h.get('pass') else 'FAIL'}")
    if neg:
        print(f"  100-130 neg ctrl:  SNR={neg.get('snr_robust')} sign={neg.get('sign_ok')} (info only)")
    print(f"  rejected:          130-160, 160-190 (blind review)")
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
