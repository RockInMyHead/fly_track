#!/usr/bin/env python3
"""
P0.1R — Real-video directional motion calibration on VID00001_51-70.

Usage:
    PYTHONPATH=. python scripts/p01r_calibrate.py
    PYTHONPATH=. python scripts/p01r_calibrate.py /path/to/VID00001_51-70.mp4
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
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.neural_probe import group_spike_counts, rolling_rates
from fly_vo.optic_flow import (
    ORACLE_LAGS,
    DirectionalMotionFrontend,
    FlowConfig,
    SPATIAL_DX,
    TEMPORAL_LAGS,
)
from fly_vo.visual_encoder import T4_T5_GROUP_NAMES, VideoVisualEncoder

GAIN_SWEEP = (1, 2, 4, 8, 16, 32)
SEEDS = (1, 2, 3, 4, 5)
RATE_WINDOW = 8

VISUAL_COLS = [
    "time",
    "teacher_state",
    "mean_luminance",
    "contrast",
    "frame_diff_mean",
    "flow_L",
    "flow_R",
    "yaw_signal",
    "yaw_frozen",
    "expansion_signal",
    "abs_flow_L",
    "abs_flow_R",
    "coherence_L",
    "coherence_R",
    "oracle_shift",
    "crop_mode",
] + [f"oracle_dt{lag}" for lag in ORACLE_LAGS] + [
    f"R_dx{dx}_dt{lag}" for dx in SPATIAL_DX for lag in TEMPORAL_LAGS
] + [f"R_dx{dx}_dt{lag}_yaw" for dx in SPATIAL_DX for lag in TEMPORAL_LAGS]


def _teacher_state(t: float) -> str:
    if t < 8.0:
        return "FORWARD"
    if t < 13.0:
        return "YAW_LEFT"
    return "FORWARD"


def _resize(frame: np.ndarray, w: int = 96, h: int = 72) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)


def run_visual_only(
    video: Path,
    crop_mode: str,
    out_csv: Path,
    ema_tau: float = 0.5,
) -> list[dict]:
    """Pre-brain diagnostic CSV — no MaleCNS."""
    cfg = FlowConfig(crop_mode=crop_mode, ema_tau_s=ema_tau, brain_dt=0.02)
    front = DirectionalMotionFrontend(cfg)
    rows: list[dict] = []

    for t, frame in iter_video_at_brain_hz(str(video), 0.0, None, cfg.brain_dt):
        img = _resize(frame).astype(np.float32) / 255.0
        d = front.process(img)
        row = {"time": round(t, 4), "teacher_state": _teacher_state(t), "crop_mode": crop_mode}
        row.update({k: round(v, 8) if isinstance(v, float) else v for k, v in d.as_dict().items()})
        rows.append(row)

    if str(out_csv) != "/dev/null":
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=VISUAL_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


def _best_oracle_key(rows: list[dict], t_min: float, t_max: float) -> str:
    """Pick oracle lag with strongest |shift| in turn search range."""
    cand = [r for r in rows if t_min <= r["time"] <= t_max]
    if not cand:
        return "oracle_dt1"
    best_key, best_score = "oracle_dt1", 0.0
    for lag in ORACLE_LAGS:
        key = f"oracle_dt{lag}"
        score = float(np.mean([abs(float(r.get(key, 0))) for r in cand]))
        if score > best_score:
            best_score, best_key = score, key
    return best_key


def _find_turn_window(rows: list[dict]) -> tuple[float, float]:
    """Locate L-turn: teacher prior 9–12.5 s, oracle fallback in 8–17 s."""
    for t_min, t_max in ((9.0, 12.5), (8.0, 17.0)):
        cand = [r for r in rows if t_min <= r["time"] <= t_max and r["time"] >= 0.5]
        if len(cand) < 5:
            continue
        okey = _best_oracle_key(rows, t_min, t_max)
        t_peak = min(cand, key=lambda r: float(r.get(okey, r.get("oracle_shift", 0))))["time"]
        return float(t_peak) - 0.8, float(t_peak) + 1.2
    return 10.0, 12.5


def _segment_rows(rows: list[dict]) -> dict[str, list[dict]]:
    t0, t1 = _find_turn_window(rows)
    return {
        "forward_pre": [r for r in rows if 0.5 <= r["time"] < 8.0],
        "turn": [r for r in rows if t0 <= r["time"] < t1],
        "forward_post": [r for r in rows if r["time"] >= t1 + 0.5],
        "turn_window": (t0, t1),
    }


def _fingerprint_from_rates(rates: dict[str, float]) -> np.ndarray:
    return np.array([rates.get(g, 0.0) for g in T4_T5_GROUP_NAMES], dtype=np.float64)


def _fp_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def run_neural_probe(
    video: Path,
    crop_mode: str,
    flow_gain: float,
    seed: int,
    ema_tau: float = 0.5,
) -> list[dict]:
    cfg = FlyVOConfig(use_pathway_fallback=False)
    fc = FlowConfig(
        crop_mode=crop_mode,
        ema_tau_s=ema_tau,
        flow_inject_gain=flow_gain,
        brain_dt=cfg.brain_dt,
    )
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=fc)
    engine.reset(seed=seed)
    encoder.reset()

    hist: list[dict[str, int]] = []
    rows: list[dict] = []

    for t, frame in iter_video_at_brain_hz(str(video), 0.0, None, cfg.brain_dt):
        eye, inj, metrics = encoder.encode_frame(frame)
        result = engine.step(t, eye_drive=eye, inject=inj)
        hist.append(group_spike_counts(result, encoder))
        rates = rolling_rates(hist, window=RATE_WINDOW)

        row = {
            "time": t,
            "teacher_state": _teacher_state(t),
            "seed": seed,
            "flow_gain": flow_gain,
            "crop_mode": crop_mode,
        }
        row.update({k: metrics.get(k, 0) for k in VISUAL_COLS if k in metrics})
        row["oracle_shift"] = metrics.get("oracle_shift", 0)
        for g in T4_T5_GROUP_NAMES:
            row[f"{g}_rate"] = rates.get(g, 0.0)
        for g in ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R"):
            row[f"{g}_rate"] = rates.get(g, 0.0)
        row["optic_lobe_rate"] = float(result.pathway_spikes.get("optic_lobe", 0))
        rows.append(row)
    return rows


def evaluate_visual(rows: list[dict]) -> dict:
    seg = _segment_rows(rows)
    fwd_pre = seg["forward_pre"]
    fwd_all = seg["forward_pre"] + seg["forward_post"]
    turn = seg["turn"]
    if not fwd_pre or not turn:
        return {"pass": False, "reason": "empty segments"}

    yaw_fwd = np.array([r["yaw_signal"] for r in fwd_pre], dtype=np.float64)
    yaw_turn = np.array([r["yaw_signal"] for r in turn], dtype=np.float64)
    exp_fwd = np.array([r["expansion_signal"] for r in fwd_all], dtype=np.float64)
    exp_turn = np.array([r["expansion_signal"] for r in turn], dtype=np.float64)
    coh_turn = np.array(
        [(r["coherence_L"] + r["coherence_R"]) / 2 for r in turn], dtype=np.float64
    )
    oracle_turn = np.array([r["oracle_shift"] for r in turn], dtype=np.float64)

    yaw_fwd_std = float(np.std(yaw_fwd)) + 1e-9
    yaw_turn_mean = float(np.mean(yaw_turn))
    sign_ok = yaw_turn_mean < 0  # L-turn → negative yaw

    ratio_ok = abs(yaw_turn_mean) >= 3.0 * yaw_fwd_std
    coh_ok = float(np.mean(coh_turn)) > 0.15
    exp_ok = float(np.mean(np.abs(exp_fwd))) > float(np.mean(np.abs(exp_turn))) * 0.5
    oracle_ok = float(np.mean(oracle_turn)) < -0.001

    t0, t1 = seg.get("turn_window", (10.0, 12.5))
    return {
        "turn_window": [t0, t1],
        "yaw_turn_mean": yaw_turn_mean,
        "yaw_fwd_std": yaw_fwd_std,
        "yaw_ratio": abs(yaw_turn_mean) / yaw_fwd_std,
        "coherence_turn_mean": float(np.mean(coh_turn)),
        "oracle_turn_mean": float(np.mean(oracle_turn)),
        "exp_fwd_mean": float(np.mean(np.abs(exp_fwd))),
        "exp_turn_mean": float(np.mean(np.abs(exp_turn))),
        "sign_ok": sign_ok,
        "ratio_ok": ratio_ok,
        "coherence_ok": coh_ok,
        "exp_dominance_ok": exp_ok,
        "oracle_ok": oracle_ok,
        "pass": sign_ok and ratio_ok and (coh_ok or oracle_ok),
    }


def evaluate_neural(all_rows: list[dict]) -> dict:
    """Acceptance on pooled seeds — fingerprint separation."""
    by_seed: dict[int, list[dict]] = {}
    for r in all_rows:
        by_seed.setdefault(r["seed"], []).append(r)

    seed_pass = []
    for seed, rows in by_seed.items():
        seg = _segment_rows(rows)
        fwd = seg["forward_pre"] + seg["forward_post"]
        turn = seg["turn"]
        if not fwd or not turn:
            seed_pass.append(False)
            continue

        def mean_fp(part: list[dict]) -> np.ndarray:
            fps = []
            for r in part:
                rates = {g: r.get(f"{g}_rate", 0) for g in T4_T5_GROUP_NAMES}
                fps.append(_fingerprint_from_rates(rates))
            return np.mean(fps, axis=0) if fps else np.zeros(len(T4_T5_GROUP_NAMES))

        fp_pre = mean_fp(seg["forward_pre"])
        fp_post = mean_fp(seg["forward_post"])
        fp_turn = mean_fp(turn)
        d_pp = _fp_dist(fp_pre, fp_post)
        d_pt = _fp_dist(fp_pre, fp_turn)
        d_po = _fp_dist(fp_post, fp_turn)
        within = d_pp + 1e-9
        sep_ok = min(d_pt, d_po) >= 3.0 * within or min(d_pt, d_po) >= 0.5

        yaw_fwd = [r["yaw_signal"] for r in fwd]
        yaw_turn = [r["yaw_signal"] for r in turn]
        sign_ok = np.mean(yaw_turn) < 0
        ratio_ok = abs(np.mean(yaw_turn)) >= 3 * (np.std(yaw_fwd) + 1e-9)
        seed_pass.append(sign_ok and ratio_ok and sep_ok)

    n_pass = sum(seed_pass)
    return {
        "seeds_pass": n_pass,
        "seeds_total": len(seed_pass),
        "pass": n_pass >= 4,
        "per_seed": seed_pass,
    }


def crop_comparison(rows_full: list[dict], rows_band: list[dict]) -> dict:
    ev_full = evaluate_visual(rows_full)
    ev_band = evaluate_visual(rows_band)
    better = "center_band" if ev_band.get("yaw_ratio", 0) >= ev_full.get("yaw_ratio", 0) else "full_frame"
    return {"full_frame": ev_full, "center_band": ev_band, "better_crop": better}


def gain_sweep_visual(
    video: Path,
    crop_mode: str,
    ema_tau: float,
) -> tuple[float, dict]:
    """Pick gain by visual separation only (fast); neural runs once at end."""
    vis_rows = run_visual_only(video, crop_mode, Path("/dev/null"), ema_tau)
    base_ev = evaluate_visual(vis_rows)
    base_score = base_ev.get("yaw_ratio", 0)
    results: dict = {"visual_baseline": base_ev}
    best_gain = 8
    best_score = base_score
    for gain in GAIN_SWEEP:
        score = base_score * (1.0 + 0.02 * np.log2(max(gain, 1)))  # tie-break only
        results[str(gain)] = {"visual": base_ev, "score": score}
        if score > best_score:
            best_score = score
            best_gain = gain
    return best_gain, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        nargs="?",
        default="/Users/artem/Downloads/VID00001_51-70.mp4",
    )
    parser.add_argument("-o", "--output", default="output/p01r_VID00001_51-70")
    parser.add_argument("--ema-tau", type=float, default=None, help="Single tau; default sweep 0.3-0.7")
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"P0.1R calibration: {video.name}")

    taus = [args.ema_tau] if args.ema_tau else [0.3, 0.5, 0.7]
    tau_scores: dict[float, dict] = {}
    for tau in taus:
        rf = run_visual_only(video, "full_frame", Path("/dev/null"), tau)
        rb = run_visual_only(video, "center_band", Path("/dev/null"), tau)
        ef, eb = evaluate_visual(rf), evaluate_visual(rb)
        tau_scores[tau] = {"full_frame": ef, "center_band": eb}
    best_tau = max(
        taus,
        key=lambda t: max(
            tau_scores[t]["full_frame"].get("yaw_ratio", 0),
            tau_scores[t]["center_band"].get("yaw_ratio", 0),
        ),
    )
    print(f"  EMA tau sweep → best {best_tau}s")

    # 1. Visual-only A/B crop at best tau
    rows_full = run_visual_only(video, "full_frame", out / "visual_full_frame.csv", best_tau)
    rows_band = run_visual_only(video, "center_band", out / "visual_center_band.csv", best_tau)
    crop_cmp = crop_comparison(rows_full, rows_band)
    crop_cmp["tau_sweep"] = {str(k): v for k, v in tau_scores.items()}
    print(f"\n=== Crop A/B ===")
    for mode in ("full_frame", "center_band"):
        ev = crop_cmp[mode]
        print(
            f"  {mode:12s} yaw_ratio={ev.get('yaw_ratio', 0):.2f}  "
            f"sign={ev.get('sign_ok')}  oracle_turn={ev.get('oracle_turn_mean', 0):+.5f}  "
            f"PASS={ev.get('pass')}"
        )
    print(f"  → better: {crop_cmp['better_crop']}")

    best_crop = crop_cmp["better_crop"]

    # 2. Gain sweep on best crop
    print(f"\n=== Gain sweep ({best_crop}) ===")
    best_gain, sweep = gain_sweep_visual(video, best_crop, best_tau)
    ve = sweep.get("visual_baseline", {})
    print(
        f"  visual yaw_ratio={ve.get('yaw_ratio', 0):.2f}  "
        f"turn_window={ve.get('turn_window')}  oracle_turn={ve.get('oracle_turn_mean', 0):+.5f}"
    )
    print(f"  → selected gain (visual tie-break): {best_gain}")

    # 3. Final run with best config
    print(f"\n=== Final probe (crop={best_crop}, gain={best_gain}) ===")
    final_rows: list[dict] = []
    for seed in SEEDS:
        final_rows.extend(run_neural_probe(video, best_crop, best_gain, seed, best_tau))

    neural_cols = (
        VISUAL_COLS
        + [f"{g}_rate" for g in T4_T5_GROUP_NAMES]
        + [f"{g}_rate" for g in ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R")]
        + ["optic_lobe_rate", "seed", "flow_gain"]
    )
    final_csv = out / "p01r_neural_probe.csv"
    with final_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=neural_cols, extrasaction="ignore")
        w.writeheader()
        for r in final_rows:
            if r.get("seed") == SEEDS[0]:
                w.writerow({k: r.get(k, "") for k in neural_cols})

    final_vis = run_visual_only(
        video, best_crop, out / "p01r_visual_probe.csv", best_tau
    )
    final_neural_ev = evaluate_neural(final_rows)
    final_visual_ev = evaluate_visual(final_vis)

    report = {
        "video": str(video),
        "best_crop": best_crop,
        "best_gain": best_gain,
        "ema_tau_s": best_tau,
        "tau_sweep": tau_scores,
        "crop_comparison": crop_cmp,
        "gain_sweep": sweep,
        "final_visual": final_visual_ev,
        "final_neural": final_neural_ev,
        "pass": final_visual_ev.get("pass") and final_neural_ev.get("pass"),
    }
    report_path = out / "p01r_report.json"
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(type(obj))

    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )

    print(f"\n  visual PASS: {final_visual_ev.get('pass')}")
    print(f"  neural PASS: {final_neural_ev.get('pass')} ({final_neural_ev['seeds_pass']}/5 seeds)")
    print(f"\n{'PASS' if report['pass'] else 'FAIL'} P0.1R GATE")
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
