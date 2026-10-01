#!/usr/bin/env python3
"""
P0.2A — Descending neuron propagation tracing (mechanistic probe).

Reference events (blind-labeled, frozen frontend):
  LEFT  — left_65   (63.5–66.5 s)
  FWD   — fwd_140   (137.5–142.5 s)
  RIGHT — right_216 (214.0–217.0 s)

Usage:
    PYTHONPATH=. python scripts/p02a_dn_tracing.py
    PYTHONPATH=. python scripts/p02a_dn_tracing.py --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.neural_probe import group_spike_counts
from fly_vo.optic_flow import FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

VIDEO = ROOT / "data/p01r/VID00001.AVI"
OUT_DIR = ROOT / "output/p02a_dn_tracing"
BASELINE_S = 2.0
SEEDS_DEFAULT = (1, 2, 3, 4, 5)

REFERENCE_EVENTS = {
    "LEFT": {"id": "left_65", "start": 63.5, "end": 66.5},
    "FWD": {"id": "fwd_140", "start": 137.5, "end": 142.5},
    "RIGHT": {"id": "right_216", "start": 214.0, "end": 217.0},
}


@dataclass
class RunTrace:
    condition: str
    seed: int
    times: list[float]
    dn_counts: np.ndarray  # (n_steps, n_dn)
    t4t5: list[dict[str, int]]
    yaw: list[float]


def _build_dn_table(engine: MaleCNSEngine) -> tuple[np.ndarray, list[dict]]:
    brain = engine.brain
    idx = np.asarray(brain.cells(["descending_neuron"]), dtype=np.int64)
    pos = {int(i): p for p, i in enumerate(idx)}
    rows = []
    for p, i in enumerate(idx):
        rows.append(
            {
                "index": int(p),
                "neuron_id": int(i),
                "cell_type": str(brain.cell_type[i]),
                "side": str(brain.side[i]),
            }
        )
    return idx, rows


def run_segment(
    engine: MaleCNSEngine,
    encoder: VideoVisualEncoder,
    video: Path,
    t0: float,
    t1: float,
    dn_idx: np.ndarray,
    dn_pos: dict[int, int],
    seed: int,
    condition: str,
) -> RunTrace:
    engine.reset(seed=seed)
    encoder.reset()

    n_dn = len(dn_idx)
    times: list[float] = []
    dn_rows: list[np.ndarray] = []
    t4t5_hist: list[dict[str, int]] = []
    yaw_hist: list[float] = []

    for t, frame in iter_video_at_brain_hz(str(video), t0, t1, engine.config.brain_dt):
        eye_drive, inject, metrics = encoder.encode_frame(frame)
        result = engine.step(t, eye_drive=eye_drive, inject=inject)
        counts = np.zeros(n_dn, dtype=np.float32)
        for fid in result.fired:
            p = dn_pos.get(int(fid))
            if p is not None:
                counts[p] += 1.0
        times.append(t)
        dn_rows.append(counts)
        t4t5_hist.append(group_spike_counts(result, encoder))
        yaw_hist.append(float(metrics.get("yaw_frozen", metrics.get("yaw_signal", 0))))

    return RunTrace(
        condition=condition,
        seed=seed,
        times=times,
        dn_counts=np.stack(dn_rows, axis=0) if dn_rows else np.zeros((0, n_dn)),
        t4t5=t4t5_hist,
        yaw=yaw_hist,
    )


def _window_mask(times: list[float], start: float, end: float) -> np.ndarray:
    t = np.asarray(times)
    return (t >= start) & (t < end)


def _rate_hz(counts: np.ndarray, dt: float) -> float:
    if counts.size == 0:
        return 0.0
    return float(np.mean(counts) / dt)


def _robust_sigma(vals: np.ndarray) -> float:
    med = float(np.median(vals))
    return 1.4826 * float(np.median(np.abs(vals - med))) + 1e-9


def _latency_s(times: list[float], rates: np.ndarray, baseline: float, thr: float) -> float | None:
    if len(times) == 0:
        return None
    win = 5
    for i in range(len(times)):
        j0 = max(0, i - win + 1)
        roll = float(np.mean(rates[j0 : i + 1]))
        if roll > baseline + thr:
            return float(times[i])
    return None


def analyze_traces(
    traces: dict[str, list[RunTrace]],
    events: dict[str, dict],
    dn_meta: list[dict],
    dt: float,
) -> tuple[list[dict], dict]:
    n_dn = len(dn_meta)
    conds = ["LEFT", "FWD", "RIGHT"]

    # Per condition × seed × neuron event/baseline rates
    rates: dict[str, dict[int, dict[str, np.ndarray]]] = {
        c: {} for c in conds
    }  # rates[c][seed] = {"event": (n_dn,), "baseline": (n_dn,)}

    yaw_medians: dict[str, list[float]] = {c: [] for c in conds}
    t4t5_summary: dict[str, dict] = {}

    for cond in conds:
        ev = events[cond]
        b0, b1 = ev["start"] - BASELINE_S, ev["start"]
        e0, e1 = ev["start"], ev["end"]
        t4_acc: dict[str, list[float]] = {}

        for tr in traces[cond]:
            t = tr.times
            m_base = _window_mask(t, b0, b1)
            m_event = _window_mask(t, e0, e1)
            base_counts = tr.dn_counts[m_base].mean(axis=0) if m_base.any() else np.zeros(n_dn)
            event_counts = tr.dn_counts[m_event].mean(axis=0) if m_event.any() else np.zeros(n_dn)
            rates[cond][tr.seed] = {
                "event": event_counts / dt,
                "baseline": base_counts / dt,
            }
            if m_event.any():
                yaw_medians[cond].append(float(np.median(np.asarray(tr.yaw)[m_event])))
            idxs = np.where(m_event)[0]
            if len(idxs) and tr.t4t5:
                for key in tr.t4t5[0].keys():
                    val = float(np.mean([tr.t4t5[i].get(key, 0) for i in idxs]))
                    t4_acc.setdefault(key, []).append(val)

        t4t5_summary[cond] = {k: float(np.mean(v)) for k, v in t4_acc.items()}

    rows: list[dict] = []
    candidates_signed: list[dict] = []
    candidates_left: list[dict] = []
    candidates_right: list[dict] = []

    for p in range(n_dn):
        meta = dn_meta[p]
        r_left = np.array([rates["LEFT"][s]["event"][p] for s in rates["LEFT"]])
        r_fwd = np.array([rates["FWD"][s]["event"][p] for s in rates["FWD"]])
        r_right = np.array([rates["RIGHT"][s]["event"][p] for s in rates["RIGHT"]])
        b_left = np.array([rates["LEFT"][s]["baseline"][p] for s in rates["LEFT"]])
        b_fwd = np.array([rates["FWD"][s]["baseline"][p] for s in rates["FWD"]])
        b_right = np.array([rates["RIGHT"][s]["baseline"][p] for s in rates["RIGHT"]])

        m_left, m_fwd, m_right = map(float, (r_left.mean(), r_fwd.mean(), r_right.mean()))
        s_left, s_fwd, s_right = map(float, (r_left.std(), r_fwd.std(), r_right.std()))

        d_lf = r_left - r_fwd
        d_rf = r_right - r_fwd
        d_lr = r_left - r_right

        # L-R differential per seed (same neuron, opposite sides compared via paired L/R types later)
        lr_diff_left_event = float(r_left.mean() - r_right.mean())

        # Sign consistency across seeds
        sc_lf = float(np.mean(d_lf > 0))  # fraction seeds where LEFT > FWD
        sc_rf = float(np.mean(d_rf > 0))
        sc_lr = float(np.mean(d_lr > 0))

        # Robust separation vs FWD (across-seed event rates)
        sep_lf = abs(m_left - m_fwd) / _robust_sigma(r_fwd) if r_fwd.size else 0.0
        sep_rf = abs(m_right - m_fwd) / _robust_sigma(r_fwd) if r_fwd.size else 0.0

        row = {
            **meta,
            "rate_LEFT": round(m_left, 4),
            "rate_FWD": round(m_fwd, 4),
            "rate_RIGHT": round(m_right, 4),
            "std_LEFT": round(s_left, 4),
            "std_FWD": round(s_fwd, 4),
            "std_RIGHT": round(s_right, 4),
            "baseline_LEFT": round(float(b_left.mean()), 4),
            "baseline_FWD": round(float(b_fwd.mean()), 4),
            "baseline_RIGHT": round(float(b_right.mean()), 4),
            "delta_LEFT_minus_FWD": round(m_left - m_fwd, 4),
            "delta_RIGHT_minus_FWD": round(m_right - m_fwd, 4),
            "delta_LEFT_minus_RIGHT": round(m_left - m_right, 4),
            "sep_LEFT_vs_FWD": round(sep_lf, 3),
            "sep_RIGHT_vs_FWD": round(sep_rf, 3),
            "sign_consistency_LEFT_gt_FWD": round(sc_lf, 3),
            "sign_consistency_RIGHT_gt_FWD": round(sc_rf, 3),
            "sign_consistency_LEFT_gt_RIGHT": round(sc_lr, 3),
        }
        rows.append(row)

        good = sep_lf >= 2.0 or sep_rf >= 2.0
        if not good:
            continue

        if m_left > m_fwd and m_left > m_right and sc_lf >= 0.8:
            candidates_left.append({**row, "class": "LEFT_selective"})
        if m_right > m_fwd and m_right > m_left and sc_rf >= 0.8:
            candidates_right.append({**row, "class": "RIGHT_selective"})

    # Signed: L/R pairs of same cell_type with opposite differential polarity
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["cell_type"], []).append(row)

    for ctype, members in by_type.items():
        lefts = [m for m in members if m["side"] == "L"]
        rights = [m for m in members if m["side"] == "R"]
        if not lefts or not rights:
            continue
        l = lefts[0]
        r = rights[0]
        diff_left_event = l["rate_LEFT"] - r["rate_LEFT"]
        diff_right_event = l["rate_RIGHT"] - r["rate_RIGHT"]
        if diff_left_event * diff_right_event < 0 and abs(diff_left_event) > 0.5 and abs(diff_right_event) > 0.5:
            candidates_signed.append(
                {
                    "cell_type": ctype,
                    "L_index": l["index"],
                    "R_index": r["index"],
                    "L_rate_LEFT": l["rate_LEFT"],
                    "R_rate_LEFT": r["rate_LEFT"],
                    "L_rate_RIGHT": l["rate_RIGHT"],
                    "R_rate_RIGHT": r["rate_RIGHT"],
                    "diff_at_LEFT_event": round(diff_left_event, 4),
                    "diff_at_RIGHT_event": round(diff_right_event, 4),
                    "polarity_flip": True,
                }
            )

    summary = {
        "reference_events": events,
        "seeds": sorted({tr.seed for cond in traces for tr in traces[cond]}),
        "yaw_frozen_median": {c: float(np.median(v)) if v else None for c, v in yaw_medians.items()},
        "t4_t5_mean_event": t4t5_summary,
        "n_descending": n_dn,
        "n_candidates_left": len(candidates_left),
        "n_candidates_right": len(candidates_right),
        "n_candidates_signed_pairs": len(candidates_signed),
        "acceptance_note": "P0.2A probe — not P0.1R+ closure",
    }
    candidates_left.sort(key=lambda r: -r["sep_LEFT_vs_FWD"])
    candidates_right.sort(key=lambda r: -r["sep_RIGHT_vs_FWD"])

    return rows, {
        "summary": summary,
        "candidates_left": candidates_left[:30],
        "candidates_right": candidates_right[:30],
        "candidates_signed": candidates_signed[:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("-o", "--output", default=str(OUT_DIR))
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    flow_cfg = FlowConfig(crop_mode="center_band", ema_tau_s=0.5)
    cfg = FlyVOConfig(visual_gain=0.9)
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=flow_cfg)

    dn_idx, dn_meta = _build_dn_table(engine)
    dn_pos = {int(i): p for p, i in enumerate(dn_idx)}
    dt = cfg.brain_dt

    print(f"P0.2A DN tracing — {len(dn_idx)} descending neurons")
    print(f"Events: LEFT={REFERENCE_EVENTS['LEFT']}  FWD={REFERENCE_EVENTS['FWD']}  RIGHT={REFERENCE_EVENTS['RIGHT']}")
    print(f"Seeds: {args.seeds}")

    traces: dict[str, list[RunTrace]] = {c: [] for c in REFERENCE_EVENTS}

    for cond, ev in REFERENCE_EVENTS.items():
        t0 = ev["start"] - BASELINE_S
        t1 = ev["end"]
        for seed in args.seeds:
            print(f"  run {cond} seed={seed}  t={t0:.1f}–{t1:.1f}s …", flush=True)
            tr = run_segment(engine, encoder, video, t0, t1, dn_idx, dn_pos, seed, cond)
            traces[cond].append(tr)

    rows, report = analyze_traces(traces, REFERENCE_EVENTS, dn_meta, dt)

    csv_path = out / "dn_responses.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"\n=== P0.2A summary ===")
    print(f"  yaw_frozen: LEFT={s['yaw_frozen_median'].get('LEFT')}  FWD={s['yaw_frozen_median'].get('FWD')}  RIGHT={s['yaw_frozen_median'].get('RIGHT')}")
    print(f"  candidates: LEFT_sel={s['n_candidates_left']}  RIGHT_sel={s['n_candidates_right']}  SIGNED_pairs={s['n_candidates_signed_pairs']}")

    if report["candidates_signed"]:
        print("\n=== Top SIGNED L/R pairs (polarity flip) ===")
        for c in report["candidates_signed"][:8]:
            print(
                f"  {c['cell_type']:20s}  "
                f"Δ@LEFT={c['diff_at_LEFT_event']:+.2f}  Δ@RIGHT={c['diff_at_RIGHT_event']:+.2f}"
            )

    if report["candidates_left"]:
        print("\n=== Top LEFT-selective DNs ===")
        for c in report["candidates_left"][:5]:
            print(
                f"  {c['cell_type']:20s} {c['side']} idx={c['index']}  "
                f"L={c['rate_LEFT']:.2f} F={c['rate_FWD']:.2f} R={c['rate_RIGHT']:.2f}  sep={c['sep_LEFT_vs_FWD']:.1f}"
            )

    if report["candidates_right"]:
        print("\n=== Top RIGHT-selective DNs ===")
        for c in report["candidates_right"][:5]:
            print(
                f"  {c['cell_type']:20s} {c['side']} idx={c['index']}  "
                f"L={c['rate_LEFT']:.2f} F={c['rate_FWD']:.2f} R={c['rate_RIGHT']:.2f}  sep={c['sep_RIGHT_vs_FWD']:.1f}"
            )

    print(f"\nWrote {out}/")
    print("P0.2A probe complete — not a P0.1R+ gate.")


if __name__ == "__main__":
    main()
