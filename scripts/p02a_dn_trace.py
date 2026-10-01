#!/usr/bin/env python3
"""
P0.2A — Descending-neuron propagation tracing (mechanistic probe).

Reference states (from P0.1R+ event benchmark):
  left_65   visual LEFT  yaw_frozen = -0.219
  right_216 visual RIGHT yaw_frozen = +0.191
  fwd_140   FWD          yaw_frozen = -0.002

Frozen frontend → MaleCNS → ALL 1314 descending neurons, 5 seeds.

Does NOT:
  - select DN types by hand (no DNa02=yaw prior)
  - tune thresholds / dopamine / trajectory / motion_readout
  - claim P0.1R+ closure

Outputs:
  output/p02a_dn_trace/dn_cells.csv      per DN cell × state
  output/p02a_dn_trace/dn_families.csv   per (cell_type, side)
  output/p02a_dn_trace/signed_pairs.csv  L/R differential sign-flip candidates
  output/p02a_dn_trace/report.json

Usage:
    PYTHONPATH=. python scripts/p02a_dn_trace.py
    PYTHONPATH=. python scripts/p02a_dn_trace.py --seeds 1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import VideoVisualEncoder

EMA_TAU = 0.5          # frozen
PRE_ROLL_S = 1.5       # baseline before event
EVENT_DUR_S = 3.0      # comparable response window
ROLL_WIN_STEPS = 10    # 0.2 s at 50 Hz
LATENCY_SIGMA = 3.0

STATES = {
    "LEFT": {"start": 63.5, "source": "left_65", "yaw_frozen": -0.219},
    "FWD": {"start": 137.5, "source": "fwd_140", "yaw_frozen": -0.002},
    "RIGHT": {"start": 214.0, "source": "right_216", "yaw_frozen": 0.191},
}


# ---------------------------------------------------------------- frontend cache
def cache_event(video: Path, t0: float, t1: float, cfg: FlyVOConfig) -> list[dict]:
    """Run frozen frontend once per event (seed-independent)."""
    enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain)
    frames: list[dict] = []
    for t, frame in iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt):
        eye, inject, metrics = enc.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject, "yaw": metrics.get("yaw_frozen", 0.0)})
    return frames


# ---------------------------------------------------------------- DN bookkeeping
def dn_table(brain) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (indices, slot_map, cell_type_per_slot, side_per_slot)."""
    sc = np.asarray(brain.superclass)
    idx = np.where(sc == "descending_neuron")[0]
    ct = np.asarray(brain.cell_type)[idx]
    sd = np.asarray(brain.side)[idx]
    slot = np.full(int(np.asarray(brain.superclass).shape[0]), -1, dtype=np.int64)
    slot[idx] = np.arange(len(idx), dtype=np.int64)
    return idx, slot, ct, sd


def run_state(
    engine: MaleCNSEngine,
    frames: list[dict],
    slot: np.ndarray,
    n_dn: int,
    seed: int,
    dt: float,
) -> np.ndarray:
    """Replay cached frontend into brain; return per-step DN counts [n_dn, n_steps]."""
    engine.reset(seed=seed)
    counts = np.zeros((n_dn, len(frames)), dtype=np.int32)
    for i, fr in enumerate(frames):
        res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
        fired = res.fired
        if len(fired):
            hits = slot[fired]
            hits = hits[hits >= 0]
            if len(hits):
                np.add.at(counts[:, i], hits, 1)
    return counts


# ---------------------------------------------------------------- metrics
def _robust_sigma(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    med = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - med)))


def _step_sigma(counts: np.ndarray, lo: int, hi: int, dt: float) -> np.ndarray:
    """Per-cell robust sigma of per-step firing rate (Hz) over steps [lo, hi)."""
    if hi - lo < 3:
        return np.ones(counts.shape[0], dtype=np.float64)
    seg = counts[:, lo:hi].astype(np.float64) / dt  # [n_dn, n_steps] Hz
    med = np.median(seg, axis=1, keepdims=True)
    mad = np.median(np.abs(seg - med), axis=1)
    return 1.4826 * mad


def _window_rate(counts: np.ndarray, lo: int, hi: int, dt: float) -> np.ndarray:
    """Per-seed mean firing rate (Hz) over steps [lo, hi)."""
    seg = counts[:, lo:hi]
    return seg.sum(axis=1) / max((hi - lo) * dt, 1e-9)


def _separation(a: np.ndarray, b: np.ndarray) -> float:
    """Robust separation between two per-seed rate distributions."""
    if not len(a) or not len(b):
        return 0.0
    sigma = np.sqrt((_robust_sigma(a) ** 2 + _robust_sigma(b) ** 2) / 2.0)
    if sigma < 1e-9:
        return 999.0 if abs(float(np.median(a)) - float(np.median(b))) > 1e-9 else 0.0
    return abs(float(np.median(a)) - float(np.median(b))) / sigma


def _latency(counts: np.ndarray, base_lo: int, base_hi: int, ev_hi: int, dt: float) -> float | None:
    """First step after event onset exceeding baseline + LATENCY_SIGMA*robust_sigma."""
    total = counts.sum(axis=0).astype(float) / dt  # Hz per step, pooled seeds
    base = total[base_lo:base_hi]
    if len(base) < 3:
        return None
    thr = float(np.median(base)) + LATENCY_SIGMA * (_robust_sigma(base) + 1e-9)
    for i in range(base_hi, min(ev_hi, len(total))):
        lo = max(base_hi, i - ROLL_WIN_STEPS + 1)
        if float(np.mean(total[lo : i + 1])) >= thr:
            return (i - base_hi) * dt
    return None


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/p01r/VID00001.AVI")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("-o", "--output", default="output/p02a_dn_trace")
    parser.add_argument("--dry-run", action="store_true", help="1 seed, short windows")
    args = parser.parse_args()

    seeds = [1] if args.dry_run else list(range(1, args.seeds + 1))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = EMA_TAU
    dt = cfg.brain_dt

    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain
    dn_idx, slot, ct_slot, sd_slot = dn_table(brain)
    n_dn = len(dn_idx)
    print(f"MaleCNS {brain.n} neurons | {n_dn} descending neurons | device={brain.device}")
    print(f"seeds={seeds}  states={list(STATES)}  pre_roll={PRE_ROLL_S}s  event={EVENT_DUR_S}s")

    # --- cache frontend per state ---
    stages: dict[str, list[dict]] = {}
    for name, meta in STATES.items():
        t0 = meta["start"] - PRE_ROLL_S
        t1 = meta["start"] + EVENT_DUR_S
        if args.dry_run:
            t1 = meta["start"] + 1.0
        t_start = time.time()
        stages[name] = cache_event(video, t0, t1, cfg)
        print(f"  cached {name}: {len(stages[name])} steps ({time.time()-t_start:.1f}s frontend)")

    steps_per_state = {k: len(v) for k, v in stages.items()}
    pre_steps = int(round(PRE_ROLL_S / dt))

    # --- run all seeds ---
    rates: dict[str, np.ndarray] = {}  # state → [n_seeds, n_dn] event rate Hz
    base_rates: dict[str, np.ndarray] = {}
    base_sigma: dict[str, np.ndarray] = {}
    latencies: dict[str, list[float | None]] = {}
    yaw_med: dict[str, float] = {}

    for name, frames in stages.items():
        ev_hi = len(frames)
        base_hi = min(pre_steps, ev_hi)
        rate_mat = np.zeros((len(seeds), n_dn), dtype=np.float64)
        base_mat = np.zeros((len(seeds), n_dn), dtype=np.float64)
        sigma_mat = np.zeros((len(seeds), n_dn), dtype=np.float64)
        lat: list[float | None] = []
        for si, seed in enumerate(seeds):
            t0 = time.time()
            counts = run_state(engine, frames, slot, n_dn, seed, dt)
            rate_mat[si] = _window_rate(counts, base_hi, ev_hi, dt)
            base_mat[si] = _window_rate(counts, 0, base_hi, dt)
            sigma_mat[si] = _step_sigma(counts, 0, base_hi, dt)
            lat.append(_latency(counts, 0, base_hi, ev_hi, dt))
            print(f"    {name} seed={seed} ({time.time()-t0:.1f}s)")
        rates[name] = rate_mat
        base_rates[name] = base_mat
        base_sigma[name] = sigma_mat
        latencies[name] = lat
        yaw_vals = [f["yaw"] for f in frames[base_hi:]]
        yaw_med[name] = float(np.median(yaw_vals))

    # --- gate: LEFT-selective / RIGHT-selective per cell ---
    def sel(a: np.ndarray, b: np.ndarray, c: np.ndarray, want: str) -> tuple[bool, float, float]:
        """want 'high' → a above b and c."""
        sep_ab = _separation(a, b)
        sep_ac = _separation(a, c)
        med_a = float(np.median(a))
        if want == "high":
            ok = med_a > float(np.median(b)) and med_a > float(np.median(c))
        else:
            ok = med_a < float(np.median(b)) and med_a < float(np.median(c))
        return ok, sep_ab, sep_ac

    rows_cells: list[dict] = []
    for i in range(n_dn):
        L = rates["LEFT"][:, i]
        F = rates["FWD"][:, i]
        R = rates["RIGHT"][:, i]
        base = base_rates["LEFT"][:, i]

        left_sel, sep_lf, sep_lr = sel(L, F, R, "high")
        right_sel, sep_rf, sep_rl = sel(R, F, L, "high")

        # sign consistency across seeds
        cons_l = float(np.mean(L > F)) if len(L) else 0.0
        cons_r = float(np.mean(R > F)) if len(R) else 0.0

        sigma_L = float(np.mean(base_sigma["LEFT"][:, i]))
        sigma_R = float(np.mean(base_sigma["RIGHT"][:, i]))
        sigma_F = float(np.mean(base_sigma["FWD"][:, i]))

        def _sep_vs_base(ev: np.ndarray, base_med: float, sigma: float) -> float:
            if sigma < 1e-9:
                return 0.0
            return abs(float(np.median(ev)) - base_med) / sigma

        base_L_med = float(np.median(base))
        base_R_med = float(np.median(base_rates["RIGHT"][:, i]))
        base_F_med = float(np.median(base_rates["FWD"][:, i]))

        rows_cells.append(
            {
                "cell_index": int(dn_idx[i]),
                "cell_type": str(ct_slot[i]),
                "side": str(sd_slot[i]),
                "rate_LEFT": float(np.median(L)),
                "rate_FWD": float(np.median(F)),
                "rate_RIGHT": float(np.median(R)),
                "rate_baseline_LEFT": base_L_med,
                "rate_std_LEFT": float(np.std(L)),
                "left_minus_fwd": float(np.median(L) - np.median(F)),
                "right_minus_fwd": float(np.median(R) - np.median(F)),
                "left_minus_right": float(np.median(L) - np.median(R)),
                "sep_LEFT_FWD": round(sep_lf, 3),
                "sep_RIGHT_FWD": round(sep_rf, 3),
                "sep_LEFT_RIGHT": round(sep_lr, 3),
                "sep_LEFT_vs_baseline": round(_sep_vs_base(L, base_L_med, sigma_L), 3),
                "sep_RIGHT_vs_baseline": round(_sep_vs_base(R, base_R_med, sigma_R), 3),
                "sigma_step_LEFT": round(sigma_L, 4),
                "consistency_LEFT_gt_FWD": cons_l,
                "consistency_RIGHT_gt_FWD": cons_r,
                "latency_LEFT_s": latencies["LEFT"][int(np.argmax(L))] if len(L) else None,
                "latency_RIGHT_s": latencies["RIGHT"][int(np.argmax(R))] if len(R) else None,
                "latency_FWD_s": latencies["FWD"][int(np.argmax(F))] if len(F) else None,
                "LEFT_selective": left_sel,
                "RIGHT_selective": right_sel,
                "yaw_LEFT": yaw_med["LEFT"],
                "yaw_FWD": yaw_med["FWD"],
                "yaw_RIGHT": yaw_med["RIGHT"],
            }
        )

    # --- aggregate to family × side ---
    fam_side: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(n_dn):
        fam_side[(str(ct_slot[i]), str(sd_slot[i]))].append(i)

    rows_fams: list[dict] = []
    for (ctype, side), idxs in fam_side.items():
        L = rates["LEFT"][:, idxs].sum(axis=1)
        F = rates["FWD"][:, idxs].sum(axis=1)
        R = rates["RIGHT"][:, idxs].sum(axis=1)
        rows_fams.append(
            {
                "cell_type": ctype,
                "side": side,
                "n_cells": len(idxs),
                "rate_LEFT": float(np.median(L)),
                "rate_FWD": float(np.median(F)),
                "rate_RIGHT": float(np.median(R)),
                "left_minus_fwd": float(np.median(L) - np.median(F)),
                "right_minus_fwd": float(np.median(R) - np.median(F)),
                "left_minus_right": float(np.median(L) - np.median(R)),
                "sep_LEFT_FWD": round(_separation(L, F), 3),
                "sep_RIGHT_FWD": round(_separation(R, F), 3),
                "consistency_LEFT": float(np.mean(L > F)),
                "consistency_RIGHT": float(np.mean(R > F)),
                "LEFT_selective": float(np.median(L)) > float(np.median(F)) and float(np.median(L)) > float(np.median(R)),
                "RIGHT_selective": float(np.median(R)) > float(np.median(F)) and float(np.median(R)) > float(np.median(L)),
            }
        )

    # --- signed L/R differential (same DN type, opposite sides) ---
    rows_signed: list[dict] = []
    common_types = {c for (c, _s) in fam_side if (c, "L") in fam_side and (c, "R") in fam_side}
    for ctype in sorted(common_types):
        il, ir = fam_side[(ctype, "L")], fam_side[(ctype, "R")]
        Ldiff_left = rates["LEFT"][:, il].sum(axis=1) - rates["LEFT"][:, ir].sum(axis=1)
        Ldiff_right = rates["RIGHT"][:, il].sum(axis=1) - rates["RIGHT"][:, ir].sum(axis=1)
        Ldiff_fwd = rates["FWD"][:, il].sum(axis=1) - rates["FWD"][:, ir].sum(axis=1)
        m_left = float(np.median(Ldiff_left))
        m_right = float(np.median(Ldiff_right))
        m_fwd = float(np.median(Ldiff_fwd))

        # Deflection relative to the no-yaw (FWD) lateralization baseline.
        defl_left = m_left - m_fwd
        defl_right = m_right - m_fwd
        genuine_flip = defl_left * defl_right < 0
        min_defl = min(abs(defl_left), abs(defl_right))

        cons_flip = float(np.mean(np.sign(Ldiff_left - Ldiff_fwd) != np.sign(Ldiff_right - Ldiff_fwd)))

        rows_signed.append(
            {
                "cell_type": ctype,
                "n_L": len(il),
                "n_R": len(ir),
                "diff_LEFT": m_left,
                "diff_FWD": m_fwd,
                "diff_RIGHT": m_right,
                "defl_LEFT": defl_left,
                "defl_RIGHT": defl_right,
                "min_abs_defl": min_defl,
                "sign_flip": bool(genuine_flip),
                "consistency_flip": cons_flip,
            }
        )

    # --- write CSVs ---
    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "dn_cells.csv", rows_cells)
    write_csv(out / "dn_families.csv", rows_fams)
    write_csv(out / "signed_pairs.csv", rows_signed)

    # --- cross-reference with synthetic inject control (if available) ---
    ctrl_csv = out / "inject_control" / "families.csv"
    cross: list[dict] = []
    if ctrl_csv.exists():
        syn = {
            r["cell_type"]: r
            for r in csv.DictReader(ctrl_csv.open())
            if r["sign_flip"] == "True" and float(r["consistency_flip"]) >= 0.8
        }
        for r in rows_signed:
            if not r["sign_flip"] or r["cell_type"] not in syn:
                continue
            s = syn[r["cell_type"]]
            cross.append(
                {
                    "cell_type": r["cell_type"],
                    "syn_defl_LEFT": float(s["defl_LEFT"]),
                    "syn_defl_RIGHT": float(s["defl_RIGHT"]),
                    "real_defl_LEFT": r["defl_LEFT"],
                    "real_defl_RIGHT": r["defl_RIGHT"],
                    "real_consistency_flip": r["consistency_flip"],
                    "syn_min_abs_defl": round(
                        min(abs(float(s["defl_LEFT"])), abs(float(s["defl_RIGHT"]))), 3
                    ),
                }
            )
        cross.sort(key=lambda r: -min(r["syn_min_abs_defl"], abs(r["real_defl_RIGHT"])))
        write_csv(out / "signed_candidates.csv", cross)

    # --- candidates ---
    def top(rows: list[dict], key: str, pred, n: int = 15) -> list[dict]:
        cand = [r for r in rows if pred(r)]
        return sorted(cand, key=lambda r: -abs(float(r[key])))[:n]

    left_cells = top(
        rows_cells, "left_minus_fwd",
        lambda r: r["LEFT_selective"] and r["consistency_LEFT_gt_FWD"] >= 0.8
        and (r["sep_LEFT_FWD"] >= 2.0 or r["sep_LEFT_vs_baseline"] >= 3.0), 25,
    )
    right_cells = top(
        rows_cells, "right_minus_fwd",
        lambda r: r["RIGHT_selective"] and r["consistency_RIGHT_gt_FWD"] >= 0.8
        and (r["sep_RIGHT_FWD"] >= 2.0 or r["sep_RIGHT_vs_baseline"] >= 3.0), 25,
    )
    signed_pairs = top(
        rows_signed, "min_abs_defl",
        lambda r: r["sign_flip"] and r["consistency_flip"] >= 0.8, 25,
    )

    # --- inject-strength audit (per event, per T4/T5 subtype) ---
    from fly_vo.brain_clock import iter_video_at_brain_hz
    from fly_vo.config import FlyVOConfig as _Cfg

    audit_cfg = _Cfg()
    audit_enc = VideoVisualEncoder(MaleCNSEngine(audit_cfg), gain=audit_cfg.visual_gain)
    inject_audit: dict[str, dict] = {}
    for name, meta in STATES.items():
        t0 = meta["start"] - PRE_ROLL_S
        t1 = meta["start"] + EVENT_DUR_S
        agg: dict[str, float] = defaultdict(float)
        n = 0
        for _t, frame in iter_video_at_brain_hz(str(video), t0, t1, audit_cfg.brain_dt):
            _eye, _inj, m = audit_enc.encode_frame(frame)
            for k, v in m["inject"].items():
                agg[k] += v
            n += 1
        per = {k: v / max(n, 1) for k, v in agg.items()}
        t4 = {s: sum(per.get(f"{g}_{s}", 0.0) for g in ("T4a", "T4b", "T4c", "T4d")) for s in ("L", "R")}
        t5 = {s: sum(per.get(f"{g}_{s}", 0.0) for g in ("T5a", "T5b", "T5c", "T5d")) for s in ("L", "R")}
        inject_audit[name] = {
            "T4_L": t4["L"], "T4_R": t4["R"], "T4_side_diff": t4["L"] - t4["R"],
            "T5_L": t5["L"], "T5_R": t5["R"], "T5_side_diff": t5["L"] - t5["R"],
            "combined_side_diff": (t4["L"] + t5["L"]) - (t4["R"] + t5["R"]),
        }

    report = {
        "probe": "P0.2A DN propagation tracing (mechanistic only)",
        "frozen": {"ema_tau": EMA_TAU, "pre_roll_s": PRE_ROLL_S, "event_s": EVENT_DUR_S, "crop": "center_band"},
        "states": {k: {**v, "yaw_frozen_median": yaw_med[k], "n_steps": steps_per_state[k]} for k, v in STATES.items()},
        "inject_audit": inject_audit,
        "n_descending_neurons": n_dn,
        "n_dn_types": len(fam_side),
        "seeds": seeds,
        "counts": {
            "LEFT_selective_cells": sum(1 for r in rows_cells if r["LEFT_selective"]),
            "RIGHT_selective_cells": sum(1 for r in rows_cells if r["RIGHT_selective"]),
            "left_cells_sep_ge2": len(left_cells),
            "right_cells_sep_ge2": len(right_cells),
            "signed_pairs_signflip": sum(1 for r in rows_signed if r["sign_flip"]),
            "signed_pairs_signflip_fwd0": len(signed_pairs),
            "signed_defl_range": {
                "max_min_abs_defl": max((r["min_abs_defl"] for r in rows_signed), default=0.0),
                "n_defl_gt_0.5Hz": sum(1 for r in rows_signed if r["min_abs_defl"] > 0.5),
            },
        },
        "candidates": {
            "LEFT_selective": left_cells,
            "RIGHT_selective": right_cells,
            "signed_pairs": signed_pairs,
            "signed_candidates_crossref": cross[:25],
        },
        "top_family_by_left_fwd": sorted(rows_fams, key=lambda r: -r["left_minus_fwd"])[:10],
        "top_family_by_right_fwd": sorted(rows_fams, key=lambda r: -r["right_minus_fwd"])[:10],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # --- console summary ---
    print(f"\n=== Selective cell counts (of {n_dn}) ===")
    print(f"  LEFT-selective cells:  {report['counts']['LEFT_selective_cells']}")
    print(f"  RIGHT-selective cells: {report['counts']['RIGHT_selective_cells']}")
    print(f"  LEFT  sep>=2 vs FWD:   {report['counts']['left_cells_sep_ge2']}")
    print(f"  RIGHT sep>=2 vs FWD:   {report['counts']['right_cells_sep_ge2']}")
    print(f"  signed L/R sign-flip:  {report['counts']['signed_pairs_signflip']}")

    def show(title: str, rows: list[dict], keys: list[str]) -> None:
        print(f"\n=== {title} ===")
        for r in rows[:10]:
            print("  " + "  ".join(f"{k}={r[k]}" for k in keys))

    show("LEFT-selective (top)", left_cells, ["cell_type", "side", "rate_LEFT", "rate_FWD", "sep_LEFT_FWD", "sep_LEFT_vs_baseline", "consistency_LEFT_gt_FWD"])
    show("RIGHT-selective (top)", right_cells, ["cell_type", "side", "rate_RIGHT", "rate_FWD", "sep_RIGHT_FWD", "sep_RIGHT_vs_baseline", "consistency_RIGHT_gt_FWD"])
    show("SIGNED L/R pairs (top)", signed_pairs, ["cell_type", "diff_LEFT", "diff_FWD", "diff_RIGHT", "defl_LEFT", "defl_RIGHT", "consistency_flip"])

    print(f"\nWrote {out}/")
    print("NOTE: mechanistic probe only — does not close P0.1R+.")


if __name__ == "__main__":
    main()
