#!/usr/bin/env python3
"""
P0.1X — the fly's visual cells on the three confirmed real turns.

Why this replaces every earlier readout number
---------------------------------------------
Every accuracy figure produced so far was scored against labels that included a
confirmed mislabelled reference. Those figures are retired. What remains is one
camera-LEFT turn and two camera-RIGHT windows, each watched and decided by a human:

    camera RIGHT   left_65   (63.5 - 66.5)   confirmed
    camera RIGHT   right_67  (66.0 - 68.5)   confirmed, lower confidence
    camera LEFT    left_108  (106.0 - 109.5) confirmed

Caveat stated up front: left_65 and right_67 overlap in time by 0.5 s and sit on
the same physical turn. The confirmed set therefore contains TWO distinct camera
states, not three independent turns. This is reported as a limitation, not hidden.

What is measured
----------------
For every visual cell type, its firing rate in each of the three windows, over five
seeds, original and mirrored. From that:

    D = mean rate over the camera-RIGHT windows  -  rate on the camera-LEFT window

Mirroring reverses the physical rotation, so a cell that carries direction must
flip the sign of D. That is the decisive, polarity-free test: it needs no
preferred direction to be assumed from any earlier, contaminated fit.

No tuning. No polarities carried over from P0.1L. The frozen Lucas-Kanade frontend
feeds T4/T5; the connectome does the rest.

Usage:
    PYTHONPATH=. python scripts/p01x_confirmed_readout.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import VideoVisualEncoder
from scripts.p01s_lk_gate import flow_config

VIDEO = ROOT / "data/p01r/VID00001.AVI"
LABELS_JSON = ROOT / "data/p01r/VID00001_labels_v2.json"
P01L_REPORT = ROOT / "output/p01l_real_video/report.json"

# The three confirmed windows. value = camera direction from watching the footage.
EVENTS = {
    "left_65":  (63.5, "RIGHT"),
    "right_67": (66.0, "RIGHT"),
    "left_108": (106.0, "LEFT"),
}
RIGHT_EVENTS = [k for k, v in EVENTS.items() if v[1] == "RIGHT"]
LEFT_EVENTS = [k for k, v in EVENTS.items() if v[1] == "LEFT"]
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
SEEDS = 5
T4T5_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
MODE = "lk"
ENCODE = (192, 108)


def candidate_types() -> list[tuple[str, str]]:
    """The 20 types P0.1L found, plus all T4/T5. No polarity is taken from P0.1L."""
    keys: list[tuple[str, str]] = []
    if P01L_REPORT.exists():
        rep = json.loads(P01L_REPORT.read_text(encoding="utf-8"))
        keys += [(r["cell_type"], r["side"]) for r in rep["all_types"]
                 if r.get("passes") and r.get("mirror_equivalence")]
    return keys


def capture(start: float, cfg: FlyVOConfig, mirror: bool) -> dict:
    t0 = max(0.0, start - PRE_ROLL_S)
    t1 = start + EVENT_DUR_S
    encoder = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain,
                                 n_azimuth_bins=128, flow_config=flow_config(MODE))
    frames = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, cfg.brain_dt):
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        eye, inject, _m = encoder.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject})
    n_pre = min(int(round(PRE_ROLL_S / cfg.brain_dt)), max(0, len(frames) - 1))
    return {"frames": frames, "n_pre": n_pre}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("-o", "--output", default="output/p01x_confirmed")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    print("P0.1X — visual cells on the three confirmed real turns")
    print(f"  camera RIGHT : {', '.join(RIGHT_EVENTS)}")
    print(f"  camera LEFT  : {', '.join(LEFT_EVENTS)}")
    print(f"  rejected     : right_216 (mixed motion, no clean horizontal rotation)")
    print(f"  frontend     : {MODE} (Lucas-Kanade), frozen")
    print(f"  runs         : original and mirrored, {args.seeds} seeds\n")
    print(f"  LIMITATION: left_65 and right_67 overlap by 0.5 s on the same physical turn,")
    print(f"  so this compares TWO distinct camera states, not three independent turns.\n")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    cfg.encode_width, cfg.encode_height = ENCODE
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain

    keys = candidate_types()
    if not keys:
        raise SystemExit("no candidate types; run P0.1L first")
    groups = {k: np.asarray(brain.cells([k[0]], side=k[1]), dtype=np.int64) for k in keys}
    # add T4/T5 directly, they are the injected population
    for t in T4T5_TYPES:
        for side in ("L", "R"):
            idx = np.asarray(brain.cells([t], side=side), dtype=np.int64)
            if len(idx):
                groups.setdefault((t, side), idx)
    keys = list(groups.keys())

    all_idx = np.unique(np.concatenate([v for v in groups.values() if len(v)]))
    union_slot = np.full(int(brain.n), -1, dtype=np.int64)
    union_slot[all_idx] = np.arange(len(all_idx), dtype=np.int64)
    local_pos = {k: union_slot[v] for k, v in groups.items()}

    print(f"  cell types read: {len(keys)}\n")
    rates: dict[tuple[str, bool, str], np.ndarray] = {}
    for run_name, mirror in (("orig", False), ("mirror", True)):
        for eid, (start, camdir) in EVENTS.items():
            cap = capture(start, cfg, mirror)
            lo, hi = cap["n_pre"], len(cap["frames"])
            span = max(hi - lo, 1) * dt
            counts = np.zeros((args.seeds, len(all_idx)), dtype=np.float64)
            for si, seed in enumerate(range(1, args.seeds + 1)):
                engine.reset(seed=seed)
                c = np.zeros(len(all_idx), dtype=np.int64)
                for i, fr in enumerate(cap["frames"]):
                    res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
                    if i < lo:
                        continue
                    f = np.asarray(res.fired)
                    if not len(f):
                        continue
                    s = union_slot[f]
                    s = s[s >= 0]
                    if len(s):
                        np.add.at(c, s, 1)
                counts[si] = c / span
            for k in keys:
                rates[(run_name, eid, k)] = counts[:, local_pos[k]].sum(axis=1)
            print(f"  {run_name:6s} {eid:10s} ({camdir:5s}) span={span:.2f}s")

    # ---- D per seed, original and mirrored ----
    print(f"\n  D = mean rate over the camera-RIGHT windows minus the camera-LEFT window")
    print(f"  a directional cell must have sign(D_mirror) = -sign(D_orig)\n")
    rows, records = [], []
    for k in keys:
        ctype, side = k
        rec = {"cell_type": ctype, "side": side, "is_injected": ctype in T4T5_TYPES}
        for run_name in ("orig", "mirror"):
            d_seeds = np.array([
                np.mean([rates[(run_name, e, k)][s] for e in RIGHT_EVENTS])
                - np.mean([rates[(run_name, e, k)][s] for e in LEFT_EVENTS])
                for s in range(args.seeds)
            ])
            med = float(np.median(d_seeds))
            mad = float(np.median(np.abs(d_seeds - med)))
            sigma = 1.4826 * mad
            rec[f"D_{run_name}"] = med
            rec[f"D_{run_name}_sigma"] = sigma
            rec[f"D_{run_name}_z"] = med / sigma if sigma > 1e-9 else float("inf") if med else 0.0
            rec[f"D_{run_name}_sign_consistency"] = float(np.mean(np.sign(d_seeds) == np.sign(med)))
            for e in EVENTS:
                rec[f"rate_{run_name}_{e}"] = float(np.median(rates[(run_name, e, k)]))

        flips = (np.sign(rec["D_orig"]) != np.sign(rec["D_mirror"]))
        strong = (abs(rec["D_orig_z"]) >= 2.0 and abs(rec["D_mirror_z"]) >= 2.0)
        consistent = (rec["D_orig_sign_consistency"] >= 0.8
                      and rec["D_mirror_sign_consistency"] >= 0.8)
        rec["mirror_flips_sign"] = bool(flips)
        rec["well_separated"] = bool(strong)
        rec["seed_consistent"] = bool(consistent)
        rec["direction_selective"] = bool(flips and strong and consistent)
        # magnitude agreement: mirroring should not change |D| much
        rec["mirror_magnitude_ratio"] = (
            min(abs(rec["D_orig"]), abs(rec["D_mirror"]))
            / max(abs(rec["D_orig"]), abs(rec["D_mirror"]), 1e-9)
        )
        rows.append(rec)
        records.append(rec)

    rows.sort(key=lambda r: (-int(r["direction_selective"]), -abs(r["D_orig_z"])))

    print(f"  {'type':10s} {'sd':2s} {'inj':3s} "
          f"{'D orig':>9s} {'z':>6s} {'cons':>5s} | {'D mirror':>9s} {'z':>6s} {'cons':>5s} "
          f"| {'flip':>5s} {'sep':>4s} {'=>':>5s}")
    for r in rows:
        print(f"  {r['cell_type'][:10]:10s} {r['side']:2s} "
              f"{'yes' if r['is_injected'] else '-':3s} "
              f"{r['D_orig']:+9.3f} {r['D_orig_z']:+6.2f} {r['D_orig_sign_consistency']:5.1f} | "
              f"{r['D_mirror']:+9.3f} {r['D_mirror_z']:+6.2f} {r['D_mirror_sign_consistency']:5.1f} | "
              f"{'yes' if r['mirror_flips_sign'] else 'no':>5s} "
              f"{'yes' if r['well_separated'] else 'no':>4s} "
              f"{'PASS' if r['direction_selective'] else '.':>5s}")

    passing = [r for r in rows if r["direction_selective"]]
    passing_dn = [r for r in passing if not r["is_injected"]]
    print(f"\n  direction-selective (flip + |z|>=2 both runs + seed-consistent):")
    print(f"    all types            : {len(passing)}/{len(rows)}")
    print(f"    downstream of T4/T5  : {len(passing_dn)}/{len([r for r in rows if not r['is_injected']])}")

    # ---- how do the two confirmed right turns compare with each other? ----
    print(f"\n  agreement between the two camera-RIGHT windows (they overlap; this is a weak check):")
    agree = 0
    for r in rows:
        a = r["rate_orig_left_65"]
        b = r["rate_orig_right_67"]
        c = r["rate_orig_left_108"]
        same_side = (a - c) * (b - c) > 0
        agree += int(same_side)
    print(f"    types where left_65 and right_67 both sit on the same side of left_108: "
          f"{agree}/{len(rows)} ({agree / len(rows):.0%})")

    # ---- what never got tested ----
    print(f"\n  the test this run cannot perform:")
    print(f"    an independent check of 'camera LEFT' beyond a single window cannot be made,")
    print(f"    and the two camera-RIGHT windows are the same physical turn. A stronger set")
    print(f"    would need several verified turns of both signs.")

    inject_rows = [r for r in rows if r["is_injected"]]
    if inject_rows:
        print(f"\n  T4/T5 (the injected population) for reference:")
        for r in inject_rows:
            print(f"    {r['cell_type']:5s} {r['side']}: D_orig {r['D_orig']:+8.3f} "
                  f"(z {r['D_orig_z']:+5.2f})  flip {r['mirror_flips_sign']}")

    report = {
        "probe": "P0.1X — visual cells on the three confirmed real turns",
        "labels_source": str(LABELS_JSON),
        "events": {k: {"start": v[0], "camera_direction": v[1]} for k, v in EVENTS.items()},
        "rejected": {"right_216": "mixed motion, no clean horizontal rotation"},
        "frontend": MODE,
        "statistic": "D = mean rate over camera-RIGHT windows minus camera-LEFT window; "
                     "mirroring must flip its sign",
        "limitation": labels["usable_for_rotation_tests"]["caveat"],
        "retired": labels["retired"],
        "n_types": len(rows),
        "n_direction_selective": len(passing),
        "n_direction_selective_downstream": len(passing_dn),
        "two_right_windows_agree": [agree, len(rows)],
        "per_type": records,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    verdict = (
        f"{len(passing)} of {len(rows)} visual types reverse their right-minus-left rate when the "
        f"video is mirrored ({len(passing_dn)} of them downstream of T4/T5), with seed consistency "
        "and separated from the seed noise. On this evidence the fly's own visual cells do carry "
        "the direction of the confirmed real turns."
        if passing else
        "No visual type reverses its right-minus-left rate under mirroring with any margin: on "
        "this evidence the fly's cells do not read the direction of the confirmed real turns."
    )
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
