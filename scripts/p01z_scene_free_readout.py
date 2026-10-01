#!/usr/bin/env python3
"""
P0.1Z — the fly's visual cells on the hand-marked turns, scene-free.

Why this replaces P0.1X
-----------------------
P0.1X used three windows: two camera-RIGHT windows that overlapped and shared a
scene, and one camera-LEFT window 40 s away in a different place. Its statistic

    D = mean rate over camera-RIGHT windows  -  rate on the camera-LEFT window

therefore compared mostly *scenes*, not directions. Luminance, contrast and texture
all differed more across the direction split than within it, which is why the
result came out with |z| up to 420 and every cell pointing the same way. That was
a scene difference wearing the costume of a direction effect.

This run uses 13 hand-marked turns, and the mapping of a window to a direction was
decided by eye from the footage. Seven of them have a readable motion field and
agree with the frozen measurement; those are the ones used here, and they are
interleaved in time rather than grouped by scene:

    camera LEFT   48.5, 80.5, 103.6, 108.9 s
    camera RIGHT  64.1, 148.0, 613.0 s

The statistic is paired and scene-free
--------------------------------------
For each event, and each cell, take the response to the original clip and to the
mirrored clip. Mirroring reverses the physical direction of image motion and
changes nothing else about the scene. So

    c(event) = rate_original(event) - rate_mirrored(event)

is measured *within one scene*. A direction-selective cell must have c(event) of
one sign on camera-LEFT events and the opposite sign on camera-RIGHT events.

That is the whole test:

    separation = |mean c(RIGHT) - mean c(LEFT)| / spread
    consistency = fraction of events whose c has the sign its group predicts

A permutation test over the event labels gives the null.

No tuning, no polarities carried from any earlier fit, no threshold beyond the
frozen measurement's own coherence.

Usage:
    PYTHONPATH=. python scripts/p01z_scene_free_readout.py
    PYTHONPATH=. python scripts/p01z_scene_free_readout.py --min-coherence 0.5
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
SCORED = ROOT / "output/p01y_marked/report.json"
P01L_REPORT = ROOT / "output/p01l_real_video/report.json"

PRE_ROLL_S = 0.0          # the marked window is used exactly as marked
SEEDS = 5
MODE = "lk"
ENCODE = (192, 108)
T4T5_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def candidate_types() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if P01L_REPORT.exists():
        rep = json.loads(P01L_REPORT.read_text(encoding="utf-8"))
        keys += [(r["cell_type"], r["side"]) for r in rep["all_types"]
                 if r.get("passes") and r.get("mirror_equivalence")]
    for t in T4T5_TYPES:
        for side in ("L", "R"):
            keys.append((t, side))
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def capture(start: float, end: float, cfg: FlyVOConfig, mirror: bool) -> dict:
    encoder = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain,
                                 n_azimuth_bins=128, flow_config=flow_config(MODE))
    frames = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), start, end, cfg.brain_dt):
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        eye, inject, _m = encoder.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject})
    return {"frames": frames}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coherence", type=float, default=0.40)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("-o", "--output", default="output/p01z_scene_free")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    scored = json.loads(SCORED.read_text(encoding="utf-8"))
    turns = scored["turns"]
    usable = [t for t in turns
              if t["agrees"] is True and t["coherence"] >= args.min_coherence]
    left = [t for t in usable if t["marked_camera"] == "LEFT"]
    right = [t for t in usable if t["marked_camera"] == "RIGHT"]

    print("P0.1Z — scene-free readout on hand-marked turns")
    print(f"  {len(turns)} marked turns; {len(usable)} usable "
          f"(measurement agrees and coherence >= {args.min_coherence})")
    print(f"  camera LEFT  ({len(left)}): "
          + ", ".join(f"{t['t0']:.1f}-{t['t1']:.1f}" for t in sorted(left, key=lambda x: x['t0'])))
    print(f"  camera RIGHT ({len(right)}): "
          + ", ".join(f"{t['t0']:.1f}-{t['t1']:.1f}" for t in sorted(right, key=lambda x: x['t0'])))
    if len(left) < 2 or len(right) < 2:
        raise SystemExit("need at least two usable turns of each direction")
    print(f"\n  statistic: c(event) = orig - mirrored, measured WITHIN one scene;")
    print(f"  a directional cell must have opposite signs on LEFT and RIGHT events\n")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    cfg.encode_width, cfg.encode_height = ENCODE
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain

    keys = candidate_types()
    groups = {k: np.asarray(brain.cells([k[0]], side=k[1]), dtype=np.int64) for k in keys}
    groups = {k: v for k, v in groups.items() if len(v)}
    keys = list(groups.keys())
    all_idx = np.unique(np.concatenate(list(groups.values())))
    union_slot = np.full(int(brain.n), -1, dtype=np.int64)
    union_slot[all_idx] = np.arange(len(all_idx), dtype=np.int64)
    local_pos = {k: union_slot[v] for k, v in groups.items()}

    events = [(t["id"], t["t0"], t["t1"], t["marked_camera"]) for t in usable]
    rates: dict[tuple[str, str], np.ndarray] = {}
    for run_name, mirror in (("orig", False), ("mirror", True)):
        for eid, t0, t1, cam in events:
            cap = capture(float(t0), float(t1), cfg, mirror)
            n = len(cap["frames"])
            span = max(n, 1) * dt
            counts = np.zeros((args.seeds, len(all_idx)), dtype=np.float64)
            for si, seed in enumerate(range(1, args.seeds + 1)):
                engine.reset(seed=seed)
                c = np.zeros(len(all_idx), dtype=np.int64)
                for fr in cap["frames"]:
                    res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
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
            print(f"  {run_name:6s} {eid:9s} t={t0:.1f}-{t1:.1f} ({cam:5s}) steps={n}")

    # ---------------------------------------------------------------- c(event)
    ev_ids = [e[0] for e in events]
    ev_dir = {e[0]: e[3] for e in events}
    print(f"\n  {'type':10s} {'sd':2s} {'inj':3s} "
          f"{'mean c LEFT':>12s} {'mean c RIGHT':>13s} {'sep z':>7s} {'consist':>8s} "
          f"{'flip':>5s} {'=>':>5s}")
    rows = []
    for k in keys:
        c_by_event, seed_sd = {}, {}
        for eid in ev_ids:
            o, m = rates[("orig", eid, k)], rates[("mirror", eid, k)]
            d = o - m
            c_by_event[eid] = float(np.median(d))
            seed_sd[eid] = float(1.4826 * np.median(np.abs(d - np.median(d))))
        c_left = np.array([c_by_event[e] for e in ev_ids if ev_dir[e] == "LEFT"])
        c_right = np.array([c_by_event[e] for e in ev_ids if ev_dir[e] == "RIGHT"])
        # pooled spread from the per-seed noise, so z is not inflated by n
        spread = float(np.mean([seed_sd[e] for e in ev_ids]))
        sep = (np.mean(c_right) - np.mean(c_left)) / spread if spread > 1e-9 else 0.0
        # the sign each group predicts, taken from the group means themselves
        pred_left = -np.sign(np.mean(c_right) - np.mean(c_left)) or 1
        pred_right = -pred_left
        hits = 0
        for e in ev_ids:
            want = pred_left if ev_dir[e] == "LEFT" else pred_right
            hits += int(np.sign(c_by_event[e]) == want)
        consistent = hits / len(ev_ids)
        # direction selectivity needs a big separation AND the same sign in every event
        selective = (abs(sep) >= 2.0 and consistent >= 0.8)

        rows.append({
            "cell_type": k[0], "side": k[1], "is_injected": k[0] in T4T5_TYPES,
            "mean_c_left": float(np.mean(c_left)), "mean_c_right": float(np.mean(c_right)),
            "separation_z": float(sep), "consistency": float(consistent),
            "n_events": len(ev_ids), "predicted_left_sign": int(pred_left),
            "per_event_c": c_by_event, "per_event_seed_sd": seed_sd,
            "direction_selective": bool(selective),
        })
        print(f"  {k[0][:10]:10s} {k[1]:2s} {'yes' if k[0] in T4T5_TYPES else '-':3s} "
              f"{np.mean(c_left):+12.2f} {np.mean(c_right):+13.2f} {sep:+7.2f} "
              f"{consistent:8.2f} {'yes' if abs(sep) >= 2 else 'no':>5s} "
              f"{'PASS' if selective else '.':>5s}")

    rows.sort(key=lambda r: (-int(r["direction_selective"]), -abs(r["separation_z"])))
    passing = [r for r in rows if r["direction_selective"]]
    passing_dn = [r for r in passing if not r["is_injected"]]
    n_dn = sum(1 for r in rows if not r["is_injected"])

    # ------------------------------------------------------- permutation null
    rng = np.random.default_rng(0)
    n_perm = 2000
    dirs = [ev_dir[e] for e in ev_ids]
    null = []
    for _ in range(n_perm):
        perm = list(rng.permutation(dirs))
        cnt = 0
        for r in rows:
            c = np.array([r["per_event_c"][e] for e in ev_ids])
            cl = np.array([c[i] for i, d in enumerate(perm) if d == "LEFT"])
            cr = np.array([c[i] for i, d in enumerate(perm) if d == "RIGHT"])
            spread = float(np.mean([r["per_event_seed_sd"][e] for e in ev_ids]))
            if spread <= 1e-9:
                continue
            sep = (cr.mean() - cl.mean()) / spread
            pl = -np.sign(cr.mean() - cl.mean()) or 1
            hits = sum(int(np.sign(c[i]) == (pl if perm[i] == "LEFT" else -pl))
                       for i in range(len(perm)))
            if abs(sep) >= 2.0 and hits / len(perm) >= 0.8:
                cnt += 1
        null.append(cnt)
    null = np.asarray(null)
    p_value = float(np.mean(null >= len(passing)))

    print(f"\n=== summary ===")
    print(f"  direction-selective              : {len(passing)}/{len(rows)}")
    print(f"  downstream of T4/T5              : {len(passing_dn)}/{n_dn}")
    print(f"  permutation null (labels shuffled): mean {null.mean():.2f}, "
          f"p95 {np.percentile(null, 95):.0f}, observed {len(passing)}, p = {p_value:.4f}")

    verdict = (
        f"{len(passing)} of {len(rows)} visual types ({len(passing_dn)} of them downstream of "
        f"T4/T5) respond to mirroring with a sign that tracks the camera direction across "
        f"{len(left)} LEFT and {len(right)} RIGHT hand-marked turns, with |z| >= 2 against the "
        f"seed noise and consistent across at least 80% of events (permutation p = {p_value:.4f}). "
        "Because each contrast is measured inside a single scene, a scene difference cannot "
        "produce this pattern."
        if passing else
        "No visual type shows a mirror contrast that tracks the camera direction across the "
        "hand-marked turns: within-scene mirror contrasts do not separate LEFT from RIGHT. "
        "The fly's cells do not read the direction of these real turns on this evidence."
    )
    print(f"\n=== VERDICT ===\n{verdict}")

    (out / "report.json").write_text(json.dumps({
        "probe": "P0.1Z — scene-free readout on hand-marked turns",
        "why": "P0.1X's cross-window statistic compared scenes, not directions",
        "thresholds": {"min_coherence": args.min_coherence, "min_abs_z": 2.0,
                       "min_consistency": 0.8},
        "events": [{"id": e[0], "t0": e[1], "t1": e[2], "camera": e[3]} for e in events],
        "n_left": len(left), "n_right": len(right),
        "n_types": len(rows), "n_direction_selective": len(passing),
        "n_direction_selective_downstream": len(passing_dn),
        "permutation": {"n_perm": n_perm, "mean": float(null.mean()),
                        "p95": float(np.percentile(null, 95)),
                        "observed": len(passing), "p_value": p_value},
        "per_type": rows, "verdict": verdict,
    }, indent=2), encoding="utf-8")

    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_type", "side", "is_injected", "mean_c_left", "mean_c_right",
                    "separation_z", "consistency", "direction_selective"])
        for r in rows:
            w.writerow([r["cell_type"], r["side"], r["is_injected"],
                        f"{r['mean_c_left']:.3f}", f"{r['mean_c_right']:.3f}",
                        f"{r['separation_z']:.3f}", f"{r['consistency']:.3f}",
                        r["direction_selective"]])
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
