#!/usr/bin/env python3
"""
P0.1Z2 — fix the statistical flaw in P0.1Z and re-test.

The flaw
-------
P0.1Z measured, per cell and per event,

    c(event) = rate(original) - rate(mirrored)

and then read off the polarity from the same numbers ("pred_left is whichever sign
fits"). Doing that makes the test circular: the data is free to choose the
convention that fits, so with 4 LEFT and 3 RIGHT events a shuffled labelling fits
just as well. The permutation test said so: mean 4.9, p95 32 of 32, p = 0.121.
The observed 32/32 was therefore not evidence of anything.

The reported |z| was also wrong: it divided by the seed-to-seed spread, which
measures simulation noise, while the contrast is dominated by a structured change.
A large z against the wrong denominator is meaningless.

The corrected test
------------------
Two stages, with the polarity fixed BEFORE the real data is looked at:

  stage 1  synthetic: two clips built from one real frame, content moving right and
           content moving left. For each cell, p = sign(rate(content_right) -
           rate(content_left)). This is the cell's motion polarity, decided on
           stimuli whose direction was manufactured, not measured.

  stage 2  real: for each hand-marked event, expected sign of
               c(event) = rate(original) - rate(mirrored)
           is p for a camera-LEFT event (content moves right, like stage 1's
           content_right) and -p for camera-RIGHT. A cell passes only if it matches
           on at least 6 of 7 events.

The permutation null now shuffles the camera labels while keeping p fixed, so the
test has real power: a cell whose polarity was fitted to these events can no
longer pass by construction.

No tuning. Stage 1's polarity is not adjustable, and nothing in stage 2 may be
changed after seeing the result.

Usage:
    PYTHONPATH=. python scripts/p01z2_polarity_test.py
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

SEEDS = 5
MODE = "lk"
ENCODE = (192, 108)
T4T5_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
SYNTH_BASE_S = 139.5
SYNTH_PX_PER_FRAME = -2.0     # content moves LEFT for "left" clip
MIN_CONSISTENCY = 6 / 7


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


def synth_clip(base, direction: str, n: int) -> list[np.ndarray]:
    """Content translated horizontally at a fixed number of pixels per frame."""
    import cv2

    h, w = base.shape[:2]
    sign = 1.0 if direction == "right" else -1.0
    frames = []
    for k in range(n):
        M = np.float32([[1, 0, sign * abs(SYNTH_PX_PER_FRAME) * k * (w / ENCODE[0])],
                        [0, 1, 0]])
        bgr = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        frames.append(bgr)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01z2_polarity")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    scored = json.loads(SCORED.read_text(encoding="utf-8"))
    usable = [t for t in scored["turns"]
              if t["agrees"] is True and t["coherence"] >= 0.40]
    left = [t for t in usable if t["marked_camera"] == "LEFT"]
    right = [t for t in usable if t["marked_camera"] == "RIGHT"]

    print("P0.1Z2 — polarity fixed on synthetic, tested on real turns")
    print(f"  stage 1 polarity: synthetic clips from one real frame")
    print(f"  stage 2 test    : {len(left)} camera-LEFT and {len(right)} camera-RIGHT "
          f"hand-marked turns\n")

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

    def rates_for_frames(frames: list[np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
        import cv2

        enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain,
                                 n_azimuth_bins=128, flow_config=flow_config(MODE))
        prepared = []
        for bgr in frames:
            eye, inject, _m = enc.encode_frame(bgr)
            prepared.append({"eye": eye, "inject": inject})
        span = max(len(prepared), 1) * dt
        counts = np.zeros((SEEDS, len(all_idx)), dtype=np.float64)
        for si, seed in enumerate(range(1, SEEDS + 1)):
            engine.reset(seed=seed)
            c = np.zeros(len(all_idx), dtype=np.int64)
            for i, fr in enumerate(prepared):
                res = engine.step(i * dt, eye_drive=fr["eye"], inject=fr["inject"])
                f = np.asarray(res.fired)
                if not len(f):
                    continue
                s = union_slot[f]
                s = s[s >= 0]
                if len(s):
                    np.add.at(c, s, 1)
            counts[si] = c / span
        return {k: counts[:, local_pos[k]].sum(axis=1) for k in keys}

    # ------------------------------------------------------------ stage 1
    import cv2
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_MSEC, SYNTH_BASE_S * 1000)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read the synthetic base frame")

    n_synth = 90
    print("  stage 1: synthetic clips")
    r_right = rates_for_frames(synth_clip(base, "right", n_synth))
    print("    content-right clip done")
    r_left = rates_for_frames(synth_clip(base, "left", n_synth))
    print("    content-left clip done")
    polarity = {k: float(np.median(r_right[k]) - np.median(r_left[k])) for k in keys}
    p_sign = {k: (1 if polarity[k] >= 0 else -1) for k in keys}
    n_determined = sum(1 for k in keys if abs(polarity[k]) > 1e-9)
    print(f"    polarity determined for {n_determined}/{len(keys)} cells")

    # ------------------------------------------------------------ stage 2
    print("\n  stage 2: real hand-marked turns")
    events = [(t["id"], float(t["t0"]), float(t["t1"]), t["marked_camera"]) for t in usable]

    def cap_real(t0: float, t1: float, mirror: bool):
        enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain,
                                 n_azimuth_bins=128, flow_config=flow_config(MODE))
        prepared = []
        for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, dt):
            if mirror:
                frame = np.ascontiguousarray(frame[:, ::-1])
            eye, inject, _m = enc.encode_frame(frame)
            prepared.append({"t": t, "eye": eye, "inject": inject})
        span = max(len(prepared), 1) * dt
        counts = np.zeros((SEEDS, len(all_idx)), dtype=np.float64)
        for si, seed in enumerate(range(1, SEEDS + 1)):
            engine.reset(seed=seed)
            c = np.zeros(len(all_idx), dtype=np.int64)
            for fr in prepared:
                res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
                f = np.asarray(res.fired)
                if not len(f):
                    continue
                s = union_slot[f]
                s = s[s >= 0]
                if len(s):
                    np.add.at(c, s, 1)
            counts[si] = c / span
        return {k: counts[:, local_pos[k]].sum(axis=1) for k in keys}

    c_by: dict[str, dict[tuple[str, str], float]] = {}
    for eid, t0, t1, cam in events:
        r_orig = cap_real(t0, t1, False)
        r_mirr = cap_real(t0, t1, True)
        c_by[eid] = {k: float(np.median(r_orig[k]) - np.median(r_mirr[k])) for k in keys}
        print(f"    {eid:9s} t={t0:.1f}-{t1:.1f} ({cam:5s}) done")

    ev_ids = [e[0] for e in events]
    ev_dir = {e[0]: e[3] for e in events}

    rows = []
    for k in keys:
        p = p_sign[k]
        hits = 0
        detail = {}
        for eid in ev_ids:
            want = p if ev_dir[eid] == "LEFT" else -p
            got = np.sign(c_by[eid][k])
            hit = int(got == want)
            hits += hit
            detail[eid] = {"c": c_by[eid][k], "expected_sign": int(want),
                           "got_sign": int(got), "hit": bool(hit)}
        consistency = hits / len(ev_ids)
        rows.append({
            "cell_type": k[0], "side": k[1], "is_injected": k[0] in T4T5_TYPES,
            "synthetic_polarity": p, "polarity_magnitude": polarity[k],
            "hits": hits, "n_events": len(ev_ids), "consistency": consistency,
            "passes": bool(consistency >= MIN_CONSISTENCY), "per_event": detail,
        })

    rows.sort(key=lambda r: (-r["hits"], r["cell_type"]))
    print(f"\n  {'type':10s} {'sd':2s} {'inj':3s} {'pol':>4s} {'hits':>6s} {'consist':>8s} "
          f"{'=>':>5s}   per-event")
    for r in rows:
        marks = "".join("Y" if r["per_event"][e]["hit"] else "." for e in ev_ids)
        print(f"  {r['cell_type'][:10]:10s} {r['side']:2s} "
              f"{'yes' if r['is_injected'] else '-':3s} {r['synthetic_polarity']:+4d} "
              f"{r['hits']}/{r['n_events']:<4d} {r['consistency']:8.2f} "
              f"{'PASS' if r['passes'] else '.':>5s}   {marks}")

    passing = [r for r in rows if r["passes"]]
    passing_dn = [r for r in passing if not r["is_injected"]]
    n_dn = sum(1 for r in rows if not r["is_injected"])

    # ------------------------------------------- permutation, polarity fixed
    rng = np.random.default_rng(0)
    n_perm = 5000
    dirs = [ev_dir[e] for e in ev_ids]
    null = []
    for _ in range(n_perm):
        perm = list(rng.permutation(dirs))
        cnt = 0
        for r in rows:
            p = r["synthetic_polarity"]
            hits = 0
            for i, eid in enumerate(ev_ids):
                want = p if perm[i] == "LEFT" else -p
                hits += int(np.sign(r["per_event"][eid]["c"]) == want)
            if hits / len(ev_ids) >= MIN_CONSISTENCY:
                cnt += 1
        null.append(cnt)
    null = np.asarray(null)
    p_value = float(np.mean(null >= len(passing)))
    print(f"\n  permutation null (camera labels shuffled, polarity fixed):")
    print(f"    mean {null.mean():.2f}, p95 {np.percentile(null, 95):.0f}, "
          f"observed {len(passing)}, p = {p_value:.4f}")

    # ------------------------------------------------ how far from perfect?
    print(f"\n  event-level accuracy, pooled over cells:")
    for eid in ev_ids:
        hits = sum(1 for r in rows if r["per_event"][eid]["hit"])
        print(f"    {eid:9s} t={dict((e[0], e[1]) for e in events)[eid]:6.1f} "
              f"({ev_dir[eid]:5s})  {hits}/{len(rows)} cells hit")

    verdict = (
        f"{len(passing)}/{len(rows)} visual types ({len(passing_dn)}/{n_dn} downstream of T4/T5) "
        f"have a sign of mirror contrast that matches the camera direction on at least "
        f"{MIN_CONSISTENCY:.0%} of {len(ev_ids)} hand-marked turns, using a polarity fixed on "
        f"synthetic stimuli and never fitted to these events (permutation p = {p_value:.4f})."
        if passing and p_value < 0.05 else
        f"No visual type predicts the camera direction of the hand-marked turns from a polarity "
        f"fixed on synthetic stimuli: {len(passing)}/{len(rows)} reach "
        f"{MIN_CONSISTENCY:.0%} agreement and the permutation null gives p = {p_value:.4f}. "
        "With this many events the polarity cannot be established, so this run neither "
        "supports nor refutes direction selectivity."
    )
    print(f"\n=== VERDICT ===\n{verdict}")

    (out / "report.json").write_text(json.dumps({
        "probe": "P0.1Z2 — polarity fixed on synthetic, tested on real turns",
        "fix": "P0.1Z fitted the polarity to the same events it tested, so it had no power; "
               "its permutation p was 0.121",
        "stage1_synthetic": {"base_frame_s": SYNTH_BASE_S,
                             "px_per_frame": SYNTH_PX_PER_FRAME, "frames": n_synth},
        "events": [{"id": e[0], "t0": e[1], "t1": e[2], "camera": e[3]} for e in events],
        "criterion": {"min_consistency": MIN_CONSISTENCY},
        "n_types": len(rows), "n_passing": len(passing),
        "n_passing_downstream": len(passing_dn),
        "permutation": {"n_perm": n_perm, "mean": float(null.mean()),
                        "p95": float(np.percentile(null, 95)),
                        "observed": len(passing), "p_value": p_value},
        "per_type": rows, "verdict": verdict,
    }, indent=2), encoding="utf-8")

    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_type", "side", "is_injected", "synthetic_polarity",
                    "hits", "n_events", "consistency", "passes"])
        for r in rows:
            w.writerow([r["cell_type"], r["side"], r["is_injected"], r["synthetic_polarity"],
                        r["hits"], r["n_events"], f"{r['consistency']:.3f}", r["passes"]])
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
