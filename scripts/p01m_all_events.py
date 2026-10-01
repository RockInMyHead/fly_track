#!/usr/bin/env python3
"""
P0.1M — the 20 frozen visual cells on ALL seven real events.

P0.1L found 20 (cell_type, side) pairs that satisfy both a mirror-flip test and a
magnitude-level mirror-equivalence test on three calibration events. That set is
now FROZEN. This test does not re-select anything: it takes those 20 and asks one
question on the full event set:

    do they read the direction of a real turn, or did they only work on the
    two favourable examples?

Events (blind labels, unchanged):
    LEFT   left_65, left_81, left_108
    RIGHT  right_67, right_210, right_216
    FWD    fwd_140

Method
------
Each type carries a polarity frozen from P0.1L (which physical direction raises
its rate). Within each run (original / mirrored) rates are z-scored across events,
so the run's overall activity level drops out and the two runs are comparable.

    score(type, event, run) = polarity * z(run)[event]
    expected(event, run)    = label(event)          for the original run
                            = -label(event)         for the mirrored run
                              (mirroring reverses the physical direction)

A type is correct on an event when sign(score) == expected. FWD is judged
separately: its |z| should be small.

Nothing is trained or tuned. No trajectory, no descending neurons.

Usage:
    PYTHONPATH=. python scripts/p01m_all_events.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.visual_encoder import VideoVisualEncoder

# label: +1 LEFT, -1 RIGHT, 0 FWD. Blind labels, unchanged.
EVENTS = {
    "left_65":   (63.5, +1),
    "left_81":   (78.5, +1),
    "left_108":  (106.0, +1),
    "right_67":  (66.0, -1),
    "right_210": (209.0, -1),
    "right_216": (214.0, -1),
    "fwd_140":   (137.5, 0),
}
LEFT_EVENTS = [k for k, v in EVENTS.items() if v[1] == +1]
RIGHT_EVENTS = [k for k, v in EVENTS.items() if v[1] == -1]
FWD_EVENTS = [k for k, v in EVENTS.items() if v[1] == 0]

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
FWD_NEUTRAL_Z = 0.50      # |z| below this counts as "near zero" on the forward event

P01L_REPORT = ROOT / "output/p01l_real_video/report.json"


def load_frozen_cells() -> list[dict]:
    rep = json.loads(P01L_REPORT.read_text(encoding="utf-8"))
    frozen = [
        r for r in rep["all_types"] if r["passes"] and r["mirror_equivalence"]
    ]
    if not frozen:
        raise SystemExit(f"{P01L_REPORT} has no cells passing both criteria")
    return frozen


def capture(video: Path, start: float, cfg: FlyVOConfig, mirror: bool) -> dict:
    t0 = max(0.0, start - PRE_ROLL_S)
    t1 = start + EVENT_DUR_S
    encoder = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain)
    frames = []
    for t, frame in iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt):
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        eye, inject, _m = encoder.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject})
    n_pre = min(int(round(PRE_ROLL_S / cfg.brain_dt)), max(0, len(frames) - 1))
    return {"frames": frames, "n_pre": n_pre}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("-o", "--output", default="output/p01m_all_events")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    seeds = list(range(1, args.seeds + 1))
    frozen = load_frozen_cells()
    print(f"P0.1M — frozen cells on all seven events")
    print(f"  frozen cells (from P0.1L, not re-selected): {len(frozen)}")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain

    keys = [(r["cell_type"], r["side"]) for r in frozen]
    # polarity frozen from P0.1L: +1 if the LEFT calibration event was higher
    polarity = {
        (r["cell_type"], r["side"]): (1 if r["orig_dir_index"] > 0 else -1)
        for r in frozen
    }

    groups = {k: np.asarray(brain.cells([k[0]], side=k[1]), dtype=np.int64) for k in keys}
    all_idx = np.unique(np.concatenate([v for v in groups.values() if len(v)]))
    union_slot = np.full(int(brain.n), -1, dtype=np.int64)
    union_slot[all_idx] = np.arange(len(all_idx), dtype=np.int64)
    local_pos = {k: union_slot[v] for k, v in groups.items()}

    # ---- run everything ----
    # rates[(run, event)][key] -> [seeds] mean firing rate Hz
    raw: dict[tuple[str, str], dict] = {}
    for run in ("orig", "mirror"):
        for eid, (_start, _lab) in ((e, v) for e, v in EVENTS.items()):
            start = EVENTS[eid][0]
            cap = capture(video, start, cfg, run == "mirror")
            lo, hi = cap["n_pre"], len(cap["frames"])
            span = max(hi - lo, 1) * dt
            counts = np.zeros((len(seeds), len(all_idx)), dtype=np.float64)
            for si, seed in enumerate(seeds):
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
            raw[(run, eid)] = {k: counts[:, local_pos[k]].sum(axis=1) for k in keys}
            print(f"  {run:6s} {eid:10s} span={span:.2f}s")

    # ---- z-score within each run, across events ----
    def zscores(run: str) -> dict[tuple[str, str], np.ndarray]:
        out_z: dict[tuple[str, str], np.ndarray] = {}
        for k in keys:
            # median across the seven events, per seed
            mat = np.stack([raw[(run, eid)][k] for eid in EVENTS], axis=1)  # [seeds, events]
            med = np.median(mat, axis=1, keepdims=True)
            mad = np.median(np.abs(mat - med), axis=1, keepdims=True)
            scale = 1.4826 * mad
            scale[scale < 1e-9] = 1e-9
            out_z[k] = (mat - med) / scale
        return out_z

    z = {run: zscores(run) for run in ("orig", "mirror")}
    event_index = {e: i for i, e in enumerate(EVENTS)}

    # ---- per-type per-event correctness ----
    type_rows = []
    cell_events: list[dict] = []
    for k in keys:
        ctype, side = k
        pol = polarity[k]
        # per-event aggregate score = median over seeds of polarity * z
        scores: dict[tuple[str, str], float] = {}
        for run in ("orig", "mirror"):
            for eid in EVENTS:
                s = pol * z[run][k][:, event_index[eid]]
                scores[(run, eid)] = float(np.median(s))

        def correct(run: str, eid: str) -> bool:
            lab = EVENTS[eid][1]
            exp = lab if run == "orig" else -lab
            if exp == 0:
                return abs(scores[(run, eid)]) < FWD_NEUTRAL_Z
            return np.sign(scores[(run, eid)]) == exp

        left_ok = sum(correct("orig", e) for e in LEFT_EVENTS)
        right_ok = sum(correct("orig", e) for e in RIGHT_EVENTS)
        fwd_ok = sum(correct("orig", e) for e in FWD_EVENTS)
        mirror_ok = sum(
            correct("mirror", e) for e in LEFT_EVENTS + RIGHT_EVENTS
        )
        mirror_n = len(LEFT_EVENTS) + len(RIGHT_EVENTS)

        for eid in EVENTS:
            for run in ("orig", "mirror"):
                lab = EVENTS[eid][1]
                exp = lab if run == "orig" else -lab
                sc = scores[(run, eid)]
                cell_events.append({
                    "cell_type": ctype,
                    "side": side,
                    "polarity": pol,
                    "run": run,
                    "event": eid,
                    "label": lab,
                    "expected": exp,
                    "score": sc,
                    "correct": bool(correct(run, eid)),
                    "rate": float(np.median(raw[(run, eid)][k])),
                })

        type_rows.append({
            "cell_type": ctype,
            "side": side,
            "polarity": pol,
            "left_correct": left_ok,
            "left_n": len(LEFT_EVENTS),
            "right_correct": right_ok,
            "right_n": len(RIGHT_EVENTS),
            "fwd_neutral_ok": fwd_ok,
            "fwd_n": len(FWD_EVENTS),
            "turn_correct": left_ok + right_ok,
            "turn_n": len(LEFT_EVENTS) + len(RIGHT_EVENTS),
            "mirror_correct": mirror_ok,
            "mirror_n": mirror_n,
            "scores": {f"{run}:{e}": scores[(run, e)] for run in ("orig", "mirror") for e in EVENTS},
        })

    type_rows.sort(key=lambda r: (-r["turn_correct"], -r["mirror_correct"]))
    n_types = len(type_rows)

    print(f"\n=== Per-type accuracy (original run: does the sign match the label?) ===")
    print(f"{'type':22s} {'pol':>4s} {'LEFT':>6s} {'RIGHT':>6s} {'FWD ok':>7s} {'turn':>6s} {'mirror':>7s}")
    for r in type_rows:
        print(
            f"{r['cell_type'][:22]:22s} {r['polarity']:+4d} "
            f"{r['left_correct']}/{r['left_n']:<4d} {r['right_correct']}/{r['right_n']:<4d} "
            f"{'OK' if r['fwd_neutral_ok'] else 'no':>7s} "
            f"{r['turn_correct']}/{r['turn_n']:<4d} "
            f"{r['mirror_correct']}/{r['mirror_n']:<4d}"
        )

    # ---- per-type fired polarity consistency across events ----
    # Pooled view: for each direction, how many of the 6 turn events land on the
    # expected side, aggregated over all 20 types.
    agg_rows = []
    for eid in EVENTS:
        lab = EVENTS[eid][1]
        if lab == 0:
            continue
        ok = sum(1 for r in type_rows if (r["scores"][f"orig:{eid}"] > 0) == (lab > 0))
        agg_rows.append({"event": eid, "label": lab, "types_correct": ok, "n_types": n_types})
    agg_rows.sort(key=lambda r: r["event"])
    print(f"\n=== Pooled vote across the {n_types} frozen types ===")
    for r in agg_rows:
        print(
            f"  {r['event']:10s} label={'L' if r['label'] > 0 else 'R'}  "
            f"{r['types_correct']}/{r['n_types']} types agree "
            f"({r['types_correct'] / r['n_types']:.0%})"
        )
    pooled = sum(r["types_correct"] for r in agg_rows)
    pooled_n = sum(r["n_types"] for r in agg_rows)

    # ---- permutation null on the pooled vote ----
    rng = np.random.default_rng(0)
    n_perm = 2000
    observed_types_pass = sum(
        1 for r in type_rows if r["turn_correct"] == r["turn_n"]
    )
    null_counts = []
    for _ in range(n_perm):
        cnt = 0
        for k in keys:
            pol = polarity[k]
            signs = []
            for eid in EVENTS:
                lab = EVENTS[eid][1]
                if lab == 0:
                    continue
                signs.append((pol * z["orig"][k][:, event_index[eid]]).mean() > 0 == (lab > 0))
            if all(signs):
                cnt += 1
        null_counts.append(cnt)
    null_counts = np.asarray(null_counts)
    p_all = float(np.mean(null_counts >= observed_types_pass))

    print(f"\n=== Types correct on ALL six turns (3 LEFT + 3 RIGHT) ===")
    print(f"  observed: {observed_types_pass}/{n_types}")
    print(
        f"  permutation null: mean {null_counts.mean():.2f} "
        f"(95th pct {np.percentile(null_counts, 95):.0f}), p = {p_all:.4f}"
    )
    print(
        f"\n=== Pooled vote ==="
        f"\n  {pooled}/{pooled_n} = {pooled / pooled_n:.1%} of type-event decisions correct"
    )

    with (out / "type_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[c for c in type_rows[0] if c != "scores"]
            + [f"scores_{k}" for k in type_rows[0]["scores"]],
        )
        w.writeheader()
        for r in type_rows:
            row = {k: v for k, v in r.items() if k != "scores"}
            row.update({f"scores_{k}": v for k, v in r["scores"].items()})
            w.writerow(row)
    with (out / "cell_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_events[0].keys()))
        w.writeheader()
        w.writerows(cell_events)

    left_types = sum(1 for r in type_rows if r["left_correct"] == r["left_n"])
    right_types = sum(1 for r in type_rows if r["right_correct"] == r["right_n"])
    verdict = (
        f"REAL YAW READOUT HOLDS: {observed_types_pass}/{n_types} frozen types get all six "
        f"real turns right (permutation p={p_all:.4f}); pooled agreement {pooled / pooled_n:.0%}"
        if observed_types_pass >= 3
        else
        f"NOT CONFIRMED: only {observed_types_pass}/{n_types} frozen types get all six real turns "
        f"right (permutation p={p_all:.4f}). LEFT-perfect types {left_types}, "
        f"RIGHT-perfect types {right_types}. The P0.1L result does not generalise to the "
        "full event set."
    )

    report = {
        "probe": "P0.1M — frozen visual cells on all seven real events",
        "frozen": True,
        "cell_selection_source": str(P01L_REPORT),
        "n_types": n_types,
        "events": {k: {"start": v[0], "label": v[1]} for k, v in EVENTS.items()},
        "seeds": seeds,
        "criterion": {
            "score": "polarity * z(run)[event], z across the seven events within each run",
            "expected": "label for the original run, -label for the mirrored run",
            "fwd_neutral_z": FWD_NEUTRAL_Z,
        },
        "per_type": type_rows,
        "pooled_vote": agg_rows,
        "pooled_correct": pooled,
        "pooled_total": pooled_n,
        "types_all_turns_correct": observed_types_pass,
        "permutation": {
            "n_perm": n_perm,
            "mean": float(null_counts.mean()),
            "p95": float(np.percentile(null_counts, 95)),
            "observed": observed_types_pass,
            "p_value": p_all,
        },
        "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")
    raise SystemExit(0 if observed_types_pass >= 3 else 1)


if __name__ == "__main__":
    main()
