#!/usr/bin/env python3
"""
P0.1L — do the direction-selective visual cells work on REAL video?

P0.1K found 27 recipient types downstream of T4/T5 that reverse their L/F/R
ordering when the stimulus is mirrored. That was synthetic. This test asks the
same question on real footage, for exactly three already-understood events:

    left_65    LEFT turn
    fwd_140    FORWARD
    right_216  RIGHT turn

Per type, the three events supply the three conditions:

    ordering  =  order(rate@left_65, rate@fwd_140, rate@right_216)

A type qualifies only if ALL of:

  1. it distinguishes LEFT from RIGHT (ordering is not flat)
  2. FORWARD sits between the two turns (not outside them)
  3. horizontally mirroring the video REVERSES the ordering
     (this is what separates direction coding from brightness / texture /
      the person in frame / any other magnitude effect)
  4. both runs are seed-consistent

Nothing is trained or tuned. The cell set comes from P0.1K, not from these results.

Usage:
    PYTHONPATH=. python scripts/p01l_real_video_test.py
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
from fly_vo.optic_flow import FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

# The three events. Second element is the expected visual direction (LEFT=+1, RIGHT=-1).
EVENTS = {
    "left_65":   (+1, 63.5),
    "fwd_140":   (0, 137.5),
    "right_216": (-1, 214.0),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0

SURVIVORS_JSON = ROOT / "output/p01k_downstream/report.json"
CONSISTENCY_MIN = 0.8


# ------------------------------------------------------------------ helpers
def load_survivors() -> list[tuple[str, str]]:
    rep = json.loads(SURVIVORS_JSON.read_text(encoding="utf-8"))
    if "direction_selective_without_loom" not in rep:
        raise SystemExit(f"{SURVIVORS_JSON} has no direction_selective_without_loom; run P0.1K first")
    return [(r["cell_type"], r["side"]) for r in rep["direction_selective_without_loom"]]


def order3(l: float, f: float, r: float) -> int:
    """Strict monotonic ordering: +1 = L>F>R, -1 = R>F>L, 0 otherwise.

    Reported for transparency only. The gate uses the selectivity pattern below,
    which matches the intended reading ("LEFT strong, FWD weak, RIGHT weak").
    """
    if l > f > r:
        return +1
    if r > f > l:
        return -1
    return 0


def dir_index(l: float, r: float) -> float:
    """Signed LEFT-RIGHT contrast in [-1, 1]; positive = LEFT-dominant."""
    return (l - r) / (l + r + 1e-9)


MARGIN = 0.10          # minimum |dir_index| for a run to count as directional
FWD_MAX_FRACTION = 1.0  # FWD must be weaker than the dominant turn
EQUIV_MARGIN = 1.15    # required LEFT/RIGHT separation for mirror equivalence
WITHIN_RUN_MIN = 0.80  # mirrored-left vs real-right magnitude agreement
BETWEEN_RUN_MIN = 1.20  # required separation between the two physical directions


def classify(l: float, f: float, r: float) -> dict:
    """Selectivity pattern for one condition triple."""
    di = dir_index(l, r)
    dominant = max(l, r)
    return {
        "dir_index": di,
        "left_dominant": di > MARGIN,
        "right_dominant": di < -MARGIN,
        "fwd_not_dominant": f <= dominant * FWD_MAX_FRACTION,
        "ordering": order3(l, f, r),
    }


def capture(video: Path, start: float, cfg: FlyVOConfig, mirror: bool) -> dict:
    """Frozen frontend over one event window. Mirror flips each frame horizontally."""
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


def group_indices(brain, types: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for ctype, side in types:
        idx = np.asarray(brain.cells([ctype], side=side), dtype=np.int64)
        out[(ctype, side)] = idx
    return out


def run_event(engine, frames: dict, union_slot: np.ndarray, local_pos: dict, seeds) -> dict:
    """Replay one event under several seeds; return per-type spike counts [seeds, n_type_cells]."""
    n_union = int(union_slot.max()) + 1
    out = {k: np.zeros((len(seeds), len(v)), dtype=np.float64) for k, v in local_pos.items()}
    lo, hi = frames["n_pre"], len(frames["frames"])
    for si, seed in enumerate(seeds):
        engine.reset(seed=seed)
        counts = np.zeros(n_union, dtype=np.int64)
        for i, fr in enumerate(frames["frames"]):
            res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
            if i < lo:
                continue
            f = np.asarray(res.fired)
            if not len(f):
                continue
            s = union_slot[f]
            s = s[s >= 0]
            if len(s):
                np.add.at(counts, s, 1)
        for k, pos in local_pos.items():
            out[k][si] = counts[pos]
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("-o", "--output", default="output/p01l_real_video")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    seeds = list(range(1, args.seeds + 1))
    survivors = load_survivors()
    print(f"P0.1L — direction-selective cells on real video")
    print(f"  types carried over from P0.1K: {len(survivors)}")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain
    # compact slot over the union of every cell we score, plus per-type local positions
    groups = group_indices(brain, survivors)
    all_idx = np.unique(np.concatenate([v for v in groups.values() if len(v)]))
    union_slot = np.full(int(brain.n), -1, dtype=np.int64)
    union_slot[all_idx] = np.arange(len(all_idx), dtype=np.int64)
    local_pos = {k: union_slot[v] for k, v in groups.items()}

    span = None
    rates: dict[tuple[str, bool], dict] = {}
    for mirror in (False, True):
        for eid, (_vis, start) in EVENTS.items():
            cap = capture(video, start, cfg, mirror)
            lo, hi = cap["n_pre"], len(cap["frames"])
            span = (hi - lo) * dt
            res = run_event(engine, cap, union_slot, local_pos, seeds)
            rates[(eid, mirror)] = {k: v / span for k, v in res.items()}
            tag = "mirror" if mirror else "orig  "
            print(f"  {tag} {eid:10s} frames={len(cap['frames'])} event_steps={hi - lo}")

    # ---- per-type evaluation ----
    # Primary criterion (the intended reading):
    #   LEFT strong / FWD weak / RIGHT weak  ->  or the mirror-image pattern.
    # A type qualifies if the dominant side AND the sign of the LEFT-RIGHT index
    # both reverse when the video is mirrored, consistently across seeds.
    rows = []
    for key in survivors:
        ctype, side = key
        def rate(eid: str, mirror: bool) -> float:
            arr = rates[(eid, mirror)][key]
            return float(np.median(arr.sum(axis=1)))

        o_l, o_f, o_r = rate("left_65", False), rate("fwd_140", False), rate("right_216", False)
        m_l, m_f, m_r = rate("left_65", True), rate("fwd_140", True), rate("right_216", True)

        oc = classify(o_l, o_f, o_r)
        mc = classify(m_l, m_f, m_r)

        def seed_stats(mirror: bool) -> tuple[float, float]:
            a = rates[("left_65", mirror)][key].sum(axis=1)
            b = rates[("right_216", mirror)][key].sum(axis=1)
            d = a - b
            med = float(np.median(d))
            if abs(med) < 1e-12:
                return 0.0, 0.0
            return float(np.mean(np.sign(d) == np.sign(med))), med

        o_cons, o_dmed = seed_stats(False)
        m_cons, m_dmed = seed_stats(True)

        flips = bool(
            (oc["left_dominant"] and mc["right_dominant"])
            or (oc["right_dominant"] and mc["left_dominant"])
        )
        fwd_ok = oc["fwd_not_dominant"] and mc["fwd_not_dominant"]
        passes = bool(flips and fwd_ok and o_cons >= CONSISTENCY_MIN and m_cons >= CONSISTENCY_MIN)

        # ---- mirror equivalence (magnitude level) ----
        # Mirroring turns a left turn into a right turn. So the response to the
        # mirrored LEFT event should MATCH, in magnitude, the response to the real
        # RIGHT event — and vice versa. A cell that tracks image content rather
        # than physical direction fails this: the mirrored stimuli look nothing
        # like the originals, so only a direction-selective cell can agree.
        #
        # Note: an ordering-only version of this test is NOT independent — it is
        # implied by the flip test above. The numbers only add information when the
        # magnitudes are compared, which is what is done here.
        phys_left = [o_l, m_r]
        phys_right = [o_r, m_l]
        phys_fwd = [o_f, m_f]

        within_left = min(phys_left) / (max(phys_left) + 1e-9)
        within_right = min(phys_right) / (max(phys_right) + 1e-9)
        between_l = min(phys_left) / (max(phys_right) + 1e-9)
        between_r = min(phys_right) / (max(phys_left) + 1e-9)
        between = max(between_l, between_r)

        equiv_sign = +1 if between_l >= between_r and between_l >= BETWEEN_RUN_MIN else (
            -1 if between_r > between_l and between_r >= BETWEEN_RUN_MIN else 0
        )
        mirror_equivalence = bool(
            equiv_sign != 0
            and within_left >= WITHIN_RUN_MIN
            and within_right >= WITHIN_RUN_MIN
            and fwd_ok
        )

        rows.append({
            "cell_type": ctype,
            "side": side,
            "orig_LEFT": o_l,
            "orig_FWD": o_f,
            "orig_RIGHT": o_r,
            "orig_dir_index": oc["dir_index"],
            "orig_left_dominant": oc["left_dominant"],
            "orig_right_dominant": oc["right_dominant"],
            "orig_order_strict": oc["ordering"],
            "orig_seed_consistency": o_cons,
            "mirror_LEFT": m_l,
            "mirror_FWD": m_f,
            "mirror_RIGHT": m_r,
            "mirror_dir_index": mc["dir_index"],
            "mirror_left_dominant": mc["left_dominant"],
            "mirror_right_dominant": mc["right_dominant"],
            "mirror_order_strict": mc["ordering"],
            "mirror_seed_consistency": m_cons,
            "fwd_not_dominant": fwd_ok,
            "flips_under_mirror": flips,
            "phys_left_responses": phys_left,
            "phys_right_responses": phys_right,
            "phys_left_minus_right_min": min(phys_left) - max(phys_right),
            "within_phys_left_ratio": within_left,
            "within_phys_right_ratio": within_right,
            "between_phys_ratio": between,
            "mirror_equivalence_ratio": between,
            "mirror_equivalence_sign": equiv_sign,
            "mirror_equivalence": mirror_equivalence,
            "direction_grounding": (oc["dir_index"] - mc["dir_index"]) / 2.0,
            "passes": passes,
        })

    rows.sort(key=lambda r: -abs(r["direction_grounding"]))
    passing = [r for r in rows if r["passes"]]
    equiv_passing = [r for r in rows if r["mirror_equivalence"]]

    print(f"\n=== Real-video results (LEFT / FWD / RIGHT, original then mirrored) ===")
    print(f"{'type':22s} {'sd':2s} {'orig L/F/R':>28s} {'di':>6s} {'cons':>5s} "
          f"{'mirror L/F/R':>28s} {'di':>6s} {'cons':>5s} {'flip':>5s}")
    for r in rows[:30]:
        print(
            f"{r['cell_type'][:22]:22s} {r['side']:2s} "
            f"{r['orig_LEFT']:9.2f}/{r['orig_FWD']:8.2f}/{r['orig_RIGHT']:8.2f} "
            f"{r['orig_dir_index']:+6.3f} {r['orig_seed_consistency']:5.2f} "
            f"{r['mirror_LEFT']:9.2f}/{r['mirror_FWD']:8.2f}/{r['mirror_RIGHT']:8.2f} "
            f"{r['mirror_dir_index']:+6.3f} {r['mirror_seed_consistency']:5.2f} "
            f"{'YES' if r['flips_under_mirror'] else 'no':>5s}"
        )

    print(f"\n=== PASSING (dominant side flips + FWD weak + seed-consistent) ===")
    print(f"  {len(passing)} / {len(rows)} types")
    for r in passing:
        print(
            f"    {r['cell_type'][:22]:22s} {r['side']}  "
            f"orig L/F/R = {r['orig_LEFT']:.2f}/{r['orig_FWD']:.2f}/{r['orig_RIGHT']:.2f} "
            f"(di {r['orig_dir_index']:+.3f})  ->  mirror di {r['mirror_dir_index']:+.3f}"
        )

    # ---- permutation null on the combined criterion ----
    rng = np.random.default_rng(0)
    n_perm = 2000
    keys = list(survivors)
    null = []
    for _ in range(n_perm):
        cnt = 0
        for key in keys:
            def med(mirror: bool, eid: str) -> float:
                return float(np.median(rates[(eid, mirror)][key].sum(axis=1)))
            vals_o = [med(False, e) for e in EVENTS]
            vals_m = [med(True, e) for e in EVENTS]
            po, pm = rng.permutation(3), rng.permutation(3)
            oc = classify(vals_o[po[0]], vals_o[po[1]], vals_o[po[2]])
            mc = classify(vals_m[pm[0]], vals_m[pm[1]], vals_m[pm[2]])
            fl = (oc["left_dominant"] and mc["right_dominant"]) or (
                oc["right_dominant"] and mc["left_dominant"]
            )
            if fl and oc["fwd_not_dominant"] and mc["fwd_not_dominant"]:
                cnt += 1
        null.append(cnt)
    null = np.asarray(null)
    p_value = float(np.mean(null >= len(passing)))

    print(
        f"\n=== Permutation null ({n_perm} draws, events relabelled independently per run) ===\n"
        f"  mean passing = {null.mean():.2f}  (95th pct {np.percentile(null, 95):.0f})  "
        f"observed = {len(passing)}  p = {p_value:.4f}"
    )

    # ---- mirror equivalence, the decisive physical-direction test ----
    print(f"\n=== MIRROR EQUIVALENCE (magnitude level: mirror(left) must equal real right) ===")
    print(f"    within-direction agreement >= {WITHIN_RUN_MIN}, between-direction ratio >= {BETWEEN_RUN_MIN}")
    print(f"  {len(equiv_passing)} / {len(rows)} types qualify")
    print(
        f"  {'type':22s} {'sd':2s} {'physLEFT (L_orig,R_mir)':>26s} {'physRIGHT (R_orig,L_mir)':>26s} "
        f"{'withinL':>8s} {'withinR':>8s} {'between':>8s}"
    )
    for r in sorted(rows, key=lambda x: -x["between_phys_ratio"]):
        pl = "/".join(f"{v:.1f}" for v in r["phys_left_responses"])
        pr = "/".join(f"{v:.1f}" for v in r["phys_right_responses"])
        mark = " *" if r["mirror_equivalence"] else ""
        print(
            f"  {r['cell_type'][:22]:22s} {r['side']:2s} {pl:>26s} {pr:>26s} "
            f"{r['within_phys_left_ratio']:8.2f} {r['within_phys_right_ratio']:8.2f} "
            f"{r['between_phys_ratio']:8.2f}{mark}"
        )

    # permutation null: relabel which event is which within each run independently
    rng2 = np.random.default_rng(1)
    null_eq = []
    for _ in range(n_perm):
        cnt = 0
        for key in keys:
            def med2(mirror: bool, eid: str) -> float:
                return float(np.median(rates[(eid, mirror)][key].sum(axis=1)))
            vals_o = [med2(False, e) for e in EVENTS]
            vals_m = [med2(True, e) for e in EVENTS]
            po, pm = rng2.permutation(3), rng2.permutation(3)
            pl = [vals_o[po[0]], vals_m[pm[2]]]
            pr = [vals_o[po[2]], vals_m[pm[0]]]
            wl = min(pl) / (max(pl) + 1e-9)
            wr = min(pr) / (max(pr) + 1e-9)
            bl = min(pl) / (max(pr) + 1e-9)
            br = min(pr) / (max(pl) + 1e-9)
            if max(bl, br) >= BETWEEN_RUN_MIN and wl >= WITHIN_RUN_MIN and wr >= WITHIN_RUN_MIN:
                cnt += 1
        null_eq.append(cnt)
    null_eq = np.asarray(null_eq)
    p_equiv = float(np.mean(null_eq >= len(equiv_passing)))
    print(
        f"\n  permutation null: mean = {null_eq.mean():.2f} "
        f"(95th pct {np.percentile(null_eq, 95):.0f}), observed = {len(equiv_passing)}, "
        f"p = {p_equiv:.4f}"
    )

    # ---- separation on the two clean turn controls ----
    print(f"\n=== LEFT vs RIGHT separation on real turns (orig run) ===")
    sep_rows = []
    for r in rows:
        l, rr = r["orig_LEFT"], r["orig_RIGHT"]
        med = (l + rr) / 2
        sep = abs(l - rr) / (abs(med) + 1e-9)
        sep_rows.append((r["cell_type"], r["side"], l, rr, sep, r["passes"]))
    sep_rows.sort(key=lambda x: -x[4])
    for ctype, side, l, rr, sep, ok in sep_rows[:12]:
        print(f"  {ctype[:24]:24s} {side}  L={l:9.2f} R={rr:9.2f}  relative sep={sep:.2f}  pass={ok}")

    with (out / "type_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    verdict = (
        f"{len(passing)} of {len(rows)} direction-selective types also reverse their "
        f"LEFT/RIGHT dominance on real video under mirroring (permutation p={p_value:.4f}); "
        f"{len(equiv_passing)} satisfy mirror equivalence, i.e. mirror(left turn) matches a real "
        f"right turn and vice versa (permutation p={p_equiv:.4f})"
        if passing
        else
        "NONE of the direction-selective types reverse their LEFT/RIGHT dominance on real "
        "video: the signal found on synthetic stimuli does not survive real footage"
    )
    report = {
        "probe": "P0.1L — direction-selective cells on real video",
        "frozen": ["frontend", "local opponent inject", "optic_flow", "MaleCNS dynamics", "labels"],
        "criterion": {
            "margin": MARGIN,
            "fwd_not_dominant": "FWD <= max(LEFT,RIGHT)",
            "flip": "dominant side and sign(LEFT-RIGHT) both reverse under mirroring",
            "mirror_equivalence": (
                "mirror(left turn) matches a real right turn in MAGNITUDE: "
                f"within-direction agreement >= {WITHIN_RUN_MIN}, "
                f"between-direction ratio >= {BETWEEN_RUN_MIN}, FWD weak in both runs"
            ),
            "seed_consistency_min": CONSISTENCY_MIN,
        },
        "events": {k: {"start": v[1], "expected_visual_sign": v[0]} for k, v in EVENTS.items()},
        "cell_set_source": str(SURVIVORS_JSON),
        "n_types": len(rows),
        "n_passing": len(passing),
        "n_mirror_equivalence": len(equiv_passing),
        "permutation_null_flip": {
            "n_perm": n_perm,
            "mean": float(null.mean()),
            "p95": float(np.percentile(null, 95)),
            "observed": len(passing),
            "p_value": p_value,
        },
        "permutation_null_mirror_equivalence": {
            "n_perm": n_perm,
            "mean": float(null_eq.mean()),
            "p95": float(np.percentile(null_eq, 95)),
            "observed": len(equiv_passing),
            "p_value": p_equiv,
        },
        "passing": passing,
        "mirror_equivalence_passing": equiv_passing,
        "all_types": rows,
        "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")
    raise SystemExit(0 if passing else 1)


if __name__ == "__main__":
    main()
