#!/usr/bin/env python3
"""
P0.1K — does direction appear immediately downstream of T4/T5?

P0.1J injected a correct local opponent motion code and it reached the brain
(DNp01 0.4 -> 8.8 Hz) but no yaw direction survived to the descending neurons.
That is the wrong place to look first: T4/T5 hand directed motion to the large
lobula-plate cells, which pool over a wide field — that is where a whole-eye
left/right signal is expected to appear.

This test therefore stops one layer earlier and asks a single question:

    T4a/T4b/T5a/T5b -> their DIRECT recipients
        LEFT  -> one sign
        FWD   -> near zero
        RIGHT -> opposite sign
        ?

Recipients are found from the connectome, not by name. No HS/VS/T4-type prior.

Frozen: frontend (local opponent inject), optic_flow, labels, MaleCNS dynamics.
No trajectory, no descending neurons, no tuning.

Usage:
    PYTHONPATH=. python scripts/p01k_downstream_test.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder
from scripts.visual_observability_test import _make_frames

SOURCE_TYPES = ("T4a", "T4b", "T5a", "T5b")   # horizontal motion detectors
CONDITIONS = {"YAW_LEFT": +1, "FORWARD": 0, "YAW_RIGHT": -1}

SYNTH_STEPS = 120
WARMUP = 30
MIN_CELLS_PER_TYPE = 3
MONOTONIC_TOL = 0.10      # relative slack for the FWD midpoint


# ------------------------------------------------------------------ recipients
def find_recipients(brain) -> dict:
    """Direct postsynaptic partners of the horizontal T4/T5 population."""
    ct = np.asarray(brain.cell_type)
    sc = np.asarray(brain.superclass)
    sd = np.asarray(brain.side)
    indptr, indices, w = brain.indptr, brain.indices, brain.weights

    sources = np.flatnonzero(np.isin(ct, SOURCE_TYPES))
    n_cells = collections.Counter()
    weight = collections.Counter()
    for j in sources.tolist():
        for e in range(indptr[j], indptr[j + 1]):
            i = int(indices[e])
            n_cells[i] += 1
            weight[i] += abs(float(w[e]))

    recv = np.array(sorted(n_cells), dtype=np.int64)
    return {
        "cells": recv,
        "cell_type": ct[recv],
        "side": sd[recv],
        "superclass": sc[recv],
        "n_synapses": np.array([n_cells[i] for i in recv]),
        "input_weight": np.array([weight[i] for i in recv]),
        "n_sources": len(sources),
    }


# ------------------------------------------------------------------ run
def run_conditions(
    seeds: list[int],
    steps: int,
    mirror: bool = False,
    disable_loom: bool = False,
) -> dict:
    cfg = FlyVOConfig(use_pathway_fallback=False)
    fc = FlowConfig(crop_mode="center_band", flow_inject_gain=8.0, ema_tau_s=0.5)
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=fc)
    engine._ensure_loaded()

    recip = find_recipients(engine.brain)
    recv_idx = recip["cells"]
    n_recv = len(recv_idx)
    slot = np.full(engine.brain.n, -1, dtype=np.int64)
    slot[recv_idx] = np.arange(n_recv, dtype=np.int64)

    # For the loom ablation we drop the LC4/LPLC2 entries from the inject list.
    loom_cells: set[int] = set()
    if disable_loom:
        for grp in encoder.loom_groups.values():
            loom_cells.update(np.asarray(grp).tolist())

    synth = _make_frames(n=steps)
    if mirror:
        # Horizontal flip reverses the motion direction while preserving every
        # magnitude statistic of the stimulus. A direction-selective cell must
        # therefore flip its ordering; a magnitude-driven cell must not.
        synth = {k: [np.ascontiguousarray(f[:, ::-1]) for f in v] for k, v in synth.items()}

    lo, hi = steps - WARMUP, steps
    span = (hi - lo) * cfg.brain_dt

    rates: dict[str, np.ndarray] = {}
    t4t5_rates: dict[str, np.ndarray] = {}
    t4t5_slot = np.full(engine.brain.n, -1, dtype=np.int64)
    src_idx = np.flatnonzero(np.isin(np.asarray(engine.brain.cell_type), SOURCE_TYPES))
    t4t5_slot[src_idx] = np.arange(len(src_idx), dtype=np.int64)

    for cond in CONDITIONS:
        mat = np.zeros((len(seeds), n_recv), dtype=np.float64)
        tmat = np.zeros((len(seeds), len(src_idx)), dtype=np.float64)
        for si, seed in enumerate(seeds):
            encoder.reset()
            engine.reset(seed=seed)
            counts = np.zeros(n_recv, dtype=np.int64)
            tcounts = np.zeros(len(src_idx), dtype=np.int64)
            for i, bgr in enumerate(synth[cond]):
                eye, inject, _m = encoder.encode_frame(bgr)
                if loom_cells:
                    inject = [
                        (cells, amt) for cells, amt in inject
                        if not (set(np.asarray(cells).tolist()) & loom_cells)
                    ]
                res = engine.step(i * cfg.brain_dt, eye_drive=eye, inject=inject)
                if i < lo:
                    continue
                f = np.asarray(res.fired)
                if len(f):
                    s = slot[f]
                    s = s[s >= 0]
                    if len(s):
                        np.add.at(counts, s, 1)
                    t = t4t5_slot[f]
                    t = t[t >= 0]
                    if len(t):
                        np.add.at(tcounts, t, 1)
            mat[si] = counts / span
            tmat[si] = tcounts / span
        rates[cond] = mat
        t4t5_rates[cond] = tmat

    return {
        "config": cfg,
        "recipients": recip,
        "rates": rates,
        "t4t5_rates": t4t5_rates,
        "t4t5_types": [str(x) for x in np.asarray(engine.brain.cell_type)[src_idx]],
        "seeds": seeds,
        "span": span,
        "mirrored": mirror,
    }


# ------------------------------------------------------------------ stats
def _monotonic(l: float, f: float, r: float) -> int:
    """+1 if L > F > R, -1 if R > F > L, 0 otherwise (with slack)."""
    scale = max(abs(l), abs(f), abs(r), 1e-9)
    tol = MONOTONIC_TOL * scale
    if l > f + tol * 0 and f > r:
        return +1
    if r > f and f > l:
        return -1
    return 0


def _strict_monotonic(l: float, f: float, r: float) -> int:
    """Strict sign pattern L > F > R (ignoring relative size)."""
    if l > f > r:
        return +1
    if r > f > l:
        return -1
    return 0


def evaluate(run: dict, seeds: list[int]) -> dict:
    rec = run["recipients"]
    rates = run["rates"]
    t4 = run["t4t5_rates"]
    n = len(rec["cells"])

    def med(cond: str, i: int) -> float:
        return float(np.median(rates[cond][:, i]))

    # ---- sanity: the injected layer must show direction ----
    print("\n=== Sanity: injected T4/T5 layer ===")
    src_types = run["t4t5_types"]
    for t in SOURCE_TYPES:
        m = np.array([str(x) == t for x in src_types])
        if not m.any():
            continue
        l = float(np.median(t4["YAW_LEFT"][:, m].sum(axis=1)))
        f = float(np.median(t4["FORWARD"][:, m].sum(axis=1)))
        r = float(np.median(t4["YAW_RIGHT"][:, m].sum(axis=1)))
        order = _strict_monotonic(l, f, r)
        print(
            f"  {t}: L={l:9.2f}  F={f:9.2f}  R={r:9.2f}  "
            f"order={order:+d}  {'OK' if order != 0 else 'NO DIRECTION'}"
        )

    # ---- per-type aggregation ----
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[(str(rec["cell_type"][i]), str(rec["side"][i]))].append(i)

    type_rows = []
    for (ctype, side), idxs in groups.items():
        if len(idxs) < MIN_CELLS_PER_TYPE:
            continue
        idxs = np.asarray(idxs)
        L = rates["YAW_LEFT"][:, idxs].sum(axis=1)
        F = rates["FORWARD"][:, idxs].sum(axis=1)
        R = rates["YAW_RIGHT"][:, idxs].sum(axis=1)
        ml, mf, mr = float(np.median(L)), float(np.median(F)), float(np.median(R))
        order = _strict_monotonic(ml, mf, mr)
        per_seed = [_strict_monotonic(L[s], F[s], R[s]) for s in range(len(seeds))]
        cons = float(np.mean([o == order and order != 0 for o in per_seed])) if order else 0.0
        di = (ml - mr) / (ml + mr + 1e-9)
        type_rows.append({
            "cell_type": ctype,
            "side": side,
            "n_cells": len(idxs),
            "superclass": str(rec["superclass"][idxs[0]]),
            "rate_LEFT": ml,
            "rate_FWD": mf,
            "rate_RIGHT": mr,
            "directional_index": di,
            "ordering": order,
            "seed_ordering": "".join("+" if o > 0 else ("-" if o < 0 else "0") for o in per_seed),
            "consistency": cons,
        })

    type_rows.sort(key=lambda r: -abs(r["directional_index"]))

    print(f"\n=== Recipients by type (n>={MIN_CELLS_PER_TYPE}) ===")
    print(f"  total recipient types: {len(type_rows)}")
    for r in type_rows[:20]:
        print(
            f"  {r['cell_type'][:26]:26s} {r['side']} n={r['n_cells']:3d} "
            f"L={r['rate_LEFT']:8.2f} F={r['rate_FWD']:8.2f} R={r['rate_RIGHT']:8.2f} "
            f"DI={r['directional_index']:+.3f} ord={r['ordering']:+d} cons={r['consistency']:.2f} {r['seed_ordering']}"
        )

    # ---- per-cell ----
    cell_rows = []
    for i in range(n):
        ml, mf, mr = med("YAW_LEFT", i), med("FORWARD", i), med("YAW_RIGHT", i)
        order = _strict_monotonic(ml, mf, mr)
        per_seed = [
            _strict_monotonic(rates["YAW_LEFT"][s, i], rates["FORWARD"][s, i], rates["YAW_RIGHT"][s, i])
            for s in range(len(seeds))
        ]
        cons = float(np.mean([o == order and order != 0 for o in per_seed])) if order else 0.0
        cell_rows.append({
            "cell_type": str(rec["cell_type"][i]),
            "side": str(rec["side"][i]),
            "superclass": str(rec["superclass"][i]),
            "rate_LEFT": ml,
            "rate_FWD": mf,
            "rate_RIGHT": mr,
            "directional_index": (ml - mr) / (ml + mr + 1e-9),
            "ordering": order,
            "consistency": cons,
            "n_synapses_from_source": int(rec["n_synapses"][i]),
            "input_weight": float(rec["input_weight"][i]),
        })

    passing_types = [r for r in type_rows if r["ordering"] != 0 and r["consistency"] >= 0.8]
    passing_cells = [r for r in cell_rows if r["ordering"] != 0 and r["consistency"] >= 0.8]

    # ---- permutation null on the per-type ordering ----
    rng = np.random.default_rng(0)
    n_perm = 500
    null_counts = []
    type_keys = [(r["cell_type"], r["side"], r["n_cells"]) for r in type_rows]
    group_idxs = [np.asarray(groups[(t, s)]) for t, s, _n in type_keys]
    for _ in range(n_perm):
        cnt = 0
        for idxs in group_idxs:
            L = rates["YAW_LEFT"][:, idxs].sum(axis=1).copy()
            F = rates["FORWARD"][:, idxs].sum(axis=1).copy()
            R = rates["YAW_RIGHT"][:, idxs].sum(axis=1).copy()
            for s in range(len(seeds)):
                perm = rng.permutation(3)
                vals = [L[s], F[s], R[s]]
                L[s], F[s], R[s] = vals[perm[0]], vals[perm[1]], vals[perm[2]]
            o = _strict_monotonic(float(np.median(L)), float(np.median(F)), float(np.median(R)))
            if o != 0:
                cons = float(np.mean([
                    _strict_monotonic(L[s], F[s], R[s]) == o for s in range(len(seeds))
                ]))
                if cons >= 0.8:
                    cnt += 1
        null_counts.append(cnt)
    null_counts = np.asarray(null_counts)
    p_type = float(np.mean(null_counts >= len(passing_types)))

    print(f"\n=== Directional types (ordering + consistency >= 0.8) ===")
    print(f"  pass: {len(passing_types)} / {len(type_rows)} types")
    for r in passing_types[:15]:
        print(
            f"    {r['cell_type'][:26]:26s} {r['side']} n={r['n_cells']:3d} "
            f"L={r['rate_LEFT']:8.2f} F={r['rate_FWD']:8.2f} R={r['rate_RIGHT']:8.2f} "
            f"ord={r['ordering']:+d} cons={r['consistency']:.2f}"
        )
    print(
        f"\n  permutation null ({n_perm} draws): mean passing = {null_counts.mean():.2f} "
        f"(95th pct {np.percentile(null_counts, 95):.0f}), "
        f"observed = {len(passing_types)}, p = {p_type:.4f}"
    )
    print(f"  per-cell pass: {len(passing_cells)} / {n}")

    return {
        "type_rows": type_rows,
        "cell_rows": cell_rows,
        "passing_types": passing_types,
        "n_passing_types": len(passing_types),
        "n_types_tested": len(type_rows),
        "n_passing_cells": len(passing_cells),
        "n_cells_tested": n,
        "permutation": {
            "n_perm": n_perm,
            "mean_passing": float(null_counts.mean()),
            "p95_passing": float(np.percentile(null_counts, 95)),
            "observed": len(passing_types),
            "p_value": p_type,
        },
    }


def _type_orders(run: dict, seeds: list[int]) -> tuple[dict, dict]:
    """Per (cell_type, side): strict ordering of the three conditions, and the rates."""
    rec = run["recipients"]
    rates = run["rates"]
    n = len(rec["cells"])
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[(str(rec["cell_type"][i]), str(rec["side"][i]))].append(i)

    orders: dict[tuple[str, str], int] = {}
    payload: dict[tuple[str, str], dict] = {}
    for key, idxs in groups.items():
        if len(idxs) < MIN_CELLS_PER_TYPE:
            continue
        idxs = np.asarray(idxs)
        L = rates["YAW_LEFT"][:, idxs].sum(axis=1)
        F = rates["FORWARD"][:, idxs].sum(axis=1)
        R = rates["YAW_RIGHT"][:, idxs].sum(axis=1)
        ml, mf, mr = float(np.median(L)), float(np.median(F)), float(np.median(R))
        orders[key] = _strict_monotonic(ml, mf, mr)
        per_seed = [
            _strict_monotonic(L[s], F[s], R[s]) for s in range(len(seeds))
        ]
        payload[key] = {
            "n_cells": len(idxs),
            "rate_LEFT": ml,
            "rate_FWD": mf,
            "rate_RIGHT": mr,
            "superclass": str(rec["superclass"][idxs[0]]),
            "seed_ordering": per_seed,
        }
    return orders, payload


def mirror_control(seeds: list[int], steps: int, base: dict, disable_loom: bool = False) -> dict:
    """The decisive control.

    Horizontally flipping every frame reverses motion direction while leaving all
    magnitude statistics untouched. A genuinely direction-selective cell must flip
    its L/F/R ordering under the flip; a cell that merely tracks overall drive
    (or any stimulus magnitude asymmetry) will not.

    Any type whose ordering does not flip is not evidence of direction coding.
    """
    tag = " (no loom)" if disable_loom else ""
    print(f"\n=== Mirror control{tag} (reverses motion direction, keeps all magnitudes) ===")
    mir = run_conditions(seeds, steps, mirror=True, disable_loom=disable_loom)

    o_base, p_base = _type_orders(base, seeds)
    o_mir, p_mir = _type_orders(mir, seeds)

    common = [k for k in o_base if k in o_mir]
    flipping, stable, silent = [], [], []
    for k in common:
        b, m = o_base[k], o_mir[k]
        if b == 0 or m == 0:
            silent.append(k)
        elif b == -m:
            flipping.append(k)
        else:
            stable.append(k)

    # direction-selective := ordered AND flips under mirror AND consistent in both runs
    def _cons(payload, key):
        v = payload[key]["seed_ordering"]
        o = 0 if not v else (v[0] if all(x == v[0] for x in v) else 0)
        return o

    selective = []
    for k in flipping:
        cb, cm = _cons(p_base, k), _cons(p_mir, k)
        if cb != 0 and cm != 0:
            selective.append(k)

    print(f"  types tested: {len(common)}")
    print(f"    ordering flips under mirror  : {len(flipping)}")
    print(f"    ordering does NOT flip        : {len(stable)}")
    print(f"    no ordering in one/both runs  : {len(silent)}")
    print(f"    flips AND seed-consistent     : {len(selective)}")

    rows = []
    for k in sorted(common):
        ctype, side = k
        rows.append({
            "cell_type": ctype,
            "side": side,
            "n_cells": p_base[k]["n_cells"],
            "superclass": p_base[k]["superclass"],
            "order_original": o_base[k],
            "order_mirrored": o_mir[k],
            "flips": bool(o_base[k] != 0 and o_mir[k] != 0 and o_base[k] == -o_mir[k]),
            "seed_consistent_original": _cons(p_base, k) != 0,
            "seed_consistent_mirrored": _cons(p_mir, k) != 0,
            "direction_selective": k in selective,
            "rate_LEFT": p_base[k]["rate_LEFT"],
            "rate_FWD": p_base[k]["rate_FWD"],
            "rate_RIGHT": p_base[k]["rate_RIGHT"],
        })

    sel_rows = [r for r in rows if r["direction_selective"]]
    sel_rows.sort(key=lambda r: -abs(r["rate_LEFT"] - r["rate_RIGHT"]))
    print("\n  direction-selective types:")
    for r in sel_rows[:15]:
        print(
            f"    {r['cell_type'][:26]:26s} {r['side']} n={r['n_cells']:3d} "
            f"L={r['rate_LEFT']:8.2f} F={r['rate_FWD']:8.2f} R={r['rate_RIGHT']:8.2f} "
            f"orig={r['order_original']:+d} mir={r['order_mirrored']:+d}"
        )

    # T4/T5 sanity under mirror: the injected layer must itself flip
    print("\n  injected-layer check (T4b should flip):")
    tb = {}
    for tag, run in (("orig", base), ("mirror", mir)):
        types = run["t4t5_types"]
        t4r = run["t4t5_rates"]
        m = np.array([t == "T4b" for t in types])
        if m.any():
            l = float(np.median(t4r["YAW_LEFT"][:, m].sum(axis=1)))
            f = float(np.median(t4r["FORWARD"][:, m].sum(axis=1)))
            r = float(np.median(t4r["YAW_RIGHT"][:, m].sum(axis=1)))
            tb[tag] = _strict_monotonic(l, f, r)
            print(f"    T4b {tag:6s}: L={l:9.2f} F={f:9.2f} R={r:9.2f} order={tb[tag]:+d}")

    return {
        "rows": rows,
        "selective": sel_rows,
        "n_tested": len(common),
        "n_flipping": len(flipping),
        "n_stable": len(stable),
        "n_silent": len(silent),
        "n_selective": len(selective),
        "t4b_order": tb,
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=SYNTH_STEPS)
    ap.add_argument("-o", "--output", default="output/p01k_downstream")
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1K — direction immediately downstream of T4/T5")
    print(f"  sources: {SOURCE_TYPES}   conditions: {list(CONDITIONS)}   seeds: {seeds}")

    run = run_conditions(seeds, args.steps)
    rec = run["recipients"]
    print(f"  source cells: {rec['n_sources']}   direct recipients: {len(rec['cells'])}")

    ev = evaluate(run, seeds)
    mir = mirror_control(seeds, args.steps, run)

    # Loom ablation: confirm the direction signal comes from the T4/T5 path and
    # not from the LC4/LPLC2 loom channel, which also carries scene asymmetry.
    mir_noloom = mirror_control(seeds, args.steps, run, disable_loom=True)
    sel_loom = {(r["cell_type"], r["side"]) for r in mir["selective"]}
    sel_noloom = {(r["cell_type"], r["side"]) for r in mir_noloom["selective"]}
    ablation = {
        "n_with_loom": len(sel_loom),
        "n_without_loom": len(sel_noloom),
        "lost_when_loom_removed": sorted(f"{t}_{s}" for t, s in (sel_loom - sel_noloom)),
        "gained_when_loom_removed": sorted(f"{t}_{s}" for t, s in (sel_noloom - sel_loom)),
    }
    print(
        f"\n=== Loom ablation ===\n"
        f"  with loom: {ablation['n_with_loom']}   without loom: {ablation['n_without_loom']}\n"
        f"  lost when loom removed: {ablation['lost_when_loom_removed']}"
    )

    with (out / "recipient_types.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev["type_rows"][0].keys()))
        w.writeheader()
        w.writerows(ev["type_rows"])
    with (out / "recipient_cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev["cell_rows"][0].keys()))
        w.writeheader()
        w.writerows(ev["cell_rows"])

    p = ev["permutation"]["p_value"]
    # The mirror control decides. A type counts only if its ordering reverses when
    # motion direction is reversed — otherwise it is tracking stimulus magnitude.
    # The loom ablation then confirms the surviving signal travels the T4/T5 path.
    n_sel = len(sel_noloom)
    established = n_sel >= 3
    if established:
        verdict = (
            f"DIRECTION FOUND AFTER T4/T5: {n_sel} recipient types reverse their L/F/R ordering "
            f"when the stimulus is mirrored (out of {mir['n_tested']} tested). The visual system "
            "does build a whole-eye left/right signal; the descending neurons are not its readout."
        )
    elif n_sel > 0:
        verdict = (
            f"WEAK: {n_sel} of {mir['n_tested']} recipient types survive the mirror control. "
            "Not enough to establish a robust direction signal."
        )
    else:
        verdict = (
            f"NO DIRECTION AFTER T4/T5: 0 of {mir['n_tested']} recipient types reverse their "
            f"ordering under mirroring ({mir['n_flipping']} types flip, but none are seed-consistent "
            "in both runs). The earlier L/F/R orderings track stimulus magnitude, not direction. "
            "The problem is at or before T4/T5."
        )

    report = {
        "probe": "P0.1K — direction immediately downstream of T4/T5",
        "frozen": ["local opponent inject", "optic_flow", "labels", "MaleCNS dynamics"],
        "not_used": ["trajectory", "descending neurons", "tuning"],
        "seeds": seeds,
        "synthetic_steps": args.steps,
        "source_types": list(SOURCE_TYPES),
        "n_source_cells": rec["n_sources"],
        "n_recipient_cells": len(rec["cells"]),
        "n_types_tested": ev["n_types_tested"],
        "n_passing_types_raw": ev["n_passing_types"],
        "n_passing_cells_raw": ev["n_passing_cells"],
        "n_cells_tested": ev["n_cells_tested"],
        "permutation_within_stimulus": ev["permutation"],
        "mirror_control": {
            "n_tested": mir["n_tested"],
            "n_ordering_flips": mir["n_flipping"],
            "n_ordering_stable": mir["n_stable"],
            "n_no_ordering": mir["n_silent"],
            "n_direction_selective": mir["n_selective"],
            "t4b_order_original_vs_mirror": mir["t4b_order"],
            "selective_types": mir["selective"],
        },
        "loom_ablation": ablation,
        "direction_selective_without_loom": mir_noloom["selective"],
        "passing_types_raw": ev["passing_types"],
        "top_types": ev["type_rows"][:30],
        "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    with (out / "mirror_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mir["rows"][0].keys()))
        w.writeheader()
        w.writerows(mir["rows"])

    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")
    raise SystemExit(0 if established else 1)


if __name__ == "__main__":
    main()
