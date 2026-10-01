#!/usr/bin/env python3
"""
P05.1 — the exact transition at which the LEFT/RIGHT signal is lost.

No new simulation. This reads the two existing P05 recordings, checks they are
comparable, and scores every recorded group by how well it separates the confirmed
turns, depth by depth.

Why the layer score is a fraction and not a median
-------------------------------------------------
The first attempt scored a layer by the median AUC of its active groups. That measure is
blind to the shape of the answer. Most descending neurons do not carry turn direction at
all, so the median over 364 of them came out at 0.66 against a null of 0.57 and looked
like a weak loss, while 82 of those same neurons separate the turns at AUC above 0.8 on
both seeds where chance predicts none.

The layer score is therefore the *share of active groups that carry direction*, with its
own permutation null. It answers the question actually being asked: does this layer
still represent the turn, or has the representation dissolved into a minority of cells.

The four numbers per depth
--------------------------
direction_share      share of active groups whose AUC exceeds the threshold on both
                     seeds, with the null share from shuffled labels and the excess
                     in sigmas
direction_score      median AUC of the active groups, kept for continuity; note that
                     it is diluted by the non-directional majority
seed_reproducibility median correlation of the per-turn values between the two seeds
active_groups        groups whose firing reaches the activity floor

Ground truth and its one limit
------------------------------
Only the 23 turns confirmed by eye as LEFT or RIGHT are used. FWD, STATIC, REJECT and
UNCLEAR are excluded, and no old `left_*` / `right_*` names are referenced.

But depth 1 is not a neutral sample: it is the set of 27 types that P0.1K had already
selected *because* they reverse under a mirrored stimulus. Its high share is therefore
part selection, not measurement, and is reported as such. Depth 2 and 3 are the whole
descending population and were never selected, so those shares are the real result.

Usage:
    PYTHONPATH=. python scripts/p051_signal_break.py
    PYTHONPATH=. python scripts/p051_signal_break.py --carry-threshold 0.7
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REC64 = ROOT / "output/p05_layer_trace/layer_signals.npz"
REC65 = ROOT / "output/p05_layer_trace_seed65/layer_signals.npz"
GROUPS = ROOT / "output/p05_layer_trace/groups.json"
REVIEW = ROOT / "data/p01r/review_set.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
WINDOWS3 = ROOT / "data/p01r/windows_v3.json"
TRACE_SCRIPT = ROOT / "scripts/p05_layer_trace.py"
OUT = ROOT / "output/p051_signal_break"

SMOOTH_S = 0.3
MAX_DEPTH = 3
NULL_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


# --------------------------------------------------------------------- helpers
def box_mat(M: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return M
    c = np.cumsum(M, axis=0, dtype=np.float64)
    c = np.vstack([np.zeros((1, M.shape[1])), c])
    o = c[n:] - c[:-n]
    left = (n - 1) // 2
    right = n - 1 - left
    o = np.vstack([np.repeat(o[:1], left, axis=0), o, np.repeat(o[-1:], right, axis=0)])
    return o[: M.shape[0]] / n


def auc_raw(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) with ties averaged. 0.5 means no separation, 0 or 1 mean perfect."""
    if len(a) < 2 or len(b) < 2:
        return 0.5
    allv = np.concatenate([a, b])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[order] = np.arange(1, len(allv) + 1)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    rk = (s / cnt)[inv]
    return float((rk[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def auc_for_columns(V: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Oriented AUC for every column of V (shape n_turns x n_groups)."""
    out = np.empty(V.shape[1])
    for g in range(V.shape[1]):
        raw = auc_raw(V[pos, g], V[~pos, g])
        out[g] = max(raw, 1.0 - raw)
    return out


def human_turns() -> list[dict]:
    rs = json.loads(REVIEW.read_text(encoding="utf-8"))["candidates"]
    vd = json.loads(VERDICTS.read_text(encoding="utf-8"))["verdicts"]
    out = []
    for x in rs:
        v = vd.get(x["id"])
        if not v:
            continue
        truth = x["camera_direction"] if v["verdict"] == "correct" \
            else (v.get("actual_direction") or "REJECT")
        if truth in ("LEFT", "RIGHT"):
            out.append({"id": x["id"], "t0": x["t0"], "t1": x["t1"], "kind": truth})
    out.sort(key=lambda e: e["t0"])
    return out


def context_windows() -> dict[str, list[tuple[float, float]]]:
    ws = json.loads(WINDOWS3.read_text(encoding="utf-8"))["windows"]
    out: dict[str, list[tuple[float, float]]] = {"FWD": [], "STATIC": []}
    for w in ws:
        if w["kind"] in out:
            out[w["kind"]].append((w["t0"], w["t1"]))
    return out


def load_group_scheme():
    spec = importlib.util.spec_from_file_location("p05_trace", TRACE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fly_vo.config import FlyVOConfig
    from fly_vo.malecns_engine import MaleCNSEngine
    engine = MaleCNSEngine(FlyVOConfig())
    group_of, meta, depths = mod.build_groups(engine.brain)
    return group_of, meta, depths, engine.brain


# --------------------------------------------------------------------- verify
def verify(meta: list[dict]) -> dict:
    print("=== 1. СРАВНИМОСТЬ ДВУХ ЗАПИСЕЙ ===")
    a, b = np.load(REC64), np.load(REC65)
    checks = {
        "same_n_steps": a["t"].shape == b["t"].shape,
        "same_time_axis": bool(np.allclose(a["t"], b["t"])),
        "same_n_groups": a["G"].shape[1] == b["G"].shape[1],
        "matches_groups_json": a["G"].shape[1] == len(meta),
        "same_dtype": a["G"].dtype == b["G"].dtype,
    }
    print(f"  шагов: {a['t'].shape[0]:,} vs {b['t'].shape[0]:,}")
    print(f"  групп: {a['G'].shape[1]} vs {b['G'].shape[1]}, "
          f"groups.json описывает {len(meta)}")
    print(f"  временная шкала совпадает: {checks['same_time_axis']}")

    sil_a = set(np.flatnonzero(a["G"].sum(axis=0) == 0).tolist())
    sil_b = set(np.flatnonzero(b["G"].sum(axis=0) == 0).tolist())
    checks["same_silent_set"] = sil_a == sil_b
    print(f"  молчащих групп: {len(sil_a)} vs {len(sil_b)}, "
          f"набор совпадает: {sil_a == sil_b}")

    # Column-by-column agreement is measured on the busiest groups. Over all 973 it
    # comes out near zero, because a descending neuron firing 0 or 1 times per step
    # produces a series whose correlation reflects which step happened to spike, not
    # whether the two runs agree on the rate.
    act = np.flatnonzero(a["G"].sum(axis=0) > 0)
    busiest = act[np.argsort(-a["G"].sum(axis=0)[act])[:40]]
    rs = []
    for g in busiest:
        va, vb = a["G"][:, g], b["G"][:, g]
        if va.std() > 0 and vb.std() > 0:
            rs.append(float(np.corrcoef(va, vb)[0, 1]))
    med_r = float(np.median(rs)) if rs else 0.0
    checks["median_column_corr_top40"] = med_r
    print(f"  корреляция по столбцам: медиана {med_r:+.3f} "
          f"на 40 самых активных группах")

    print("\n  проверка схемы групп (пересчёт с нуля):")
    _, meta2, _, _ = load_group_scheme()
    same = (len(meta2) == len(meta)
            and all(x["cell_type"] == y["cell_type"] and x["side"] == y["side"]
                    for x, y in zip(meta, meta2)))
    checks["scheme_reproduces"] = bool(same)
    print(f"    групп пересчитано: {len(meta2)}, совпадает с записью: {same}")
    return checks


# ------------------------------------------------------------------- per-turn
def window_reduce(S: np.ndarray, t: np.ndarray, wins, reduce_fn) -> np.ndarray:
    return np.stack([reduce_fn(S[(t >= a) & (t <= b)], axis=0) if ((t >= a) & (t <= b)).any()
                     else np.zeros(S.shape[1]) for a, b in wins])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity-floor", type=float, default=0.5, help="Hz")
    ap.add_argument("--carry-threshold", type=float, default=0.8,
                    help="AUC on both seeds required to call a group directional")
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(GROUPS.read_text(encoding="utf-8"))["groups"]
    checks = verify(meta)

    a, b = np.load(REC64), np.load(REC65)
    t = a["t"].astype(np.float64)
    dt = float(np.median(np.diff(t)))
    n_sm = max(int(round(SMOOTH_S / dt)), 1)
    ncell = np.array([m["n_cells"] for m in meta], float)
    ng = len(meta)

    turns = [w for w in human_turns() if w["t0"] >= t[0] and w["t1"] <= t[-1]]
    is_R = np.array([w["kind"] == "RIGHT" for w in turns])
    ctx = context_windows()
    print(f"\n=== 2. МЕТКИ ===")
    print(f"  подтверждённых поворотов: {len(turns)} "
          f"({int(is_R.sum())} RIGHT, {int((~is_R).sum())} LEFT)")
    print(f"  FWD, STATIC, REJECT, UNCLEAR исключены; старые имена left_*/right_* "
          f"не используются")
    print(f"  контрольные окна из классификатора (только контекст, не метки): "
          f"FWD {len(ctx['FWD'])}, STATIC {len(ctx['STATIC'])}")

    # ---- per-turn values for both seeds ------------------------------------
    win_turn = [(w["t0"], w["t1"]) for w in turns]
    win_pre = [(max(0.0, w["t0"] - (w["t1"] - w["t0"])), w["t0"]) for w in turns]
    data = {}
    for name, rec in (("seed64", a), ("seed65", b)):
        M = rec["G"].astype(np.float64) / ncell[None, :]
        S = box_mat(M, n_sm)
        during = window_reduce(S, t, win_turn, np.median)
        pre = window_reduce(S, t, win_pre, np.median)
        data[name] = {
            "during": during, "pre": pre, "change": during - pre,
            "fwd": window_reduce(S, t, ctx["FWD"], np.median),
            "stat": window_reduce(S, t, ctx["STATIC"], np.median),
            "rate": np.mean(np.concatenate([during, pre], axis=0), axis=0),
        }

    # ---- validity ----------------------------------------------------------
    depth = np.array([m["mean_depth"] for m in meta])
    depth_bin = np.clip(np.round(depth).astype(int), 0, MAX_DEPTH)
    # The connectome has 1,358 cells with an empty cell_type, spanning nineteen
    # superclasses and mostly unreachable from T4/T5. Grouped by (type, side) they form
    # two grab-bags whose mean depth rounds to zero, landing among T4/T5. Aggregates
    # duplicate content already counted.
    junk = np.array([m["cell_type"] == "" for m in meta])
    is_agg = np.array([m["superclass"] == "aggregate" for m in meta])
    valid = ~(junk | is_agg)
    depth_bin[~valid] = -1
    act = ((data["seed64"]["rate"] / dt) > args.activity_floor) \
        & ((data["seed65"]["rate"] / dt) > args.activity_floor) & valid

    print(f"\n=== 3. ПРИГОДНОСТЬ И АКТИВНОСТЬ ===")
    print(f"  групп с пустым типом (свалка 19 классов): {int(junk.sum())} -> исключены")
    print(f"  агрегатных групп: {int(is_agg.sum())} -> исключены")
    print(f"  порог активности {args.activity_floor} Гц")
    print(f"  активных групп: {int(act.sum())} из {int(valid.sum())} в анализе")

    # ---- per-group metrics -------------------------------------------------
    rows = []
    for g in range(ng):
        m = meta[g]
        w64, w65 = data["seed64"]["change"][:, g], data["seed65"]["change"][:, g]
        r = float(np.corrcoef(w64, w65)[0, 1]) if (w64.std() > 0 and w65.std() > 0) \
            else 0.0
        d64 = data["seed64"]["during"][:, g]
        d65 = data["seed65"]["during"][:, g]
        c64 = data["seed64"]["change"][:, g]
        c65 = data["seed65"]["change"][:, g]
        raw64 = auc_raw(c64[is_R], c64[~is_R])
        raw65 = auc_raw(c65[is_R], c65[~is_R])
        rL, rR = float(np.median(d64[~is_R])), float(np.median(d64[is_R]))
        rF = float(np.median(data["seed64"]["fwd"][:, g]))
        rS = float(np.median(data["seed64"]["stat"][:, g]))
        rows.append({
            "group": g, "cell_type": m["cell_type"], "side": m["side"],
            "layer": m["layer"], "depth": float(m["mean_depth"]),
            "depth_bin": int(depth_bin[g]), "n_cells": m["n_cells"],
            "superclass": m["superclass"], "in_analysis": bool(valid[g]),
            "active": bool(act[g]), "rate_hz": float(data["seed64"]["rate"][g] / dt),
            "auc_change_64": max(raw64, 1 - raw64),
            "auc_change_65": max(raw65, 1 - raw65),
            "auc_during_64": float(auc_for_columns(d64[:, None], is_R)[0]),
            "auc_during_65": float(auc_for_columns(d65[:, None], is_R)[0]),
            "seed_r": r,
            "pol_64": float(np.sign(raw64 - 0.5)),
            "pol_65": float(np.sign(raw65 - 0.5)),
            "rate_L": rL, "rate_R": rR, "rate_FWD": rF, "rate_STATIC": rS,
            # a direction channel should sit differently in left and right turns while
            # forward falls between them; if forward is outside the range, the channel
            # is following something else
            "fwd_between": bool(min(rL, rR) <= rF <= max(rL, rR)),
        })
    for row in rows:
        row["carries"] = bool(min(row["auc_change_64"], row["auc_change_65"])
                              > args.carry_threshold)
        row["pol_agree"] = bool(row["pol_64"] == row["pol_65"] and row["pol_64"] != 0)

    A64 = np.array([r["auc_change_64"] for r in rows])
    A65 = np.array([r["auc_change_65"] for r in rows])
    A_min = np.minimum(A64, A65)

    # ---- null for the share -------------------------------------------------
    rng = np.random.default_rng(0)
    null_share = {thr: {d: [] for d in range(MAX_DEPTH + 1)} for thr in NULL_THRESHOLDS}
    null_median = {d: [] for d in range(MAX_DEPTH + 1)}
    for _ in range(args.n_perm):
        pos = rng.random(len(turns)) < is_R.mean()
        if pos.sum() < 3 or pos.sum() > len(turns) - 3:
            continue
        n64 = auc_for_columns(data["seed64"]["change"], pos)
        n65 = auc_for_columns(data["seed65"]["change"], pos)
        nm = np.minimum(n64, n65)
        for d in range(MAX_DEPTH + 1):
            m = (depth_bin == d) & act
            if not m.any():
                continue
            for thr in NULL_THRESHOLDS:
                null_share[thr][d].append(float(np.mean(nm[m] > thr)))
            null_median[d].append(float(np.median(n64[m])))

    print(f"\n=== 4. РАЗЛИЧИМОСТЬ ПО ГЛУБИНАМ ===")
    print(f"  доля активных групп, различающих LEFT и RIGHT (AUC > "
          f"{args.carry_threshold} на ОБОИХ seed), против нуля\n")
    print(f"  {'глубина':>8s} {'активн':>7s} {'несут':>7s} {'доля':>7s} "
          f"{'нуль':>7s} {'значим.':>9s} {'медиана AUC':>12s} {'нуль мед.':>10s} "
          f"{'repro':>7s} {'согласие':>9s} {'FWD меж':>8s}")
    layer_metrics = {}
    for d in range(MAX_DEPTH + 1):
        idx = np.flatnonzero((depth_bin == d) & valid)
        ai = np.flatnonzero((depth_bin == d) & act)
        comp = Counter(meta[g]["superclass"] for g in idx)
        if len(ai) == 0:
            layer_metrics[d] = {"n_groups": int(len(idx)), "n_active": 0,
                                "composition": dict(comp)}
            continue
        share = float(np.mean(A_min[ai] > args.carry_threshold))
        n_share = float(np.mean(null_share[args.carry_threshold][d])) \
            if null_share[args.carry_threshold][d] else 0.0
        sd_share = float(np.std(null_share[args.carry_threshold][d])) or 1e-9
        med_auc = float(np.median([np.median(A64[ai]), np.median(A65[ai])]))
        raw_auc = float(np.median([np.median(A64[ai]), np.median(A65[ai])]))
        n_med = float(np.mean(null_median[d])) if null_median[d] else 0.5
        rr = np.array([rows[g]["seed_r"] for g in ai])
        pa = np.array([rows[g]["pol_agree"] for g in ai])
        fb = np.array([rows[g]["fwd_between"] for g in ai])
        layer_metrics[d] = {
            "n_groups": int(len(idx)), "n_active": int(len(ai)),
            "n_carrying": int(np.sum(A_min[ai] > args.carry_threshold)),
            "direction_share": share,
            "direction_share_null": n_share,
            "direction_share_excess": share - n_share,
            "direction_share_sigma": (share - n_share) / sd_share,
            "direction_score_median_auc": med_auc,
            "direction_score_null_median": n_med,
            "direction_score_excess": med_auc - n_med,
            "seed_reproducibility": float(np.median(rr)),
            "polarity_agreement": float(np.mean(pa)),
            "fraction_fwd_between": float(np.mean(fb)),
            # the same share among those that also pass the forward-context control
            "direction_share_fwd_between": float(np.mean(
                A_min[ai][fb] > args.carry_threshold)) if fb.any() else float("nan"),
            "auc_p90_64": float(np.percentile(A64[ai], 90)),
            "auc_p90_65": float(np.percentile(A65[ai], 90)),
            "composition": dict(comp),
        }
        print(f"  {d:8d} {len(ai):7d} "
              f"{int(np.sum(A_min[ai] > args.carry_threshold)):7d} {share * 100:6.0f}% "
              f"{n_share * 100:6.0f}% {(share - n_share) / sd_share:8.1f}σ "
              f"{med_auc:12.3f} {n_med:10.3f} {np.median(rr):+7.2f} "
              f"{np.mean(pa) * 100:8.0f}% {np.mean(fb) * 100:7.0f}%")

    # ---- the break ---------------------------------------------------------
    print(f"\n=== 5. ПЕРЕЛОМ ===")
    print("  критерий: доля несущих направление падает, И падает воспроизводимость")
    print("  (падение одной медианы недостаточно: она разбавлена неинформативным "
          "большинством)\n")
    shares = {d: layer_metrics[d].get("direction_share", float("nan"))
              for d in range(MAX_DEPTH + 1)}
    repro = {d: layer_metrics[d].get("seed_reproducibility", float("nan"))
             for d in range(MAX_DEPTH + 1)}
    print(f"  {'переход':>12s} {'доля':>16s} {'repro':>16s} {'вердикт':>10s}")
    brk = None
    for d in range(MAX_DEPTH):
        s0, s1 = shares.get(d, np.nan), shares.get(d + 1, np.nan)
        r0, r1 = repro.get(d, np.nan), repro.get(d + 1, np.nan)
        if not all(np.isfinite(v) for v in (s0, s1, r0, r1)):
            continue
        share_dropped = s1 < s0 * 0.6
        repro_dropped = r1 < r0 - 0.25
        if brk is None and share_dropped and repro_dropped:
            brk = (d, d + 1, s0, s1, r0, r1)
            verdict = "ПЕРЕЛОМ"
        else:
            verdict = "держится"
        print(f"  {d:5d} -> {d + 1:<4d} {s0 * 100:7.0f}% -> {s1 * 100:<6.0f}% "
              f"{r0:+6.2f} -> {r1:<+6.2f} {verdict:>10s}")
    if brk is None:
        print("  перелома не найдено")
        last_good, first_bad = MAX_DEPTH, None
    else:
        last_good, first_bad = brk[0], brk[1]
        print(f"\n  переход {last_good} -> {first_bad}")
        print(f"    доля несущих:      {shares[last_good] * 100:.0f}% -> "
              f"{shares[first_bad] * 100:.0f}%")
        print(f"    воспроизводимость: {repro[last_good]:+.2f} -> "
              f"{repro[first_bad]:+.2f}")

    # ---- time-confound control --------------------------------------------
    print(f"\n=== 5b. КОНТРОЛЬ НА ВРЕМЕННОЙ СДВИГ ===")
    print("  левые повороты сгруппированы в начале, поэтому нужен контроль")
    for cut in (300.0,):
        sub = np.array([w["t0"] > cut for w in turns])
        if sub.sum() < 6:
            continue
        print(f"  только повороты после {cut:.0f} с "
              f"({int((sub & is_R).sum())} RIGHT, {int((sub & ~is_R).sum())} LEFT):")
        print(f"    {'глубина':>8s} {'активн':>7s} {'доля':>7s} {'repro':>8s}")
        for d in range(MAX_DEPTH + 1):
            ai = np.flatnonzero((depth_bin == d) & act)
            if len(ai) == 0:
                continue
            b64 = auc_for_columns(data["seed64"]["change"], sub & is_R)
            b65 = auc_for_columns(data["seed65"]["change"], sub & is_R)
            bm = np.minimum(b64, b65)
            sh = float(np.mean(bm[ai] > args.carry_threshold))
            rr = []
            for g in ai:
                v64 = data["seed64"]["change"][sub, g]
                v65 = data["seed65"]["change"][sub, g]
                if v64.std() > 0 and v65.std() > 0:
                    rr.append(float(np.corrcoef(v64, v65)[0, 1]))
            print(f"    {d:8d} {len(ai):7d} {sh * 100:6.0f}% "
                  f"{np.median(rr) if rr else float('nan'):+8.2f}")
            layer_metrics[d][f"late_share_{int(cut)}"] = sh
            layer_metrics[d][f"late_reproducibility_{int(cut)}"] = (
                float(np.median(rr)) if rr else float("nan"))

    # ---- expand the transition --------------------------------------------
    print(f"\n=== 6. РАСКРЫТИЕ ПЕРЕХОДА {last_good} -> {first_bad} ===")
    _, _, all_depths, brain = load_group_scheme()
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    sc_ = np.asarray(brain.superclass)
    ip = np.asarray(brain.indptr)
    ix = np.asarray(brain.indices)
    wt = np.asarray(brain.weights)

    src_mask = (depth_bin == last_good) & act \
        & np.array([m["superclass"] != "descending_neuron" for m in meta])
    good_types = sorted({(meta[g]["cell_type"], meta[g]["side"])
                         for g in np.flatnonzero(src_mask)})
    print(f"  источники: {len(good_types)} активных групп "
          f"({len({x for x, _ in good_types})} типов) на глубине {last_good}")
    print(f"  (нисходящие нейроны исключены как источники: 8 их групп сидят на "
          f"глубине 1, они моносинаптические от T4/T5, но нас интересует цель)")

    src_cells = np.array(sorted({int(j) for ct_name, side in good_types
                                 for j in np.flatnonzero((ct == ct_name) & (sd == side))}),
                         dtype=np.int64)
    tgt, wsum = [], []
    for j in src_cells.tolist():
        tgt.append(ix[ip[j]:ip[j + 1]])
        wsum.append(wt[ip[j]:ip[j + 1]])
    tgt = np.concatenate(tgt) if tgt else np.empty(0, np.int64)
    wsum = np.concatenate(wsum) if len(wsum) else np.empty(0, np.float32)

    trans: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"w_exc": 0.0, "w_inh": 0.0, "n_syn": 0, "cells": set()})
    for k in range(len(tgt)):
        i = int(tgt[k])
        if ct[i] == "":
            # the same grab-bag as on the source side: not a real cell type, and its
            # superclass cannot be read off a single member
            continue
        rec = trans[(str(ct[i]), str(sd[i]))]
        rec["n_syn"] += 1
        rec["cells"].add(i)
        if wsum[k] >= 0:
            rec["w_exc"] += float(wsum[k])
        else:
            rec["w_inh"] += float(-wsum[k])
    trans_rows = []
    for (tt, ts), rec in trans.items():
        tot = rec["w_exc"] + rec["w_inh"]
        cells = np.flatnonzero((ct == tt) & (sd == ts))
        t_dep = float(np.mean(all_depths[cells])) if len(cells) else float("nan")
        cls = str(sc_[cells[0]]) if len(cells) else ""
        trans_rows.append({
            "source_layer": last_good, "target_depth": t_dep,
            "target_type": tt, "target_side": ts, "target_superclass": cls,
            "n_cells": len(rec["cells"]), "n_synapses": rec["n_syn"],
            "weight_exc": rec["w_exc"], "weight_inh": rec["w_inh"],
            "weight_total": tot,
            "inh_fraction": rec["w_inh"] / tot if tot > 0 else 0.0,
            "target_is_dn": cls.startswith("descending_neuron"),
        })
    trans_rows.sort(key=lambda r: -r["weight_total"])

    class_w: dict[str, float] = defaultdict(float)
    class_cells: dict[str, int] = defaultdict(int)
    for r in trans_rows:
        class_w[r["target_superclass"]] += r["weight_total"]
        class_cells[r["target_superclass"]] += r["n_cells"]
    total_w = sum(class_w.values())
    dn_w = sum(v for k, v in class_w.items() if k.startswith("descending_neuron"))
    dn_cells = sum(v for k, v in class_cells.items() if k.startswith("descending_neuron"))

    print(f"\n  КУДА УХОДИТ ВЫХОД ХОРОШЕГО СЛОЯ:")
    print(f"  {'класс':>24s} {'вес':>9s} {'доля веса':>10s} {'клеток':>8s} "
          f"{'вес на клетку':>14s}")
    for k, v in sorted(class_w.items(), key=lambda z: -z[1])[:8]:
        per = v / class_cells[k] if class_cells[k] else 0.0
        print(f"  {k:>24s} {v:9.1f} {v / total_w * 100:9.2f}% "
              f"{class_cells[k]:8d} {per:14.4f}")
    print(f"  -> в нисходящие нейроны: {dn_w / total_w * 100:.3f}% веса, "
          f"{dn_w / dn_cells if dn_cells else 0:.4f} веса на клетку")

    print(f"\n  крупнейшие получатели по весу:")
    print(f"  {'тип':>14s} {'стор':>5s} {'глубина':>8s} {'клеток':>7s} "
          f"{'синапсов':>9s} {'вес':>8s} {'торм%':>6s} {'класс':>18s}")
    for r in trans_rows[:18]:
        print(f"  {r['target_type'][:14]:>14s} {r['target_side']:>5s} "
              f"{r['target_depth']:8.1f} {r['n_cells']:7d} {r['n_synapses']:9d} "
              f"{r['weight_total']:8.1f} {r['inh_fraction'] * 100:5.0f}% "
              f"{r['target_superclass'][:18]:>18s}")

    print(f"\n  для каждой группы последнего хорошего слоя — 5 крупнейших получателей:")
    per_group = []
    for g in np.flatnonzero(src_mask):
        m = meta[g]
        cells = np.flatnonzero((ct == m["cell_type"]) & (sd == m["side"]))
        if len(cells) == 0:
            continue
        acc: dict[tuple[str, str], float] = defaultdict(float)
        for j in cells.tolist():
            for e in range(ip[j], ip[j + 1]):
                i = int(ix[e])
                acc[(str(ct[i]), str(sd[i]))] += abs(float(wt[e]))
        per_group.append({
            "group": int(g), "cell_type": m["cell_type"], "side": m["side"],
            "auc_change_64": rows[g]["auc_change_64"],
            "auc_change_65": rows[g]["auc_change_65"], "seed_r": rows[g]["seed_r"],
            "top_recipients": [{"type": k[0], "side": k[1], "weight": v}
                               for k, v in sorted(acc.items(), key=lambda z: -z[1])[:10]],
        })
    for pg in per_group[:5]:
        print(f"    {pg['cell_type']} {pg['side']} "
              f"(AUC {pg['auc_change_64']:.2f}/{pg['auc_change_65']:.2f}, "
              f"r {pg['seed_r']:+.2f}):")
        for r in pg["top_recipients"][:5]:
            print(f"        -> {r['type'][:22]:>22s} {r['side']:>2s} вес {r['weight']:7.1f}")

    # ---- outputs -----------------------------------------------------------
    with (OUT / "group_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (OUT / "layer_metrics.csv").open("w", newline="") as f:
        keys = ["n_groups", "n_active", "n_carrying", "direction_share",
                "direction_share_null", "direction_share_excess",
                "direction_share_sigma", "direction_score_median_auc",
                "direction_score_null_median", "direction_score_excess",
                "seed_reproducibility", "polarity_agreement",
                "fraction_fwd_between", "direction_share_fwd_between",
                "late_share_300", "late_reproducibility_300", "composition"]
        w = csv.writer(f)
        w.writerow(["depth"] + keys)
        for d in range(MAX_DEPTH + 1):
            lm = layer_metrics[d]
            w.writerow([d] + [json.dumps(lm.get(k), ensure_ascii=False)
                              if isinstance(lm.get(k), dict) else lm.get(k)
                              for k in keys])

    with (OUT / "break_transition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_layer", "target_depth", "target_type",
                                          "target_side", "target_superclass", "n_cells",
                                          "n_synapses", "weight_exc", "weight_inh",
                                          "weight_total", "inh_fraction",
                                          "target_is_dn"])
        w.writeheader()
        for r in trans_rows:
            w.writerow(r)

    # ---- figures ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = list(range(MAX_DEPTH + 1))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    ax = axes[0]
    x = np.arange(len(ds))
    obs = [layer_metrics[d].get("direction_share", np.nan) * 100 for d in ds]
    nul = [layer_metrics[d].get("direction_share_null", np.nan) * 100 for d in ds]
    ax.bar(x - 0.2, obs, 0.4, color="tab:orange", label="наблюдаем")
    ax.bar(x + 0.2, nul, 0.4, color="0.65", label="нуль (метки перемешаны)")
    for i, d in enumerate(ds):
        lm = layer_metrics[d]
        if lm.get("n_active"):
            ax.text(i - 0.2, obs[i] + 2, f"{lm['n_carrying']}/{lm['n_active']}",
                    ha="center", fontsize=8)
    if first_bad is not None:
        ax.axvline(last_good + 0.5, color="tab:red", ls="--", lw=2)
        ax.text(last_good + 0.55, 92, "разрыв", color="tab:red", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"depth {d}\n({layer_metrics[d]['n_groups']} групп)" for d in ds],
                       fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel(f"доля активных групп с AUC>{args.carry_threshold}, %")
    ax.set_title("сколько клеток несут направление")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for d in ds:
        ii = [r for r in rows if r["depth_bin"] == d and r["active"]]
        if ii:
            ax.scatter([r["auc_change_64"] for r in ii],
                       [r["auc_change_65"] for r in ii], s=16, alpha=0.6,
                       label=f"depth {d} (n={len(ii)})")
    ax.axhline(args.carry_threshold, color="0.4", ls="--")
    ax.axvline(args.carry_threshold, color="0.4", ls="--")
    ax.axhline(0.5, color="k", ls=":"); ax.axvline(0.5, color="k", ls=":")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("AUC, seed 64"); ax.set_ylabel("AUC, seed 65")
    ax.set_title("различимость на двух seed\nправый верхний квадрат = несут направление")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    ax = axes[2]
    for d in ds:
        ii = [r["seed_r"] for r in rows if r["depth_bin"] == d and r["active"]]
        if ii:
            ax.hist(ii, bins=np.linspace(-1, 1, 25), alpha=0.55, label=f"depth {d}")
    ax.axvline(0.8, color="0.4", ls="--")
    ax.set_xlabel("корреляция seed 64 ↔ seed 65")
    ax.set_ylabel("групп")
    ax.set_title("совпадение двух seed")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"P05.1 — доля клеток, несущих направление поворота, по глубине. "
                 f"Перелом: depth {last_good} -> {first_bad}.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "signal_by_layer.png", dpi=125)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    for d in ds:
        ii = [r["seed_r"] for r in rows if r["depth_bin"] == d and r["active"]]
        if ii:
            ax.hist(ii, bins=np.linspace(-1, 1, 25), alpha=0.55, label=f"depth {d}")
    ax.axvline(0.8, color="tab:green", ls="--", label="порог 0.8")
    ax.set_xlabel("корреляция seed 64 ↔ seed 65")
    ax.set_ylabel("групп"); ax.set_title("совпадение seed по группам")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ag = [layer_metrics[d].get("polarity_agreement", np.nan) * 100 for d in ds]
    ax.bar(np.arange(len(ds)), ag,
           color=["tab:green" if (v == v and v >= 70) else "tab:red" for v in ag])
    for i, v in enumerate(ag):
        if v == v:
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    ax.axhline(50, color="k", ls=":", label="случай")
    ax.set_xticks(np.arange(len(ds)))
    ax.set_xticklabels([f"depth {d}" for d in ds])
    ax.set_ylim(0, 108)
    ax.set_ylabel("групп с совпавшей полярностью, %")
    ax.set_title("согласие направления связи между seed")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("P05.1 — воспроизводимость: сигнал против шума", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "seed_reproducibility.png", dpi=125)
    plt.close(fig)

    # ---- where the output goes, and what each group feeds -------------------
    fig = plt.figure(figsize=(16.5, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.95, 0.95],
                          hspace=0.42, wspace=0.45)

    ax = fig.add_subplot(gs[0, :2])
    show = trans_rows[:24]
    y = np.arange(len(show))[::-1]
    cols = ["tab:red" if r["target_is_dn"]
            else ("tab:purple" if r["inh_fraction"] > 0.5 else "tab:blue")
            for r in show]
    ax.barh(y, [r["weight_total"] for r in show], color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['target_type'][:14]} {r['target_side']}  "
                        f"({r['n_cells']} кл., глубина {r['target_depth']:.1f})"
                        for r in show], fontsize=8)
    wmax = max(r["weight_total"] for r in show)
    for yi, r in zip(y, show):
        ax.text(r["weight_total"] + wmax * 0.015, yi,
                f"{r['inh_fraction'] * 100:.0f}% торм.", va="center", fontsize=7,
                color="0.35")
    ax.set_xlim(0, wmax * 1.18)
    ax.set_xlabel("суммарный вес связей от последнего хорошего слоя")
    ax.set_title(f"куда уходит выход depth {last_good} — 24 крупнейших цели\n"
                 f"(синее — возбуждающие цели, фиолетовое — преимущественно тормозные, "
                 f"красное — нисходящие нейроны)")
    ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[0, 2])
    dn = sorted([r for r in trans_rows if r["target_is_dn"]],
                key=lambda r: -r["weight_total"])[:10]
    if dn:
        yd = np.arange(len(dn))[::-1]
        ax.barh(yd, [r["weight_total"] for r in dn], color="tab:red")
        ax.set_yticks(yd)
        ax.set_yticklabels([f"{r['target_type'][:12]} {r['target_side']}"
                            for r in dn], fontsize=8)
        ax.set_xlabel("вес")
        ax.set_title(f"крупнейшие нисходящие получатели\n"
                     f"всего {dn_w / total_w * 100:.3f}% веса уходит в DN")
        ax.grid(alpha=0.3, axis="x")

    for i, pg in enumerate(per_group[:6]):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        rec = pg["top_recipients"][:5]
        yr = np.arange(len(rec))[::-1]
        ax.barh(yr, [r["weight"] for r in rec], color="tab:blue")
        ax.set_yticks(yr)
        ax.set_yticklabels([f"{r['type'][:13]} {r['side']}" for r in rec], fontsize=7)
        ax.set_title(f"{pg['cell_type'][:14]} {pg['side']}\n"
                     f"AUC {pg['auc_change_64']:.2f}/{pg['auc_change_65']:.2f}, "
                     f"r {pg['seed_r']:+.2f}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle(f"P05.1 — переход depth {last_good} -> {first_bad}: "
                 f"куда и с какой силой. Ниже — крупнейшие получатели отдельных групп.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "transition_graph.png", dpi=125)
    plt.close(fig)

    # ---- report -----------------------------------------------------------
    (OUT / "report.json").write_text(json.dumps({
        "inputs": {"rec64": str(REC64), "rec65": str(REC65), "groups": str(GROUPS)},
        "verification": checks,
        "n_steps": int(len(t)), "duration_s": float(t[-1] - t[0]),
        "n_groups": ng, "smooth_s": SMOOTH_S, "activity_floor_hz": args.activity_floor,
        "carry_threshold": args.carry_threshold, "n_permutations": args.n_perm,
        "n_turns": len(turns), "n_right": int(is_R.sum()),
        "n_left": int((~is_R).sum()),
        "labels_used": "только подтверждённые глазами LEFT/RIGHT; FWD, STATIC, "
                       "REJECT, UNCLEAR исключены",
        "excluded_groups": {
            "empty_cell_type_grab_bags": int(junk.sum()),
            "aggregate_duplicates": int(is_agg.sum())},
        "layer_metrics": {str(k): v for k, v in layer_metrics.items()},
        "why_fraction_not_median": "медиана AUC по слою разбавлена клетками, которые "
                                   "направление не несут: у глубины 2 она даёт 0.66 "
                                   "против нуля 0.57 и выглядит как слабая потеря, "
                                   "тогда как 82 из 364 нейронов различают повороты "
                                   "с AUC>0.8 на обоих seed там, где случай не даёт "
                                   "ни одного",
        "depth1_selection_caveat": "глубина 1 — это 27 типов, отобранных в P0.1K "
                                   "именно за направленную избирательность, поэтому "
                                   "её высокая доля частично следствие отбора; "
                                   "глубины 2 и 3 не отбирались и являются честной "
                                   "оценкой",
        "break": {"last_good_layer": last_good, "first_bad_layer": first_bad,
                  "direction_share_before": shares.get(last_good),
                  "direction_share_after": shares.get(first_bad),
                  "reproducibility_before": repro.get(last_good),
                  "reproducibility_after": repro.get(first_bad)},
        "transition": {"n_source_groups": len(good_types),
                       "n_target_types": len(trans_rows),
                       "weight_by_class": dict(class_w),
                       "cells_by_class": dict(class_cells),
                       "weight_to_dn_fraction": dn_w / total_w if total_w else 0.0,
                       "top_targets": trans_rows[:25],
                       "per_group_top_recipients": per_group},
        "top_paths_that_lose_signal": [
            {"target_type": r["target_type"], "target_side": r["target_side"],
             "n_synapses": r["n_synapses"], "weight_total": r["weight_total"],
             "inh_fraction": r["inh_fraction"],
             "target_superclass": r["target_superclass"]} for r in trans_rows[:3]],
        "verbatim": "ничего не обучалось, веса MaleCNS не менялись, пороги не "
                    "подбирались, новый прогон мозга не запускался",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- final summary ----------------------------------------------------
    def fmt(v, spec=".3f"):
        return format(v, spec) if v is not None and np.isfinite(v) else "—"

    print("\n" + "=" * 66)
    print(f"LAST GOOD LAYER:  depth {last_good}")
    print(f"FIRST BAD LAYER:  depth {first_bad}")
    print(f"BREAK:            depth {last_good} -> depth {first_bad}")
    print(f"DIRECTION BEFORE: {fmt(shares.get(last_good) * 100, '.0f')}% несущих "
          f"(медиана AUC {fmt(layer_metrics[last_good].get('direction_score_median_auc'))})")
    print(f"DIRECTION AFTER:  "
          f"{fmt(shares.get(first_bad) * 100, '.0f') if first_bad is not None else '—'}% "
          f"несущих (медиана AUC "
          f"{fmt(layer_metrics.get(first_bad, {}).get('direction_score_median_auc'))})")
    print(f"SEED AGREEMENT BEFORE: {fmt(repro.get(last_good), '+.3f')}")
    print(f"SEED AGREEMENT AFTER:  {fmt(repro.get(first_bad), '+.3f')}")
    print(f"\nНУЛЬ: доля несущих при перемешанных метках — "
          + ", ".join(f"depth {d}: {fmt(layer_metrics[d].get('direction_share_null') * 100, '.1f')}%"
                      for d in ds))
    print(f"ВЫХОД ХОРОШЕГО СЛОЯ: {dn_w / total_w * 100:.3f}% веса идёт в нисходящие "
          f"нейроны")
    print("\nTOP PATHS THAT LOSE THE SIGNAL:")
    for i, r in enumerate(trans_rows[:3], 1):
        print(f"{i}. {r['target_type']} {r['target_side']} — {r['n_synapses']} "
              f"синапсов, вес {r['weight_total']:.0f}, "
              f"{r['inh_fraction'] * 100:.0f}% тормозных, класс {r['target_superclass']}")
    print("=" * 66)
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
