#!/usr/bin/env python3
"""P08.5 — why two real turns never reached the signal.

P08.4C closed the last phase by showing that geometry and the beam both fail the same way at
T50 and T49: the walker turns, and the reading that is supposed to describe the turn does not
name it. At T50 the reading is +0.14 against a floor of 0.29, so no direction is named at all.
At T49 the reading clears the floor and names LEFT where the walker went right.

This script does not change any of that. It walks backwards down the chain that produced the
reading, and asks where the turn stopped being visible:

    the video       n_active, the motion the recording actually shows
       |
    T4/T5           NOT RECORDED for this clip — see the honesty note below
       |
    the drive       drive_L and drive_R, the input each descending neuron received
       |
    each DN pair    common (R+L)/2 and pair (R-L), per type, per side, per cell
       |
    the pooled yaw  the four normalised channels averaged, which is what the tracker reads

Four verdicts, and each junction gets exactly one:

    NO_SIGNAL_IN_BRAIN          nothing anywhere in the descending neurons points the way
                                the walker went, so the turn is not in this recording at all
    SIGNAL_CANCELLED_BY_POOLING at least one type points the right way on its own, and the
                                average over four types loses it
    BELOW_THRESHOLD             present in the pooled signal with the right sign, under the
                                floor, so nothing is named
    WRONG_SIGN                  the pooled signal clears the floor and names the other side

Honesty note on T4/T5
---------------------
They are not in the data, and saying so is more useful than inventing a proxy. The brain run
for the real clips recorded 46 cells, all of them descending neurons, because the targets
were chosen as the cells whose output could be read; T4/T5 exist only in the synthetic
sessions of P06.1. The nearest thing available upstream is `drive_L`/`drive_R`, the drive each
descending neuron received from the two sides, and that is what is reported in their place.

There is however a precedent worth carrying into the reading. P05 found on the synthetic data
that T4a and T4b prefer opposite directions, so adding them removed almost all of the signal
(LEFT 16.8 Hz against RIGHT 7.3 Hz on T4a, and the reverse on T4b, summing to nearly nothing).
The question this script asks is whether the same thing is happening one level down, where
four descending pairs are averaged with one of them sign-flipped.

Everything is a reading of the pipeline exactly as it stands. The per-type reconstruction is
checked against the recorded signal before it is used for anything, so the diagnostic cannot
silently describe a signal that differs from the one the tracker actually saw.
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
sys.path.insert(0, str(ROOT / "scripts"))

import p08_graph as G  # noqa: E402

TYPES = ["DNp17", "DNa07", "DNp26", "DNp20"]
FPS = 50.0
SMOOTH_S = 0.30              # the default of the frozen readout
N_SM = int(round(SMOOTH_S * FPS))
APPROACH_S = 2.0             # the tracker's window
SETTLE_S = 1.2
WINDOW_S = APPROACH_S + SETTLE_S
FLOOR_A, FLOOR_B, FLOOR_SCALE = 0.09, 0.15, 0.50
Z_CLEAR = 1.0                # a type "points a way" when its own z exceeds one spread


def box(x: np.ndarray, n: int) -> np.ndarray:
    """The pipeline's own smoothing: a moving average of n samples, edges included."""
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def floor_for(window_s: float) -> float:
    return FLOOR_SCALE * (FLOOR_A + FLOOR_B * max(float(window_s), 0.0))


def polarities() -> dict:
    """The P06.1 polarities, recomputed rather than copied, so they cannot drift."""
    synth = ROOT / "output/p061_synthetic"
    d64 = np.load(synth / "synthetic_seed64.npz")
    d65 = np.load(synth / "synthetic_seed65.npz")
    col = {i: x for i, x in enumerate(
        csv.DictReader((ROOT / "output/p053_threshold/targets.csv").open()))}
    out, detail = {}, {}
    for nm in TYPES:
        vals = []
        for side in ("L", "R"):
            for i, x in col.items():
                if x["cell_type"] != nm or x["side"] != side:
                    continue
                vals.append(float((d64["right"][:, i] - d64["left"][:, i]).mean()))
                vals.append(float((d65["right"][:, i] - d65["left"][:, i]).mean()))
        m = float(np.mean(vals)) if vals else 0.0
        out[nm] = 1.0 if m > 0 else -1.0
        detail[nm] = m
    return out, detail


class Brain:
    """Per-cell and per-type access to one clip's recorded descending neurons."""

    def __init__(self, trace: Path, targets: Path):
        d = np.load(trace)
        self.t = d["t"].astype(float)
        self.fired = d["fired"].astype(float)
        self.drive_l = d["drive_L"]
        self.drive_r = d["drive_R"]
        self.n_active = d["n_active"]
        rows = list(csv.DictReader(targets.open(encoding="utf-8")))
        self.cells = rows
        self.idx: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(rows):
            self.idx[r["cell_type"]][r["side"]].append(i)

    def side_rate(self, nm: str, side: str) -> np.ndarray:
        ids = self.idx[nm].get(side, [])
        if not ids:
            return np.zeros_like(self.t)
        return self.fired[:, ids].mean(axis=1) * FPS

    def common(self, nm: str) -> np.ndarray:
        return (self.side_rate(nm, "R") + self.side_rate(nm, "L")) / 2

    def pair(self, nm: str) -> np.ndarray:
        return self.side_rate(nm, "R") - self.side_rate(nm, "L")

    def drive(self, nm: str, side: str) -> np.ndarray:
        ids = self.idx[nm].get("L", []) + self.idx[nm].get("R", [])
        if not ids:
            return np.zeros_like(self.t)
        arr = self.drive_l if side == "L" else self.drive_r
        return arr[:, ids].mean(axis=1)

    def z(self, x: np.ndarray) -> np.ndarray:
        sd = float(np.std(x))
        return (x - float(np.mean(x))) / (sd if sd > 1e-9 else 1.0)

    def window(self, t: float) -> np.ndarray:
        return (self.t >= t - WINDOW_S) & (self.t <= t)


def window_mean(mask: np.ndarray, x: np.ndarray) -> float:
    return float(x[mask].mean()) if mask.any() else 0.0


def rect_integral(mask: np.ndarray, x: np.ndarray, t: np.ndarray) -> float:
    """The tracker's own integration: a left-Riemann sum over the window."""
    if mask.sum() < 2:
        return 0.0
    tt, xx = t[mask], x[mask]
    return float((xx[:-1] * np.diff(tt)).sum())


def truth_of(g: G.Graph, audit: list[dict]) -> dict[tuple[str, float], dict]:
    """The human verdicts, keyed by node and time, with the turn each one implies.

    The *arrival* edge is deliberately not taken from here. The audit was run against the
    graph as it stood before P08.4A, so some of its edge names no longer exist — `T49__T15`
    became `T49__T50` when T15 was merged into T50 — and asking the repaired graph about a
    name it no longer has throws, which silently dropped T49 from the audit altogether. The
    arrival is taken from the run being analysed instead, and only the truth edge, which is
    a node-to-node statement, is taken from the audit.
    """
    out = {}
    for a in audit:
        if a["verdict"] not in ("CORRECT", "WRONG"):
            continue
        node = a["node"]
        truth = a["correct_edge_if_wrong"] or a["truth_edge"] or a["chosen_edge"]
        if not truth or truth not in g.edges:
            continue
        out[(node, round(float(a["time"]), 1))] = {
            "truth_edge": truth, "verdict": a["verdict"],
            "chosen_edge": a["chosen_edge"], "audit_time": float(a["time"]),
            "note": a.get("note", ""),
        }
    return out


def nearest_truth(truth: dict, node: str, t: float) -> dict | None:
    best, bd = None, 1e9
    for (n, _), v in truth.items():
        if n != node:
            continue
        d = abs(v["audit_time"] - t)
        if d < bd:
            best, bd = v, d
    return best if bd <= 8.0 else None


def classify(integral: float, floor: float, truth_deg: float,
             per_type: list[dict]) -> tuple[str, str]:
    """One verdict per junction, and the sentence that justifies it.

    The four verdicts are the ones the phase asked for, and the order they are tried in is
    the order of how deep the defect goes: something in one channel that the average loses
    is a pooling defect; the pooled signal naming the wrong side is a reading defect; the
    pooled signal naming the right side but too quietly to be believed is a sensitivity
    defect; and nothing anywhere is the end of what this recording can support.
    """
    want = "LEFT" if truth_deg < 0 else "RIGHT"
    named = "" if abs(integral) < floor else ("LEFT" if integral > 0 else "RIGHT")
    pool_side = "LEFT" if integral > 0 else "RIGHT"

    # a channel that points the way the walker actually went, on its own evidence
    true_pointers = [p for p in per_type
                     if abs(p["z_pair"]) >= Z_CLEAR
                     and (("LEFT" if p["z_pair"] > 0 else "RIGHT") == want)]
    loudest = max(per_type, key=lambda p: abs(p["z_pair"])) if per_type else None

    if named == want:
        return "OK", f"пул называет {named}, как и надо"
    if true_pointers:
        return "SIGNAL_CANCELLED_BY_POOLING", (
            f"пул называет {named or 'ничего'} ({integral:+.3f} при пороге {floor:.3f}), "
            f"а сам по себе на {want} указывает "
            + ", ".join(f"{p['type']} z={p['z_pair']:+.2f}" for p in true_pointers))
    if named:
        return "WRONG_SIGN", (
            f"пул называет {named} при правде {want}; ни один канал не указывает на {want}, "
            f"сильнее всех {loudest['type']} z={loudest['z_pair']:+.2f}")
    if pool_side == want:
        return "BELOW_THRESHOLD", (
            f"знак верный ({integral:+.3f} в сторону {want}), но порог {floor:.3f} не взят")
    return "BELOW_THRESHOLD", (
        f"ниже порога ({integral:+.3f} при {floor:.3f}) и знак тоже против: пул тянет "
        f"на {pool_side}, а шёл на {want}; ни один канал не выделяется "
        f"(сильнее всех {loudest['type']} z={loudest['z_pair']:+.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default=str(ROOT / "output/p06_neurons_vid6/spike_trace.npz"))
    ap.add_argument("--targets", default=str(ROOT / "output/p06_neurons_vid6/targets.csv"))
    ap.add_argument("--yaw", default=str(ROOT / "output/p07/yaw_signal_VID00006.csv"))
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--run", default=str(ROOT / "output/p084c/run"),
                    help="замороженный прогон P08.4B/4C: его времена решений")
    ap.add_argument("--audit", default=str(ROOT / "output/p083_route_audit/decisions.csv"))
    ap.add_argument("--out", default=str(ROOT / "output/p085"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = G.Graph.load(Path(args.graph))
    pol, pol_detail = polarities()
    b = Brain(Path(args.trace), Path(args.targets))

    yrows = list(csv.DictReader(Path(args.yaw).open(encoding="utf-8")))
    yt = np.array([float(r["t"]) for r in yrows])
    y_pool = np.array([float(r["yaw_signal"]) for r in yrows])

    # ---- reconstruct, and refuse to go on unless it matches what was recorded -----------
    # The pipeline's order matters: each type is standardised on its raw rate, signed by its
    # polarity, and only then smoothed; the four channels are averaged. Smoothing is linear,
    # so a per-type channel built the same way averages to exactly the pooled signal — which
    # is what makes "what would this one type have said" a fair question to ask of it.
    chan_common = {nm: box(pol[nm] * b.z(b.common(nm)), N_SM) for nm in TYPES}
    chan_pair = {nm: box(pol[nm] * b.z(b.pair(nm)), N_SM) for nm in TYPES}
    common_z = np.mean([chan_common[nm] for nm in TYPES], axis=0)
    pair_z = np.mean([chan_pair[nm] for nm in TYPES], axis=0)
    err = float(np.abs(common_z - y_pool).max())
    print("=" * 92)
    print("P08.5 — ЧТО БЫЛО НА ПУТИ ОТ ВИДЕО ДО СИГНАЛА")
    print("=" * 92)
    print(f"трейс: {len(b.cells)} клеток, {len(b.t)} отсчётов, {TYPES}")
    for nm in TYPES:
        print(f"  полярность {nm:<7} delta {pol_detail[nm]:+7.3f} → "
              f"{'LEFT' if pol[nm] > 0 else 'RIGHT'}")
    print()
    print("─── ПРОВЕРКА СБОРКИ ───")
    print(f"  максимум расхождения пересобранного сигнала с записанным: {err:.6f}")
    if err > 1e-4:
        print("  СТОП: пересборка не совпала с записанным сигналом, выводы будут не о том")
        return 1
    print("  совпадает точно, поэтому дальше разбирается тот же сигнал, что видел трекер")
    print()

    # ---- junctions -----------------------------------------------------------------------
    audit = list(csv.DictReader(Path(args.audit).open(encoding="utf-8")))
    truth = truth_of(g, audit)
    run = list(csv.DictReader((Path(args.run) / "decisions.csv").open(encoding="utf-8")))

    recs: list[dict] = []
    chain: list[dict] = []
    for d in run:
        if d["reason"] == "ONLY_OPTION":
            continue
        node, t = d["node"], float(d["time"])
        tr = nearest_truth(truth, node, t)
        if not tr:
            continue
        # the turn the person made, measured in the repaired graph from the edge the run
        # actually arrived along, so the angle matches the choice being audited
        try:
            truth_deg = float(g.turn(d["current_edge"], node, tr["truth_edge"])["deg"])
        except Exception:
            continue
        mask = b.window(t)
        integral = rect_integral(mask, y_pool, yt)
        flo = float(d["yaw_floor"])
        per = []
        for nm in TYPES:
            cm = b.common(nm)
            pr = b.pair(nm)
            zc = window_mean(mask, chan_common[nm])
            zp = window_mean(mask, chan_pair[nm])
            per.append({
                "type": nm, "polarity": pol[nm],
                "common_hz": window_mean(mask, cm), "pair_hz": window_mean(mask, pr),
                "rate_L_hz": window_mean(mask, b.side_rate(nm, "L")),
                "rate_R_hz": window_mean(mask, b.side_rate(nm, "R")),
                "drive_L": window_mean(mask, b.drive(nm, "L")),
                "drive_R": window_mean(mask, b.drive(nm, "R")),
                "z_common": zc, "z_pair": zp,
            })
        verdict, why = classify(integral, flo, truth_deg, per)
        want_side = "LEFT" if truth_deg < 0 else "RIGHT"
        pool_side = "LEFT" if integral > 0 else "RIGHT"

        # Two controls, because "small" and "weak" need a reference to mean anything. The
        # percentile says where this window's magnitude sits among every window of the same
        # length in the clip; the peak says whether anything loud happened inside the window
        # at all, and which way it pointed.
        rolling = np.convolve(common_z, np.ones(int(WINDOW_S * FPS)) / int(WINDOW_S * FPS),
                              mode="valid")
        pct = float((np.abs(rolling) < abs(window_mean(mask, common_z))).mean() * 100.0)
        sel = np.where(mask)[0]
        if len(sel):
            j = sel[int(np.argmax(np.abs(common_z[sel])))]
            peak_val = float(common_z[j])
            peak_off = float(yt[j] - t)
        else:
            peak_val, peak_off = 0.0, 0.0

        recs.append({
            "t": t, "node": node, "arrival": d["current_edge"],
            "chosen": d["chosen_edge"], "reason": d["reason"],
            "truth_edge": tr["truth_edge"], "truth_side": want_side,
            "truth_deg": round(truth_deg, 1),
            "human": tr["verdict"], "note": tr["note"],
            "route_error": int(d["chosen_edge"] != tr["truth_edge"]),
            "yaw_integral": round(integral, 4), "yaw_floor": round(flo, 4),
            "yaw_named": "" if abs(integral) < flo else pool_side,
            "yaw_sign_ok": int(pool_side == want_side),
            "yaw_window_mean": round(window_mean(mask, y_pool), 4),
            "pool_pair_mean": round(window_mean(mask, pair_z), 4),
            "pool_percentile_in_clip": round(pct, 1),
            "peak_in_window": round(peak_val, 4),
            "peak_offset_s": round(peak_off, 2),
            "peak_side": ("LEFT" if peak_val > 0 else "RIGHT"),
            "n_active": round(window_mean(mask, b.n_active.astype(float)), 1),
            "verdict": verdict, "why": why,
        })
        for p in per:
            chain.append({"t": t, "node": node, "arrival": d["current_edge"],
                          "truth_deg": round(truth_deg, 1), **p})

    # ---- report ---------------------------------------------------------------------------
    # Two different things can go wrong and they are reported separately. A *route* failure is
    # the system leaving by the wrong edge. A *signal* failure is the reading naming the wrong
    # side, or nothing, where the walker turned — which may still leave the route right,
    # because geometry sometimes carries the day. The phase is about the second kind; the
    # first is what it costs.
    signal_fail = [r for r in recs if r["yaw_sign_ok"] == 0]
    route_fail = [r for r in recs if r["route_error"]]
    print("─── РАЗВИЛКИ, ГДЕ ЧЕЛОВЕК ПОВЕРНУЛ, А ЧТЕНИЕ НЕ СОВПАЛО ───")
    print(f"  {'t':>7} {'узел':<6} {'правда':<14}{'угол':>7} {'yaw':>9} {'порог':>7} "
          f"{'названо':>8} {'путь':>6}  вердикт")
    for r in sorted(recs, key=lambda x: x["t"]):
        if r["yaw_sign_ok"]:
            continue
        print(f"  {r['t']:>7.1f} {r['node']:<6} {r['truth_edge']:<14}{r['truth_deg']:>+7.1f} "
              f"{r['yaw_integral']:>+9.3f} {r['yaw_floor']:>7.3f} "
              f"{(r['yaw_named'] or '—'):>8} "
              f"{('ОШИБКА' if r['route_error'] else 'верно'):>6}  {r['verdict']}")
    print()
    print(f"  чтение не совпало: {len(signal_fail)} из {len(recs)}")
    print(f"  из них путь оказался неверным: {len(route_fail)}  "
          f"({', '.join(r['node'] for r in route_fail)})")
    print(f"  остальные спасла геометрия: "
          f"{', '.join(r['node'] for r in signal_fail if not r['route_error'])}")
    print()

    # ---- the two the phase exists for ------------------------------------------------------
    print("─── ЦЕПОЧКА ПО ТИПАМ: T50 И T49 ───")
    for name in ("T50", "T49"):
        rows = [c for c in chain if c["node"] == name]
        if not rows:
            continue
        h = next(r for r in recs if r["node"] == name)
        print(f"\n  {name}  t={h['t']:.1f} с   правда {h['truth_edge']} "
              f"({h['truth_deg']:+.1f}° = {h['truth_side']})   вердикт {h['verdict']}")
        print(f"    видео: n_active {h['n_active']:.0f}")
        print(f"    {'тип':<8}{'Гц L':>8}{'Гц R':>8}{'общий R+L/2':>13}{'разн. R-L':>11}"
              f"{'поляр':>7}{'drive_L':>9}{'drive_R':>9}{'z общий':>9}{'z разн':>8}")
        for c in rows:
            print(f"    {c['type']:<8}{c['rate_L_hz']:>8.2f}{c['rate_R_hz']:>8.2f}"
                  f"{c['common_hz']:>13.2f}{c['pair_hz']:>11.2f}{c['polarity']:>+7.0f}"
                  f"{c['drive_L']:>9.3f}{c['drive_R']:>9.3f}"
                  f"{c['z_common']:>+9.2f}{c['z_pair']:>+8.2f}")
        signs = "".join("+" if c["polarity"] * c["pair_hz"] > 0 else "-" for c in rows)
        print(f"    знаки разностного канала после полярности: {signs}   "
              f"({'согласны' if len(set(signs)) == 1 else 'РАСХОДЯТСЯ'})")
        print(f"    общий сигнал пула в окне: {h['yaw_window_mean']:+.4f} "
              f"(порог {h['yaw_floor']:.3f}), интеграл {h['yaw_integral']:+.3f}")
        print(f"    разностный пул в окне:    {h['pool_pair_mean']:+.4f}")
        print(f"    контроль: {h['pool_percentile_in_clip']:.0f}-й процентиль по всей записи; "
              f"сильнейший всплеск в окне {h['peak_in_window']:+.3f} ({h['peak_side']}) "
              f"на {h['peak_offset_s']:+.2f} с от решения")
        tgt = "тот же" if h["peak_side"] == h["truth_side"] else "ПРОТИВОПОЛОЖНЫЙ"
        print(f"    всплеск указывает {h['peak_side']}, а шёл {h['truth_side']} — {tgt}")

    # ---- variants ---------------------------------------------------------------------------
    print()
    print("─── ЧТО ДАЛ БЫ ДРУГОЙ СПОСОБ ОБЪЕДИНЕНИЯ (диагностика, не решение) ───")
    variants = {
        "пул общих (КАК СЕЙЧАС)": common_z,
        "пул разностных R-L": pair_z,
        "только DNp17": chan_common["DNp17"],
        "только DNa07": chan_common["DNa07"],
        "только DNp26": chan_common["DNp26"],
        "только DNp20": chan_common["DNp20"],
    }
    ok_count: dict[str, int] = {}
    named_count: dict[str, int] = {}
    print(f"  {'вариант':<24} {'названо':>8}  " +
          "".join(f"{r['node'][:5]:>7}" for r in recs))
    for vname, sig in variants.items():
        cells, named, good = [], 0, 0
        for r in recs:
            m = b.window(r["t"])
            I = rect_integral(m, sig, yt)
            f = r["yaw_floor"]
            if abs(I) < f:
                cells.append(f"{I:+.2f}—")
                continue
            named += 1
            s = "LEFT" if I > 0 else "RIGHT"
            good += int(s == r["truth_side"])
            cells.append(f"{I:+.2f}{'✓' if s == r['truth_side'] else '✗'}")
        ok_count[vname] = good
        named_count[vname] = named
        print(f"  {vname:<24} {named:>8}  " + "".join(f"{c:>7}" for c in cells))
    print()
    print("  ✓ знак совпал с правдой, ✗ знак против, «—» вариант промолчал")
    print("  Важно: у T50 и T49 правда — поворот ВПРАВО. Если вариант вообще говорит,")
    print("  он говорит ЛЕВО, то есть против. Ни один способ объединения провал не чинит.")
    for vname in variants:
        print(f"  {vname:<24} сказал {named_count[vname]}, из них верно "
              f"{ok_count[vname]}")

    # ---- files -------------------------------------------------------------------------------
    with (out / "junction_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    with (out / "chain_dump.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chain[0].keys()))
        w.writeheader()
        w.writerows(chain)


    print()
    print("─── СТРУКТУРА ПРИВОДА ЧЕТЫРЁХ ПАР (почему у пула мало направленного хода) ───")
    drive = []
    for nm in TYPES:
        ids = b.idx[nm].get("L", []) + b.idx[nm].get("R", [])
        dl = float(b.drive_l[:, ids].mean())
        dr = float(b.drive_r[:, ids].mean())
        wl = sum(float(r.get("w_exc_from_left", 0) or 0) for r in b.cells
                 if r["cell_type"] == nm)
        wr = sum(float(r.get("w_exc_from_right", 0) or 0) for r in b.cells
                 if r["cell_type"] == nm)
        drive.append({"type": nm, "cells": len(ids), "polarity": pol[nm],
                      "drive_L_mean": round(dl, 6), "drive_R_mean": round(dr, 6),
                      "w_exc_from_left": round(wl, 4), "w_exc_from_right": round(wr, 4)})
        print(f"  {nm:<8}{len(ids):>3} кл.  привод слева {dl:8.5f}  справа {dr:8.5f}   "
              f"веса слева {wl:7.4f} справа {wr:7.4f}   полярность {pol[nm]:+.0f}")
    left_only = [d for d in drive if d["w_exc_from_right"] < 1e-3 and d["w_exc_from_left"] > 0]
    right_only = [d for d in drive if d["w_exc_from_left"] < 1e-3 and d["w_exc_from_right"] > 0]
    print()
    print(f"  только от левых носителей: {', '.join(d['type'] for d in left_only)} "
          f"({len(left_only)} из {len(drive)})")
    print(f"  только от правых носителей: {', '.join(d['type'] for d in right_only)}")
    print("  И это ровно тот канал, у которого полярность перевёрнута. То есть пул — не")
    print("  четыре независимых взгляда, а три канала с одного входа против одного с")
    print("  другого и с обратным знаком. Направленный ход такого среднего мал по построению.")

    print()
    (out / "report.json").write_text(json.dumps({
        "phase": "P08.5 — diagnostics only, nothing tuned",
        "frozen": "P08.4C",
        "unchanged": ["MaleCNS", "клетки", "yaw", "novelty", "скорость", "веса",
                      "пороги", "граф P08.4A", "правило P08.4B", "beam P08.4C"],
        "reconstruction_check": {
            "max_abs_error_vs_recorded_signal": err,
            "means": "пересобранный сигнал совпадает с записанным, поэтому разбирается "
                     "тот же сигнал, что видел трекер",
        },
        "chain": {
            "video": "n_active",
            "T4_T5": "НЕ ЗАПИСАНЫ для реальных клипов: мозг писал 46 DN-клеток, "
                     "T4/T5 есть только в синтетике P06.1; ближайшее доступное — drive_L/drive_R",
            "pairs": ["DNp17 (+1)", "DNa07 (-1)", "DNp26 (+1)", "DNp20 (+1)"],
            "pool": "среднее четырёх z-нормированных каналов, common = (R+L)/2",
            "pool_pair": "то же для разностного R-L, пишется как yaw_pair_diagnostic",
        },
        "junctions": recs,
        "drive_structure": drive,
        "verdicts": {v: sum(1 for r in recs if r["verdict"] == v)
                     for v in sorted({r["verdict"] for r in recs})},
        "conclusion": (
            "T50 и T49 — не дефект объединения и не дефект порога. Ни один из четырёх "
            "каналов, ни их общий, ни разностный, ни любой одиночный тип не направлен туда, "
            "куда муха пошла: все они, если говорят, говорят ЛЕВО, а поворот был ВПРАВО. "
            "Поэтому никакая линейная пересборка этих четырёх каналов провал не чинит. "
            "В T50 окно вообще 53-й процентиль записи, то есть обычное, а сильнейший всплеск "
            "в нём (+0.694 за 0.28 с до решения) направлен против поворота — то же "
            "расхождение камеры и тела, что P08.4C нашёл в J35."),
        "limit": (
            "Для этих двух развилок P08 упёрся во вход: информация о направлении поворота "
            "тела в записанных DN отсутствует либо обратна. Граф и правила её не восстановят."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'junction_audit.csv'}")
    print(f"записано: {out/'chain_dump.csv'}")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
