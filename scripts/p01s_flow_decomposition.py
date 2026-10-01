#!/usr/bin/env python3
"""
P0.1S follow-up — separate camera rotation from forward translation in the field.

P0.1S showed the Lucas-Kanade field is reliable (it recovers known shifts to
0.001 px and has no systematic vertical leak) and that on real events its cells
are now *coherent* rather than random: on the failed events ~63% of cells agree
with each other on the wrong sign. Coherent-and-wrong is not noise, so something
systematic is in the field. The obvious candidate is the other motion that is
always present in this footage: forward translation.

A camera moving forward makes the image flow radially away from the focus of
expansion, so the horizontal component grows linearly with azimuth:

    dx(x)  =  a  +  b * xn(x)          xn = azimuth in [-1, +1]

    a = uniform part  -> camera ROTATION (yaw)
    b = slope part    -> camera TRANSLATION (forward speed)

This is the standard decomposition and it is label-free: a and b are fitted to
the measured field alone. The probe asks whether the rotation coefficient `a`
separates LEFT/FWD/RIGHT better than the raw pooled mean — i.e. whether the
'mixed' events were really translation domination rather than a measurement
error.

Also reported: the per-frame trace of `a`, so the P0.1P question ("is there a
stable correct stretch inside a turn?") can be re-asked on a translation-free
signal.

Frontend only. No MaleCNS, no injection, nothing trained.

Usage:
    PYTHONPATH=. python scripts/p01s_flow_decomposition.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS = {
    "left_65":   (63.5, "LEFT"),
    "left_81":   (78.5, "LEFT"),
    "left_108":  (106.0, "LEFT"),
    "right_67":  (66.0, "RIGHT"),
    "right_210": (209.0, "RIGHT"),
    "right_216": (214.0, "RIGHT"),
    "fwd_140":   (137.5, "FWD"),
}
# Frozen convention verified on left_65 / right_216: LEFT -> negative rotation.
WANT = {"LEFT": -1, "RIGHT": +1, "FWD": 0}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DT = 0.02
GRID_W, GRID_H = 16, 8
XN = np.linspace(-1.0, 1.0, GRID_W)


def design() -> np.ndarray:
    """[cells, 2]: intercept (rotation) and azimuth slope (expansion)."""
    xn = np.tile(XN, GRID_H)
    return np.stack([np.ones_like(xn), xn], axis=1)


def capture(mode: str, ew: int, eh: int, t0: float, t1: float) -> list[dict]:
    front = DirectionalMotionFrontend(
        FlowConfig(n_azimuth_bins=128, ema_tau_s=0.5, brain_dt=DT, crop_mode="center_band",
                   field_mode=mode, grid_w=GRID_W, grid_h=GRID_H)
    )
    out = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, DT):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (ew, eh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        d = front.process(img)
        field = d.motion_field_2d if d.motion_field_2d else None
        out.append({"t": t, "field": np.asarray(field, dtype=np.float64).reshape(-1)
                    if field is not None else None,
                    "yaw": float(d.yaw_frozen), "held": bool(d.detector_bank.get("lk_held", 0.0))})
    return out


def fit(rows: list[dict], design_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame least-squares (a, b); frames without a field are dropped."""
    keep = [r for r in rows if r["field"] is not None]
    f = np.stack([r["field"] for r in keep], axis=0) if keep else np.zeros((0, design_m.shape[0]))
    if not len(f):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    pinv = np.linalg.pinv(design_m)
    ab = f @ pinv.T                       # [frames, 2]
    return keep and np.array([r["t"] for r in keep]), ab[:, 0], ab[:, 1]


def runs(mask: np.ndarray) -> tuple[int, int]:
    best = cur = 0
    best_at = 0
    for i, v in enumerate(mask):
        cur = cur + 1 if v else 0
        if cur > best:
            best, best_at = cur, i
    return best, best_at - best + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01s_decomposition")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    D = design()

    print("P0.1S — flow decomposition: dx(x) = a + b * azimuth")
    print("  a = uniform (rotation/yaw),  b = slope (forward translation / expansion)\n")
    print(f"  {'event':10s} {'label':5s} {'raw mean':>10s} {'a (yaw)':>10s} {'b (exp)':>10s} "
          f"{'a/b':>7s}  {'a sign ok':>9s}")

    rows, traces = [], {}
    for eid, (start, lab) in EVENTS.items():
        t0, t1 = max(0.0, start - PRE_ROLL_S), start + EVENT_DUR_S
        n_pre = int(round(PRE_ROLL_S / DT))
        rec = {"event": eid, "label": lab}

        for tag, mode, ew, eh in (("lk", "lk", 192, 108), ("grid2d", "grid2d", 192, 108)):
            cap = capture(mode, ew, eh, t0, t1)
            _, a, b = fit(cap[n_pre:], D)
            fields = np.stack([r["field"] for r in cap[n_pre:] if r["field"] is not None], axis=0)
            raw = fields.mean(axis=0)
            want = WANT[lab]
            stats = {"a_median": float(np.median(a)), "b_median": float(np.median(b)),
                     "raw_median": float(np.median(raw))}
            if want:
                stats["a_sign_correct"] = bool(np.sign(np.median(a)) == want)
                stats["b_abs_median"] = float(np.abs(np.median(b)))
                stats["raw_abs_median"] = float(abs(np.median(raw)))
                stats["a_over_b"] = float(
                    abs(np.median(a)) / (abs(np.median(b)) + 1e-9)
                )
                corr = np.sign(a) == want
                longest, at = runs(corr)
                stats["a_frac_correct"] = float(corr.mean())
                stats["a_longest_correct_s"] = longest * DT
                stats["a_longest_start_t"] = float(cap[n_pre:][at]["t"]) if longest else None
            stats["a"] = a.tolist()
            stats["b"] = b.tolist()
            stats["t"] = [r["t"] for r in cap[n_pre:] if r["field"] is not None]
            rec[tag] = stats

        rows.append(rec)
        lk = rec["lk"]
        print(f"  {eid:10s} {lab:5s} {lk['raw_median']:+10.3f} {lk['a_median']:+10.3f} "
              f"{lk['b_median']:+10.3f} "
              f"{lk.get('a_over_b', float('nan')):7.2f}  "
              f"{str(lk.get('a_sign_correct', '-')):>9s}")

        traces[eid] = {"t": lk["t"], "a": lk["a"], "b": lk["b"], "label": lab}

    def acc(key: str, sign_key: str = "a_sign_correct") -> tuple[int, int]:
        ok = tot = 0
        for r in rows:
            if r["label"] == "FWD":
                continue
            tot += 1
            ok += int(bool(r["lk"].get(sign_key)))
        return ok, tot

    a_acc = acc("a")
    print(f"\n  rotation coefficient a: sign accuracy {a_acc[0]}/{a_acc[1]}")
    for eid in ("left_65", "left_81", "left_108", "right_67", "right_210", "right_216"):
        r = next(x for x in rows if x["event"] == eid)
        lk = r["lk"]
        print(f"    {eid:10s} raw {lk['raw_median']:+7.3f}  a {lk['a_median']:+7.3f}  "
              f"b {lk['b_median']:+7.3f}   a correct {lk['a_frac_correct']:.0%} of frames, "
              f"longest correct run {lk['a_longest_correct_s']:.2f}s "
              f"(starts t={lk['a_longest_start_t']})")

    # ---- per-frame trace figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["left_65", "left_81", "left_108", "right_67", "right_210", "right_216", "fwd_140"]
    fig, axes = plt.subplots(len(order), 2, figsize=(14, 2.0 * len(order)), sharex="col")
    for ax_row, eid in zip(axes, order):
        tr = traces[eid]
        t = np.asarray(tr["t"]) - (EVENTS[eid][0] - PRE_ROLL_S)
        want = WANT[tr["label"]]
        for ax, vals, name in ((ax_row[0], tr["a"], "a = rotation (yaw)"),
                               (ax_row[1], tr["b"], "b = expansion (forward)")):
            v = np.asarray(vals)
            if want:
                cols = ["tab:green" if np.sign(x) == want else "tab:red" for x in v]
            else:
                cols = ["tab:gray"] * len(v)
            ax.axhspan(-0.05, 0.05, color="0.9")
            ax.axvline(PRE_ROLL_S, color="0.6", lw=0.8)
            ax.scatter(t, v, s=3, c=cols)
            ax.set_ylabel(f"{eid}\n{name}", fontsize=7)
            ax.grid(alpha=0.25, lw=0.4)
    axes[-1][0].set_xlabel("time in window (s)")
    axes[-1][1].set_xlabel("time in window (s)")
    fig.suptitle("P0.1S — Lucas-Kanade field decomposed into rotation (a) and expansion (b)\n"
                 "green = correct rotation sign for the blind label", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "decomposition.png", dpi=130)
    plt.close(fig)

    # ---- is the improvement real? ----
    raw_acc = sum(
        1 for r in rows
        if r["label"] != "FWD" and np.sign(r["lk"]["raw_median"]) == WANT[r["label"]]
    )
    n_turn = sum(1 for r in rows if r["label"] != "FWD")
    verdict = (
        f"TRANSLATION WAS MASKING ROTATION: on the raw pooled mean the sign is right "
        f"{raw_acc}/{n_turn}; on the rotation coefficient a the sign is right {a_acc[0]}/{n_turn}. "
        "The local field is a reliable measurement, but it mixes rotation with forward-motion "
        "expansion, and the expansion part is large enough to dominate the pooled sign on the "
        "'mixed' events."
        if a_acc[0] > raw_acc
        else f"NOT THE EXPLANATION: the rotation coefficient a gets the sign right "
             f"{a_acc[0]}/{n_turn}, the raw mean {raw_acc}/{n_turn}. Removing the translational "
             "common mode does not recover the sign, so the disagreement on those events is not "
             "caused by forward motion dominating the field."
    )
    print(f"\n  raw pooled mean sign accuracy : {raw_acc}/{n_turn}")
    print(f"  rotation coefficient a        : {a_acc[0]}/{n_turn}")
    print(f"\n=== VERDICT ===\n{verdict}")

    with (out / "events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "label", "mode", "raw_median", "a_median", "b_median",
                    "a_over_b", "a_sign_correct", "a_frac_correct", "a_longest_correct_s"])
        for r in rows:
            for tag in ("lk", "grid2d"):
                s = r[tag]
                w.writerow([r["event"], r["label"], tag, f"{s['raw_median']:.5f}",
                            f"{s['a_median']:.5f}", f"{s['b_median']:.5f}",
                            f"{s.get('a_over_b', float('nan')):.4f}",
                            s.get("a_sign_correct", ""),
                            f"{s.get('a_frac_correct', float('nan')):.4f}",
                            f"{s.get('a_longest_correct_s', float('nan')):.3f}"])
    (out / "report.json").write_text(
        json.dumps({"rows": [{k: v for k, v in r.items() if k != "lk" and k != "grid2d"}
                             | {tag: {kk: vv for kk, vv in r[tag].items()
                                      if kk not in ("a", "b", "t")}
                                for tag in ("lk", "grid2d")} for r in rows],
                    "raw_sign_accuracy": [raw_acc, n_turn],
                    "a_sign_accuracy": [a_acc[0], n_turn],
                    "verdict": verdict}, indent=2),
        encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
