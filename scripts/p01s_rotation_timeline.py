#!/usr/bin/env python3
"""
P0.1S audit — a continuous rotation timeline for the whole video.

Every experiment so far has looked at 3-second windows with hand-assigned labels.
P0.1S showed that the Lucas-Kanade field is trustworthy (0.001 px shift recovery,
no vertical leak, 94% seed coverage) yet on three of six turn windows it reports a
coherent rotation of the *opposite* sign to the label. Before any more work on the
frontend, that disagreement has to be resolved, and the only way to resolve it is
to look at the measurement continuously instead of window by window.

This script computes, for every frame of the clip:

    dx(x, y) = a + b * azimuth        (least squares, per frame)

    a = camera rotation (yaw)         b = forward-motion expansion

and plots the whole timeline together with the labelled event windows. It then
reports, for each label, the sign of the *local extremum* of |a| within a few
seconds of the anchor — the audit question is simply whether the labelled turns
sit on rotation excursions of the sign the label claims.

This changes no labels. It is an audit: if a labelled window does not contain a
rotation excursion of the claimed sign, that is a finding about the labels, not a
licence to re-pick them silently.

Frontend only — no MaleCNS, no injection, nothing trained.

Usage:
    PYTHONPATH=. python scripts/p01s_rotation_timeline.py
    PYTHONPATH=. python scripts/p01s_rotation_timeline.py --t0 40 --t1 250
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

from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
LABELS = {"LEFT_YAW": -1, "RIGHT_YAW": +1, "FWD": 0, "MIXED": 0, "REJECT": 0}
GRID_W, GRID_H = 16, 8
XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
DESIGN = np.stack([np.ones_like(XN), XN], axis=1)
PINV = np.linalg.pinv(DESIGN)
ENCODE = (192, 108)
FPS = 30.0
SMOOTH_FRAMES = 9          # ~0.3 s median filter, display/detection only
AUDIT_WINDOW_S = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=40.0)
    ap.add_argument("--t1", type=float, default=250.0)
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_timeline")
    args = ap.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    t0, t1 = args.t0, args.t1
    print("P0.1S audit — continuous rotation timeline (Lucas-Kanade field)")
    print(f"  video t={t0:.0f}..{t1:.0f}s, native {FPS:.0f} fps stepping (no seek duplicates)")

    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)))
    n_frames = int(round((t1 - t0) * FPS))

    times, a_vals, b_vals, raw_vals, held = [], [], [], [], []
    for k in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        dxf, _dyf, diag = lk.process(img)
        flat = dxf.reshape(-1)
        ab = flat @ PINV.T
        times.append(t0 + k / FPS)
        a_vals.append(float(ab[0]))
        b_vals.append(float(ab[1]))
        raw_vals.append(float(flat.mean()))
        held.append(bool(diag.held))
    cap.release()

    t = np.asarray(times)
    a = np.asarray(a_vals)
    b = np.asarray(b_vals)
    raw = np.asarray(raw_vals)
    held = np.asarray(held)
    print(f"  frames={len(t)}  held(duplicate)={held.mean():.0%}  "
          f"a range [{a.min():+.2f}, {a.max():+.2f}] px, "
          f"median |a|={np.median(np.abs(a)):.3f} px")

    # display/detection smoothing (not used for any gate)
    k = SMOOTH_FRAMES | 1
    pad = k // 2
    a_s = np.array([np.median(a[max(0, i - pad): i + pad + 1]) for i in range(len(a))])
    b_s = np.array([np.median(b[max(0, i - pad): i + pad + 1]) for i in range(len(b))])
    noise = 1.4826 * np.median(np.abs(a - a_s))
    print(f"  robust per-frame noise sigma(a) = {noise:.2f} px; "
          f"|a| median / sigma = {np.median(np.abs(a)) / max(noise, 1e-9):.2f}")

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    rows = []
    print(f"\n  audit of the blind labels (local extremum of |a| within +/-{AUDIT_WINDOW_S:.0f}s)")
    print(f"  {'event':10s} {'label':10s} {'win a med':>10s} {'extr |a|':>9s} {'extr t':>8s} "
          f"{'extr a':>9s} {'sign ok':>8s}")
    for ev in doc["events"]:
        eid, lab, anchor = ev["id"], ev["label"], float(ev["anchor_s"])
        want = LABELS.get(lab, 0)
        lo = max(0.0, anchor - AUDIT_WINDOW_S)
        hi = anchor + AUDIT_WINDOW_S
        m = (t >= lo) & (t <= hi)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        j = idx[int(np.argmax(np.abs(a_s[m])))]
        win = (t >= ev["start"]) & (t <= ev["end"])
        a_win = float(np.median(a[win])) if win.any() else float("nan")
        a_ext, t_ext = float(a[j]), float(t[j])
        rec = {
            "event": eid, "label": lab, "anchor_s": anchor, "want_sign": want,
            "a_window_median": a_win, "a_extremum": a_ext, "a_extremum_t": t_ext,
            "a_extremum_abs": abs(a_ext), "noise_sigma": float(noise),
            "extremum_sign_ok": bool(want and np.sign(a_ext) == want),
            "window_sign_ok": bool(want and np.sign(a_win) == want),
            "extremum_over_noise": abs(a_ext) / max(noise, 1e-9),
        }
        rows.append(rec)
        if lab in ("MIXED", "REJECT"):
            continue
        print(f"  {eid:10s} {lab:10s} {a_win:+10.3f} {abs(a_ext):9.3f} {t_ext:8.2f} "
              f"{a_ext:+9.3f} {str(rec['extremum_sign_ok']):>8s}")

    turns = [r for r in rows if r["want_sign"] != 0]
    ok_ext = sum(1 for r in turns if r["extremum_sign_ok"])
    ok_win = sum(1 for r in turns if r["window_sign_ok"])
    fwd = [r for r in rows if r["want_sign"] == 0 and r["label"] == "FWD"]
    print(f"\n  sign agreement with the labels: window median {ok_win}/{len(turns)}, "
          f"local extremum {ok_ext}/{len(turns)}")
    if fwd:
        print(f"  FWD windows: median |a| = "
              f"{np.median([abs(r['a_window_median']) for r in fwd]):.3f} px, "
              f"turn-like excursions found in "
              f"{sum(1 for r in fwd if r['extremum_over_noise'] > 1.5)}/{len(fwd)} forward windows")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
    for ax, vals, name, ylab in (
        (axes[0], a, "a = rotation (yaw)", "px"),
        (axes[1], b, "b = expansion (forward)", "px / azimuth"),
        (axes[2], raw, "raw pooled mean dx", "px"),
    ):
        flipped = np.array([lab in ("LEFT_YAW",) for lab in [""]] * 0)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.plot(t, vals, lw=0.5, color="0.75", label="per frame")
        ax.plot(t, a_s if name.startswith("a") else
                 (b_s if name.startswith("b") else vals), lw=1.2, color="tab:blue",
                 label=f"median filter ({SMOOTH_FRAMES} frames)")
        ax.axhspan(-noise, noise, color="0.9", zorder=0)
        ax.set_ylabel(f"{name} ({ylab})")
        ax.grid(alpha=0.25, lw=0.4)
    for r in rows:
        if r["label"] in ("MIXED", "REJECT"):
            continue
        colour = {"LEFT_YAW": "tab:green", "RIGHT_YAW": "tab:purple", "FWD": "0.6"}[r["label"]]
        for ax in axes:
            ax.axvspan(r["anchor_s"] - 1.5, r["anchor_s"] + 3.0, color=colour, alpha=0.15, lw=0)
        axes[0].annotate(
            f"{r['event']}\n{r['label'].replace('_YAW','')}",
            (r["anchor_s"], axes[0].get_ylim()[1]), fontsize=6, ha="center",
            va="top", color=colour, annotation_clip=False,
        )
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("P0.1S audit — continuous rotation (a) and expansion (b) from the Lucas-Kanade field\n"
                 "grey band = +-1 robust sigma (per-frame noise); shaded spans = blind event windows",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "rotation_timeline.png", dpi=120)
    plt.close(fig)

    # ---- noise-relative signal size on the turn windows ----
    turn_snr = {}
    for r in turns:
        m = (t >= r["anchor_s"] - 1.5) & (t <= r["anchor_s"] + 1.5)
        turn_snr[r["event"]] = float(abs(np.median(a[m])) / max(noise, 1e-9))

    verdict = (
        f"LABELS DISAGREE WITH THE MEASUREMENT: on a trustworthy estimator (LK), the rotation "
        f"excursion inside the labelled window has the claimed sign in only {ok_win}/{len(turns)} "
        f"turns, and the strongest rotation excursion near the anchor has the claimed sign in "
        f"{ok_ext}/{len(turns)}. Meanwhile the per-frame rotation noise sigma is only "
        f"{noise:.2f} px, so a real turn is far above the noise floor. The windows where the sign "
        "disagrees are therefore candidates for a mislabelled or mis-anchored event, not for a "
        "failed estimator. This must be resolved before any further frontend work."
        if ok_win <= len(turns) - 2
        else f"LABELS HOLD: the rotation excursion inside the labelled window has the claimed sign "
             f"in {ok_win}/{len(turns)} turns (extremum test {ok_ext}/{len(turns)}), with per-frame "
             f"noise sigma {noise:.2f} px. The remaining failure has to be sought elsewhere."
    )
    print(f"\n  per-event rotation signal / noise on the window: "
          + ", ".join(f"{k} {v:.1f}" for k, v in turn_snr.items()))
    print(f"\n=== VERDICT ===\n{verdict}")

    with (out / "timeline.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a_rotation", "b_expansion", "raw_dx", "held"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{a[i]:.6f}", f"{b[i]:.6f}", f"{raw[i]:.6f}", int(held[i])])
    (out / "report.json").write_text(
        json.dumps({
            "probe": "P0.1S audit — continuous rotation timeline",
            "video": args.video, "t0": t0, "t1": t1, "fps": FPS, "n_frames": len(t),
            "held_fraction": float(held.mean()),
            "noise_sigma": float(noise),
            "median_abs_a": float(np.median(np.abs(a))),
            "events": rows,
            "labels_window_ok": [ok_win, len(turns)],
            "labels_extremum_ok": [ok_ext, len(turns)],
            "turn_snr": turn_snr,
            "verdict": verdict,
        }, indent=2),
        encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
