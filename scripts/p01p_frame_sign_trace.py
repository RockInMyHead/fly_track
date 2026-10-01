"""
P0.1P — is there a stable correct-sign stretch inside a real turn?

Per-frame sign trace of the frozen frontend for four events, with a timeline
strip of frames underneath so the plot can be read against what was on screen.

Sign convention (from the frontend itself, verified in P0.1N):
    yaw_frozen < 0  = LEFT motion
    yaw_frozen > 0  = RIGHT motion

Reported per event
  - per-frame sign, and a "dead band" of |yaw| < 0.05 drawn as 0
  - the longest run of consecutive frames holding the correct sign
  - the longest run holding a stable sign at all, correct or not
  - frame strip at ~0.25 s spacing

Usage:
    PYTHONPATH=. python scripts/p01p_frame_sign_trace.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

EVENTS = {
    "left_65":   (63.5, -1, "LEFT"),
    "left_108":  (106.0, -1, "LEFT"),
    "right_210": (209.0, -1 + 2, "RIGHT"),
    "right_216": (214.0, +1, "RIGHT"),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DEAD_BAND = 0.05          # |yaw| below this is plotted as 0 (no clear direction)
FRAME_STRIP_STEP_S = 0.25

PLOT_W, PLOT_H = 1500, 260
STRIP_W = 1500


def frame_strip(video: Path, t0: float, t1: float, step_s: float) -> np.ndarray:
    imgs, times = [], []
    t = t0
    while t < t1:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        if ok:
            imgs.append(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)))
            times.append(t)
        t += step_s
    if not imgs:
        return np.zeros((10, STRIP_W, 3), np.uint8)
    h = max(i.shape[0] for i in imgs)
    w = sum(i.shape[1] for i in imgs) + 4 * (len(imgs) - 1)
    canvas = np.full((h + 22, w, 3), 245, np.uint8)
    x = 0
    for t, im in zip(times, imgs):
        canvas[22:22 + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, f"{t:.2f}", (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        x += im.shape[1] + 4
    scale = STRIP_W / canvas.shape[1]
    return cv2.resize(canvas, (STRIP_W, int(canvas.shape[0] * scale)))


def longest_run(signs: np.ndarray, want: int) -> tuple[int, int]:
    best = cur = 0
    best_end = 0
    for i, s in enumerate(signs):
        if s == want:
            cur += 1
            if cur > best:
                best, best_end = cur, i
        else:
            cur = 0
    return best, best_end - best + 1


def longest_stable(signs: np.ndarray) -> tuple[int, int]:
    """Longest run of any one non-zero sign."""
    best = 0
    best_val = 0
    best_start = 0
    for want in (-1, 1):
        n, start = longest_run(signs, want)
        if n > best:
            best, best_val, best_start = n, want, start
    return best, best_start


def main() -> None:
    out = ROOT / "output/p01p_frame_sign"
    out.mkdir(parents=True, exist_ok=True)
    video = ROOT / "data/p01r/VID00001.AVI"
    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    fc = FlowConfig(crop_mode="center_band", ema_tau_s=cfg.ema_tau_s, brain_dt=cfg.brain_dt)
    dt = cfg.brain_dt
    n_pre = int(round(PRE_ROLL_S / dt))

    summary = {}
    for eid, (start, lab, labname) in EVENTS.items():
        enc = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain, flow_config=fc)
        t0, t1 = max(0.0, start - PRE_ROLL_S), start + EVENT_DUR_S
        times, yaws = [], []
        for i, (t, frame) in enumerate(iter_video_at_brain_hz(str(video), t0, t1, dt)):
            _e, _inj, m = enc.encode_frame(frame)
            if i < n_pre:
                continue
            times.append(t)
            yaws.append(float(m["yaw_frozen"]))
        y = np.asarray(yaws)
        t_arr = np.asarray(times)
        signs = np.where(np.abs(y) < DEAD_BAND, 0, np.sign(y)).astype(int)

        n_correct, start_correct = longest_run(signs, lab)
        n_stable, start_stable = longest_stable(signs)
        stable_val = signs[start_stable] if n_stable else 0
        frac_correct = float(np.mean(signs == lab))
        # runs of correct sign, all of them
        runs = []
        cur = 0
        for i, s in enumerate(signs):
            if s == lab:
                cur += 1
            elif cur:
                runs.append((cur, i - cur))
                cur = 0
        if cur:
            runs.append((cur, len(signs) - cur))
        runs.sort(reverse=True)

        print(f"\n=== {eid} ({labname})  t={t0:.1f}..{t1:.1f}s ===")
        print(f"  frames={len(signs)}  dead-band |yaw|<{DEAD_BAND}: "
              f"{int((signs == 0).sum())} frames ({(signs == 0).mean():.0%})")
        print(f"  fraction of time with correct sign : {frac_correct:.0%}")
        print(f"  longest correct-sign run           : {n_correct} frames = {n_correct * dt:.2f}s "
              f"(starts t={t_arr[start_correct]:.2f}s)" if n_correct else "  longest correct-sign run: none")
        print(f"  longest stable run (any direction) : {n_stable} frames = {n_stable * dt:.2f}s "
              f"direction={stable_val:+d} starts t={t_arr[start_stable]:.2f}s")
        print(f"  all correct-sign runs >=0.4s        : "
              + (", ".join(f"{c * dt:.2f}s@t={t_arr[s]:.2f}" for c, s in runs if c * dt >= 0.4) or "none"))

        report = {
            "event": eid,
            "label": labname,
            "frames": int(len(signs)),
            "dead_band": DEAD_BAND,
            "frames_dead": int((signs == 0).sum()),
            "frac_correct": frac_correct,
            "longest_correct_frames": int(n_correct),
            "longest_correct_seconds": n_correct * dt,
            "longest_correct_start_t": float(t_arr[start_correct]) if n_correct else None,
            "longest_stable_frames": int(n_stable),
            "longest_stable_seconds": n_stable * dt,
            "longest_stable_direction": int(stable_val),
            "longest_stable_start_t": float(t_arr[start_stable]) if n_stable else None,
            "correct_runs_ge_0p4s": [
                {"seconds": c * dt, "start_t": float(t_arr[s])} for c, s in runs if c * dt >= 0.4
            ],
        }
        summary[eid] = report

        with (out / f"{eid}_trace.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "yaw_frozen", "sign", "correct"])
            for t, v, s in zip(t_arr, y, signs):
                w.writerow([f"{t:.3f}", f"{v:.6f}", int(s), int(s == lab)])

        # ---- plot ----
        img = np.full((PLOT_H, PLOT_W, 3), 255, np.uint8)
        pad_l, pad_r, pad_t, pad_b = 90, 30, 46, 26
        xw = PLOT_W - pad_l - pad_r
        yh = PLOT_H - pad_t - pad_b
        mid = pad_t + yh // 2
        lim = max(0.35, float(np.percentile(np.abs(y), 98)))

        def px(i: int) -> int:
            return pad_l + int(i / max(len(y) - 1, 1) * xw)

        def py(v: float) -> int:
            return int(mid - np.clip(v / lim, -1, 1) * (yh / 2 - 6))

        # dead band shading
        cv2.rectangle(img, (pad_l, py(DEAD_BAND)), (pad_l + xw, py(-DEAD_BAND)), (238, 238, 238), -1)
        cv2.line(img, (pad_l, mid), (pad_l + xw, mid), (170, 170, 170), 1)
        cv2.putText(img, "LEFT (-)", (pad_l - 82, mid - yh // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 60, 60), 1)
        cv2.putText(img, "RIGHT (+)", (pad_l - 82, mid + yh // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 90, 180), 1)
        cv2.putText(img, f"+{lim:.2f}", (pad_l - 60, py(lim) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(img, f"-{lim:.2f}", (pad_l - 60, py(-lim) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        col_correct = (70, 160, 70)
        col_wrong = (60, 60, 220)
        for i in range(len(y) - 1):
            c = col_correct if signs[i] == lab else (col_wrong if signs[i] != 0 else (190, 190, 190))
            cv2.line(img, (px(i), py(y[i])), (px(i + 1), py(y[i + 1])), c, 1)

        # mark the longest correct run
        if n_correct:
            x0, x1 = px(start_correct), px(start_correct + n_correct - 1)
            cv2.rectangle(img, (x0, pad_t + 4), (x1, pad_t + 18), col_correct, -1)
            cv2.putText(img, f"longest correct: {n_correct * dt:.2f}s", (x0 + 4, pad_t + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.putText(img, f"{eid} ({labname})   frac correct {frac_correct:.0%}   "
                         f"longest correct {n_correct * dt:.2f}s   longest stable {n_stable * dt:.2f}s",
                    (pad_l, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"window t={t0:.1f}..{t1:.1f}s   green=correct side, red=wrong side, grey=dead band",
                    (pad_l, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        strip = frame_strip(video, t0, t1, FRAME_STRIP_STEP_S)
        combo = np.vstack([img, np.full((6, PLOT_W, 3), 255, np.uint8), strip[:, :PLOT_W]])
        cv2.imwrite(str(out / f"{eid}_sign_timeline.png"), combo)

    # ---- verdict ----
    long_enough = 0.6  # seconds
    verdicts = {}
    for eid, r in summary.items():
        verdicts[eid] = {
            "has_stable_correct_stretch": r["longest_correct_seconds"] >= long_enough,
            "longest_correct_s": r["longest_correct_seconds"],
            "frac_correct": r["frac_correct"],
            "flips": int(r["frames"] - r["longest_stable_frames"]) > 0,
        }
    n_with = sum(1 for v in verdicts.values() if v["has_stable_correct_stretch"])
    if n_with >= 3:
        verdict = (
            f"WINDOW TOO LARGE: {n_with}/4 events contain a stable correct-sign stretch "
            f">= {long_enough}s inside the 3 s window. A turn detector that reads the signal only "
            "during that stretch would work; the failure comes from averaging over the whole window."
        )
    elif n_with >= 1:
        verdict = (
            f"PARTIAL: only {n_with}/4 events contain a stable correct-sign stretch >= {long_enough}s. "
            "Some events are recoverable by windowing, others are not."
        )
    else:
        verdict = (
            "LOCAL MOTION IS THE PROBLEM: no event holds the correct sign for even "
            f"{long_enough}s. Changing the window cannot help; the per-frame motion estimate "
            "itself is unstable during real turns."
        )

    report = {
        "probe": "P0.1P — per-frame sign trace inside real turns",
        "sign_convention": "yaw_frozen < 0 = LEFT, > 0 = RIGHT",
        "dead_band": DEAD_BAND,
        "stable_threshold_s": long_enough,
        "events": summary,
        "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
