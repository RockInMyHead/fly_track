#!/usr/bin/env python3
"""
P0.1S — instrument checks: does the kymograph encode direction correctly, and
what does the measured motion actually look like on the imagery?

Part 1 — objective kymograph check
    Build a clip whose content is translated right by a known amount, then track
    the column position of the strongest feature in the kymograph over time. If
    the instrument is correct this must increase monotonically. This removes any
    doubt about the display being mirrored.

Part 2 — displacement overlay
    Draw the measured per-patch displacement as arrows on the real frames. The
    arrows are in image coordinates, so anyone can check by eye whether they
    follow the texture. This is the only artifact in this whole phase where the
    measurement and the imagery are visible together, and it is the fastest way
    to see whether the estimator tracks the scene or not.

Part 3 — the same overlay as video, and a side-by-side of the raw frame pair with
    the temporal difference, which shows edge dipoles whose polarity encodes the
    direction directly.

Frontend only. Nothing trained, nothing injected.

Usage:
    PYTHONPATH=. python scripts/p01s_instrument_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.lk_motion_field import LKMotionField

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
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
FPS = 30.0
ENCODE = (192, 108)
GRID_W, GRID_H = 16, 8
BAND_ROWS = (20, 54, 88)
BAND_HALF = 3
KYMO_WIDTH = 900


# --------------------------------------------------- part 1: instrument check
def kymograph(grays: list[np.ndarray], row: int, half: int = BAND_HALF) -> np.ndarray:
    y0, y1 = max(0, row - half), row + half + 1
    strip = np.stack([g[y0:y1].mean(axis=0) for g in grays], axis=0)
    return strip


def instrument_check() -> dict:
    """Translate a real frame right by a known amount; track feature column over time."""
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_MSEC, 139.5 * 1000)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read base frame")
    h, w = base.shape[:2]
    gray0 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    small0 = cv2.resize(gray0, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    results = {}
    for name, px_per_frame in (("content_right", +2.0), ("content_left", -2.0)):
        grays = []
        for k in range(90):
            M = np.float32([[1, 0, px_per_frame * k * (w / ENCODE[0])], [0, 1, 0]])
            bgr = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            grays.append(cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)

        kymo = kymograph(grays, 54)                      # [time, width]
        # follow the brightest column restricted to a band so it cannot wrap out
        col0 = int(np.argmax(kymo[0, 20:80])) + 20
        track = []
        col = col0
        for t in range(len(kymo)):
            lo, hi = max(0, col - 6), min(kymo.shape[1], col + 7)
            if hi <= lo:
                break
            col = lo + int(np.argmax(kymo[t, lo:hi]))
            track.append(col)
        track = np.asarray(track)
        # slope of column vs time in the kymograph array
        slope = float(np.polyfit(np.arange(len(track)), track, 1)[0]) if len(track) > 2 else 0.0
        results[name] = {
            "kymograph_slope_px_per_frame": slope,
            "expected_sign": +1 if px_per_frame > 0 else -1,
            "instrument_correct": bool(np.sign(slope) == np.sign(px_per_frame)),
            "tracked_columns": track.tolist(),
        }
        print(f"  {name:14s} kymograph column slope {slope:+.3f} px/frame "
              f"(expected {'+' if px_per_frame > 0 else '-'})  "
              f"correct: {results[name]['instrument_correct']}")
    return results


# ------------------------------------------------------- part 2/3: overlay
def motion_overlay(bgr: np.ndarray, dxf: np.ndarray, dyf: np.ndarray,
                   scale: int = 6, stride: int = 1) -> np.ndarray:
    """Arrows for the measured per-cell displacement, drawn on the frame."""
    out = bgr.copy()
    h, w = out.shape[:2]
    gh, gw = dxf.shape
    cw, ch = w / gw, h / gh
    for gy in range(0, gh, stride):
        for gx in range(0, gw, stride):
            dx, dy = float(dxf[gy, gx]), float(dyf[gy, gx])
            if abs(dx) < 0.15 and abs(dy) < 0.15:
                continue
            x = int((gx + 0.5) * cw)
            y = int((gy + 0.5) * ch)
            x2 = int(x + dx * scale)
            y2 = int(y + dy * scale)
            mag = float(np.hypot(dx, dy))
            colour = (0, 0, 255) if mag > 1.5 else (0, 180, 255)
            cv2.arrowedLine(out, (x, y), (x2, y2), colour, 2, cv2.LINE_AA, tipLength=0.35)
    return out


def process_event(eid: str, anchor: float, out: Path, make_video: bool = True) -> dict:
    t0 = max(0.0, anchor - PRE_ROLL_S)
    t1 = anchor + EVENT_DUR_S
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)))
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    records, idx = [], 0
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = t0 + idx / FPS
        if t > t1:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        h = small.shape[0]
        band = np.ascontiguousarray(small[int(0.30 * h): int(0.70 * h)])
        dxf, dyf, _d = lk.process(band)
        records.append({"t": t, "dx": dxf.copy(), "dy": dyf.copy(), "frame": frame.copy()})
        idx += 1
    cap.release()

    n_pre = int(round(PRE_ROLL_S * FPS))
    ev = records[n_pre:]
    dx_stack = np.stack([r["dx"] for r in ev], axis=0)
    dy_stack = np.stack([r["dy"] for r in ev], axis=0)

    # a few annotated stills
    picks = [0, len(ev) // 3, 2 * len(ev) // 3, len(ev) - 1]
    tiles = []
    for i in picks:
        r = ev[i]
        ov = motion_overlay(r["frame"], r["dx"], r["dy"], scale=3)
        diff = np.abs(np.asarray(
            cv2.cvtColor(ev[max(0, i - 1)]["frame"], cv2.COLOR_BGR2GRAY), dtype=np.float32
        ) - np.asarray(cv2.cvtColor(r["frame"], cv2.COLOR_BGR2GRAY), dtype=np.float32))
        d = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        pair = np.hstack([cv2.resize(o, (480, 270)) for o in (ov, d)])
        cv2.putText(pair, f"t={r['t']:.2f}s   left: measured displacement arrows   "
                          f"right: |frame difference|",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(pair)
    cv2.imwrite(str(out / f"{eid}_overlay.png"), np.vstack(tiles))

    if make_video:
        writer = cv2.VideoWriter(str(out / f"{eid}_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                 (ev[0]["frame"].shape[1], ev[0]["frame"].shape[0]))
        for r in ev:
            writer.write(motion_overlay(r["frame"], r["dx"], r["dy"], scale=3))
        writer.release()

    mean_dx = float(dx_stack.mean())
    mean_dy = float(dy_stack.mean())
    return {
        "event": eid, "mean_dx_px": mean_dx, "mean_dy_px": mean_dy,
        "mean_abs_dx_px": float(np.abs(dx_stack).mean()),
        "frac_cells_positive": float((dx_stack > 0).mean()),
        "cell_agreement": float(abs((dx_stack > 0).mean() - 0.5) * 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01s_instrument")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.output
    (out / "overlay").mkdir(parents=True, exist_ok=True)

    print("P0.1S instrument check — part 1: is the kymograph orientation correct?")
    check = instrument_check()

    print("\n  part 2/3: displacement arrows on the real frames")
    print(f"  {'event':10s} {'label':5s} {'mean dx':>9s} {'mean |dx|':>10s} "
          f"{'cells +':>8s} {'agreement':>10s}")
    rows = []
    for eid, (anchor, lab) in EVENTS.items():
        rec = process_event(eid, anchor, out / "overlay", make_video=not args.no_video)
        rec["label"] = lab
        rows.append(rec)
        print(f"  {eid:10s} {lab:5s} {rec['mean_dx_px']:+9.3f} {rec['mean_abs_dx_px']:10.3f} "
              f"{rec['frac_cells_positive']:8.0%} {rec['cell_agreement']:10.0%}")

    print(f"\n  {'event':10s} {'frames where every cell agrees on the sign':>44s}")
    for r in rows:
        print(f"  {r['event']:10s} mean dx {r['mean_dx_px']:+.3f} px  -> "
              f"{'content moves RIGHT' if r['mean_dx_px'] > 0 else 'content moves LEFT'}")

    (out / "instrument.json").write_text(json.dumps(
        {"kymograph_check": check, "overlay": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/  (open overlay/<event>_overlay.png and .mp4)")


if __name__ == "__main__":
    main()
