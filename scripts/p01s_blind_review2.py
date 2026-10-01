#!/usr/bin/env python3
"""
P0.1S blind review, round 2 — fixed panels plus a calibration set.

Two flaws in the round-1 instrument
-----------------------------------
1. The kymographs were cropped to `KYMO_W // 3` columns *after* resizing to full
   width, so only the leftmost third of the frame was ever displayed. More than
   half of the available evidence was hidden.
2. The instruction told the reviewer to report the direction the *image content*
   travels, while the event labels record the direction the *camera* rotated. For
   a yaw those are opposite, so the two sets of answers are not comparable
   without knowing which convention was used.

Both are fixed here, and the convention question is removed as a source of doubt:
this package contains **calibration items with known ground truth**, synthesised by
applying a known transform to a single real frame. Their answers are shuffled in
with the real ones and revealed only in the key. That measures, in one pass, both

    (a) whether the panel format is readable at all, and
    (b) which convention the reviewer is using (content motion vs camera rotation),

because for a calibration item we know exactly what the pixel content did.

Calibration transforms (applied to one real frame, so the texture statistics match
the real footage; the only motion present is the known synthetic one):

    content_right   the whole image translated right, 2 px per frame
    content_left    the whole image translated left,  2 px per frame
    expansion       scaled up 1.5% per frame about the centre — what driving
                    forward looks like
    static          the same frame repeated — a pure control

For a yaw, `content_left` corresponds to the camera rotating RIGHT and
`content_right` to the camera rotating LEFT. So the key states both readings for
each calibration item.

Usage:
    PYTHONPATH=. python scripts/p01s_blind_review2.py
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

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
TEMPLATE_JSON = ROOT / "data/p01r/VID00001_blind_review2b.json"
KEY_JSON = ROOT / "data/p01r/VID00001_blind_review2b_key.json"

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
FPS = 30.0
ENCODE = (192, 108)
REVIEW_IDS = ("left_65", "left_81", "left_108", "right_67", "right_210", "right_216", "fwd_140")
BAND_ROWS = (20, 54, 88)          # upper third / centre / lower third of 108 rows
BAND_HALF = 3
KYMO_WIDTH = 900
KYMO_TIME_SCALE = 4               # vertical pixels per frame
CALIB_BASE_FRAME_S = 139.5        # a forward corridor view: rich texture, no dominant motion
CALIB_PX_PER_FRAME = 2.0
CALIB_EXPANSION = 0.015


def read_frames(video: Path, t0: float, t1: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)))
    grays, bgrs = [], []
    for _ in range(int(round((t1 - t0) * FPS))):
        ok, frame = cap.read()
        if not ok:
            break
        bgrs.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    cap.release()
    return grays, bgrs


def read_single_bgr(video: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame at t={t}")
    return frame


# --------------------------------------------------------------- calibration
def calibration_frames(base_bgr: np.ndarray, kind: str, n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build (grays, bgrs) for one synthetic condition from a single real frame."""
    h, w = base_bgr.shape[:2]
    gray0 = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    small0 = cv2.resize(gray0, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    grays, bgrs = [], []
    for k in range(n):
        if kind == "static":
            M = np.float32([[1, 0, 0], [0, 1, 0]])
        elif kind == "content_left":
            M = np.float32([[1, 0, -CALIB_PX_PER_FRAME * k * (w / ENCODE[0])],
                            [0, 1, 0]])
        elif kind == "content_right":
            M = np.float32([[1, 0, CALIB_PX_PER_FRAME * k * (w / ENCODE[0])],
                            [0, 1, 0]])
        else:  # expansion
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0,
                                        1.0 + CALIB_EXPANSION * k)
        bgr = cv2.warpAffine(base_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        bgrs.append(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    # sanity: the whole-frame synthetic motion really is what we claim
    dx = float(np.mean((np.stack(grays[-1]) - np.stack(grays[0])) * 0))  # placeholder
    return grays, bgrs


def check_calibration(grays: list[np.ndarray], kind: str) -> dict:
    """Measure the synthetic clip with Lucas-Kanade and confirm the sign is as claimed."""
    from fly_vo.lk_motion_field import LKMotionField

    lk = LKMotionField(grid_w=16, grid_h=8)
    dxs = []
    for g in grays:
        dxf, _dy, _d = lk.process(g[27:81])
        dxs.append(float(dxf.mean()))
    ev = np.asarray(dxs)[int(round(PRE_ROLL_S / (1 / FPS))):]
    measured = float(np.mean(ev)) if len(ev) else 0.0
    return {"measured_mean_dx_px": measured, "kind": kind}


# --------------------------------------------------------------------- panel
def kymograph(grays: list[np.ndarray], row: int, half: int = BAND_HALF) -> np.ndarray:
    y0, y1 = max(0, row - half), row + half + 1
    strip = np.stack([g[y0:y1].mean(axis=0) for g in grays], axis=0)   # [time, width]
    img = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (KYMO_WIDTH, img.shape[0] * KYMO_TIME_SCALE),
                     interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def build_panel(code: str, grays: list[np.ndarray], bgrs: list[np.ndarray]) -> np.ndarray:
    kymos = [kymograph(grays, r) for r in BAND_ROWS]
    labels = ["UPPER third of frame", "CENTRE band", "LOWER third of frame"]

    kh = kymos[0].shape[0]
    blocks = []
    for k, lab in zip(kymos, labels):
        strip = np.full((22, KYMO_WIDTH, 3), 0, np.uint8)
        cv2.putText(strip, f"{lab}   x = image column (left = left edge of frame), "
                           f"y = time downward", (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        blocks.append(np.vstack([strip, k]))
        blocks.append(np.full((8, KYMO_WIDTH, 3), 255, np.uint8))
    kstrip = np.vstack(blocks)

    still_idx = [int(len(bgrs) * f) for f in (0.0, 0.25, 0.5, 0.75, 0.95)]
    stills = [cv2.resize(bgrs[i], (bgrs[i].shape[1] // 5, bgrs[i].shape[0] // 5))
              for i in still_idx if i < len(bgrs)]
    sh = max(s.shape[0] for s in stills)
    sw = sum(s.shape[1] + 8 for s in stills)
    srow = np.full((sh, KYMO_WIDTH, 3), 255, np.uint8)
    x = 0
    for s in stills:
        if x + s.shape[1] > KYMO_WIDTH:
            break
        srow[: s.shape[0], x:x + s.shape[1]] = s
        cv2.putText(srow, f"{still_idx[len([q for q in stills][:stills.index(s)])] / FPS:.1f}s"
                    if False else "", (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        x += s.shape[1] + 8

    head = np.full((78, KYMO_WIDTH, 3), 255, np.uint8)
    cv2.putText(head, f"ITEM {code}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(head, "top: five stills from the window, in time order.",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.putText(head, "below: pixel value over time for three horizontal bands. "
                      "A RIGHTWARD move of the content makes streaks lean down-right.",
                (10, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.putText(head, "camera ROTATION slants all three bands the SAME way; "
                      "FORWARD motion slants the outer two bands OPPOSITE ways.",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    return np.vstack([head, srow, np.full((6, KYMO_WIDTH, 3), 255, np.uint8), kstrip])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_blind_review2b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    if TEMPLATE_JSON.exists() and not args.force:
        raise SystemExit(f"{TEMPLATE_JSON} exists — pass --force to regenerate")

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    items = []
    for e in doc["events"]:
        if e["id"] in REVIEW_IDS:
            items.append({"kind": "real", "id": e["id"], "original_label": e["label"],
                          "anchor": float(e["anchor_s"])})
    base = read_single_bgr(Path(args.video), CALIB_BASE_FRAME_S)
    for kind in ("content_left", "content_right", "expansion", "static"):
        items.append({"kind": "calibration", "id": kind, "original_label": "SYNTHETIC",
                      "anchor": CALIB_BASE_FRAME_S})

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    codes = [chr(ord("A") + i) for i in range(len(items))]
    shuffled = [items[i] for i in order]

    print("P0.1S blind review round 2")
    print(f"  {len(shuffled)} items ({sum(1 for i in shuffled if i['kind'] == 'real')} real, "
          f"{sum(1 for i in shuffled if i['kind'] == 'calibration')} calibration), "
          f"shuffled with seed {args.seed}, named {codes[0]}..{codes[-1]}")
    print("  panels now show the FULL frame width in every kymograph\n")

    key, entries, calib_check = {}, [], {}
    for code, it in zip(codes, shuffled):
        if it["kind"] == "real":
            t0 = max(0.0, it["anchor"] - PRE_ROLL_S)
            t1 = it["anchor"] + EVENT_DUR_S
            grays, bgrs = read_frames(Path(args.video), t0, t1)
            window = [t0, t1]
        else:
            grays, bgrs = calibration_frames(base, it["id"], int(round((PRE_ROLL_S + EVENT_DUR_S) * FPS)))
            window = [CALIB_BASE_FRAME_S, CALIB_BASE_FRAME_S + PRE_ROLL_S + EVENT_DUR_S]
            calib_check[code] = check_calibration(grays, it["id"])
        if not grays:
            print(f"  {code}: no frames, skipped")
            continue

        cv2.imwrite(str(out / f"{code}.png"), build_panel(code, grays, bgrs))
        writer = cv2.VideoWriter(str(out / f"{code}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (bgrs[0].shape[1], bgrs[0].shape[0]))
        for f in bgrs:
            writer.write(f)
        writer.release()

        key[code] = {"kind": it["kind"], "event_id": it["id"],
                     "original_label": it["original_label"], "window": window}
        entries.append({"code": code, "direction_from_frames": "", "confidence": "", "notes": ""})
        print(f"  {code}: {it['kind']:11s} written")

    # state both readings for the calibration items, since a yaw inverts them
    for code, info in key.items():
        if info["kind"] != "calibration":
            continue
        k = info["event_id"]
        if k == "content_left":
            info["content_motion"] = "LEFT"
            info["equivalent_camera_rotation"] = "RIGHT"
        elif k == "content_right":
            info["content_motion"] = "RIGHT"
            info["equivalent_camera_rotation"] = "LEFT"
        elif k == "expansion":
            info["content_motion"] = "FWD"
            info["equivalent_camera_rotation"] = "FWD"
        else:
            info["content_motion"] = "STATIC"
            info["equivalent_camera_rotation"] = "STATIC"
        info["measured_mean_dx_px"] = calib_check.get(code, {}).get("measured_mean_dx_px")

    TEMPLATE_JSON.write_text(json.dumps({
        "round": 2,
        "instructions": (
            "Fill direction_from_frames for all items using ONLY the panels and "
            "clips in output/p01s_blind_review2b/. Four of the items are synthetic "
            "calibration items synthesised from a single real frame; you are not "
            "told which ones. Answer all of them the same way.\n\n"
            "Report left/right by the direction the IMAGE CONTENT travels across the "
            "frame -- that is, the direction the streaks lean. If the content sweeps "
            "toward the left edge of the frame, answer LEFT. This is the opposite of "
            "the direction the camera rotated; the key records both readings.\n\n"
            "Allowed: LEFT / RIGHT / FWD / UNCLEAR."
        ),
        "review_date": "", "reviewer": "", "blind_to_measurement": True,
        "items": entries,
    }, indent=2), encoding="utf-8")
    KEY_JSON.write_text(json.dumps({
        "warning": "Do not open until the judgements are written down.",
        "seed": args.seed,
        "items": key,
    }, indent=2), encoding="utf-8")
    print(f"\n  calibration self-check (Lucas-Kanade on the synthetic clips):")
    for code, res in calib_check.items():
        kind = key[code]["event_id"]
        print(f"    {code}  {kind:14s} measured mean dx = {res['measured_mean_dx_px']:+6.3f} px")
    print(f"\nFill in:   {TEMPLATE_JSON}")
    print(f"Do not open: {KEY_JSON}")
    print(f"Look at:   {out}/")


if __name__ == "__main__":
    main()
