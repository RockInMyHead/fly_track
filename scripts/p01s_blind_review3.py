#!/usr/bin/env python3
"""
P0.1S — blind review round 3, with the convention ambiguity removed.

What went wrong in rounds 1 and 2
---------------------------------
Round 1 asked for "the direction the image content travels" but the labels record
the direction the camera rotated. Those are opposite for a yaw, and the inversion
was never stated, so the answers could not be compared to the labels.
Round 1 also cropped every kymograph to the leftmost third of the frame, hiding
most of the evidence.
Round 2 fixed the crop but kept the ambiguous wording and had no way to tell which
convention the reviewer was using.

How round 3 removes the ambiguity
---------------------------------
It asks for ONE thing only, everywhere: **the direction the image content travels**.
That is directly visible and, unlike camera direction, it can be verified on
synthetic items. Four calibration clips are synthesised from a single real frame by
applying a known transform, so their true content direction is known exactly; they
are shuffled in with the real windows and revealed only in the key. That gives, in
a single pass:

    (a) whether the reviewer is reading content direction correctly at all, and
    (b) the error rate of the reading,

so a real answer can be weighted by a measured reliability instead of faith.

The conversion to camera labels happens afterwards, in the key, with the mapping
written out per item. Nobody has to hold the inversion in their head.

Panels
------
Full frame width in every plot. The large plot is a space-time image: the frame
averaged over height, high-passed in time so static structure disappears and moving
texture leaves slanted streaks. x = image column, y = time. A rotation slants all
three band plots the same way; forward motion slants the outer two opposite ways.

Usage:
    PYTHONPATH=. python scripts/p01s_blind_review3.py
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

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
TEMPLATE_JSON = ROOT / "data/p01r/VID00001_blind_review3.json"
KEY_JSON = ROOT / "data/p01r/VID00001_blind_review3_key.json"

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
FPS = 30.0
ENCODE = (192, 108)
REVIEW_IDS = ("left_65", "left_81", "left_108", "right_67", "right_210", "right_216", "fwd_140")
BAND_ROWS = (20, 54, 88)
BAND_HALF = 3
FULL_W = 1100
TIME_SCALE = 5
CALIB_BASE_S = 139.5
CALIB_PX_PER_FRAME = 2.0
CALIB_EXPANSION = 0.015


def read_frames(video: Path, t0: float, t1: float):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)))
    grays, bgrs = [], []
    for _ in range(int(round((t1 - t0) * FPS))):
        ok, frame = cap.read()
        if not ok:
            break
        bgrs.append(frame)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    cap.release()
    return grays, bgrs


def read_single(video: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame at t={t}")
    return frame


def synth_frames(base: np.ndarray, kind: str, n: int):
    h, w = base.shape[:2]
    grays, bgrs = [], []
    for k in range(n):
        if kind == "static":
            M = np.float32([[1, 0, 0], [0, 1, 0]])
        elif kind == "content_left":
            M = np.float32([[1, 0, -CALIB_PX_PER_FRAME * k * (w / ENCODE[0])], [0, 1, 0]])
        elif kind == "content_right":
            M = np.float32([[1, 0, CALIB_PX_PER_FRAME * k * (w / ENCODE[0])], [0, 1, 0]])
        else:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 1.0 + CALIB_EXPANSION * k)
        bgr = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bgrs.append(bgr)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    return grays, bgrs


# ------------------------------------------------------------------ plotting
def _render(strip: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    """strip: [time, columns] already high-passed; -> BGR image with a title bar."""
    lo, hi = np.percentile(strip, 2), np.percentile(strip, 98)
    v = np.clip((strip - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (v * 255).astype(np.uint8)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((22, width, 3), np.uint8)
    cv2.putText(bar, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([bar, img])


def spacetime(grays: list[np.ndarray]) -> np.ndarray:
    strip = np.stack([g.mean(axis=0) for g in grays], axis=0)      # [T, W]
    return strip - strip.mean(axis=0, keepdims=True)               # kill static structure


def band_kymo(grays: list[np.ndarray], row: int, half: int = BAND_HALF) -> np.ndarray:
    y0, y1 = max(0, row - half), row + half + 1
    strip = np.stack([g[y0:y1].mean(axis=0) for g in grays], axis=0)
    return strip - strip.mean(axis=0, keepdims=True)


def build_panel(code: str, grays: list[np.ndarray], bgrs: list[np.ndarray]) -> np.ndarray:
    t = len(grays)
    big = _render(spacetime(grays), FULL_W, t * TIME_SCALE,
                  "FULL frame averaged over height  —  x = image column (left edge of frame at left), "
                  "y = time downward")
    bands = []
    for row, lab in zip(BAND_ROWS, ("UPPER third", "CENTRE", "LOWER third")):
        bands.append(_render(band_kymo(grays, row), FULL_W, t * (TIME_SCALE - 2),
                             f"{lab} of frame"))
    stills = []
    for f in (0.02, 0.25, 0.5, 0.75, 0.98):
        i = min(len(bgrs) - 1, int(len(bgrs) * f))
        s = cv2.resize(bgrs[i], (FULL_W // 5 - 6, int(bgrs[i].shape[0] / bgrs[i].shape[1] * (FULL_W // 5 - 6))))
        stills.append(s)
    sh = max(s.shape[0] for s in stills)
    srow = np.full((sh, FULL_W, 3), 255, np.uint8)
    x = 0
    for s in stills:
        if x + s.shape[1] > FULL_W:
            break
        srow[: s.shape[0], x:x + s.shape[1]] = s
        x += s.shape[1] + 6

    head = np.full((104, FULL_W, 3), 255, np.uint8)
    cv2.putText(head, f"ITEM {code}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2)
    cv2.putText(head, "top: five stills in time order.", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1)
    cv2.putText(head, "below: pixel value over time. Bright streaks are moving texture; "
                      "the way they lean is the way the content moves.",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1)
    cv2.putText(head, "Answer where the IMAGE CONTENT goes, not where the camera goes. "
                      "Streaks leaning toward the left edge of a plot = content moved LEFT.",
                (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 180), 1)
    cv2.putText(head, "FWD only if the content spreads outward from the middle: long streaks "
                      "leaning RIGHT in the upper band and LEFT in the lower band.",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 180), 1)
    return np.vstack([head, srow, np.full((6, FULL_W, 3), 255, np.uint8), big,
                      np.full((6, FULL_W, 3), 255, np.uint8), *bands])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_blind_review3")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    if TEMPLATE_JSON.exists() and not args.force:
        raise SystemExit(f"{TEMPLATE_JSON} exists — pass --force to regenerate")

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    items = [{"kind": "real", "id": e["id"], "old_label": e["label"], "anchor": float(e["anchor_s"])}
             for e in doc["events"] if e["id"] in REVIEW_IDS]
    base = read_single(Path(args.video), CALIB_BASE_S)
    for kind in ("content_left", "content_right", "expansion", "static"):
        items.append({"kind": "calibration", "id": kind, "old_label": "SYNTHETIC",
                      "anchor": CALIB_BASE_S})

    rng = np.random.default_rng(args.seed)
    codes = [chr(ord("A") + i) for i in range(len(items))]
    shuffled = [items[i] for i in rng.permutation(len(items))]

    print("P0.1S blind review round 3")
    print(f"  {len(shuffled)} items, shuffled seed {args.seed}, named {codes[0]}..{codes[-1]}")
    print("  you are asked for CONTENT motion only; the camera mapping is done in the key")
    print("  panels show the FULL frame width; four items are synthetic with known answers\n")

    key, entries = {}, []
    for code, it in zip(codes, shuffled):
        if it["kind"] == "real":
            t0 = max(0.0, it["anchor"] - PRE_ROLL_S)
            grays, bgrs = read_frames(Path(args.video), t0, it["anchor"] + EVENT_DUR_S)
            window = [t0, it["anchor"] + EVENT_DUR_S]
        else:
            grays, bgrs = synth_frames(base, it["id"], int(round((PRE_ROLL_S + EVENT_DUR_S) * FPS)))
            window = [CALIB_BASE_S, CALIB_BASE_S + PRE_ROLL_S + EVENT_DUR_S]
        if not grays:
            print(f"  {code}: no frames, skipped")
            continue

        cv2.imwrite(str(out / f"{code}.png"), build_panel(code, grays, bgrs))
        vw = 960
        vh = int(bgrs[0].shape[0] * vw / bgrs[0].shape[1])
        writer = cv2.VideoWriter(str(out / f"{code}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (vw, vh))
        for f in bgrs:
            writer.write(cv2.resize(f, (vw, vh)))
        writer.release()

        rec = {"kind": it["kind"], "event_id": it["id"], "old_label_camera_sense": None,
               "window": window}
        if it["kind"] == "calibration":
            cm = {"content_left": "LEFT", "content_right": "RIGHT",
                  "expansion": "FWD", "static": "STATIC"}[it["id"]]
            rec["true_content_direction"] = cm
            rec["true_camera_direction"] = {"LEFT": "RIGHT", "RIGHT": "LEFT",
                                            "FWD": "FWD", "STATIC": "STATIC"}[cm]
        else:
            rec["old_label_camera_sense"] = {
                "LEFT_YAW": "LEFT", "RIGHT_YAW": "RIGHT", "FWD": "FWD"}.get(it["old_label"], "UNCLEAR")
            rec["old_label_content_sense"] = {
                "LEFT": "RIGHT", "RIGHT": "LEFT", "FWD": "FWD"}.get(
                    rec["old_label_camera_sense"], "UNCLEAR")
            if it["id"] == "left_65":
                rec["note"] = ("corrected by eye: the camera turns RIGHT, so the old "
                               "LEFT_YAW label is wrong")
        key[code] = rec
        entries.append({"code": code, "content_direction": "", "confidence": "", "notes": ""})
        print(f"  {code}: {it['kind']:11s} written")

    TEMPLATE_JSON.write_text(json.dumps({
        "round": 3,
        "instructions": (
            "ONE question, identical for every item: which way does the IMAGE CONTENT "
            "travel across the frame?\n"
            "Look at the .mp4 and/or the .png in output/p01s_blind_review3/.\n\n"
            "Definitions, stated exactly:\n"
            "  LEFT    the content slides toward the LEFT  edge of the frame\n"
            "  RIGHT   the content slides toward the RIGHT edge of the frame\n"
            "  FWD     the content SPREADS OUTWARD from the middle: what is left of the\n"
            "          centre moves further left, what is right of the centre moves\n"
            "          further right. In the band plots this is long streaks leaning\n"
            "          RIGHT in the upper band and LEFT in the lower band.\n"
            "          FWD means THIS and nothing else. It does NOT mean 'I think the\n"
            "          camera is driving forward'.\n"
            "  UNCLEAR you cannot tell, or the motions contradict each other\n\n"
            "Answer about the content only. Never convert to camera direction -- that\n"
            "conversion is done in the key and is not your job here.\n\n"
            "Four of the items are synthetic clips built from a single real frame with a\n"
            "known content motion. You are not told which. Answer them the same way as\n"
            "everything else; they are how the readability of these panels gets measured."
        ),
        "review_date": "", "reviewer": "", "blind_to_measurement": True,
        "items": entries,
    }, indent=2), encoding="utf-8")
    KEY_JSON.write_text(json.dumps({
        "warning": "Do not open until the judgements are written down.",
        "seed": args.seed,
        "conversion": "content LEFT -> camera RIGHT;  content RIGHT -> camera LEFT;  "
                      "content FWD -> FWD (not a yaw)",
        "items": key,
    }, indent=2), encoding="utf-8")
    print(f"\nFill in:    {TEMPLATE_JSON}")
    print(f"Do not open: {KEY_JSON}")
    print(f"Look at:    {out}/")


if __name__ == "__main__":
    main()
