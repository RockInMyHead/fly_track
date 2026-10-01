#!/usr/bin/env python3
"""
P0.1S — blind review package for the disputed event windows.

Purpose
-------
Three estimators agree with each other and with a z = 5..24 integrated rotation
measurement on the sign of all seven labelled windows, yet three of them disagree
with the hand labels. The measurement cannot settle that on its own, so this
script prepares a package for a *blind* re-review of the frames.

Blindness rules, implemented here
---------------------------------
1. Nothing about our motion measurement is shown. No traces, no numbers, no
   correction, no yaw_frozen, no indication of which windows are disputed.
2. Frames carry no burned-in timestamps, so a reviewer cannot place an item on the
   timeline and recognise it from a previous session.
3. Item order is shuffled with a fixed seed and items are named A, B, C, ...
4. The set is padded with windows that are not under dispute, so the disputed ones
   are not identifiable by count.
5. The mapping lives in a separate key file that must not be opened until the
   judgements are written down.

The visual aids are deliberately model-free
-------------------------------------------
* `frames/`  the raw footage, no overlay.
* `clip`     the same window as a short video, for anyone who reads motion better
             in motion than as stills.
* kymographs two of the three 1-D bands of the frame, each plotted as pixel value
  over time: x = image column, y = time. A camera yaw makes the *same* slant in
  all bands. Forward translation makes the upper and lower bands slant in
  *opposite* directions, because the image flows radially away from the focus of
  expansion. That distinction is raw geometry, not our estimator, so it can be
  read directly off the image.

Files written
-------------
    data/p01r/VID00001_blind_review2.json   <- fill this in
    data/p01r/VID00001_blind_review2_key.json <- do NOT open until filled in
    output/p01s_blind_review/<code>.png
    output/p01s_blind_review/<code>.mp4

Usage:
    PYTHONPATH=. python scripts/p01s_blind_review.py
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
TEMPLATE_JSON = ROOT / "data/p01r/VID00001_blind_review2.json"
KEY_JSON = ROOT / "data/p01r/VID00001_blind_review2_key.json"

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
FPS = 30.0
ENCODE = (192, 108)
# Only the seven scored windows are reviewed, plus a-priori fillers so the
# disputed ones are not identifiable by count.
REVIEW_IDS = ("left_65", "left_81", "left_108", "right_67", "right_210", "right_216", "fwd_140")
EXTRA_ANCHORS = [51.0, 93.0, 179.5]
BAND_ROWS = (27, 54, 81)      # upper third, centre, lower third of a 108-row frame
BAND_HALF = 2                 # average this many rows either side for noise
KYMO_W = 768


def read_window(video: Path, t0: float, t1: float) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    """Return (times, gray frames, bgr frames) at native fps for [t0, t1)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)))
    times, grays, bgrs = [], [], []
    for k in range(int(round((t1 - t0) * FPS))):
        ok, frame = cap.read()
        if not ok:
            break
        times.append(t0 + k / FPS)
        bgrs.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    cap.release()
    return times, grays, bgrs


def kymograph(grays: list[np.ndarray], row: int, half: int = BAND_HALF) -> np.ndarray:
    """Pixel value over time for one horizontal band: x = column, y = time."""
    y0, y1 = max(0, row - half), row + half + 1
    strip = np.stack([g[y0:y1].mean(axis=0) for g in grays], axis=0)  # [time, width]
    img = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (KYMO_W, img.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, f"image column ->   band y={row}/108", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return img


def _pad_to(img: np.ndarray, width: int, height: int | None = None) -> np.ndarray:
    h = height if height is not None else img.shape[0]
    canvas = np.full((h, width, 3), 255, np.uint8)
    canvas[: min(h, img.shape[0]), : min(width, img.shape[1])] = \
        img[: min(h, img.shape[0]), : min(width, img.shape[1])]
    return canvas


def build_panel(code: str, t0: float, t1: float, grays: list[np.ndarray], bgrs: list[np.ndarray]) -> np.ndarray:
    kymos = [kymograph(grays, r)[:, : KYMO_W // 3] for r in BAND_ROWS]
    kh = min(k.shape[0] for k in kymos)
    kstrip = np.hstack([
        part
        for k in kymos
        for part in (k[:kh], np.full((kh, 10, 3), 255, np.uint8))
    ])
    # a few stills for context
    still_idx = [int(len(bgrs) * f) for f in (0.15, 0.5, 0.85)]
    stills = [cv2.resize(bgrs[i], (bgrs[i].shape[1] // 5, bgrs[i].shape[0] // 5))
              for i in still_idx if i < len(bgrs)]
    sh = max(s.shape[0] for s in stills)
    srow = np.hstack([
        part for s in stills for part in (s, np.full((sh, 8, 3), 255, np.uint8))
    ]) if stills else np.full((10, 10, 3), 255, np.uint8)

    width = max(kstrip.shape[1], srow.shape[1])
    head = np.full((60, width, 3), 255, np.uint8)
    cv2.putText(head, f"ITEM {code}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(head,
                "top: stills from the window.   below: pixel value over time, three horizontal "
                "bands (upper / centre / lower).  x = image column, y = time.",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
    cv2.putText(head,
                "camera ROTATION slants all three bands the SAME way; forward motion slants the "
                "upper and lower bands OPPOSITE ways.",
                (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
    return np.vstack([
        head,
        _pad_to(srow, width, sh if stills else None),
        np.full((6, width, 3), 255, np.uint8),
        _pad_to(kstrip, width, kh),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_blind_review")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing template (discards recorded judgements)")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    if TEMPLATE_JSON.exists() and not args.force:
        raise SystemExit(
            f"{TEMPLATE_JSON} already exists — refusing to overwrite recorded judgements.\n"
            "Pass --force only if you really want to discard them."
        )

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    wanted = set(REVIEW_IDS)
    items = [{"id": e["id"], "label": e["label"], "start": float(e["anchor_s"]),
              "source": "labelled_event"} for e in doc["events"] if e["id"] in wanted]
    for a in EXTRA_ANCHORS:
        items.append({"id": f"extra_{a:.0f}", "label": "FWD", "start": float(a),
                      "source": "extra_forward_window"})

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    codes = [chr(ord("A") + i) for i in range(len(items))]
    shuffled = [items[i] for i in order]

    print("P0.1S — blind review package")
    print(f"  {len(shuffled)} items, shuffled with seed {args.seed}, named {codes[0]}..{codes[-1]}")
    print(f"  NO measurement is shown; frames carry no timestamps\n")

    key = {}
    entries = []
    for code, it in zip(codes, shuffled):
        t0 = max(0.0, it["start"] - PRE_ROLL_S)
        t1 = it["start"] + EVENT_DUR_S
        times, grays, bgrs = read_window(Path(args.video), t0, t1)
        if not grays:
            print(f"  {code}: no frames read, skipping")
            continue
        panel = build_panel(code, t0, t1, grays, bgrs)
        cv2.imwrite(str(out / f"{code}.png"), panel)

        writer = cv2.VideoWriter(
            str(out / f"{code}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
            (bgrs[0].shape[1], bgrs[0].shape[0]),
        )
        for f in bgrs:
            writer.write(f)
        writer.release()

        key[code] = {"event_id": it["id"], "original_label": it["label"],
                     "window": [t0, t1], "source": it["source"]}
        entries.append({
            "code": code,
            "direction_from_frames": "",        # LEFT | RIGHT | FWD | UNCLEAR
            "confidence": "",                   # high | medium | low
            "notes": "",
        })
        print(f"  {code}: panel + clip written  ({len(grays)} frames)")

    TEMPLATE_JSON.write_text(json.dumps({
        "instructions": (
            "Fill direction_from_frames for every item by looking at "
            "output/p01s_blind_review/<code>.png and <code>.mp4 ONLY. "
            "Do not open the key file until every line is filled. "
            "Use LEFT / RIGHT / FWD / UNCLEAR. "
            "The kymographs are raw pixels: a rotation slants all three bands the same way, "
            "forward motion slants the outer bands opposite ways."
        ),
        "review_date": "",
        "reviewer": "",
        "blind_to_measurement": True,
        "items": entries,
    }, indent=2), encoding="utf-8")
    KEY_JSON.write_text(json.dumps({
        "warning": "Do not open until every judgement is written into "
                   f"{TEMPLATE_JSON.name}.",
        "seed": args.seed,
        "items": key,
    }, indent=2), encoding="utf-8")

    print(f"\nFill in: {TEMPLATE_JSON}")
    print(f"Do not open: {KEY_JSON}")
    print(f"Look at: {out}/")


if __name__ == "__main__":
    main()
