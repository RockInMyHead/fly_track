#!/usr/bin/env python3
"""
P0.1S — targeted look at one window, with a mirror self-check.

The blind review measures how well direction can be read across a shuffled set.
This script does the opposite for a single named window: it puts everything about
that one window in front of the eye, so a person can decide it without any context
from the rest of the set.

What is produced for the window
-------------------------------
    <name>_raw.mp4       the footage, unmodified, no overlay
    <name>_overlay.mp4   the measured per-patch displacement drawn as arrows
    <name>_panel.png     five stills, a full-width space-time image, and three band
                         kymographs -- the same view the blind review uses
    <name>_mirror.mp4    the same window mirrored left-right
    <name>_mirror_panel.png

The mirrored copy is the self-check that needs no ground truth. Mirroring reverses
the content motion exactly. So if a reviewer reports content moving the same way in
the original and the mirrored clip, the reading is not tracking direction; if the
two answers are opposite, the reading is real. The same argument applies to the
measurement, and mirroring is also the cleanest test that the estimator is not
picking up some fixed scene bias.

Usage:
    PYTHONPATH=. python scripts/p01s_targeted_window.py --event right_216
    PYTHONPATH=. python scripts/p01s_targeted_window.py --event left_65
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.lk_motion_field import LKMotionField
from scripts.p01s_blind_review3 import build_panel, read_frames
from scripts.p01s_instrument_check import motion_overlay

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
FPS = 30.0
ENCODE = (192, 108)
GRID_W, GRID_H = 16, 8


def measure(grays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    dxs, dys = [], []
    for g in grays:
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(0.30 * h): int(0.70 * h)])
        dxf, dyf, _d = lk.process(band)
        dxs.append(dxf)
        dys.append(dyf)
    n_pre = int(round(PRE_ROLL_S * FPS))
    dx = np.stack(dxs[n_pre:], axis=0)
    dy = np.stack(dys[n_pre:], axis=0)
    dxz = dx - dx.mean(axis=(1, 2), keepdims=True)       # remove the uniform component
    resid = float(np.abs(dxz).mean())
    return dx, dy, resid


def report_direction(name: str, dx: np.ndarray, resid: float) -> dict:
    mean_dx = float(dx.mean())
    mean_abs = float(np.abs(dx).mean())
    coherence = abs(mean_dx) / max(mean_abs, 1e-9)
    frac_pos = float((dx > 0).mean())
    content = "RIGHT" if mean_dx > 0 else "LEFT"
    camera = "LEFT" if mean_dx > 0 else "RIGHT"
    print(f"  {name}")
    print(f"    mean content dx            {mean_dx:+.3f} px/frame  ->  content moves {content}")
    print(f"    mean |dx| (local activity) {mean_abs:.3f} px/frame")
    print(f"    coherence |mean|/mean|dx|  {coherence:.2f}"
          + ("   (readable)" if coherence >= 0.40 else "   (residue of opposed motions)"))
    print(f"    cells agreeing with the mean {max(frac_pos, 1 - frac_pos):.0%}")
    print(f"    if the label means camera rotation, this window is CAMERA {camera}")
    print(f"    after removing the uniform component, mean |residual| {resid:.3f} px "
          f"(pure deformation / divergence, no yaw)")
    return {"mean_dx": mean_dx, "mean_abs_dx": mean_abs, "coherence": coherence,
            "frac_agreeing": max(frac_pos, 1 - frac_pos),
            "content_direction": content, "camera_direction": camera,
            "residual_deformation": resid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="right_216")
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_targeted")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    ev = next((e for e in doc["events"] if e["id"] == args.event), None)
    if ev is None:
        raise SystemExit(f"unknown event {args.event!r}; known: "
                         + ", ".join(e["id"] for e in doc["events"]))
    anchor = float(ev["anchor_s"])
    t0 = max(0.0, anchor - PRE_ROLL_S)
    t1 = anchor + EVENT_DUR_S

    print(f"P0.1S targeted window — {args.event}")
    print(f"  old label: {ev['label']}  (camera sense)")
    print(f"  window t={t0:.1f}-{t1:.1f}s\n")

    grays, bgrs = read_frames(Path(args.video), t0, t1)
    if not grays:
        raise SystemExit("no frames read")

    # ---- raw ----
    vw = 1280
    vh = int(bgrs[0].shape[0] * vw / bgrs[0].shape[1])
    w = cv2.VideoWriter(str(out / f"{args.event}_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        FPS, (vw, vh))
    for f in bgrs:
        w.write(cv2.resize(f, (vw, vh)))
    w.release()

    # ---- panel ----
    cv2.imwrite(str(out / f"{args.event}_panel.png"), build_panel(args.event, grays, bgrs))

    # ---- overlay + measurement ----
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    w = cv2.VideoWriter(str(out / f"{args.event}_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        FPS, (bgrs[0].shape[1], bgrs[0].shape[0]))
    dxs, dys = [], []
    for g, bgr in zip(grays, bgrs):
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(0.30 * h): int(0.70 * h)])
        dxf, dyf, _d = lk.process(band)
        dxs.append(dxf)
        dys.append(dyf)
        w.write(motion_overlay(bgr, dxf, dyf, scale=4))
    w.release()

    n_pre = int(round(PRE_ROLL_S * FPS))
    dx = np.stack(dxs[n_pre:], axis=0)
    dy = np.stack(dys[n_pre:], axis=0)
    dxz = dx - dx.mean(axis=(1, 2), keepdims=True)
    resid = float(np.abs(dxz).mean())
    orig = report_direction(f"{args.event}  (as recorded)", dx, resid)

    # ---- mirrored ----
    m_grays = [np.ascontiguousarray(g[:, ::-1]) for g in grays]
    m_bgrs = [np.ascontiguousarray(b[:, ::-1]) for b in bgrs]
    cv2.imwrite(str(out / f"{args.event}_mirror_panel.png"), build_panel(args.event + "M", m_grays, m_bgrs))
    w = cv2.VideoWriter(str(out / f"{args.event}_mirror.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        FPS, (vw, vh))
    for f in m_bgrs:
        w.write(cv2.resize(f, (vw, vh)))
    w.release()

    mlk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    mdxs = []
    for g in m_grays:
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(0.30 * h): int(0.70 * h)])
        dxf, _dyf, _d = mlk.process(band)
        mdxs.append(dxf)
    mdx = np.stack(mdxs[n_pre:], axis=0)
    mdxz = mdx - mdx.mean(axis=(1, 2), keepdims=True)
    mir = report_direction(f"{args.event}  (mirrored)", mdx, float(np.abs(mdxz).mean()))

    flips = np.sign(orig["mean_dx"]) != np.sign(mir["mean_dx"])
    mirror_test = abs(orig["mean_dx"] + mir["mean_dx"]) < 0.25 * max(
        abs(orig["mean_dx"]), 1e-9
    )
    print(f"\n  mirror self-checks")
    print(f"    content direction reverses under mirroring: {flips}")
    print(f"    mirrored mean dx ≈ -original mean dx: {mirror_test} "
          f"({orig['mean_dx']:+.3f} vs {mir['mean_dx']:+.3f})")

    summary = {
        "event": args.event,
        "old_label": ev["label"],
        "window": [t0, t1],
        "as_recorded": orig,
        "mirrored": mir,
        "mirror_reverses_direction": bool(flips),
        "mirror_magnitude_matches": bool(mirror_test),
        "files": {
            "raw": f"{args.event}_raw.mp4",
            "overlay": f"{args.event}_overlay.mp4",
            "panel": f"{args.event}_panel.png",
            "mirror": f"{args.event}_mirror.mp4",
            "mirror_panel": f"{args.event}_mirror_panel.png",
        },
        "how_to_read": (
            "In the raw clip: which edge of the frame does the scenery sweep toward? "
            "Content toward the LEFT edge means content motion LEFT, which for a camera "
            "yaw means the camera rotated RIGHT. The overlay clip draws the measured "
            "per-patch displacement as arrows; check by eye whether the arrows follow "
            "the texture. The mirror clip shows the same window flipped, so its content "
            "motion must be the opposite of the raw clip."
        ),
    }
    (out / f"{args.event}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
