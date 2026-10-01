#!/usr/bin/env python3
"""
Blind event frame review for full VID00001 — NO encoder / yaw CSV.

Extracts frame strips around candidate turn timestamps for visual labeling.

Usage:
    PYTHONPATH=. python scripts/p01r_event_review.py
    PYTHONPATH=. python scripts/p01r_event_review.py --video /path/to/VID00001.AVI

After reviewing frames, edit data/p01r/VID00001_events.json then run:
    PYTHONPATH=. python scripts/p01r_event_gate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data/p01r/VID00001.AVI"
FALLBACK_VIDEO = Path("/Users/artem/Downloads/febf7cf1-053c-4ec9-9b1a-a20a4aa5aac6_VID00001.AVI")
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
OUT_DIR = ROOT / "output/p01r_event_benchmark/frame_review"

# Candidate turn anchors (seconds) from prior VID00001 annotation.
CANDIDATE_ANCHORS = [51, 81, 108, 119, 123, 156, 168, 181, 216, 226, 236]

# Additional forward candidate anchors (between turns, for FWD windows).
FWD_CANDIDATE_ANCHORS = [42, 72, 95, 140, 200, 250]


def resolve_video(path: Path | None) -> Path:
    if path and path.exists():
        return path
    if DEFAULT_VIDEO.exists():
        return DEFAULT_VIDEO
    if FALLBACK_VIDEO.exists():
        return FALLBACK_VIDEO
    raise SystemExit("VID00001 not found. Pass --video or symlink to data/p01r/VID00001.AVI")


def extract_strip(
    video: Path,
    anchor_s: float,
    out_dir: Path,
    half_window_s: float = 2.5,
    step_s: float = 0.5,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")

    saved = []
    t = max(0.0, anchor_s - half_window_s)
    t_end = anchor_s + half_window_s
    while t <= t_end + 0.01:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        rel = t - anchor_s
        fname = f"frame_t{rel:+05.1f}s.jpg"
        path = out_dir / fname
        cv2.imwrite(str(path), frame)
        saved.append(str(path.relative_to(ROOT)))
        t += step_s
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--half-window", type=float, default=2.5)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("-o", "--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    video = resolve_video(args.video)
    print(f"Video: {video}")
    print(f"Extracting frame strips (±{args.half_window}s, step={args.step}s)")
    print("Assign labels in data/p01r/VID00001_events.json BEFORE running encoder.\n")

    manifest: dict = {
        "video": str(video),
        "candidate_anchors_s": CANDIDATE_ANCHORS,
        "fwd_candidate_anchors_s": FWD_CANDIDATE_ANCHORS,
        "half_window_s": args.half_window,
        "strips": {},
    }

    all_anchors = [(t, "turn_candidate") for t in CANDIDATE_ANCHORS]
    all_anchors += [(t, "fwd_candidate") for t in FWD_CANDIDATE_ANCHORS]

    for anchor, kind in all_anchors:
        strip_dir = args.output / f"anchor_{anchor:04d}s"
        frames = extract_strip(video, anchor, strip_dir, args.half_window, args.step)
        manifest["strips"][str(anchor)] = {
            "kind": kind,
            "anchor_s": anchor,
            "window": [anchor - args.half_window, anchor + args.half_window],
            "frames": frames,
        }
        print(f"  anchor {anchor:4d}s ({kind}): {len(frames)} frames → {strip_dir.relative_to(ROOT)}")

    manifest_path = args.output / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path.relative_to(ROOT)}")
    print(f"Edit {EVENTS_JSON.relative_to(ROOT)} with blind labels, then:")
    print("  PYTHONPATH=. python scripts/p01r_event_gate.py")


if __name__ == "__main__":
    main()
