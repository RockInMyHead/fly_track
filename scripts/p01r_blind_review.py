#!/usr/bin/env python3
"""
Blind frame review — extract frames ONLY. No optic_flow / yaw CSV.

Usage:
    PYTHONPATH=. python scripts/p01r_blind_review.py data/p01r/controlled.mp4
    PYTHONPATH=. python scripts/p01r_blind_review.py /path/to/controlled.mp4 --step 0.5

After review, edit data/p01r/controlled_segments.json (frames only), then:
    PYTHONPATH=. python scripts/p01r_controlled_gate.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS_JSON = ROOT / "data/p01r/controlled_segments.json"
SPEC_JSON = ROOT / "data/p01r/controlled_recording_spec.json"


def extract_frames(video: Path, out_dir: Path, step_s: float = 1.0) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 30.0
    saved = []
    t = 0.0
    while t <= duration + 0.01:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        path = out_dir / f"frame_t{t:05.1f}s.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(str(path.relative_to(ROOT)))
        t += step_s
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind frame review (no encoder)")
    parser.add_argument("video", type=Path, help="Path to controlled.mp4")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "output/p01r_blind_review/controlled")
    parser.add_argument("--step", type=float, default=1.0, help="Frame interval in seconds")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    frames = extract_frames(video, args.output, args.step)
    print(f"Extracted {len(frames)} frames → {args.output}")
    print("Review frames visually. Look for: FWD → LEFT → FWD → RIGHT → FWD")
    print(f"Then edit: {SEGMENTS_JSON}")
    print("Set blind_review_done=true and actual segment boundaries.")
    print("Do NOT run p01r_visual_only or inspect yaw CSV before editing segments.")

    record = {
        "video": str(video.relative_to(ROOT)) if video.is_relative_to(ROOT) else str(video),
        "review_date": str(date.today()),
        "frame_step_s": args.step,
        "frames": frames,
        "nominal_spec": json.loads(SPEC_JSON.read_text()) if SPEC_JSON.exists() else {},
        "next_step": "Edit data/p01r/controlled_segments.json, then run scripts/p01r_controlled_gate.py",
    }
    out_json = args.output / "review_manifest.json"
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
