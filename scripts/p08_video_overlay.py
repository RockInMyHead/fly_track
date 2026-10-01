#!/usr/bin/env python3
"""
P08 — the video beside the plan, with the position the graph implies.

The point of this figure is that the position no longer comes from integrating anything.
It is the interpolation along the edge the tracker says the walker is on, so what moves on
the right is a consequence of the edge sequence, and a wrong sequence is visible as a dot
that goes somewhere a person did not.

Frames are read sequentially rather than by seeking. The source clips declare more frames
than they contain, so a seek lands in the wrong place, which the project already recorded
in `fly_vo/video_reader.py`; that reader is used instead.

Usage:
    PYTHONPATH=. python scripts/p08_video_overlay.py
    PYTHONPATH=. python scripts/p08_video_overlay.py --t0 60 --t1 300
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402

OUT = ROOT / "output/p08"
MEDIA = ROOT / "webapp/media"
GRAPH_PATH = ROOT / "data/p08/graph.json"

PANEL_H = 480
VID_W = 854
FPS = 30


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def fit_panel(img: np.ndarray, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(int(img.shape[1] * s), 1), h), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID00001")
    ap.add_argument("--graph", default=str(GRAPH_PATH))
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--out", default=str(OUT / "trajectory_video.mp4"))
    args = ap.parse_args()

    g = Graph.load(args.graph)
    plan_file = ROOT / "data/p08" / (g.doc.get("image") or "plan.png")
    plan = cv2.imread(str(plan_file))
    if plan is None:
        print(f"нет плана {plan_file} — сначала постройте граф в редакторе")
        return
    plan = fit_panel(plan, PANEL_H)

    vpath = MEDIA / f"{args.video}_fixed.mp4"
    if not vpath.exists():
        vpath = MEDIA / f"{args.video}.mp4"
    if not vpath.exists():
        print(f"нет видео {vpath}")
        return

    traj = read_csv(OUT / "graph_trajectory.csv")
    if not traj:
        print("нет output/p08/graph_trajectory.csv — сначала запустите трекер")
        return
    tt = np.array([float(r["t"]) for r in traj])
    tx = np.array([float(r["x_norm"]) for r in traj])
    ty = np.array([float(r["y_norm"]) for r in traj])

    seq = read_csv(OUT / "edge_sequence.csv")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"не открылось видео {vpath}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    src_fps = fps if 25 <= fps <= 60 else FPS
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    sw, sh = plan.shape[1], plan.shape[0]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (VID_W + sw, PANEL_H))

    # node positions and edge endpoints in plan-panel pixels
    def to_panel(nid: str) -> tuple[int, int]:
        x, y = g.pos(nid)
        return int(x / g.img_w * sw), int(y / g.img_h * sh)

    print(f"P08 — видео рядом с планом")
    print(f"  видео {vpath.name}, {src_fps:.0f} кадров/с, объявлено кадров {n_total}")
    print(f"  панель плана {sw}x{sh}, итог {VID_W + sw}x{PANEL_H}")
    if args.t0 is not None:
        print(f"  отрезок {args.t0:.0f}..{args.t1:.0f} с")

    i = 0
    written = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        tnow = i / src_fps
        i += 1
        if args.t0 is not None and tnow < args.t0:
            continue
        if args.t1 is not None and tnow > args.t1:
            break
        ok, frame = cap.retrieve()
        if not ok:
            continue

        left = cv2.resize(frame, (VID_W, PANEL_H), interpolation=cv2.INTER_AREA)
        right = plan.copy()

        # the graph itself, faint, so the route is readable against the plan
        for e in g.edges.values():
            cv2.line(right, to_panel(e["from"]), to_panel(e["to"]), (150, 110, 60), 2)
        for nid in g.nodes:
            cv2.circle(right, to_panel(nid), 5, (255, 200, 92), -1)

        # travelled part of the sequence, so the route so far is visible
        for s in seq:
            if float(s["time_start"]) > tnow:
                continue
            col = (255, 140, 60) if float(s["time_end"]) >= tnow else (120, 220, 120)
            cv2.line(right, to_panel(s["from"]), to_panel(s["to"]), col, 3)

        if tt[0] <= tnow <= tt[-1]:
            x = int(np.interp(tnow, tt, tx) * sw)
            y = int(np.interp(tnow, tt, ty) * sh)
            cv2.circle(right, (x, y), 11, (30, 30, 30), -1)
            cv2.circle(right, (x, y), 9, (70, 220, 255), -1)

        cv2.putText(right, f"t = {tnow:6.1f} s", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(left, "video", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        writer.write(np.hstack([left, right]))
        written += 1
        if written % 900 == 0:
            print(f"  записано кадров {written} (t = {tnow:.0f} с)")

    cap.release()
    writer.release()
    print(f"  готово: {written} кадров → {args.out}")


if __name__ == "__main__":
    main()
