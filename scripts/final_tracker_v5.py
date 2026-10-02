#!/usr/bin/env python3
"""FINAL TRACKER V5 — V1's route, re-timed by the frozen stand rule.

The user compared V1-V4 from the same start and kept finding V1 best at junctions.
Two attempts to put standing into the walk itself failed on VID00020:
    V4                 stands 47 % at V1's speed: 65 m instead of 81, never turns back.
    V4 --keep-distance same distance as V1, but reaches J21 at another moment, takes
                       another branch and circles the M5-M7-E ring four times.
V1's junction choices only hold at V1's own pace, so V5 does not touch the walk:

    1. V1 runs exactly as frozen (pixel STOP, frozen V_WALK); the output is the one
       continuous path of its final best hypothesis (as in V4, not the per-second leader).
    2. Along that fixed path the dot is moved by the stand rule from V3
       (data/final_tracker_v3/FROZEN_STAND.json): it holds while the rule says
       "standing" and advances while "walking", at the pace that brings it to the end
       of V1's path at the end of the clip:

           a(t) = A_total * W(t) / W_total,   W(t) = walking seconds up to t

Route, junction decisions and route_meters are V1's. Only time-along-route changes.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v5.py --video VID00020 --start-like-v1
"""

from __future__ import annotations

import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_tracker as V1  # noqa: E402
import final_tracker_v3 as V3  # noqa: E402
import final_tracker_v4 as V4  # noqa: E402

OUT_BASE = ROOT / "output/final_tracker_v5"
_CH: dict = {}


class V1PaceChannels(V3.StandChannels):
    """Walks with V1's pixel STOP; the stand rule is computed alongside, not used to walk."""

    def __init__(self, video_id, path, t, freeze):
        super().__init__(video_id, path, t, freeze)
        self.stop_threshold = f"{freeze['stop']} (как V1); стояние по правилу V3 — только для времени"
        _CH["ch"] = self

    def is_stop(self, at: float) -> bool:
        return super(V3.StandChannels, self).is_stop(at)

    def standing(self, at: float) -> bool:
        return V3.StandChannels.is_stop(self, at)


def retime(rows: list[dict], path: list[tuple], ch: V1PaceChannels) -> dict:
    """Move each per-second row along the fixed path by walking time; returns a summary."""
    ts = np.array([p[0] for p in path])
    xy = np.array([[p[1], p[2]] for p in path])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    a_total = float(arc[-1])

    grid = ch.stand_grid
    walking = ~ch.stand_flag
    w_cum = np.concatenate([[0.0], np.cumsum(walking[:-1] * np.diff(grid))])
    w_total = float(w_cum[-1]) or 1e-9

    for r in rows:
        t = float(r["time"])
        w = float(np.interp(t, grid, w_cum))
        target = a_total * w / w_total
        i = int(np.searchsorted(arc, target, side="right")) - 1
        i = max(0, min(i, len(path) - 1))
        if i + 1 < len(path) and arc[i + 1] > arc[i]:
            f = (target - arc[i]) / (arc[i + 1] - arc[i])
            x = xy[i, 0] + f * (xy[i + 1, 0] - xy[i, 0])
            y = xy[i, 1] + f * (xy[i + 1, 1] - xy[i, 1])
        else:
            x, y = xy[i]
        _t, _x, _y, edge, prog = path[i]
        stand = ch.standing(t)
        r.update({"x": round(float(x), 6), "y": round(float(y), 6), "edge": edge,
                  "progress_m": round(float(prog), 3), "is_stop": int(stand),
                  "speed": 0.0 if stand else r.get("speed", 0.0)})
    return {"path_px": round(a_total, 1), "walking_s": round(w_total, 1),
            "standing_share": round(float(ch.stand_flag.mean()), 3),
            "path_end_t_v1": round(float(ts[-1]), 2)}


def main() -> int:
    if not V3.FROZEN_STAND.exists():
        print(f"ОТКАЗ: нет {V3.FROZEN_STAND} — сначала scripts/stand_freeze.py", file=sys.stderr)
        return 1
    argv = sys.argv[1:]
    if "--start-progress" in argv:
        i = argv.index("--start-progress")
        del argv[i:i + 2]
    if "--start-like-v1" in argv:
        argv.remove("--start-like-v1")
        vid = argv[argv.index("--video") + 1]
        rep = json.loads((V1.OUT_BASE / vid / "report.json").read_text(encoding="utf-8"))
        argv += ["--start-edge", rep["start"]["edge"], "--start-from", rep["start"]["from"]]
    if "--out" not in argv:
        vid = argv[argv.index("--video") + 1]
        argv += ["--out", str(OUT_BASE / Path(vid).stem)]
    sys.argv = [sys.argv[0]] + argv

    print("FINAL TRACKER V5 — маршрут V1, точка стоит, когда человек стоит (правило V3)")
    V1.Channels = V1PaceChannels
    V1.Hyp = V4.TracedHyp
    original_write = V1.write_outputs
    note: dict = {}

    def write_retimed(out_dir, g, res, checks, freeze):
        path = V4.continuous_path(res["best"])
        V4.rewrite_rows(res["rows"], path, res["best"].id)
        note.update(retime(res["rows"], path, _CH["ch"]))
        return original_write(out_dir, g, res, checks, freeze)

    V1.write_outputs = write_retimed
    code = V1.main()
    out_dir = Path(argv[argv.index("--out") + 1])
    rp = out_dir / "report.json"
    if code == 0 and rp.exists():
        doc = json.loads(rp.read_text(encoding="utf-8"))
        frozen = json.loads(V3.FROZEN_STAND.read_text(encoding="utf-8"))
        doc["phase"] = "FINAL TRACKER V5"
        doc.setdefault("rules", {})["stop"] = "V1 pixel STOP for the walk; FROZEN_STAND for the dot"
        doc["stand_rule"] = {"frozen_at": frozen["frozen_at"], "train_clip": frozen["train_clip"],
                             **frozen["rule"]}
        doc["retime"] = {"rule": "a(t) = A_total * W(t) / W_total along V1's final best path", **note}
        rp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  точка переставлена по маршруту V1: стоит {note['standing_share']:.0%} времени, "
              f"идёт {note['walking_s']:.0f} с")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V1.CheckFailed as e:
        print(f"ОТКАЗ: {e}", file=sys.stderr)
        raise SystemExit(1)
