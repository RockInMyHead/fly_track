#!/usr/bin/env python3
"""FINAL TRACKER V4 — V1 with one change: who decides "standing".

On VID00020 the user compared V1, V2 and V3 side by side from the same start and found
V1 the most accurate at junctions. V1 and V2 both read "standing" from pixel motion,
which is at chance on the user's stand labels (VID00020 and VID00010). V3 fixed that
with a frozen rule but sits on V2's junction logic.

V4 keeps every V1 rule and replaces only `is_stop` with V3's frozen stand rule
(data/final_tracker_v3/FROZEN_STAND.json, unchanged). V1 itself is not edited:
its Channels class is swapped for V3.StandChannels before V1.main runs.

V1 has no --start-progress: the walk starts at the node --start-from.

Output path. V1 writes each second the position of whichever hypothesis leads at that
moment; when the lead changes the dot jumps to another branch and the drawn line tears
(VID00020: jumps of 500-700 px in one second). Here every hypothesis logs its own steps
and its parent, and trajectory.csv / edge_sequence.csv are rewritten from the one
continuous path of the final best hypothesis — the same one route_meters already uses.

--keep-distance (off by default) rescales V_WALK so the clip's total distance equals V1's.
On VID00020 it made the walk reach a junction at another moment, take another branch and
circle the M5-M7-E ring four times: V1's junction choices only hold at V1's own pace.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v4.py --video VID00020 --start-like-v1
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v4.py --video VID00020 \\
        --start-edge J11__T12 --start-from T12
"""

from __future__ import annotations

import bisect
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_tracker as V1  # noqa: E402
import final_tracker_v3 as V3  # noqa: E402

OUT_BASE = ROOT / "output/final_tracker_v4"

_BY_HISTORY: dict[tuple, "TracedHyp"] = {}
MAX_SPEED_RATIO = 3.0


class DistanceKeepingChannels(V3.StandChannels):
    """Stand rule from V3, walking speed rescaled so the clip's total distance equals V1's.

    V_WALK was frozen with pixel STOP, which almost never fires (11 % on VID00020), so it is
    in effect the mean speed including standing. With real standing (47 %) the same speed
    leaves the walk ~40 % short and the route never reaches its turn-back.
    """

    def __init__(self, video_id, path, t, freeze):
        super().__init__(video_id, path, t, freeze)
        pixel = np.array([super(V3.StandChannels, self).is_stop(float(x)) for x in self.stand_grid])
        walk_v1 = 1.0 - float(pixel.mean())
        walk_v4 = max(1e-6, 1.0 - float(self.stand_flag.mean()))
        ratio = min(MAX_SPEED_RATIO, walk_v1 / walk_v4)
        self.speed_note = {"frozen_v_walk": float(freeze["v_walk"]), "walk_share_v1": round(walk_v1, 3),
                           "walk_share_v4": round(walk_v4, 3), "ratio": round(ratio, 3)}
        freeze["v_walk"] = round(float(freeze["v_walk"]) * ratio, 4)
        self.speed_note["v_walk"] = freeze["v_walk"]
        _SPEED.update(self.speed_note)


_SPEED: dict = {}


class _LoggedTrace(deque):
    def __init__(self, owner: "TracedHyp", items=()):
        super().__init__(items, maxlen=400)
        self.owner = owner

    def append(self, item):
        super().append(item)
        o = self.owner
        o.log.append((float(item[0]), float(item[1]), float(item[2]), o.edge, float(o.progress)))


class TracedHyp(V1.Hyp):
    __slots__ = ("log", "parent", "_trace")

    def __init__(self, hid, edge, entry, g, score, t0, history=None):
        self.log: list[tuple] = []
        super().__init__(hid, edge, entry, g, score, t0, history=history)
        self.parent = _BY_HISTORY.get(tuple(self.history[:-1])) if len(self.history) > 1 else None
        _BY_HISTORY[tuple(self.history)] = self

    # V1 replaces a child's trace with a plain deque copied from the parent.
    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, value):
        self._trace = _LoggedTrace(self, value)


def continuous_path(best: TracedHyp) -> list[tuple]:
    chain = []
    h = best
    while h is not None and h not in chain:
        chain.append(h)
        h = h.parent
    chain.reverse()
    out: list[tuple] = []
    for k, h in enumerate(chain):
        until = chain[k + 1].born if k + 1 < len(chain) else float("inf")
        out.extend(p for p in h.log if p[0] < until and (not out or p[0] > out[-1][0]))
    return out


def rewrite_rows(rows: list[dict], path: list[tuple], best_id: int) -> int:
    """Put the continuous path into the per-second rows; returns how many rows moved."""
    if not path:
        return 0
    ts = [p[0] for p in path]
    moved = 0
    for r in rows:
        i = max(0, bisect.bisect_right(ts, float(r["time"])) - 1)
        _t, x, y, edge, prog = path[i]
        if r["edge"] != edge or abs(float(r["x"]) - x) > 1e-6:
            moved += 1
        r.update({"x": round(x, 6), "y": round(y, 6), "edge": edge,
                  "progress_m": round(prog, 3), "best_hypothesis": best_id})
    return moved


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
        s = rep["start"]
        argv += ["--start-edge", s["edge"], "--start-from", s["from"]]
    if "--out" not in argv:
        vid = argv[argv.index("--video") + 1]
        argv += ["--out", str(OUT_BASE / Path(vid).stem)]
    sys.argv = [sys.argv[0]] + argv

    print("FINAL TRACKER V4 — развилки как в V1, «стоит» решают муха и ритм шагов (V3)")
    keep_distance = "--keep-distance" in argv
    if keep_distance:
        argv.remove("--keep-distance")
        sys.argv = [sys.argv[0]] + argv
    V1.Channels = DistanceKeepingChannels if keep_distance else V3.StandChannels
    V1.Hyp = TracedHyp
    original_write = V1.write_outputs
    path_note: dict = {}

    def write_continuous(out_dir, g, res, checks, freeze):
        path = continuous_path(res["best"])
        path_note["rows_moved"] = rewrite_rows(res["rows"], path, res["best"].id)
        path_note["rows"] = len(res["rows"])
        return original_write(out_dir, g, res, checks, freeze)

    V1.write_outputs = write_continuous
    code = V1.main()
    out_dir = Path(argv[argv.index("--out") + 1])
    rp = out_dir / "report.json"
    if code == 0 and rp.exists():
        doc = json.loads(rp.read_text(encoding="utf-8"))
        frozen = json.loads(V3.FROZEN_STAND.read_text(encoding="utf-8"))
        doc["phase"] = "FINAL TRACKER V4"
        doc.setdefault("rules", {})["stop"] = "FROZEN_STAND"
        doc["stand_rule"] = {"frozen_at": frozen["frozen_at"], "train_clip": frozen["train_clip"],
                             **frozen["rule"]}
        doc["output_path"] = {"mode": "continuous path of the final best hypothesis", **path_note}
        if _SPEED:
            doc["speed"] = {"rule": "same total distance as V1, walked only while not standing", **_SPEED}
        rp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  непрерывный путь лучшей гипотезы: переписано {path_note.get('rows_moved')} "
              f"из {path_note.get('rows')} секунд")
        if _SPEED:
            print(f"  скорость: {_SPEED.get('frozen_v_walk')} → {_SPEED.get('v_walk')} м/с "
                  f"(ходьба V1 {_SPEED.get('walk_share_v1')}, V4 {_SPEED.get('walk_share_v4')})")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V1.CheckFailed as e:
        print(f"ОТКАЗ: {e}", file=sys.stderr)
        raise SystemExit(1)
