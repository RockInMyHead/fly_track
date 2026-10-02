#!/usr/bin/env python3
"""FINAL TRACKER V3 — V2 with one change: who decides "standing".

V2 stops the walk when the frame moves less than a threshold. With the camera on the
head that is useless: a person standing and looking around moves the frame as much as a
person walking (on VID00020 it caught 11 % of the standing seconds, balanced 50 %).

V3 keeps every V2 rule and replaces only `is_stop`. Standing is read from two inputs:

    fly      the net_displacement channel, signed: +confidence for NO_NET, -for MOVE.
             The class is nearly always MOVE; the confidence is what drops when standing.
    bob      share of the vertical image shift power at 1.4-2.6 Hz over 6 s — the step
             rhythm of a head-mounted camera. Low when the person is not stepping.

    p = sigmoid(b + w_fly * fly + w_bob * bob), averaged over SMOOTH_S, standing if p > thr

Weights, smoothing and threshold come from data/final_tracker_v3/FROZEN_STAND.json and
are not changed here. They were fitted on VID00020 only; a result on VID00020 is in-sample.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v3.py --video VID00020 --start-like-v2
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_v3.py --video VID00020 \\
        --start-edge J11__T12 --start-from T12
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_tracker as V1  # noqa: E402
import final_tracker_v2 as V2  # noqa: E402

OUT_BASE = ROOT / "output/final_tracker_v3"
FROZEN_STAND = ROOT / "data/final_tracker_v3/FROZEN_STAND.json"
SHIFT_CACHE = OUT_BASE / "cache"

STEP_BAND = (1.4, 2.6)
BOB_WIN_S = 6.0
GRID_S = 0.5


def image_shifts(video: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Frame-to-frame global shift (phase correlation on 320x180 grey), cached by size+mtime."""
    st = video.stat()
    cache = SHIFT_CACHE / f"shift_{video.stem}.npz"
    if cache.exists():
        d = np.load(cache)
        if int(d["size"]) == st.st_size and abs(float(d["mtime"]) - st.st_mtime) < 1.0:
            return d["dx"], d["dy"], float(d["fps"])
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, win, dx, dy = None, None, [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (sx, sy), _ = cv2.phaseCorrelate(prev, g, win)
            dx.append(sx)
            dy.append(sy)
        prev = g
    dx, dy = np.array(dx), np.array(dy)
    SHIFT_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, dx=dx, dy=dy, fps=np.float64(fps),
             size=np.int64(st.st_size), mtime=np.float64(st.st_mtime))
    return dx, dy, fps


def step_bob(dy: np.ndarray, fps: float, centers: np.ndarray) -> np.ndarray:
    t = np.arange(len(dy)) / fps
    out = np.full(len(centers), np.nan)
    for k, c in enumerate(centers):
        m = (t >= c - BOB_WIN_S / 2) & (t < c + BOB_WIN_S / 2)
        if m.sum() < fps * 2:
            continue
        s = dy[m] - dy[m].mean()
        p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        f = np.fft.rfftfreq(len(s), 1 / fps)
        tot = p[(f > 0.3) & (f < 6)].sum()
        if tot > 0:
            out[k] = p[(f >= STEP_BAND[0]) & (f <= STEP_BAND[1])].sum() / tot
    good = np.isfinite(out)
    if good.any() and not good.all():
        out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return out


def stand_features(ch: V1.Channels, video: Path, grid: np.ndarray) -> np.ndarray:
    """[len(grid), 2]: signed fly confidence, step-rhythm share."""
    fly = []
    for x in grid:
        nd = ch.net_at(float(x))
        fly.append((1.0 if nd["class"] == "NO_NET" else -1.0) * float(nd["confidence"]))
    _dx, dy, fps = image_shifts(video)
    return np.column_stack([np.array(fly), step_bob(dy, fps, grid)])


def stand_probability(F: np.ndarray, rule: dict) -> np.ndarray:
    z = rule["bias"] + F @ np.array([rule["w_fly"], rule["w_bob"]])
    p = 1.0 / (1.0 + np.exp(-z))
    n = max(1, int(round(rule["smooth_s"] / GRID_S)))
    k = np.ones(n) / n
    pad = np.pad(p, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


class StandChannels(V1.Channels):
    """V1 channels, with is_stop answered by the frozen stand rule."""

    def __init__(self, video_id: str, path: Path, t: np.ndarray, freeze: dict):
        super().__init__(video_id, path, t, freeze)
        rule = json.loads(FROZEN_STAND.read_text(encoding="utf-8"))["rule"]
        self.stand_rule = rule
        self.stand_grid = np.arange(float(t[0]), float(t[-1]) + 1e-9, GRID_S)
        F = stand_features(self, path, self.stand_grid)
        self.stand_p = stand_probability(F, rule)
        self.stand_flag = self.stand_p > float(rule["threshold"])
        self.stop_threshold = f"V3 стоит, если p > {rule['threshold']}"

    def is_stop(self, at: float) -> bool:
        i = int(np.searchsorted(self.stand_grid, at))
        i = max(0, min(i, len(self.stand_grid) - 1))
        return bool(self.stand_flag[i])


def main() -> int:
    if not FROZEN_STAND.exists():
        print(f"ОТКАЗ: нет {FROZEN_STAND} — сначала scripts/stand_freeze.py", file=sys.stderr)
        return 1
    argv = sys.argv[1:]
    if "--start-like-v2" in argv:
        argv.remove("--start-like-v2")
        vid = argv[argv.index("--video") + 1]
        rep = json.loads((V2.OUT_BASE / vid / "report.json").read_text(encoding="utf-8"))
        s = rep["start"]
        argv += ["--start-edge", s["edge"], "--start-from", s["from"],
                 "--start-progress", str(s.get("progress_m", 0.0))]
    if "--out" not in argv:
        vid = argv[argv.index("--video") + 1]
        argv += ["--out", str(OUT_BASE / Path(vid).stem)]
    sys.argv = [sys.argv[0]] + argv

    print("FINAL TRACKER V3 — как V2, но «стоит» решают муха и ритм шагов")
    V1.Channels = StandChannels
    code = V2.main()
    out_dir = Path(argv[argv.index("--out") + 1])
    rp = out_dir / "report.json"
    if code == 0 and rp.exists():
        doc = json.loads(rp.read_text(encoding="utf-8"))
        frozen = json.loads(FROZEN_STAND.read_text(encoding="utf-8"))
        doc["phase"] = "FINAL TRACKER V3"
        doc["rules"]["stop"] = "FROZEN_STAND"
        doc["stand_rule"] = {"frozen_at": frozen["frozen_at"], "train_clip": frozen["train_clip"],
                             **frozen["rule"]}
        rp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V1.CheckFailed as e:
        print(f"ОТКАЗ: {e}", file=sys.stderr)
        raise SystemExit(1)
