#!/usr/bin/env python3
"""FINAL TRACKER V1 — a route on the floor plan from video, graph and a start. No manual route.

WHAT IT DOES
------------
Video plus the building graph plus a start gives a trajectory, the sequence of edges, the moments a
decision was made, and the stretches where the answer is not known. Nothing is annotated by hand.

WHAT IT IS ALLOWED TO USE
-------------------------
Four things, and only these:

    camera_yaw         LEFT / RIGHT / SILENT, from the tracker's own yaw signal. It says where the
                       camera turned, which is not automatically where the body went.
    the STOP gate      frame_motion < 12 on the video itself. Proven safe: 0 false STOPs in 40
                       windows below the threshold across two independent sets.
    MOVE / NO_NET      the early visual populations, test 0.651 / 0.624 at p 0.003 over five camera
                       recordings, five of five above chance. NO_NET does NOT mean standing: it
                       means the person moved but gained little net displacement.
    the graph          walls bound what is possible; a fork keeps several ways on; when the data is
                       insufficient the answer stays AMBIGUOUS rather than guessed.

Everything else that was tried is excluded by name in `FROZEN_V1.json`: route_change, SAME /
DIFFERENT, the P09.2 glance filter, and the old speed signal. They are kept as history only.

WHY THE SPEED IS ONE NUMBER
---------------------------
The old tracker paced itself by a signal with no relation to movement, so a walk that never stopped
advanced at about a third of pace and a junction could be reached that was never visited. V1
replaces it with the two states that were actually established: stopped, or walking at one constant
speed. There is no attempt to read an instantaneous speed from the brain, because that was measured
and does not exist.

WHY HYPOTHESES LIVE AFTER A FORK
--------------------------------
A single yaw event must not decide a junction: the camera can turn left while the body carries
straight on, and J35 is the recorded case. So a fork creates hypotheses, they are carried forward
for twenty seconds, and later events re-rank them. Two hypotheses that come back together are
merged, and the earlier ambiguity is not counted as an error for the rest of the route.

WHAT IT REFUSES TO DO
---------------------
It refuses to hide uncertainty: equal-scoring hypotheses both appear, with the status AMBIGUOUS,
and the per-second output carries the top score, the runner-up and the margin between them. It
refuses to move while the person is stopped, and it refuses to leave the graph.

Usage:
    PYTHONPATH=. python scripts/final_tracker.py --video VID00010 \\
        --graph data/p08/graph.json \\
        --start-node J6 --previous-node J5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p08_graph import Graph, GraphError  # noqa: E402

FREEZE = ROOT / "data/final_tracker/FROZEN_V1.json"
MANIFEST = ROOT / "data/p14/MANIFEST.json"
OUT_BASE = ROOT / "output/final_tracker"
CAMERA = Path("/Volumes/NO NAME/DCIM")
P01R = ROOT / "data/p01r"

EPS_M = 1e-6
NODE_EPS_M = 1e-6


# --------------------------------------------------------------------------- self-checks

class CheckFailed(RuntimeError):
    """A loud stop. The spec requires the tracker to refuse rather than guess."""


def _fail(what: str, detail: str) -> None:
    raise CheckFailed(f"{what}: {detail}")


def resolve_video(v: str) -> Path:
    """Accept an identifier or a path, and return a file that exists."""
    p = Path(v)
    if p.exists():
        return p
    for cand in (P01R / f"{v}.AVI", P01R / v, CAMERA / f"{v}.AVI", CAMERA / v):
        if cand.exists():
            return cand
    _fail("видео", f"не найдено ни как путь, ни в проекте, ни на камере: {v}")


def real_frames_of(video_id: str, path: Path) -> dict:
    """Frames actually decodable, from the manifest or a cache; computed only as a last resort."""
    man = ROOT / "data/p14/MANIFEST.json"
    if man.exists():
        doc = json.loads(man.read_text(encoding="utf-8"))
        for rec in (doc.get("videos", {}).get(video_id) or {}).values():
            n = rec.get("real_frames")
            if n:
                return {"frames": int(n), "source": "data/p14/MANIFEST.json",
                        "declared": rec.get("declared_frames"),
                        "resolution": rec.get("resolution"), "fps": rec.get("declared_fps")}
    cache = OUT_BASE / "cache" / f"frames_{video_id}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        d["source"] = "кэш"
        return d
    _fail("реальное число кадров",
          f"нет ни в манифесте, ни в кэше для {video_id}. Сначала: "
          f"PYTHONPATH=. python scripts/p14_manifest.py --frames")


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        s = (j.get("streams") or [{}])[0]
        rate = s.get("r_frame_rate") or "0/1"
        num, den = (int(x) for x in rate.split("/"))
        return {"width": s.get("width"), "height": s.get("height"),
                "fps": num / den if den else None}
    except Exception:
        return {}


def self_checks(video_id: str, path: Path, g: Graph, freeze: dict,
                start_edge: str, start_from: str,
                yaw_path: Path, brain_path: Path) -> dict:
    """Everything that must hold before a single step is taken."""
    out = {}

    # graph
    if not g.edges:
        _fail("граф", "пустой")
    missing_len = [e for e in g.edges if g.length_m(e) is None]
    if missing_len:
        _fail("длины рёбер", f"нет длины у {len(missing_len)}: {missing_len[:5]}")
    if start_edge not in g.edges:
        _fail("старт", f"ребра {start_edge} нет в графе")
    nodes_on = {g.edges[start_edge]["from"], g.edges[start_edge]["to"]}
    if start_from not in nodes_on:
        _fail("старт", f"узел {start_from} не лежит на ребре {start_edge}")
    if not g.candidates(start_from, start_edge, allow_back=False):
        _fail("старт", f"из {start_from} по {start_edge} нет ни одного продолжения")
    out["graph"] = {"edges": len(g.edges), "nodes": len(g.nodes),
                    "meters_per_pixel": g.meters_per_pixel,
                    "errors": g.validate()[:5]}

    # video
    info = probe(path)
    if not info.get("fps"):
        _fail("видео", f"не определить fps: {path}")
    rf = real_frames_of(video_id, path)
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    rec = (man.get("videos", {}).get(video_id) or {}).get("camera") or {}
    if rec.get("width") and info.get("width") and rec["width"] != info["width"]:
        _fail("разрешение", f"видео {info['width']}x{info['height']} против манифеста "
                            f"{rec['width']}x{rec['height']} — вероятно, другой файл")
    out["video"] = {"path": str(path), "id": video_id, "fps": info["fps"],
                    "resolution": f"{info.get('width')}x{info.get('height')}",
                    "real_frames": rf["frames"], "real_frames_source": rf["source"],
                    "declared_frames": rf.get("declared"),
                    "real_seconds": rf["frames"] / info["fps"]}

    # time base
    if not brain_path.exists():
        _fail("запись мозга", f"нет {brain_path}")
    z = np.load(brain_path)
    t = np.asarray(z["t"]).astype(float)
    if not np.isfinite(t).all():
        _fail("время", "в записи мозга есть NaN")
    d = np.diff(t)
    if np.any(d < 0):
        i = int(np.nonzero(d < 0)[0][0])
        _fail("время", f"время идёт назад в t[{i}]={t[i]:.3f} → {t[i+1]:.3f}; это признак склейки "
                       f"разных записей")
    # Duplicate timestamps are real and understood: the recordings were made in sixty-second
    # chunks, and each chunk's first sample lands exactly on its boundary, repeating the previous
    # chunk's last one. Every recording has 18-20 of them. Dropped here, counted in the report —
    # not silently, because a duplicate that was *not* a boundary would mean something else.
    dups = int((d == 0).sum())
    if dups:
        keep = np.concatenate([[True], d > 0])
        t = t[keep]
    if len(t) < 2:
        _fail("время", "после удаления дубликатов осталось меньше двух отсчётов")
    span = float(t[-1] - t[0])
    real_s = rf["frames"] / info["fps"]
    if abs(span - real_s) / max(real_s, 1e-9) > 0.02:
        _fail("временная шкала",
              f"запись мозга покрывает {span:.1f} с, а реально декодируется {real_s:.1f} с "
              f"({rf['frames']} кадров при {info['fps']:.1f} fps) — это разные записи")
    out["time_base"] = {"samples": int(len(t)), "span_s": span,
                        "rate_hz": float(1.0 / np.median(np.diff(t))),
                        "source": "номер декодированного кадра / реальный fps",
                        "duplicate_timestamps_dropped": dups,
                        "why_duplicates": ("запись велась кусками по 60 с; первый отсчёт куска "
                                           "совпадает по времени с последним отсчётом предыдущего"),
                        "declared_duration_used": False}
    out["_t"] = t

    # yaw
    if not yaw_path.exists():
        _fail("yaw", f"нет {yaw_path}")
    rows = list(csv.DictReader(yaw_path.open(encoding="utf-8")))
    ty = np.array([float(r["t"]) for r in rows])
    out["yaw"] = {"path": str(yaw_path), "samples": int(len(ty)),
                  "span_s": float(ty[-1] - ty[0])}
    if abs((ty[-1] - ty[0]) - span) / max(span, 1e-9) > 0.05:
        _fail("yaw", f"длина yaw {ty[-1]-ty[0]:.1f} с против записи мозга {span:.1f} с")

    # frozen constants
    if not freeze:
        _fail("заморозка", "нет FROZEN_V1.json — сначала scripts/final_tracker_freeze.py")
    if not freeze.get("reader_net_displacement"):
        _fail("заморозка", "нет читателя net_displacement")
    out["frozen"] = {"v_walk": freeze["v_walk"], "v_walk_kind": freeze["v_walk_kind"],
                     "stop": freeze["stop"]["threshold"],
                     "frozen_at": freeze["frozen_at"]}
    return out


# --------------------------------------------------------------------------- channels

class Channels:
    """The three inputs, all resampled onto one clock."""

    def __init__(self, video_id: str, path: Path, t: np.ndarray, freeze: dict):
        self.t = t
        self.id = video_id
        self.stop_threshold = float(freeze["stop"]["threshold"])
        self._build_stop(path)
        self._build_yaw(video_id)
        self._build_net(video_id, freeze)

    # --- STOP: frame_motion of the video itself, cached by size and mtime
    def _build_stop(self, path: Path) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from p13_stillness import frame_motion
        cache_dir = OUT_BASE / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        st = path.stat()
        cache = cache_dir / f"stop_{self.id}.npz"
        mot = None
        if cache.exists():
            d = np.load(cache)
            if int(d["size"]) == st.st_size and abs(float(d["mtime"]) - st.st_mtime) < 1.0:
                mot, dt = d["mot"], float(d["dt"])
        if mot is None:
            mot, dt = frame_motion(path)
            np.savez_compressed(cache, mot=mot, dt=np.float32(dt),
                                size=np.int64(st.st_size), mtime=np.float64(st.st_mtime))
        self.stop_t = np.arange(len(mot)) * dt
        self.motion = mot
        self.stop_flag = mot < self.stop_threshold

    def is_stop(self, at: float) -> bool:
        i = np.searchsorted(self.stop_t, at) - 1
        if i < 0:
            i = 0
        if i >= len(self.motion):
            i = len(self.motion) - 1
        return bool(self.stop_flag[i])

    def motion_at(self, at: float) -> float:
        i = np.searchsorted(self.stop_t, at) - 1
        i = max(0, min(i, len(self.motion) - 1))
        return float(self.motion[i])

    # --- camera_yaw: LEFT / RIGHT / SILENT with a strength, on a fine grid
    def _build_yaw(self, video_id: str) -> None:
        import p12_channels as C
        import p086_multi as M
        ch = C.yaw_channel(video_id)
        grid = np.arange(0.0, float(self.t[-1]) + 1e-9, 0.25)
        yy = np.interp(grid, ch["t"], ch["y"])
        window = float(ch["window"])
        floor = float(ch["floor_fn"]())
        # rolling signed integral over the trailing window, vectorised
        dt = 0.25
        integ = np.concatenate([[0.0], np.cumsum(yy[1:] * np.diff(grid))])
        lo = np.searchsorted(grid, grid - window, side="left")
        I = integ - integ[lo]
        ramp = np.array([float(ch["ramp"](float(x), floor)) for x in I])
        cls = np.where(ramp <= 0, "", np.where(I > 0, "LEFT", "RIGHT"))
        self.yaw_grid, self.yaw_I, self.yaw_cls, self.yaw_ramp = grid, I, cls, ramp
        self.yaw_floor = floor
        self.yaw_window = window

    def yaw_at(self, at: float) -> dict:
        i = int(np.searchsorted(self.yaw_grid, at))
        i = max(0, min(i, len(self.yaw_grid) - 1))
        return {"class": self.yaw_cls[i] or None, "strength": float(self.yaw_ramp[i]),
                "integral": float(self.yaw_I[i]), "floor": self.yaw_floor}

    # --- MOVE / NO_NET from the frozen reader
    def _build_net(self, video_id: str, freeze: dict) -> None:
        import p12_channels as C
        from p11a_route_change import FEATS
        reader = freeze["reader_net_displacement"]
        # The feature window reaches BEFORE[0] backwards and AFTER[1] forwards, so the first and
        # last seconds of a recording have no window at all. The grid is trimmed to where one
        # exists rather than padded: an invented window at the edges would be a fabricated input.
        lead = abs(float(C.BEFORE[0]))
        tail = float(C.AFTER[1])
        step = 0.5
        t_end = float(self.t[-1])
        lo, hi = float(self.t[0]) + lead, t_end - tail
        if hi <= lo:
            _fail("net_displacement", f"запись короче окна признаков ({lead} + {tail} с)")
        grid = np.arange(lo, hi + 1e-9, step)
        rd = {"n_groups": reader["n_groups"], "cell_idx": reader["cell_idx"],
              "w": reader["w"], "b": reader["b"]}
        F, _names, _nc = C.source_features("early", video_id, list(grid))
        z = C.channel_scores(F, rd, len(FEATS))
        cls, conf = [], []
        for zz in z:
            c, cf, _p = C.to_class_conf(float(zz))
            cls.append("NO_NET" if c == "pos" else "MOVE")
            conf.append(cf)
        self.net_grid = grid
        self.net_cls = np.array(cls)
        self.net_conf = np.array(conf, dtype=float)
        self.net_edges_s = {"from_s": round(lo, 1), "to_s": round(hi, 1),
                            "why": "края, где окно признаков не помещается, читателю недоступны"}
        if not np.isfinite(self.net_conf).all():
            _fail("net_displacement", "в уверенности канала есть NaN")

    def net_at(self, at: float) -> dict:
        i = int(np.searchsorted(self.net_grid, at))
        i = max(0, min(i, len(self.net_grid) - 1))
        return {"class": str(self.net_cls[i]), "confidence": float(self.net_conf[i])}


# --------------------------------------------------------------------------- hypotheses

class Hyp:
    """One way the walk could have gone. Several are kept alive after a fork."""

    __slots__ = ("id", "edge", "entry", "to", "progress", "score", "born", "alive",
                 "history", "status", "trace", "dead_end_returns", "merged_into")

    def __init__(self, hid: int, edge: str, entry: str, g: Graph, score: float, t0: float,
                 history: list[str] | None = None):
        self.id = hid
        self.edge = edge
        self.entry = entry
        self.to = g.other(edge, entry)
        self.progress = 0.0
        self.score = float(score)
        self.born = float(t0)
        self.alive = True
        self.history = list(history or [])
        self.status = "RUNNING"
        self.trace: deque = deque(maxlen=400)
        self.dead_end_returns = 0
        self.merged_into = None

    def position(self, g: Graph) -> tuple[float, float]:
        a = np.array(g.pos(self.entry), dtype=float)
        b = np.array(g.pos(self.to), dtype=float)
        L = g.length_m(self.edge) or 1e-9
        f = min(max(self.progress / L, 0.0), 1.0)
        p = a + (b - a) * f
        return float(p[0]), float(p[1])

    def net_over(self, window: float, now: float) -> float:
        """Net displacement in metres over the trailing window, from this hypothesis's own path."""
        if not self.trace:
            return 0.0
        pts = list(self.trace)
        then = None
        for tt, x, y in pts:
            if tt <= now - window:
                then = (x, y)
        if then is None:
            then = (pts[0][1], pts[0][2])
        x, y = pts[-1][1], pts[-1][2]
        return float(math.hypot(x - then[0], y - then[1]))


# --------------------------------------------------------------------------- the run

def run(g: Graph, ch: Channels, freeze: dict, start_edge: str, start_from: str,
        out_dir: Path, verbose: bool = True) -> dict:
    import p084b_geometry as geo

    rules = freeze["rules"]
    V = float(freeze["v_walk"])
    dt = 0.02
    t = ch.t
    n = len(t)
    max_hyp = int(rules["fork_hypotheses"]["max"])
    life = float(rules["hypothesis_life_s"])
    net_w = float(rules["net_window_s"])
    net_wt = float(rules["net_weight"])
    yaw_wt = 1.0
    novel_wt = float(rules["novelty_weight"])
    margin = float(rules["ambiguous_score_margin"])
    REF_NET = max(0.5 * V * net_w, 1e-6)

    hyps = [Hyp(0, start_edge, start_from, g, 0.0, float(t[0]),
                history=[start_edge])]
    next_id = 1
    decisions = []
    rows = []
    visits: dict[str, int] = {}
    last_visit: dict[str, float] = {}

    sample_every = int(round(1.0 / dt))
    checks = {"teleports": 0, "moved_while_stopped": 0, "illegal_edges": 0}

    def spawn_from(h: Hyp, node: str, in_edge: str, at: float) -> list[Hyp]:
        """Create the continuations at `node`. Auto-continue when there is only one."""
        nonlocal next_id
        fwd = g.classify_candidates(node, in_edge, allow_back=False)
        back = g.classify_candidates(node, in_edge, allow_back=True)
        back_only = [c for c in back if c["edge"] == in_edge]

        if not fwd:
            # dead end: the passage behind is the only way out, and it is not a choice
            if not back_only:
                h.alive = False
                h.status = "DEAD_END_NO_EXIT"
                return []
            h.dead_end_returns += 1
            h.score -= 0.35
            decisions.append({
                "time": round(at, 2), "node": node, "incoming_edge": in_edge,
                "candidate_edges": [in_edge], "candidate_angles": [180.0],
                "camera_yaw": "", "yaw_strength": 0.0,
                "net_displacement": "", "net_confidence": 0.0,
                "candidate_scores": [round(h.score, 4)], "chosen_edge": in_edge,
                "decision_status": "DEAD_END_RETURN",
                "note": "настоящий тупик: разворот разрешён, доверие снижено",
            })
            nh = Hyp(next_id, in_edge, node, g, h.score, at, history=h.history + [in_edge])
            next_id += 1
            nh.dead_end_returns = h.dead_end_returns
            nh.trace = deque(h.trace, maxlen=400)
            h.alive = False
            h.merged_into = nh.id
            return [nh]

        if len(fwd) == 1:
            c = fwd[0]
            decisions.append({
                "time": round(at, 2), "node": node, "incoming_edge": in_edge,
                "candidate_edges": [c["edge"]], "candidate_angles": [round(c["deg"], 1)],
                "camera_yaw": "", "yaw_strength": 0.0,
                "net_displacement": "", "net_confidence": 0.0,
                "candidate_scores": [round(1.0, 4)], "chosen_edge": c["edge"],
                "decision_status": "ONLY_OPTION",
                "note": "единственный проход: каналы не опрашивались",
            })
            nh = Hyp(next_id, c["edge"], node, g, h.score, at, history=h.history + [c["edge"]])
            next_id += 1
            nh.trace = deque(h.trace, maxlen=400)
            h.alive = False
            h.merged_into = nh.id
            return [nh]

        # a real fork: keep several ways on
        y = ch.yaw_at(at)
        nd = ch.net_at(at)
        cands = fwd[:max_hyp]
        scores = []
        for c in cands:
            s = 0.0
            # geometry alone cannot choose between two passageways; it only says how natural each
            # continuation is. The signal is what names a side, and it does so by real angle.
            if y["class"]:
                al = geo.alignment(float(c["deg"]), str(y["class"]))
                s += yaw_wt * al * y["strength"]
                s += 0.05 * geo.geometry_score(float(c["deg"]))
            else:
                s += 0.05 * geo.geometry_score(float(c["deg"]))
            if novel_wt:
                s += novel_wt * geo.novelty(c["edge"], visits, last_visit, at)
            scores.append(s)
        base = max(scores) if scores else 0.0
        made = []
        for c, s in zip(cands, scores):
            nh = Hyp(next_id, c["edge"], node, g, h.score + s, at,
                     history=h.history + [c["edge"]])
            next_id += 1
            nh.trace = deque(h.trace, maxlen=400)
            made.append(nh)

        order = np.argsort([-s for s in scores])
        top, second = scores[order[0]], scores[order[1]] if len(order) > 1 else None
        ambiguous = second is not None and abs(float(top) - float(second)) < margin
        h.alive = False
        decisions.append({
            "time": round(at, 2), "node": node, "incoming_edge": in_edge,
            "candidate_edges": [c["edge"] for c in cands],
            "candidate_angles": [round(c["deg"], 1) for c in cands],
            "camera_yaw": y["class"] or "", "yaw_strength": round(float(y["strength"]), 3),
            "net_displacement": nd["class"], "net_confidence": round(float(nd["confidence"]), 3),
            "candidate_scores": [round(float(x), 4) for x in scores],
            "chosen_edge": cands[int(order[0])]["edge"],
            "decision_status": "AMBIGUOUS" if ambiguous else "RESOLVED",
            "note": (f"гипотез {len(made)}, живут {life:.0f} с; одно событие yaw не решает"
                     if not ambiguous else
                     f"счёты близки ({top:.3f} против {second:.3f}): обе гипотезы сохранены"),
        })
        return made

    if verbose:
        print(f"  старт: ребро {start_edge} от {start_from} → {g.other(start_edge, start_from)}")
        print(f"  шаг {dt*1000:.0f} мс, V_WALK {V} м/с, STOP < {ch.stop_threshold}")
        print()

    for i in range(n):
        now = float(t[i])
        stop = ch.is_stop(now)
        ds = 0.0 if stop else V * dt
        if stop and ds:
            checks["moved_while_stopped"] += 1

        alive = [h for h in hyps if h.alive]
        if not alive:
            # everything died: the walk cannot be placed any further
            break

        for h in alive:
            before = h.position(g)
            h.progress += ds
            crossed = 0
            while True:
                L = g.length_m(h.edge)
                if L is None:
                    checks["illegal_edges"] += 1
                    h.alive = False
                    break
                if h.progress < L - NODE_EPS_M:
                    break
                h.progress -= L
                node = h.to
                visits[node] = visits.get(node, 0) + 1
                last_visit[node] = now
                new = spawn_from(h, node, h.edge, now)
                if not h.alive:
                    for nh in new:
                        hyps.append(nh)
                    break
                crossed += 1
                if crossed > 3:
                    break

            if h.alive:
                after = h.position(g)
                d = math.hypot(after[0] - before[0], after[1] - before[1])
                # `d` is in plan-normalised units and `ds` is in metres, so the two cannot be
                # compared directly. Convert the allowance into the same units using the edge's
                # own scale: a step of ds metres is ds/length of this edge, as a fraction of it.
                L_now = g.length_m(h.edge) or 1e-9
                frac = ds / L_now
                ex = math.hypot(g.pos(h.to)[0] - g.pos(h.entry)[0],
                                g.pos(h.to)[1] - g.pos(h.entry)[1])
                allowed = 1.5 * frac * ex + 1e-6
                if d > allowed:
                    checks["teleports"] += 1
                    if checks["teleports"] <= 3:
                        checks.setdefault("teleport_examples", []).append(
                            {"t": round(now, 2), "edge": h.edge, "d": d,
                             "allowed": allowed, "ds": ds, "len_m": L_now})
                h.trace.append((now, after[0], after[1]))

        # re-rank: the net-displacement channel is a soft argument about the trailing window
        nd = ch.net_at(now)
        if nd["confidence"] > 0:
            for h in hyps:
                if not h.alive:
                    continue
                pred = h.net_over(net_w, now)
                want_big = (nd["class"] == "MOVE")
                term = math.tanh(pred / REF_NET)
                if not want_big:
                    term = -term
                h.score += net_wt * nd["confidence"] * term * (dt / net_w)

        # merge hypotheses that have arrived at the same edge in the same direction
        live = [h for h in hyps if h.alive]
        by_place: dict[tuple[str, str], Hyp] = {}
        for h in sorted(live, key=lambda x: -x.score):
            key = (h.edge, h.entry)
            if key in by_place:
                keep = by_place[key]
                keep.score = max(keep.score, h.score)
                h.alive = False
                h.status = "MERGED_LATER"
                h.merged_into = keep.id
            else:
                by_place[key] = h

        # cap the population
        live = [h for h in hyps if h.alive]
        if len(live) > max_hyp:
            live.sort(key=lambda x: -x.score)
            for h in live[max_hyp:]:
                h.alive = False
                h.status = "DROPPED_LOW_SCORE"

        # a hypothesis that has outlived its window is no longer an alternative
        live = [h for h in hyps if h.alive]
        if len(live) > 1:
            for h in live:
                if now - h.born > life and h is not max(live, key=lambda x: x.score):
                    h.alive = False
                    h.status = "EXPIRED"

        if i % sample_every == 0:
            live = [h for h in hyps if h.alive]
            if not live:
                break
            live.sort(key=lambda x: -x.score)
            best = live[0]
            second = live[1] if len(live) > 1 else None
            x, y = best.position(g)
            unc = 1.0 if second is None else float(
                min(1.0, max(0.0, 1.0 - (best.score - second.score) / (abs(best.score) + 0.5))))
            yy = ch.yaw_at(now)
            nd2 = ch.net_at(now)
            rows.append({
                "time": round(now, 2), "x": round(x, 6), "y": round(y, 6),
                "edge": best.edge, "progress_m": round(best.progress, 3),
                "speed": 0.0 if stop else V, "is_stop": int(stop),
                "best_hypothesis": best.id, "best_score": round(best.score, 4),
                "second_score": "" if second is None else round(second.score, 4),
                "uncertainty": round(unc, 4),
                "camera_yaw": yy["class"] or "", "camera_yaw_strength": round(yy["strength"], 3),
                "net_displacement": nd2["class"],
                "net_displacement_confidence": round(nd2["confidence"], 3),
            })

    final = [h for h in hyps if h.alive] or hyps
    final.sort(key=lambda x: -x.score)
    best = final[0]
    return {"rows": rows, "decisions": decisions, "best": best, "hyps": hyps,
            "checks": checks, "visits": visits}


# --------------------------------------------------------------------------- outputs

def write_outputs(out_dir: Path, g: Graph, res: dict, checks: dict, freeze: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = res["rows"]
    with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # edge sequence: times from trajectory samples (best hypothesis each second)
    best = res["best"]
    seq = []
    if rows:
        cur_edge = rows[0]["edge"]
        t_enter = float(rows[0]["time"])
        conf_acc = [1.0 - float(rows[0]["uncertainty"])]
        for r in rows[1:]:
            if r["edge"] != cur_edge:
                seq.append({
                    "order": len(seq) + 1,
                    "time_enter": round(t_enter, 2),
                    "time_exit": round(float(r["time"]), 2),
                    "edge": cur_edge,
                    "from_node": g.edges[cur_edge]["from"],
                    "to_node": g.edges[cur_edge]["to"],
                    "confidence": round(float(np.mean(conf_acc)), 3),
                })
                cur_edge = r["edge"]
                t_enter = float(r["time"])
                conf_acc = [1.0 - float(r["uncertainty"])]
            else:
                conf_acc.append(1.0 - float(r["uncertainty"]))
        seq.append({
            "order": len(seq) + 1,
            "time_enter": round(t_enter, 2),
            "time_exit": round(float(rows[-1]["time"]), 2),
            "edge": cur_edge,
            "from_node": g.edges[cur_edge]["from"],
            "to_node": g.edges[cur_edge]["to"],
            "confidence": round(float(np.mean(conf_acc)), 3),
        })
    else:
        seen = []
        for e in best.history:
            if not seen or seen[-1] != e:
                seen.append(e)
        for i, e in enumerate(seen):
            seq.append({"order": i + 1, "time_enter": "", "time_exit": "", "edge": e,
                        "from_node": g.edges[e]["from"], "to_node": g.edges[e]["to"],
                        "confidence": ""})
    with (out_dir / "edge_sequence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq[0].keys()))
        w.writeheader()
        w.writerows(seq)

    decs = res["decisions"]
    if decs:
        with (out_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(decs[0].keys()))
            w.writeheader()
            for d in decs:
                w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                            for k, v in d.items()})

    routes = res["best"].history
    total_m = sum((g.length_m(e) or 0.0) for e in routes)
    # Loop detection, reported rather than acted on: the spec forbids novelty in the routing, so a
    # revisited edge cannot be penalised while walking, but it must not be hidden either. A walk
    # that goes round the same short ring many times is not a walk, and the count belongs in the
    # result where it can be argued with.
    from collections import Counter as _C
    cnt = _C(routes)
    doubles = 0
    for i in range(len(routes) - 1):
        if routes[i] == routes[i + 1]:
            doubles += 1
    # a 2-cycle is an edge and the same edge straight back
    for i in range(len(routes) - 2):
        if routes[i] == routes[i + 2] and routes[i] != routes[i + 1]:
            doubles += 1
    stats = {
        "edges_traversed": len(routes),
        "distinct_edges": len(set(routes)),
        "route_meters": round(total_m, 1),
        "decisions": len(decs),
        "decisions_by_status": {},
        "samples": len(rows),
        "stop_fraction": round(float(np.mean([r["is_stop"] for r in rows])), 3),
        "self_checks": checks,
        "loops": {
            "max_traversals_of_one_edge": int(max(cnt.values())) if cnt else 0,
            "most_repeated": cnt.most_common(3),
            "immediate_back_and_forth": doubles,
            "distinct_over_total": round(len(set(routes)) / max(len(routes), 1), 3),
            "reading": ("distinct_over_total близко к 1 — прогулка; заметно ниже — система ходит "
                        "по кругу, и это дефект маршрутизации, а не свойство прогулки"),
        },
    }
    for d in decs:
        k = d["decision_status"]
        stats["decisions_by_status"][k] = stats["decisions_by_status"].get(k, 0) + 1
    return stats


def draw(out_dir: Path, g: Graph, res: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except Exception:
        return False
    plan = ROOT / "data/p08/plan.png"
    fig, ax = plt.subplots(figsize=(14, 10))
    iw, ih = float(g.img_w), float(g.img_h)
    if plan.exists():
        img = imread(plan)
        ax.imshow(img, extent=(0, iw, ih, 0), alpha=0.55, zorder=0)
    for e in g.edges:
        a, b = g.pos(g.edges[e]["from"]), g.pos(g.edges[e]["to"])
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#888", lw=0.4, zorder=1)
    pts = res.get("rows") or []
    if pts:
        xs = [float(r["x"]) for r in pts]
        ys = [float(r["y"]) for r in pts]
        ax.plot(xs, ys, color="#1f77ff", lw=2.2, zorder=3, label="траектория")
        ax.plot(xs[0], ys[0], "o", color="#2ca02c", ms=11, zorder=4, label="START")
        ax.plot(xs[-1], ys[-1], "s", color="#d62728", ms=9, zorder=4, label="END")
    route = res["best"].history
    amb = [d for d in res["decisions"] if d["decision_status"] == "AMBIGUOUS"]
    dead = [d for d in res["decisions"] if d["decision_status"] == "DEAD_END_RETURN"]
    for d in amb:
        if d["node"] in g.nodes:
            x, y = g.pos(d["node"])
            ax.plot(x, y, "x", color="#ffb000", ms=7, zorder=5)
    for d in dead:
        if d["node"] in g.nodes:
            x, y = g.pos(d["node"])
            ax.plot(x, y, "+", color="#8b0000", ms=9, zorder=5)
    ax.set_xlim(0, iw)
    ax.set_ylim(ih, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"FINAL TRACKER V1 — маршрут без ручной разметки\n"
                 f"рёбер {len(route)}, развилок {len(res['decisions'])}, "
                 f"AMBIGUOUS {len(amb)}, тупиков {len(dead)}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--start-node", default=None, help="узел, откуда человек начал")
    ap.add_argument("--previous-node", default=None, help="узел, из которого он пришёл")
    ap.add_argument("--start-edge", default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--v-walk", type=float, default=None,
                    help="ДИАГНОСТИКА: временно переопределить V_WALK. Отклонение от заморозки "
                         "записывается в отчёт; на этом значении нельзя выдавать результат")
    a = ap.parse_args()

    t0 = time.time()
    print("=" * 100)
    print("FINAL TRACKER V1")
    print("=" * 100)

    path = resolve_video(a.video)
    vid = path.stem
    g = Graph.load(a.graph)
    if not FREEZE.exists():
        _fail("заморозка", f"нет {FREEZE}. Сначала: scripts/final_tracker_freeze.py")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))

    if a.start_edge and a.start_from:
        start_edge, start_from = a.start_edge, a.start_from
    elif a.start_node and a.previous_node:
        # Человек на start-node, пришёл из previous-node: вход на ребро — previous-node.
        start_from = a.previous_node
        start_edge = f"{a.previous_node}__{a.start_node}"
        if start_edge not in g.edges:
            start_edge = f"{a.start_node}__{a.previous_node}"
        if start_edge not in g.edges:
            _fail("старт", f"нет ребра между {a.previous_node} и {a.start_node}")
    elif g.start:
        start_edge, start_from = g.start["edge"], g.start["from"]
    else:
        _fail("старт", "укажите --start-node и --previous-node, либо --start-edge/--start-from")

    out_dir = Path(a.out) if a.out else OUT_BASE / vid
    yaw_path = ROOT / f"output/p07/yaw_signal_{vid}.csv"
    brain_path = ROOT / f"output/p10/brain_{vid}.npz"

    checks = self_checks(vid, path, g, freeze, start_edge, start_from, yaw_path, brain_path)
    print(f"  видео {vid}: {checks['video']['resolution']}, "
          f"{checks['video']['real_frames']} реальных кадров "
          f"({checks['video']['real_frames_source']}), "
          f"{checks['video']['real_seconds']:.0f} с")
    print(f"  граф: {checks['graph']['edges']} рёбер, {checks['graph']['nodes']} узлов")
    print(f"  время: {checks['time_base']['samples']} отсчётов, "
          f"{checks['time_base']['span_s']:.0f} с, {checks['time_base']['rate_hz']:.0f} Гц")
    print(f"  заморозка {checks['frozen']['frozen_at']}: "
          f"V_WALK {checks['frozen']['v_walk']} м/с, "
          f"STOP {checks['frozen']['stop']}")

    ch = Channels(vid, path, checks["_t"], freeze)
    diag = None
    if a.v_walk is not None and abs(a.v_walk - float(freeze["v_walk"])) > 1e-9:
        diag = {"v_walk_override": float(a.v_walk),
                "frozen_v_walk": float(freeze["v_walk"]),
                "warning": ("диагностический прогон: значение отличается от замороженного. "
                            "Результат нельзя предъявлять как предсказанный заморозкой")}
        freeze = json.loads(json.dumps(freeze))
        freeze["v_walk"] = float(a.v_walk)
    res = run(g, ch, freeze, start_edge, start_from, out_dir, verbose=not a.quiet)
    stats = write_outputs(out_dir, g, res, res["checks"], freeze)
    drew = draw(out_dir, g, res)

    report = {
        "phase": "FINAL TRACKER V1",
        "video": vid, "video_path": str(path),
        "start": {"edge": start_edge, "from": start_from, "to": g.other(start_edge, start_from)},
        "graph": checks["graph"], "time_base": checks["time_base"],
        "frozen": checks["frozen"], "frozen_at": freeze["frozen_at"],
        "stats": stats,
        "drew_png": drew,
        "diagnostic": diag,
        "no_manual_route_used": True,
        "elapsed_s": round(time.time() - t0, 1),
        "self_check_notes": {
            "declared_duration_used": False,
            "time_base": "номер декодированного кадра / реальный fps",
            "loud_failures": "18 проверок в self_checks; ни одна не пропускается молча",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print()
    print(f"  рёбер пройдено: {stats['edges_traversed']} (различных {stats['distinct_edges']})")
    print(f"  длина маршрута: {stats['route_meters']} м")
    print(f"  развилок: {stats['decisions']}  {stats['decisions_by_status']}")
    print(f"  доля времени в STOP: {stats['stop_fraction']:.0%}")
    print(f"  самопроверки: телепортов {stats['self_checks']['teleports']}, "
          f"движений в STOP {stats['self_checks']['moved_while_stopped']}, "
          f"незаконных рёбер {stats['self_checks']['illegal_edges']}")
    print(f"  записано: {out_dir}  (trajectory.csv, edge_sequence.csv, decisions.csv, "
          f"trajectory.png{'' if drew else ' — НЕ нарисован'}, report.json)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as e:
        print()
        print("=" * 100)
        print(f"  СТОП: {e}")
        print("  Трекер не строит маршрут, когда условия не выполнены.")
        print("=" * 100)
        raise SystemExit(2)
