#!/usr/bin/env python3
"""
Local annotation server for the trajectory tool.

Serves the browser app, the transcoded video, and a small save endpoint so the
annotations land on disk instead of having to be mailed around. The save file is
plain JSON and is written next to the event labels.

Endpoints
    GET  /                     the app
    GET  /media/<file>         video, with HTTP Range support so seeking works
    GET  /api/state            current annotations + the existing event labels
    POST /api/save             write annotations to data/p01r/trajectory_annotations.json
    POST /api/export           write a plain table of turns to data/p01r/turns_export.csv

Usage:
    PYTHONPATH=. .venv/bin/python webapp/server.py            # http://127.0.0.1:8765
    PYTHONPATH=. .venv/bin/python webapp/server.py --port 9000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_backend  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MEDIA = ROOT / "media"
DATA = PROJECT / "data" / "p01r"
VIDEO = DATA / os.environ.get("REVIEW_VIDEO", "VID00001.AVI")
# The review bank is switchable so that additional rounds and additional clips can be
# labelled without overwriting earlier verdicts. Two knobs:
#
#   REVIEW_TAG=vid2        suffix for every file, e.g. review_set_vid2.json
#   REVIEW_ROUND=2         shorthand for REVIEW_TAG=_v2, kept for the first clip
#
# With neither set, the first clip's round 1 is used and only its own files are touched.
_ROUND = os.environ.get("REVIEW_ROUND")
_TAG = os.environ.get("REVIEW_TAG")
if not _TAG and _ROUND not in (None, "", "1"):
    _TAG = f"_v{_ROUND}"
_TAG = _TAG or ""
if _TAG and not _TAG.startswith("_"):
    _TAG = "_" + _TAG


def _bank(name: str) -> Path:
    return DATA / f"{name}{_TAG}.json"


VERDICTS_JSON = _bank("turn_verdicts")
CANDIDATES_JSON = DATA / "candidate_turns.json"
REVIEW_SET_JSON = _bank("review_set")
EXPORT_CSV = DATA / f"turns_export{_TAG}.csv"
# The drawn route must follow the tag as well. It used to be a single fixed file, so
# drawing a route for a new clip would have overwritten the first clip's drawing.
SAVE_JSON = _bank("trajectory_annotations")
LABELS_V2 = DATA / "VID00001_labels_v2.json"
EVENTS_JSON = DATA / "VID00001_events.json"

# P08: the graph a human draws over a plan of the room. Kept apart from the review
# banks, because it belongs to a clip and a room rather than to a labelling round.
P08_DATA = PROJECT / "data" / "p08"
P08_OUT = PROJECT / "output" / "p08"
P083 = PROJECT / "output" / "p083_route_audit"
FINAL_OUT = PROJECT / "output" / "final_tracker"
FINAL_V2 = PROJECT / "output" / "final_tracker_v2"
FINAL_V3 = PROJECT / "output" / "final_tracker_v3"
FINAL_V4 = PROJECT / "output" / "final_tracker_v4"
FINAL_V5 = PROJECT / "output" / "final_tracker_v5"
RUN_BASES = {"v5": FINAL_V5, "v4": FINAL_V4, "v3": FINAL_V3, "v2": FINAL_V2, "v1": FINAL_OUT}


def _versions(clip: str) -> list[str]:
    return [v for v, base in RUN_BASES.items() if (base / clip / "trajectory.csv").exists()]


def _run_dir(clip: str, ver: str | None = None) -> tuple[str, Path]:
    have = _versions(clip)
    if ver in have:
        return ver, RUN_BASES[ver] / clip
    if have:
        return have[0], RUN_BASES[have[0]] / clip
    return "v1", FINAL_OUT / clip


def _run_names() -> set[str]:
    names = set()
    for base in RUN_BASES.values():
        if base.exists():
            for d in base.iterdir():
                if d.is_dir() and d.name != "cache":
                    names.add(d.name)
    return names
P01R = PROJECT / "data" / "p01r"
P08_GRAPH = P08_DATA / "graph.json"
P08_TRUTH = P08_DATA / "truth_route.json"
P08_PLAN_STEM = P08_DATA / "plan"
START_PROFILES: dict[str, dict] = {
    "VID00001": {
        "media": MEDIA / "VID00001_legacy_fixed.mp4",
        "title": "Старт VID00001",
        "note": "Старая запись из Downloads, около 20 минут. Не трёхсекундный файл с карты.",
    },
    "VID00020": {
        "media": MEDIA / "VID00020_5min_fixed.mp4",
        "title": "Старт 5 минут (VID00020)",
        "note": "5 мин без первых 50 с (одевание камеры вырезано). Укажите точку и направление, затем «Старт».",
    },
}
DEFAULT_START_CLIP = "VID00001"
_START_LOCK = threading.Lock()
_START_JOB = {"state": "idle", "log": "", "error": "", "place": None, "code": None, "clip": None}


def _start_clip_from_path(path: str, query: str) -> str:
    if path in ("/start5", "/start5.html"):
        return "VID00020"
    from urllib.parse import parse_qs
    clip = (parse_qs(query).get("clip") or [DEFAULT_START_CLIP])[0]
    if not re.fullmatch(r"VID[0-9]{5}", clip):
        return DEFAULT_START_CLIP
    return clip


def _start_profile(clip: str) -> dict | None:
    if clip in START_PROFILES:
        return START_PROFILES[clip]
    media = MEDIA / f"{clip}_fixed.mp4"
    if not media.exists() and not app_backend.clip_record(clip):
        return None
    return {
        "media": media,
        "title": f"Старт {app_backend.clip_title(clip)}",
        "note": "Укажите точку и направление, затем «Старт».",
    }


def _fly_motion_samples(clip: str) -> list[dict]:
    """Same LEFT/RIGHT/SILENT reading as final_tracker (p12 yaw_at on 0.25 s grid)."""
    yaw_path = PROJECT / "output" / "p07" / f"yaw_signal_{clip}.csv"
    if not yaw_path.exists():
        return []
    sys.path.insert(0, str(PROJECT / "scripts"))
    import numpy as np
    import p12_channels as C

    ch = C.yaw_channel(clip)
    t_end = float(ch["t"][-1])
    out: list[dict] = []
    t = 0.0
    while t <= t_end + 1e-9:
        y = C.yaw_at(ch, t)
        cls = y["class"]
        out.append({
            "t": round(t, 2),
            "dir": "STRAIGHT" if not cls else cls,
            "yaw": round(float(y["integral"]), 5),
            "strength": round(float(y["confidence"]), 4),
        })
        t += 0.25
    return out


def _stand_path(clip: str) -> Path:
    return PROJECT / "data" / "final_tracker" / f"{clip}_stand_labels.json"


def _tracker_inputs_ready(clip: str) -> bool:
    brain = PROJECT / "output" / "p10" / f"brain_{clip}.npz"
    yaw = PROJECT / "output" / "p07" / f"yaw_signal_{clip}.csv"
    return brain.exists() and yaw.exists()


def _start_video_ready(profile: dict) -> bool:
    media = profile["media"]
    return media.exists() and media.stat().st_size > 1_000_000

CHUNK = 1 << 20
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mp4": "video/mp4",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webm": "video/webm",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def _load_graph():
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "scripts"))
    from p08_graph import Graph
    return Graph.load(P08_GRAPH)


def snap_on_graph(x: float, y: float, toward: str | None = None) -> dict:
    """Stick a plan-pixel click to the nearest edge and, if asked, face one end."""
    g = _load_graph()
    best = None
    for eid, e in g.edges.items():
        ax, ay = g.pos(e["from"])
        bx, by = g.pos(e["to"])
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        if span < 1e-6:
            continue
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / span))
        px, py = ax + dx * t, ay + dy * t
        node = None
        if math.hypot(px - ax, py - ay) <= 10:
            t, px, py, node = 0.0, ax, ay, e["from"]
        elif math.hypot(px - bx, py - by) <= 10:
            t, px, py, node = 1.0, bx, by, e["to"]
        dist = math.hypot(x - px, y - py)
        if best is None or dist < best["dist"]:
            best = {"edge": eid, "from_node": e["from"], "to_node": e["to"],
                    "t": t, "x": px, "y": py, "node": node, "dist": dist,
                    "ax": ax, "ay": ay, "bx": bx, "by": by}
    if best is None or best["dist"] > 80:
        return {"ok": False, "error": "кликните ближе к линии графа"}
    ends = [
        {"id": best["from_node"], "x": best["ax"], "y": best["ay"]},
        {"id": best["to_node"], "x": best["bx"], "y": best["by"]},
    ]
    out = {"ok": True, "edge": best["edge"], "x": round(best["x"], 2), "y": round(best["y"], 2),
           "dist_px": round(best["dist"], 1), "ends": ends, "on_node": best["node"]}
    if not toward:
        return out
    if toward not in (best["from_node"], best["to_node"]):
        return {"ok": False, "error": f"узел {toward} не на ребре {best['edge']}"}
    start_from = best["to_node"] if toward == best["from_node"] else best["from_node"]
    length = g.length_m(best["edge"])
    progress = (best["t"] if start_from == best["from_node"] else 1.0 - best["t"]) * float(length)
    out.update({
        "toward": toward,
        "start_from": start_from,
        "progress_m": round(max(0.0, progress), 3),
        "length_m": round(float(length), 3),
    })
    return out


TRACKER_SCRIPTS = {"v1": "final_tracker.py", "v2": "final_tracker_v2.py",
                   "v3": "final_tracker_v3.py", "v4": "final_tracker_v4.py",
                   "v5": "final_tracker_v5.py"}


def _start_worker(place: dict, clip: str, ver: str = "v5") -> None:
    vers = list(TRACKER_SCRIPTS) if ver == "all" else [ver]
    lines: list[str] = []
    code = 0
    for v in vers:
        lines.append(f"=== {v.upper()} ===\n")
        code = _run_tracker(place, clip, v, lines)
        if code != 0:
            lines.append(f"{v.upper()} остановился (код {code})\n")
            break
    with _START_LOCK:
        _START_JOB["code"] = code
        _START_JOB["log"] = "".join(lines[-60:])
        if code == 0:
            _START_JOB["state"] = "done"
            _START_JOB["error"] = ""
        else:
            _START_JOB["state"] = "error"
            text = "".join(lines).strip().splitlines()
            _START_JOB["error"] = text[-1] if text else "трекер остановился"


def _run_tracker(place: dict, clip: str, ver: str, lines: list[str]) -> int:
    cmd = [
        app_backend.python_exe(),
        str(PROJECT / "scripts" / TRACKER_SCRIPTS[ver]),
        "--video", clip,
        "--start-edge", place["edge"],
        "--start-from", place["start_from"],
    ]
    # V1 (и V4/V5 поверх него) без --start-progress: стартуют от узла start_from.
    if ver not in ("v1", "v4", "v5"):
        cmd += ["--start-progress", f"{place['progress_m']:.4f}"]
    try:
        proc = subprocess.Popen(cmd, **app_backend.popen_kwargs())
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            with _START_LOCK:
                _START_JOB["log"] = "".join(lines[-60:])
        return proc.wait()
    except Exception as exc:
        lines.append(str(exc) + "\n")
        return 1


def _track_hook(clip: str, start: dict) -> tuple[int, str]:
    """Run V1..V5 from a plan start for the clip pipeline; waits if a manual run is going."""
    place = snap_on_graph(float(start["x"]), float(start["y"]), str(start["toward"]))
    if not place.get("ok"):
        return 1, place.get("error") or "старт не на графе"
    while True:
        with _START_LOCK:
            if _START_JOB["state"] != "running":
                _START_JOB.update(state="running", log="запуск…\n", error="",
                                  place=place, code=None, clip=clip, ver="all")
                break
        time.sleep(1.0)
    _start_worker(place, clip, "all")
    with _START_LOCK:
        return int(_START_JOB.get("code") or 0), str(_START_JOB.get("error") or "")


app_backend.track_hook = _track_hook


def _runs_summary(clip: str) -> dict:
    out = {}
    for v in _versions(clip):
        rep = read_json(RUN_BASES[v] / clip / "report.json", {})
        stats = rep.get("stats") or {}
        out[v] = {"meters": stats.get("route_meters"), "stop_fraction": stats.get("stop_fraction"),
                  "start": rep.get("start")}
    return out


def _diagnostics() -> dict:
    import platform
    import shutil as sh

    checks = []

    def add(cid, title, ok, message):
        checks.append({"id": cid, "title": title, "ok": bool(ok), "message": message})

    add("backend", "Локальный сервер", True, f"Python {platform.python_version()}, {platform.system()} {platform.release()}")
    for tool in ("ffmpeg", "ffprobe"):
        where = sh.which(tool)
        add(tool, tool, where, where or "не найден — переустановите программу")
    brain_dir = Path(os.environ.get("FLY_DATA") or PROJECT / "data" / "malecns")
    for name, min_mb in (("brain.npz", 40), ("weights.npz", 150)):
        f = brain_dir / name
        mb = f.stat().st_size / 1e6 if f.exists() else 0
        add(name, f"Мозг мухи: {name}", mb >= min_mb, f"{mb:.0f} МБ" if mb else f"нет файла в {brain_dir}")
    try:
        import numba  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import flybrain  # noqa: F401
        add("packages", "Пакеты расчёта", True, f"numpy {numpy.__version__}, numba {numba.__version__}")
    except Exception as e:
        add("packages", "Пакеты расчёта", False, str(e))
    add("graph", "План и граф цеха", P08_GRAPH.exists(), str(P08_GRAPH.name) if P08_GRAPH.exists() else "нет графа")
    free = sh.disk_usage(PROJECT).free / 1e9
    add("disk", "Свободное место", free > 5, f"{free:.1f} ГБ (нужно ~2 ГБ на час видео)")
    cams = app_backend.scan_camera().get("cameras", [])
    add("camera", "Камера", bool(cams),
        ", ".join(f"{c['label']}: {len(c['files'])} AVI" for c in cams) if cams else "не подключена (это нормально, если загружаете файл)")
    return {"ok": all(c["ok"] for c in checks if c["id"] != "camera"), "checks": checks,
            "workspace": str(PROJECT)}


UI_DIST = PROJECT / "desktop" / "ui" / "dist"


def final_video_url(clip: str) -> str:
    """The file the tracker page should play.

    VID00001_fixed.mp4 is the 3-second camera stub. The run on this page is the
    older 20-minute recording, which lives in VID00001_legacy_fixed.mp4.
    """
    prof = _start_profile(clip)
    if prof and _start_video_ready(prof):
        return f"/media/{prof['media'].name}"
    for name in (f"{clip}_fixed.mp4", f"{clip}.mp4"):
        if (MEDIA / name).exists():
            return f"/media/{name}"
    if (P01R / f"{clip}.AVI").exists():
        return f"/api/final/video?clip={clip}"
    return ""


def serve_video_name() -> str:
    """The clip the browser plays.

    `*_fixed.mp4` is the one whose timeline matches the content: it is built with
    `ffmpeg -r 30 -i <avi>`, which reinterprets the source as contiguous 30 fps. The
    plain transcode keeps the source's own stretched timeline, so every timestamp in
    the browser lands about 18% late and drifts to over 200 s by the end of the clip.
    Both source clips declare 45000 frames and contain about 36840, so the defect is
    present in both and this applies to each. Fall back only if the fixed file is
    missing.
    """
    stem = VIDEO.stem
    fixed = MEDIA / f"{stem}_fixed.mp4"
    return f"{stem}_fixed.mp4" if fixed.exists() else f"{stem}.mp4"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class Handler(BaseHTTPRequestHandler):
    server_version = "flyvo-annotate/1.0"

    # ---------------------------------------------------------------- helpers
    def log_message(self, fmt: str, *args) -> None:
        if "/media/" in self.path:
            return
        print(f"  {self.address_string()} {fmt % args}")

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------- body
    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/ui" or path.startswith("/ui/"):
            rel = path[len("/ui"):].lstrip("/") or "index.html"
            target = (UI_DIST / rel).resolve()
            if not str(target).startswith(str(UI_DIST.resolve())) or not target.is_file():
                target = UI_DIST / "index.html"
            return self._serve(target)
        if path in ("/app", "/app.html") or (path == "/" and os.environ.get("FLY_APP") == "1"):
            return self._serve(ROOT / "app.html")
        if path == "/api/app/clips":
            app_backend.ensure_worker()
            clips = app_backend.clips_overview(_tracker_inputs_ready)
            for c in clips:
                c["runs"] = _runs_summary(c["id"])
            return self._json({"ok": True, "clips": clips,
                               "import": app_backend.import_status(),
                               "tracker": {k: _START_JOB.get(k) for k in ("state", "clip", "error")},
                               "app_mode": os.environ.get("FLY_APP") == "1"})
        if path == "/api/app/graph":
            return self._json({"ok": True, "graph": read_json(P08_GRAPH, {}), "plan_url": "/api/p08/plan"})
        if path == "/api/app/diagnostics":
            return self._json(_diagnostics())
        if path == "/api/camera/scan":
            return self._json(app_backend.scan_camera())
        if path == "/api/camera/import":
            return self._json(app_backend.import_status())

        if path in ("/", "/index.html"):
            return self._serve(ROOT / "index.html")
        if path in ("/review", "/review.html"):
            return self._serve(ROOT / "review.html")

        if path == "/api/state":
            return self._json({
                "saved": read_json(SAVE_JSON, {"trajectory": [], "turns": [], "notes": ""}),
                "existing_labels": read_json(LABELS_V2, {}),
                "events_file": read_json(EVENTS_JSON, {}),
                "save_path": str(SAVE_JSON),
                "export_path": str(EXPORT_CSV),
            })

        if path == "/api/candidates":
            # Prefer the balanced set when it exists: the class mix is what makes the
            # review informative, because agreeing with every item must not be possible.
            doc = read_json(REVIEW_SET_JSON, {})
            if not doc:
                doc = read_json(CANDIDATES_JSON, {})
            verdicts = read_json(VERDICTS_JSON, {"verdicts": {}})
            return self._json({
                "candidates": doc.get("candidates", []),
                "meta": {k: v for k, v in doc.items() if k != "candidates"},
                "verdicts": verdicts.get("verdicts", {}),
                "verdict_path": str(VERDICTS_JSON),
                "review_set_path": str(REVIEW_SET_JSON),
                # the page must play the clip these windows belong to; a hardcoded
                # name would silently pair one clip's timestamps with another's video
                "video": serve_video_name(),
                "source_video": VIDEO.name,
            })

        if path in ("/graph", "/graph.html"):
            return self._serve(ROOT / "p08_graph_editor.html")

        if path in ("/p083", "/p083.html"):
            return self._serve(ROOT / "p083.html")

        if path in ("/final", "/final.html"):
            return self._serve(ROOT / "final.html")

        if path in ("/start", "/start.html", "/start5", "/start5.html"):
            return self._serve(ROOT / "start.html")

        if path == "/api/start/state":
            from urllib.parse import parse_qs
            q = self.path.split("?", 1)[-1] if "?" in self.path else ""
            clip = _start_clip_from_path(path, q)
            prof = _start_profile(clip)
            if not prof:
                return self._json({"ok": False, "error": f"неизвестный clip {clip}"}, 400)
            vready = _start_video_ready(prof)
            tready = _tracker_inputs_ready(clip)
            note = prof["note"]
            if not tready:
                note += " Сейчас готовятся brain и yaw — «Старт» включится, когда цепочка дойдёт."
            return self._json({
                "ok": True,
                "clip": clip,
                "title": prof["title"],
                "video_url": f"/media/{prof['media'].name}" if vready else "",
                "tracker_ready": tready,
                "note": note,
                "plan_url": "/api/p08/plan",
            })

        if path == "/api/start/status":
            with _START_LOCK:
                job = dict(_START_JOB)
            return self._json({"ok": True, **job})

        if path in ("/human", "/human.html", "/human_route", "/human_route.html"):
            return self._serve(ROOT / "human_route.html")

        if path.startswith("/api/human/state"):
            from urllib.parse import parse_qs
            clip = (parse_qs(self.path.split("?", 1)[-1]).get("clip") or ["VID00010"])[0]
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._text("bad clip id", 400)
            graph = read_json(P08_GRAPH, {})
            sketch_path = PROJECT / "data" / "final_tracker" / f"{clip}_human_sketch.json"
            sketch = read_json(sketch_path, {})
            tracker = []
            traj_path = FINAL_OUT / clip / "trajectory.csv"
            if traj_path.exists():
                rows = list(csv.DictReader(traj_path.open(encoding="utf-8")))
                stride = max(1, len(rows) // 800)
                tracker = [[float(r["x"]), float(r["y"])] for r in rows[::stride]]
            return self._json({
                "ok": True, "clip": clip, "graph": graph, "tracker": tracker,
                "sketch": sketch if sketch.get("points") else None,
                "sketch_path": str(sketch_path),
                "video_url": final_video_url(clip),
            })

        if path.startswith("/api/stand/state"):
            from urllib.parse import parse_qs
            clip = (parse_qs(self.path.split("?", 1)[-1]).get("clip") or [""])[0]
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad clip id"}, 400)
            doc = read_json(_stand_path(clip), {})
            return self._json({"ok": True, "clip": clip, "stand": doc.get("stand", []),
                               "reviewed_until": doc.get("reviewed_until", 0),
                               "saved_at": doc.get("saved_at", "")})

        if path.startswith("/api/fly/motion"):
            from urllib.parse import parse_qs
            clip = (parse_qs(self.path.split("?", 1)[-1]).get("clip") or [""])[0]
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad clip id"}, 400)
            samples = _fly_motion_samples(clip)
            return self._json({
                "ok": True, "clip": clip, "samples": samples,
                "method": "p12_yaw_at",
                "note": "Тот же канал, что у трекера: интеграл yaw за окно + порог (не сырой P07). "
                        "STRAIGHT = ниже порога. yaw>0 → камера влево.",
            })

        if path.startswith("/api/final/tracks"):
            from urllib.parse import parse_qs
            q = parse_qs(self.path.split("?", 1)[-1]) if "?" in self.path else {}
            want = (q.get("ver") or [None])[0]
            tracks = []
            for name in sorted(_run_names()):
                ver, run_dir = _run_dir(name, want)
                traj_path = run_dir / "trajectory.csv"
                if not traj_path.exists():
                    continue
                pts = []
                with traj_path.open(encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        pts.append([round(float(r["x"]), 1), round(float(r["y"]), 1)])
                tracks.append({"id": name, "version": ver, "points": pts})
            return self._json({"tracks": tracks})

        if path == "/api/final/list":
            clips = []
            for name in sorted(_run_names()):
                ver, run_dir = _run_dir(name)
                rep = run_dir / "report.json"
                clips.append({"id": name, "ok": rep.exists(), "version": ver,
                              "versions": _versions(name),
                              "report_path": str(rep) if rep.exists() else ""})
            return self._json({"clips": clips})

        if path.startswith("/api/final/state"):
            q = self.path.split("?", 1)
            clip, want = "VID00010", None
            if len(q) > 1:
                from urllib.parse import parse_qs
                qs = parse_qs(q[1])
                clip = (qs.get("clip") or [clip])[0]
                want = (qs.get("ver") or [None])[0]
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._text("bad clip id", 400)
            ver, run_dir = _run_dir(clip, want)
            traj_path = run_dir / "trajectory.csv"
            dec_path = run_dir / "decisions.csv"
            rep_path = run_dir / "report.json"
            if not traj_path.exists():
                return self._json({"ok": True, "clip": clip, "version": None, "versions": [],
                                   "trajectory": [], "decisions": [], "alternatives": [],
                                   "report": None, "duration": 0.0, "tracker_duration": 0.0,
                                   "start_xy": None, "video_url": final_video_url(clip),
                                   "graph": read_json(P08_GRAPH, {}),
                                   "note": "прогона трекера ещё нет"})
            traj = list(csv.DictReader(traj_path.open(encoding="utf-8")))
            decs = list(csv.DictReader(dec_path.open())) if dec_path.exists() else []
            alt_path = run_dir / "alternatives.csv"
            alts = list(csv.DictReader(alt_path.open(encoding="utf-8"))) if alt_path.exists() else []
            report = read_json(rep_path, {})
            duration = float(traj[-1]["time"]) if traj else 0.0
            start_xy = None
            if traj:
                start_xy = {"x": float(traj[0]["x"]), "y": float(traj[0]["y"]),
                            "edge": traj[0].get("edge"), "t": float(traj[0]["time"])}
            return self._json({
                "ok": True, "clip": clip, "version": ver, "versions": _versions(clip),
                "trajectory": traj, "decisions": decs, "alternatives": alts,
                "report": report, "duration": duration, "tracker_duration": duration,
                "start_xy": start_xy,
                "video_url": final_video_url(clip),
                "graph": read_json(P08_GRAPH, {}),
                "trajectory_png": f"/api/final/png?clip={clip}&ver={ver}",
            })

        if path.startswith("/api/final/png"):
            from urllib.parse import parse_qs
            qs = parse_qs(self.path.split("?", 1)[-1])
            clip = (qs.get("clip") or ["VID00010"])[0]
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._text("bad clip id", 400)
            _ver, run_dir = _run_dir(clip, (qs.get("ver") or [None])[0])
            png = run_dir / "trajectory.png"
            if not png.exists():
                return self._text("no png", 404)
            return self._serve(png)

        if path.startswith("/api/final/video"):
            from urllib.parse import parse_qs
            clip = (parse_qs(self.path.split("?", 1)[-1]).get("clip") or [""])[0]
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._text("bad clip id", 400)
            avi = P01R / f"{clip}.AVI"
            if not avi.exists():
                return self._text("no video", 404)
            return self._serve_video(avi)

        if path == "/api/p083/list":
            # What the audit produced, plus whatever verdicts have been recorded. The verdict
            # columns live in the same CSV that the audit wrote, so the file stays the single
            # record of the stage rather than being split across two places.
            rows = []
            csv_path = P083 / "decisions.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(csv_path.open()))
            items = []
            for r in rows:
                if not r.get("clip"):
                    continue
                items.append({
                    "n": int(r["n"]),
                    "time": float(r["time"]),
                    "node": r["node"],
                    "arrival_edge": r.get("arrival_edge", ""),
                    "candidate_edges": r.get("candidate_edges", ""),
                    "candidate_sides": r.get("candidate_sides", ""),
                    "candidate_angles": r.get("candidate_angles", ""),
                    "male_direction": r.get("male_direction", ""),
                    "yaw_integral": r.get("yaw_integral", ""),
                    "yaw_floor": r.get("yaw_floor", ""),
                    "chosen_edge": r.get("chosen_edge", ""),
                    "reason": r.get("reason", ""),
                    "final_scores": r.get("final_scores", ""),
                    "geometry_scores": r.get("geometry_scores", ""),
                    "novelty_scores": r.get("novelty_scores", ""),
                    "clip_file": r["clip"],
                    "picture": r.get("picture", ""),
                    "strip": r.get("strip", ""),
                    "rotation_coherence": r.get("rotation_coherence", ""),
                    "turn_visible": r.get("turn_visible", ""),
                    "verdict": r.get("verdict", ""),
                    "truth_edge": r.get("truth_edge", ""),
                    "truth_side": r.get("truth_side", ""),
                    "correct_edge_if_wrong": r.get("correct_edge_if_wrong", ""),
                    "note": r.get("note", ""),
                })
            return self._json({"items": items, "video": VIDEO.name,
                               "csv": str(csv_path)})

        if path.startswith("/p083/"):
            name = Path(path).name
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
                return self._text("bad name", 400)
            target = P083 / "junctions" / name
            if not target.exists():
                return self._text("no picture", 404)
            return self._serve(target)

        if path.startswith("/p083clip/"):
            name = Path(path).name
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
                return self._text("bad name", 400)
            target = P083 / "clips" / name
            if not target.exists():
                return self._text("no clip", 404)
            return self._serve_video(target)

        if path == "/api/p08/state":
            # The editor is loaded from a fresh page often, so it asks for the saved
            # graph and the plan in one request rather than assuming either exists.
            plan = None
            for ext in (".png", ".jpg", ".jpeg"):
                if (P08_PLAN_STEM.with_suffix(ext)).exists():
                    plan = f"/api/p08/plan"
                    break
            doc = read_json(P08_GRAPH, {})
            return self._json({
                "graph": doc,
                "graph_exists": bool(doc),
                "graph_path": str(P08_GRAPH),
                "plan_url": plan,
                "out_dir": str(P08_OUT),
            })

        if path == "/api/p08/plan":
            for ext in (".png", ".jpg", ".jpeg"):
                t = P08_PLAN_STEM.with_suffix(ext)
                if t.exists():
                    return self._serve(t)
            return self._text("no plan", 404)

        if path == "/api/p08/plan_overview":
            # The plan sized to be read, not to be measured. Drawn straight from the full plan
            # it comes out blank at the width of its panel, because shrinking averages the thin
            # walls away; see p08_plan_overview. It is derived, so it is rebuilt whenever the
            # plan is newer than it, and the page never has to know whether it exists.
            plan = None
            for ext in (".png", ".jpg", ".jpeg"):
                t = P08_PLAN_STEM.with_suffix(ext)
                if t.exists():
                    plan = t
                    break
            if plan is None:
                return self._text("no plan", 404)
            dest = P08_OUT / "plan_overview.png"
            try:
                stale = (not dest.exists()
                         or dest.stat().st_mtime < plan.stat().st_mtime)
                if stale:
                    P08_OUT.mkdir(parents=True, exist_ok=True)
                    sys.path.insert(0, str(PROJECT / "scripts"))
                    from p08_plan_overview import make_overview
                    w, h = make_overview(plan, dest)
                    print(f"  p08 план обзор пересобран: {dest.name} {w}x{h}")
                return self._serve(dest)
            except Exception as exc:  # noqa: BLE001
                return self._text(f"overview failed: {type(exc).__name__}: {exc}", 500)

        if path == "/api/p08/trajectory":
            # Where the tracker walked, for drawing the route over the whole plan. Thinned and
            # rounded: the file holds a row every 0.02 s, which over the width of the picture is
            # many points per pixel, so the extra precision could not be seen even if it were
            # sent. Parallel arrays rather than objects, because there are thousands of points.
            src = P08_OUT / "graph_trajectory.csv"
            if not src.exists():
                return self._json({"t": [], "x": [], "y": [], "note": "трекер не запускался"})
            rows = list(csv.DictReader(src.open()))
            stride = max(1, len(rows) // 2500)
            pts = rows[::stride]
            return self._json({
                "t": [round(float(r["t"]), 2) for r in pts],
                "x": [round(float(r["x_px"]), 1) for r in pts],
                "y": [round(float(r["y_px"]), 1) for r in pts],
                "stride": stride,
                "n_total": len(rows),
            })

        if path == "/api/p08/tracking":
            # What the tracker wrote, for the playback view. Absent until the tracker
            # has been run, and the page says so rather than showing an empty frame.
            seq = []
            seq_path = P08_OUT / "edge_sequence.csv"
            if seq_path.exists():
                for line in seq_path.read_text(encoding="utf-8").splitlines()[1:]:
                    p = line.split(",")
                    if len(p) >= 5:
                        seq.append({"time_start": float(p[0]), "time_end": float(p[1]),
                                    "edge": p[2], "from": p[3], "to": p[4]})
            return self._json({
                "available": bool(seq),
                "edge_sequence": seq,
                "video": f"/media/{serve_video_name()}",
                "video_name": VIDEO.name,
                "edge_sequence_path": str(seq_path),
                "decisions_path": str(P08_OUT / "decisions.csv"),
            })

        if path == "/api/p08/truth":
            doc = read_json(P08_TRUTH, {})
            return self._json({"truth": doc,
                               "edges": doc.get("edges") or [],
                               "nodes": doc.get("nodes") or [],
                               "path": str(P08_TRUTH),
                               "exists": bool(doc)})

        if path == "/api/p08/compare":
            # Comparison runs on request rather than on save, so the drawing can be edited
            # and re-checked without the file changing under it.
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "p08_compare", PROJECT / "scripts/p08_compare_truth.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                from p08_graph import Graph  # noqa: F401
                sys.path.insert(0, str(PROJECT / "scripts"))
                from p08_graph import Graph as G2
                g = G2.load(P08_GRAPH)
                truth = read_json(P08_TRUTH, {})
                truth_edges = truth.get("edges") or []
                if len(truth_edges) < 2:
                    return self._json({"ok": False,
                                       "problems": ["нарисованный маршрут пуст"],
                                       "lines": ["нарисованный маршрут пуст: кликните рёбра "
                                                 "в закладке ТРЕКИНГ и сохраните"]})
                seq = []
                sp = P08_OUT / "edge_sequence.csv"
                if sp.exists():
                    for line in sp.read_text(encoding="utf-8").splitlines()[1:]:
                        p = line.split(",")
                        if len(p) >= 5:
                            seq.append({"time_start": float(p[0]), "time_end": float(p[1]),
                                        "edge": p[2], "from": p[3], "to": p[4]})
                dec = []
                dp = P08_OUT / "decisions.csv"
                if dp.exists():
                    dec = list(csv.DictReader(dp.open()))
                res, lines = mod.analyse(g, truth_edges, seq, dec)
                res["lines"] = lines
                (P08_OUT / "truth_comparison.json").parent.mkdir(parents=True, exist_ok=True)
                (P08_OUT / "truth_comparison.json").write_text(
                    json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  p08 compare: совпало {res.get('matched_nodes')} узлов")
                return self._json(res)
            except Exception as exc:  # noqa: BLE001
                return self._json({"ok": False, "lines":
                                   [f"сравнение не удалось: {type(exc).__name__}: {exc}"]},
                                  500)

        if path.startswith("/p083v/"):
            name = Path(path).name
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
                return self._text("bad name", 400)
            target = P083 / "clips" / name
            if not target.exists():
                return self._text("no clip", 404)
            return self._serve_video(target)

        if path.startswith("/media/"):
            name = Path(path).name
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
                return self._text("bad name", 400)
            # The pages ask for VID00001.mp4. Serve whichever file currently has the
            # correct timeline, so the HTML never has to be edited when it changes.
            if name == "VID00001.mp4":
                name = serve_video_name()
            target = MEDIA / name
            if not target.exists():
                return self._text(f"missing {name}", 404)
            return self._serve_video(target)

        # anything else under the app dir
        rel = path.lstrip("/")
        if re.fullmatch(r"[A-Za-z0-9_./\-]+", rel):
            target = (ROOT / rel).resolve()
            if str(target).startswith(str(ROOT)) and target.is_file():
                return self._serve(target)

        return self._text("not found", 404)

    def _serve(self, target: Path) -> None:
        if not target.exists():
            return self._text("not found", 404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self, target: Path) -> None:
        """Serve with Range support; seeking in the player depends on it."""
        size = target.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))
        length = end - start + 1

        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        with target.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(CHUNK, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        payload = self._body()

        if path == "/api/camera/import":
            paths = payload.get("paths") or []
            if not isinstance(paths, list):
                return self._json({"ok": False, "error": "paths: список файлов"}, 400)
            start = payload.get("start")
            if start is not None:
                try:
                    start = {"x": float(start["x"]), "y": float(start["y"]), "toward": str(start["toward"])}
                except (KeyError, TypeError, ValueError):
                    return self._json({"ok": False, "error": "start: нужны x, y и toward"}, 400)
                place = snap_on_graph(start["x"], start["y"], start["toward"])
                if not place.get("ok"):
                    return self._json(place, 400)
            res = app_backend.start_import([str(p) for p in paths], start)
            return self._json(res, 200 if res.get("ok") else 409)
        if path == "/api/app/track":
            clip = str(payload.get("clip") or "")
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad clip id"}, 400)
            try:
                start = {"x": float(payload["x"]), "y": float(payload["y"]), "toward": str(payload["toward"])}
            except (KeyError, TypeError, ValueError):
                return self._json({"ok": False, "error": "нужны x, y и toward"}, 400)
            place = snap_on_graph(start["x"], start["y"], start["toward"])
            if not place.get("ok"):
                return self._json(place, 400)
            return self._json(app_backend.request_track(clip, start))
        if path == "/api/app/retry":
            clip = str(payload.get("clip") or "")
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad clip id"}, 400)
            return self._json(app_backend.retry(clip))

        if path == "/api/start/place":
            try:
                x, y = float(payload.get("x")), float(payload.get("y"))
            except (TypeError, ValueError):
                return self._json({"ok": False, "error": "нужны координаты на плане"}, 400)
            toward = payload.get("toward") or None
            if toward is not None:
                toward = str(toward)
            return self._json(snap_on_graph(x, y, toward))

        if path == "/api/start/run":
            try:
                x, y = float(payload.get("x")), float(payload.get("y"))
            except (TypeError, ValueError):
                return self._json({"ok": False, "error": "нужны координаты на плане"}, 400)
            clip = str(payload.get("clip") or DEFAULT_START_CLIP)
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad clip id"}, 400)
            if not _start_profile(clip):
                return self._json({"ok": False, "error": f"clip {clip} не настроен на /start"}, 400)
            if not _tracker_inputs_ready(clip):
                return self._json({
                    "ok": False,
                    "error": "brain/yaw для этого ролика ещё не готовы — подождите минуту и обновите страницу",
                }, 503)
            toward = str(payload.get("toward") or "")
            if not toward:
                return self._json({"ok": False, "error": "выберите направление"}, 400)
            ver = str(payload.get("ver") or "all")
            if ver not in (*TRACKER_SCRIPTS, "all"):
                return self._json({"ok": False, "error": "ver: v1…v5 или all"}, 400)
            place = snap_on_graph(x, y, toward)
            if not place.get("ok"):
                return self._json(place, 400)
            with _START_LOCK:
                if _START_JOB["state"] == "running":
                    return self._json({"ok": False, "error": "прогон уже идёт"}, 409)
                _START_JOB.update(state="running", log="запуск…\n", error="",
                                  place=place, code=None, clip=clip, ver=ver)
            threading.Thread(target=_start_worker, args=(place, clip, ver), daemon=True).start()
            return self._json({"ok": True, "state": "running", "place": place, "clip": clip, "ver": ver})

        if path == "/api/stand/save":
            clip = str(payload.get("clip") or "")
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad clip id"}, 400)
            stand = []
            for iv in payload.get("stand") or []:
                try:
                    a, b = float(iv[0]), float(iv[1])
                except (TypeError, ValueError, IndexError):
                    return self._json({"ok": False, "error": f"плохой отрезок {iv}"}, 400)
                if b <= a:
                    return self._json({"ok": False, "error": f"конец раньше начала: {iv}"}, 400)
                stand.append([round(a, 2), round(b, 2)])
            stand.sort()
            reviewed = float(payload.get("reviewed_until") or 0)
            doc = {"clip": clip, "stand": stand, "reviewed_until": round(reviewed, 2),
                   "time_base": "секунды видео /final (= время трекера при сдвиге 0)",
                   "rule": "всё внутри reviewed_until и вне отрезков stand считается ходьбой",
                   "source": "человек, по видео",
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            out = _stand_path(clip)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            return self._json({"ok": True, **doc})

        if path == "/api/human/save":
            clip = str(payload.get("video") or "")
            if not re.fullmatch(r"VID[0-9]{5}", clip):
                return self._json({"ok": False, "error": "bad video id"}, 400)
            pts = payload.get("points") or []
            out = PROJECT / "data" / "final_tracker" / f"{clip}_human_sketch.json"
            if not pts:
                if out.exists():
                    out.unlink()
                print(f"  human sketch {clip}: cleared")
                return self._json({"ok": True, "cleared": True, "points": 0})
            if len(pts) < 2:
                return self._json({"ok": False, "error": "нужно хотя бы 2 точки"}, 400)
            doc = {
                "video": clip,
                "coordinate_system": "plan_pixels",
                "img_w": payload.get("img_w"),
                "img_h": payload.get("img_h"),
                "points": pts,
                "note": payload.get("note") or "",
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  human sketch {clip}: {len(pts)} points -> {out}")
            return self._json({"ok": True, "path": str(out), "saved_at": doc["saved_at"],
                               "points": len(pts)})

        if path == "/api/p08/save":
            nodes = payload.get("nodes") or []
            edges = payload.get("edges") or []
            ids = [n.get("id") for n in nodes]
            if len(set(ids)) != len(ids):
                return self._json({"ok": False, "error": "duplicate node id"}, 400)
            known = set(ids)
            bad = [e for e in edges
                   if e.get("from") not in known or e.get("to") not in known]
            if bad:
                return self._json({"ok": False,
                                   "error": f"edge with unknown endpoint: {bad[0]}"}, 400)
            payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            P08_DATA.mkdir(parents=True, exist_ok=True)
            P08_GRAPH.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            # The tracker's output names passages by their ids. Replacing the graph with one
            # that uses different ids leaves that output pointing at passages that no longer
            # exist, and a comparison against a drawn route would silently match nothing.
            # Only the passages the tracker actually walked matter: the graph legitimately
            # has many more edges than any one route visits, so comparing whole sets would
            # warn every time. The check is reported rather than enforced, because replacing
            # the graph is a legitimate thing to do.
            stale = None
            seq_path = P08_OUT / "edge_sequence.csv"
            if seq_path.exists():
                walked = set()
                for line in seq_path.read_text(encoding="utf-8").splitlines()[1:]:
                    p = line.split(",")
                    if len(p) >= 5:
                        walked.add(p[2])
                new = {str(e.get("id")) for e in edges}
                gone = walked - new
                if gone:
                    stale = (f"{len(gone)} из {len(walked)} пройденных рёбер исчезли из "
                             f"графа (например {sorted(gone)[:3]}). Результаты трекера "
                             f"относятся к прежнему графу: их надо перестроить, иначе "
                             f"сравнение с разметкой ничего не найдёт.")
            print(f"  p08 graph saved: {len(nodes)} nodes, {len(edges)} edges "
                  f"-> {P08_GRAPH}" + (" (STALE outputs)" if stale else ""))
            return self._json({"ok": True, "path": str(P08_GRAPH),
                               "nodes": len(nodes), "edges": len(edges),
                               "stale_output": stale})

        if path == "/api/p08/truth":
            edges = payload.get("edges") or []
            if not edges:
                return self._json({"ok": False, "error": "empty route"}, 400)
            known = set()
            for d in read_json(P08_GRAPH, {}).get("edges", []):
                known.add(d.get("id"))
            bad = [r for r in edges if r.get("edge") not in known]
            if bad:
                return self._json({"ok": False,
                                   "error": f"unknown edge {bad[0].get('edge')}"}, 400)
            payload["edges"] = edges
            payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            P08_DATA.mkdir(parents=True, exist_ok=True)
            P08_TRUTH.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"  p08 truth route saved: {len(edges)} edges -> {P08_TRUTH}")
            return self._json({"ok": True, "path": str(P08_TRUTH), "edges": len(edges)})

        if path == "/api/p083/verdict":
            n = str(payload.get("n", ""))
            verdict = str(payload.get("verdict", ""))
            if verdict not in ("", "CORRECT", "WRONG", "MISSING", "UNCLEAR"):
                return self._json({"ok": False, "error": "unknown verdict"}, 400)
            csv_path = P083 / "decisions.csv"
            if not csv_path.exists():
                return self._json({"ok": False, "error": "no audit yet"}, 400)
            rows = list(csv.DictReader(csv_path.open()))
            fields = list(rows[0].keys())
            # The audit writes this file, so a column added by the page has to be appended to
            # the header here rather than assumed present: DictWriter raises on a key that is
            # not in fieldnames, and older files predate the column.
            for extra in ("truth_edge", "truth_side"):
                if extra not in fields:
                    fields.append(extra)
                    for r in rows:
                        r.setdefault(extra, "")
            hit = False
            for r in rows:
                if r["n"] == n:
                    r["verdict"] = verdict
                    r["truth_edge"] = str(payload.get("truth_edge", ""))
                    r["truth_side"] = str(payload.get("truth_side", ""))
                    r["correct_edge_if_wrong"] = str(payload.get("correct_edge_if_wrong", ""))
                    r["note"] = str(payload.get("note", r.get("note", "")))
                    hit = True
            if not hit:
                return self._json({"ok": False, "error": f"no decision {n}"}, 404)
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"  p083 verdict {n}: {verdict or 'cleared'}")
            return self._json({"ok": True, "n": n, "verdict": verdict})

        if path == "/api/p08/snap":
            # A drawn line is read as a route over the graph here rather than in the page, so
            # that the page and the comparison use one reading. Drawing a line over the plan is
            # much quicker than clicking passages one at a time, and it needs no knowledge of
            # the graph at all — only of the building.
            pts = payload.get("points") or []
            if len(pts) < 2:
                return self._json({"ok": False, "error": "нужно хотя бы две точки линии"}, 400)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "p08_snap", PROJECT / "scripts/p08_snap.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.path.insert(0, str(PROJECT / "scripts"))
                from p08_graph import Graph as G2
                g = G2.load(P08_GRAPH)
                res = mod.snap_polyline(g, [(float(q[0]), float(q[1])) for q in pts])
                print(f"  p08 snap: {len(pts)} точек -> {res.get('n_edges')} рёбер, "
                      f"rms {res.get('mean_dist_to_graph_px')} px, "
                      f"перезахватов {res.get('reanchors')}")
                return self._json(res)
            except Exception as exc:  # noqa: BLE001
                return self._json({"ok": False,
                                   "error": f"{type(exc).__name__}: {exc}"}, 500)

        if path == "/api/p08/plan":
            data = str(payload.get("data", ""))
            m = re.match(r"data:image/(png|jpeg|jpg);base64,(.*)$", data, re.S)
            if not m:
                return self._json({"ok": False, "error": "expected a base64 png or jpeg"},
                                  400)
            import base64
            ext = ".png" if m.group(1) == "png" else ".jpg"
            blob = base64.b64decode(m.group(2))
            P08_DATA.mkdir(parents=True, exist_ok=True)
            # one plan at a time: the other extension is removed so the page cannot
            # load a stale image after a replacement
            for other in (".png", ".jpg", ".jpeg"):
                if other != ext and P08_PLAN_STEM.with_suffix(other).exists():
                    P08_PLAN_STEM.with_suffix(other).unlink()
            target = P08_PLAN_STEM.with_suffix(ext)
            target.write_bytes(blob)
            print(f"  p08 plan saved: {target.name} ({len(blob) / 1e6:.2f} MB) "
                  f"from {payload.get('name')}")
            return self._json({"ok": True, "saved_as": target.name,
                               "path": str(target), "bytes": len(blob)})

        if path == "/api/measure":
            # Run the FROZEN measurement on a window and report what it says, in camera
            # terms. This is what removes the need to draw anything by hand: mark the
            # start and end of a turn, and the answer for the direction comes from the
            # validated estimator rather than from a 1500-click sketch.
            t0 = float(payload.get("t0", 0.0))
            t1 = float(payload.get("t1", 0.0))
            if t1 <= t0:
                return self._json({"ok": False, "error": "t1 must exceed t0"}, 400)
            try:
                from fly_vo.content_motion import measure_window
                m = measure_window(str(VIDEO), t0, t1, "live")
                out = m.as_dict()
                out.update({
                    "ok": True,
                    "content_dx": m.content_dx,
                    "mean_dx": m.mean_dx,
                    "coherence": m.coherence,
                    "reliable_fraction": m.reliable_fraction,
                    "local_activity": m.local_activity,
                    # sign convention: content moving right means the camera turned left
                    "suggested_camera": m.camera_direction,
                    "suggested_confidence": (
                        "high" if m.coherence >= 0.50 else
                        "medium" if m.coherence >= 0.30 else "low"
                    ),
                    "reading": (
                        "content moved LEFT, so the camera turned RIGHT"
                        if m.content_dx < 0 else
                        "content moved RIGHT, so the camera turned LEFT"
                        if m.content_dx > 0 else "no consistent sideways motion"
                    ),
                })
                print(f"  measured t={t0:.2f}-{t1:.2f}: {out['reading']} "
                      f"(coherence {m.coherence:.2f})")
                return self._json(out)
            except Exception as exc:  # noqa: BLE001
                return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

        if path == "/api/verdict":
            # One human judgement on one system proposal. The verdict is what turns a
            # proposal into a label; the proposal itself is kept alongside it.
            cid = str(payload.get("id", ""))
            if not cid:
                return self._json({"ok": False, "error": "missing id"}, 400)
            doc = read_json(VERDICTS_JSON, {"verdicts": {}})
            doc.setdefault("verdicts", {})
            prev = doc["verdicts"].get(cid, {})
            doc["verdicts"][cid] = {
                "id": cid,
                "system_direction": payload.get("system_direction"),
                "verdict": payload.get("verdict"),          # correct | wrong | skip
                "actual_direction": payload.get("actual_direction"),
                "notes": payload.get("notes", prev.get("notes", "")),
                "coherence": payload.get("coherence"),
                "confidence": payload.get("confidence"),
                "t0": payload.get("t0"), "t1": payload.get("t1"),
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            VERDICTS_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
            n = len(doc["verdicts"])
            print(f"  verdict {cid}: {payload.get('verdict')} "
                  f"(system said {payload.get('system_direction')}"
                  + (f", actually {payload['actual_direction']}" if payload.get("actual_direction") else "")
                  + f")  [{n} reviewed]")
            return self._json({"ok": True, "n_verdicts": n})

        if path == "/api/verdict_export":
            doc = read_json(VERDICTS_JSON, {"verdicts": {}})
            rows = sorted(doc.get("verdicts", {}).values(), key=lambda v: v.get("t0") or 0)
            lines = ["id,t0,t1,system_direction,coherence,confidence,verdict,actual_direction,notes"]
            for v in rows:
                notes = str(v.get("notes", "")).replace(",", ";")
                lines.append(
                    f"{v.get('id')},{v.get('t0')},{v.get('t1')},{v.get('system_direction')},"
                    f"{v.get('coherence')},{v.get('confidence')},{v.get('verdict')},"
                    f"{v.get('actual_direction') or ''},{notes}"
                )
            EXPORT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  exported {len(rows)} verdicts -> {EXPORT_CSV.name}")
            return self._json({"ok": True, "path": str(EXPORT_CSV), "n": len(rows)})

        if path == "/api/save":
            payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            SAVE_JSON.parent.mkdir(parents=True, exist_ok=True)
            SAVE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            n_turns = len(payload.get("turns") or [])
            n_path = len(payload.get("trajectory") or [])
            print(f"  saved {n_turns} turns, {n_path} path points -> {SAVE_JSON.name}")
            return self._json({"ok": True, "path": str(SAVE_JSON),
                               "turns": n_turns, "trajectory": n_path})

        if path == "/api/export":
            turns = payload.get("turns") or []
            lines = ["id,t_start,t_end,direction,confidence,notes"]
            for i, t in enumerate(turns, 1):
                notes = str(t.get("notes", "")).replace(",", ";")
                lines.append(
                    f"{t.get('id', i)},{t.get('t0', '')},{t.get('t1', '')},"
                    f"{t.get('direction', '')},{t.get('confidence', '')},{notes}"
                )
            EXPORT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  exported {len(turns)} turns -> {EXPORT_CSV.name}")
            return self._json({"ok": True, "path": str(EXPORT_CSV), "turns": len(turns)})

        return self._text("not found", 404)


class _DualStackHTTPServer(ThreadingHTTPServer):
    """Listen on IPv6 :: with V6ONLY=0 so http://localhost works on macOS (::1 first)."""

    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def create_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Bound server for the desktop shell; port 0 picks a free one."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    app_backend.ensure_worker()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="::", help=":: = localhost IPv4+IPv6 (default on macOS)")
    args = ap.parse_args()

    MEDIA.mkdir(parents=True, exist_ok=True)
    video = MEDIA / serve_video_name()
    print("fly_vo annotation tool")
    print(f"  app      http://127.0.0.1:{args.port}  (и http://localhost:{args.port})")
    print(f"  video    {video}  ({'present' if video.exists() else 'MISSING - transcode still running?'})")
    print(f"  saves to {SAVE_JSON}")
    print(f"  exports  {EXPORT_CSV}")
    print("  ctrl-c to stop\n")

    if ":" in args.host:
        server = _DualStackHTTPServer((args.host, args.port), Handler)
    else:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
