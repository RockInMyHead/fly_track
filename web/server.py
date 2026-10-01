#!/usr/bin/env python3
"""Live MaleCNS VO web app — upload video, set start/direction, stream trajectory."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

logger = logging.getLogger("flyvo.web")

from dataclasses import asdict

from fly_vo.config import FlyVOConfig
from fly_vo.dopamine import (
    compare_trajectories,
    dopamine_optimize,
    load_adaptive,
    save_adaptive,
    split_polyline,
    video_key_from_path,
)
from fly_vo.feedback import apply_feedback, load_feedback_state
from fly_vo.streaming import StreamingMaleCNS, TrajectoryPoint, split_neural_cache
from fly_vo.video_io import get_video_info
from fly_vo.video_preview import extract_frame_jpeg, is_browser_playable, probe_codec

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="MaleCNS VO Live")
UPLOAD_DIR.mkdir(exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# MaleCNS streaming config — 50 Hz brain, downscaled video for speed
STREAM_CONFIG = FlyVOConfig(max_frame_size=480, chunk_seconds=60.0, brain_dt=0.020)
STREAM_EMIT_EVERY = 2  # emit every 2 brain steps (~25 Hz UI)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB via browser; use local path for larger files


def _malecns_stream_worker(
    video_path: str,
    start: float,
    end: float | None,
    x0: float,
    y0: float,
    heading: float,
    out_q: queue.Queue,
    include_viz: bool = True,
    include_frame: bool = True,
) -> None:
    """Run MaleCNS brain simulation in a dedicated thread."""
    try:
        adaptive_path = ROOT / "data" / "dopamine_state.json"
        vk = video_key_from_path(video_path)
        params = load_adaptive(adaptive_path, video_key=vk)
        processor = StreamingMaleCNS(
            STREAM_CONFIG,
            adaptive=params,
            emit_every_n_steps=STREAM_EMIT_EVERY,
        )
        processor.reset(x0=x0, y0=y0, heading_deg=heading)
        out_q.put(("ready", processor.device))
        for pt in processor.stream(
            video_path,
            start_seconds=start,
            end_seconds=end,
            include_viz=include_viz,
        ):
            if not include_frame:
                pt = TrajectoryPoint(
                    timestamp=pt.timestamp,
                    x=pt.x,
                    y=pt.y,
                    heading_deg=pt.heading_deg,
                    vx_body=pt.vx_body,
                    vy_body=pt.vy_body,
                    speed=pt.speed,
                    confidence=pt.confidence,
                    frame_jpeg=None,
                    pathways=pt.pathways,
                    descending=pt.descending,
                    n_spikes=pt.n_spikes,
                    eye_preview=pt.eye_preview,
                    yaw_rate=pt.yaw_rate,
                    turning=pt.turning,
                )
            out_q.put(("point", pt))
        out_q.put(("done", None))
    except Exception as exc:
        logger.exception("malecns worker failed")
        out_q.put(("error", exc))


async def _iter_malecns_stream(
    video_path: str,
    start: float,
    end: float | None,
    x0: float,
    y0: float,
    heading: float,
    include_viz: bool = True,
    include_frame: bool = True,
) -> Iterator[tuple[str, object]]:
    out_q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_malecns_stream_worker,
        args=(video_path, start, end, x0, y0, heading, out_q, include_viz, include_frame),
        daemon=True,
    )
    thread.start()
    loop = asyncio.get_event_loop()
    while True:
        kind, data = await loop.run_in_executor(None, out_q.get)
        yield kind, data
        if kind in ("done", "error"):
            break


class LocalPathRequest(BaseModel):
    path: str


class TeacherPoint(BaseModel):
    x: float
    y: float


class DopamineRequest(BaseModel):
    video_id: str
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None
    x0: float = 0.0
    y0: float = 0.0
    heading_deg: float = 90.0
    teacher: List[TeacherPoint]
    epochs: int = 1


class FeedbackRequest(BaseModel):
    """Fly drew a path — human judges and optionally shows correction."""

    video_id: str
    verdict: str  # "correct" | "wrong"
    fly_path: List[TeacherPoint]
    correction: Optional[List[TeacherPoint]] = None
    issues: List[str] = []
    note: str = ""
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None
    x0: float = 0.0
    y0: float = 0.0
    heading_deg: float = 90.0


def _resolve_video(video_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="video not found")
    path = matches[0]
    if path.is_symlink():
        return path.resolve()
    return path


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".avi", ".mov"):
        return "video/x-msvideo" if ext == ".avi" else "video/quicktime"
    if ext == ".webm":
        return "video/webm"
    return "video/mp4"


def _register_video_path(src: Path) -> dict:
    src = src.expanduser().resolve()
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {src}")
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    home = Path.home().resolve()
    if not str(src).startswith(str(home)):
        raise HTTPException(
            status_code=403,
            detail="For security, path must be under your home directory",
        )

    try:
        duration, fps, w, h = get_video_info(str(src))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read video: {exc}") from exc

    vid = uuid.uuid4().hex
    link = UPLOAD_DIR / f"{vid}{src.suffix.lower() or '.mp4'}"
    link.symlink_to(src)

    return {
        "video_id": vid,
        "filename": src.name,
        "duration": duration,
        "fps": fps,
        "width": w,
        "height": h,
        "size_mb": round(src.stat().st_size / 1e6, 1),
        "source": "local",
        "browser_playable": is_browser_playable(src),
        "codec": probe_codec(src) or None,
    }


def _pick_video_path() -> Path | None:
    """Open native file picker (macOS AppleScript or tkinter fallback)."""
    if sys.platform == "darwin":
        script = (
            'set f to choose file with prompt "Выберите видео" '
            'of type {"public.movie", "mp4", "avi", "mov", "mkv"}\n'
            "POSIX path of f"
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        path = result.stdout.strip()
        return Path(path) if path else None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Выберите видео",
        filetypes=[
            ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


@app.get("/")
async def index() -> HTMLResponse:
    html = STATIC_DIR / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


def _pt_dict(p: TrajectoryPoint) -> dict:
    return {
        "t": p.timestamp,
        "x": p.x,
        "y": p.y,
        "heading": p.heading_deg,
    }


def _run_dopamine_training(req: DopamineRequest, video_path: Path) -> dict:
    """Sync training pass (runs in thread pool)."""
    params = load_adaptive(ROOT / "data" / "dopamine_state.json")
    end = req.end_seconds
    if end is None:
        duration, _, _, _ = get_video_info(str(video_path))
        end = min(duration, req.start_seconds + 19.0)

    teach_xy = [(t.x, t.y) for t in req.teacher]
    proc = StreamingMaleCNS(STREAM_CONFIG, adaptive=params)
    cache = proc.collect_neural_cache(
        str(video_path),
        start_seconds=req.start_seconds,
        end_seconds=end,
    )

    best_params, loss_before, loss_after, pts_after, trials = dopamine_optimize(
        params,
        cache,
        proc,
        teach_xy,
        x0=req.x0,
        y0=req.y0,
        heading_deg=req.heading_deg,
    )
    pts_before = proc.integrate_from_cache(
        cache,
        params,
        x0=req.x0,
        y0=req.y0,
        heading_deg=req.heading_deg,
        emit_stride=4,
    )

    save_adaptive(best_params, ROOT / "data" / "dopamine_state.json")

    old = asdict(params)
    new = asdict(best_params)
    param_delta = {
        k: round(new[k] - old[k], 4)
        for k in old
        if k in new and isinstance(old[k], (int, float)) and k != "sessions"
    }

    improved = loss_after < loss_before - 1e-4
    msg = (
        f"сессия #{best_params.sessions} · {trials} вариантов · "
        f"loss {loss_before:.3f} → {loss_after:.3f} · "
        f"speed×{best_params.speed_scale:.2f} yaw×{best_params.yaw_gain_scale:.2f} "
        f"hdg{best_params.heading_offset_deg:+.1f}°"
    )

    return {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "improved": improved,
        "message": msg,
        "trials": trials,
        "params": new,
        "param_delta": param_delta,
        "fly_path_before": [_pt_dict(p) for p in pts_before],
        "fly_path": [_pt_dict(p) for p in pts_after],
        "teacher_path": [{"x": t.x, "y": t.y} for t in req.teacher],
        "segment_seconds": end - req.start_seconds,
    }


def _xy(pts) -> list[tuple[float, float]]:
    return [(p.x, p.y) for p in pts]


def _train_worker(
    req: dict,
    video_path: Path,
    out_q: queue.Queue,
    stop: threading.Event,
) -> None:
    """Multi-epoch dopamine training with cache → train split → test holdout."""
    try:
        params = load_adaptive(ROOT / "data" / "dopamine_state.json")
        start = float(req.get("start_seconds", 0.0))
        end = req.get("end_seconds")
        if end is None:
            duration, _, _, _ = get_video_info(str(video_path))
            end = min(duration, start + STREAM_CONFIG.chunk_seconds)
        else:
            end = float(end)
        x0 = float(req.get("x0", 0.0))
        y0 = float(req.get("y0", 0.0))
        heading = float(req.get("heading_deg", 90.0))
        epochs = int(np.clip(int(req.get("epochs", 5)), 1, 30))
        teacher = [(float(t["x"]), float(t["y"])) for t in req["teacher"]]

        proc = StreamingMaleCNS(STREAM_CONFIG, adaptive=params)

        def on_cache(_stage, t, t_end, frac):
            if stop.is_set():
                return
            out_q.put(
                (
                    "cache",
                    {
                        "t": float(t),
                        "end": float(t_end),
                        "frac": float(np.clip(frac, 0.0, 1.0)),
                    },
                )
            )

        out_q.put(("status", {"message": "MaleCNS кэш — один прогон видео"}))
        cache = proc.collect_neural_cache(
            str(video_path),
            start_seconds=start,
            end_seconds=end,
            progress=on_cache,
        )
        if stop.is_set():
            out_q.put(("error", "остановлено"))
            return

        train_cache, test_cache, held_out = split_neural_cache(cache, 0.7)
        teach_np = np.asarray(teacher, dtype=np.float64)
        if held_out and len(teach_np) >= 4:
            teach_train, teach_test = split_polyline(teach_np, 0.7)
        else:
            held_out = False
            teach_train, teach_test = teach_np, teach_np
            train_cache = cache

        pts_before = proc.integrate_from_cache(
            train_cache, params, x0=x0, y0=y0, heading_deg=heading, emit_stride=4
        )
        current = params
        history: list[dict] = []
        t0 = time.time()
        last_path = pts_before

        for epoch in range(1, epochs + 1):
            if stop.is_set():
                out_q.put(("error", "остановлено"))
                return
            current, loss_before, loss_after, last_path, trials = dopamine_optimize(
                current,
                train_cache,
                proc,
                [(float(x), float(y)) for x, y in teach_train],
                x0=x0,
                y0=y0,
                heading_deg=heading,
            )
            elapsed = time.time() - t0
            speed = epoch / elapsed if elapsed > 0 else 0.0
            remain = (epochs - epoch) / speed if speed > 0 else 0.0
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss_after), 4),
                    "loss_before": round(float(loss_before), 4),
                }
            )
            out_q.put(
                (
                    "epoch",
                    {
                        "epoch": epoch,
                        "epochs": epochs,
                        "loss": float(loss_after),
                        "loss_before": float(loss_before),
                        "elapsed": elapsed,
                        "eta": remain,
                        "speed": speed,
                        "trials": trials,
                        "held_out": held_out,
                    },
                )
            )

        save_adaptive(current, ROOT / "data" / "dopamine_state.json")

        train_path = proc.integrate_from_cache(
            train_cache, current, x0=x0, y0=y0, heading_deg=heading, emit_stride=4
        )
        train_metrics = compare_trajectories(_xy(train_path), teach_train)
        before_metrics = compare_trajectories(_xy(pts_before), teach_train)

        if held_out:
            last = train_path[-1]
            test_path = proc.integrate_from_cache(
                test_cache,
                current,
                x0=last.x,
                y0=last.y,
                heading_deg=last.heading_deg,
                emit_stride=4,
            )
            test_metrics = compare_trajectories(_xy(test_path), teach_test)
        else:
            test_path = train_path
            test_metrics = train_metrics

        out_q.put(
            (
                "done",
                {
                    "epochs": epochs,
                    "held_out": held_out,
                    "params": asdict(current),
                    "history": history,
                    "train": train_metrics,
                    "test": test_metrics,
                    "before": before_metrics,
                    "fly_path_before": [_pt_dict(p) for p in pts_before],
                    "fly_path": [_pt_dict(p) for p in train_path],
                    "fly_path_test": [_pt_dict(p) for p in test_path],
                    "teacher_train": [{"x": float(x), "y": float(y)} for x, y in teach_train],
                    "teacher_test": [{"x": float(x), "y": float(y)} for x, y in teach_test],
                    "segment_seconds": end - start,
                    "elapsed": time.time() - t0,
                },
            )
        )
    except Exception as exc:
        logger.exception("train worker failed")
        out_q.put(("error", str(exc)))


@app.get("/api/feedback/stats")
async def feedback_stats() -> dict:
    state = load_feedback_state(ROOT / "data" / "feedback_log.json")
    return {
        "correct": state.correct,
        "wrong": state.wrong,
        "total": state.correct + state.wrong,
        "recent": [asdict(r) for r in state.records[-5:]],
    }


@app.post("/api/feedback")
async def fly_feedback(req: FeedbackRequest) -> dict:
    """Fly drew → user marks correct/wrong → brain adapts."""
    if req.verdict not in ("correct", "wrong"):
        raise HTTPException(status_code=400, detail="verdict must be correct or wrong")
    if len(req.fly_path) < 2:
        raise HTTPException(status_code=400, detail="Нужен путь мухи (мин. 2 точки)")
    if req.verdict == "wrong" and not req.issues and (
        not req.correction or len(req.correction) < 2
    ):
        raise HTTPException(
            status_code=400,
            detail="Отметь проблему (метки) или нарисуй как должно быть",
        )

    try:
        video_path = _resolve_video(req.video_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="video not found") from None

    end = req.end_seconds
    if end is None:
        duration, _, _, _ = get_video_info(str(video_path))
        end = min(duration, req.start_seconds + STREAM_CONFIG.chunk_seconds)

    vk = video_key_from_path(video_path)
    adaptive_path = ROOT / "data" / "dopamine_state.json"
    params = load_adaptive(adaptive_path, video_key=vk)
    proc = StreamingMaleCNS(STREAM_CONFIG, adaptive=params)
    fly_xy = [(p.x, p.y) for p in req.fly_path]
    correction = (
        [(p.x, p.y) for p in req.correction] if req.correction else None
    )

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: apply_feedback(
                params,
                req.verdict,  # type: ignore[arg-type]
                proc,
                str(video_path),
                fly_xy,
                correction=correction,
                issues=req.issues,
                note=req.note,
                start_seconds=req.start_seconds,
                end_seconds=end,
                x0=req.x0,
                y0=req.y0,
                heading_deg=req.heading_deg,
                state_path=ROOT / "data" / "feedback_log.json",
                adaptive_path=ROOT / "data" / "dopamine_state.json",
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("feedback failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/adaptive")
async def adaptive_state() -> dict:
    p = load_adaptive(ROOT / "data" / "dopamine_state.json")
    return asdict(p)


@app.post("/api/adaptive/reset")
async def adaptive_reset(video_key: str | None = None) -> dict:
    from fly_vo.dopamine import AdaptiveParams

    path = ROOT / "data" / "dopamine_state.json"
    p = AdaptiveParams()
    if video_key:
        save_adaptive(p, path, video_key=video_key)
    else:
        path.write_text(json.dumps({"by_video": {}}, indent=2), encoding="utf-8")
    return asdict(p)


@app.post("/api/dopamine")
async def dopamine_pulse(req: DopamineRequest) -> dict:
    """Reward fly with user-drawn teacher trajectory — one learning iteration."""
    if len(req.teacher) < 2:
        raise HTTPException(status_code=400, detail="Нарисуйте траекторию (мин. 2 точки)")

    try:
        video_path = _resolve_video(req.video_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="video not found") from None

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _run_dopamine_training, req, video_path)
    except Exception as exc:
        logger.exception("dopamine failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws/train")
async def train_trajectory(ws: WebSocket) -> None:
    """Multi-epoch dopamine with progress, then train/test comparison."""
    await ws.accept()
    stop = threading.Event()
    try:
        raw = await ws.receive_text()
        req = json.loads(raw)
        if len(req.get("teacher") or []) < 2:
            await ws.send_json({"type": "error", "message": "Нарисуйте эталон (мин. 2 точки)"})
            return
        try:
            video_path = _resolve_video(req["video_id"])
        except HTTPException:
            await ws.send_json({"type": "error", "message": "video not found"})
            return

        out_q: queue.Queue = queue.Queue()
        thread = threading.Thread(
            target=_train_worker,
            args=(req, video_path, out_q, stop),
            daemon=True,
        )
        thread.start()
        loop = asyncio.get_event_loop()
        while True:
            kind, data = await loop.run_in_executor(None, out_q.get)
            if kind == "cache":
                await ws.send_json({"type": "cache", **data})
            elif kind == "status":
                await ws.send_json({"type": "status", **data})
            elif kind == "epoch":
                await ws.send_json({"type": "epoch", **data})
            elif kind == "done":
                await ws.send_json({"type": "done", **data})
                break
            elif kind == "error":
                await ws.send_json({"type": "error", "message": str(data)})
                break
    except WebSocketDisconnect:
        stop.set()
    except Exception as exc:
        logger.exception("train ws failed")
        stop.set()
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()


@app.get("/api/device")
async def device_info() -> dict:
    adaptive = load_adaptive(ROOT / "data" / "dopamine_state.json")
    proc = StreamingMaleCNS(STREAM_CONFIG)
    try:
        device = proc.device
        n_neurons = proc.engine.n_neurons
    except Exception:
        device = STREAM_CONFIG.device
        n_neurons = 166_700
    fb = load_feedback_state(ROOT / "data" / "feedback_log.json")
    return {
        "device": device,
        "sim_hz": STREAM_CONFIG.sim_hz,
        "n_epg": STREAM_CONFIG.n_epg_neurons,
        "n_neurons": n_neurons,
        "connectome": "MaleCNS v1.0",
        "dopamine_sessions": adaptive.sessions,
        "feedback_correct": fb.correct,
        "feedback_wrong": fb.wrong,
    }


@app.get("/api/hex-layout")
async def hex_layout() -> dict:
    """Legacy endpoint — MaleCNS uses 1-D photoreceptor drive, not FlyVis hex layout."""
    return {"hex_x": [], "hex_y": [], "n_hexals": 0}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    vid = uuid.uuid4().hex
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{vid}{ext}"
    total = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large ({total / 1e6:.0f} MB). "
                            f"Max upload {MAX_UPLOAD_BYTES // 1_000_000} MB — "
                            "use local path for big files."
                        ),
                    )
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        duration, fps, w, h = get_video_info(str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Cannot read video: {exc}") from exc

    return {
        "video_id": vid,
        "filename": file.filename,
        "duration": duration,
        "fps": fps,
        "width": w,
        "height": h,
        "size_mb": round(total / 1e6, 1),
        "source": "upload",
        "browser_playable": is_browser_playable(dest),
        "codec": probe_codec(dest) or None,
    }


@app.post("/api/register-local")
async def register_local(req: LocalPathRequest) -> dict:
    """Use a video already on disk (recommended for large AVI/MJPEG files)."""
    src = Path(req.path.strip())
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    return _register_video_path(src)


@app.post("/api/pick-local")
async def pick_local() -> dict:
    """Open system file picker and register selected video (no upload)."""
    loop = asyncio.get_event_loop()
    picked = await loop.run_in_executor(None, _pick_video_path)
    if picked is None:
        raise HTTPException(status_code=400, detail="Выбор файла отменён")
    return _register_video_path(picked)


@app.get("/api/video/{video_id}")
async def serve_video(video_id: str) -> FileResponse:
    path = _resolve_video(video_id)
    return FileResponse(path, media_type=_media_type(path), filename=path.name)


@app.get("/api/video/{video_id}/frame.jpg")
async def video_frame(video_id: str, t: float = 0.0, w: int = 960) -> Response:
    """JPEG preview frame — used for AVI/MJPEG and other browser-incompatible formats."""
    path = _resolve_video(video_id)
    loop = asyncio.get_event_loop()
    try:
        jpeg = await loop.run_in_executor(
            None, lambda: extract_frame_jpeg(path, t_seconds=t, max_width=w)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=jpeg, media_type="image/jpeg")


def _point_payload(
    pt: TrajectoryPoint,
    *,
    include_viz: bool = True,
    include_frame: bool = True,
) -> dict:
    payload = {
        "type": "point",
        "t": pt.timestamp,
        "x": pt.x,
        "y": pt.y,
        "heading": pt.heading_deg,
        "speed": pt.speed,
        "confidence": pt.confidence,
        "vx_body": pt.vx_body,
        "vy_body": pt.vy_body,
    }
    if include_frame and pt.frame_jpeg:
        payload["frame"] = base64.b64encode(pt.frame_jpeg).decode("ascii")
    if include_viz:
        if pt.pathways:
            payload["pathways"] = pt.pathways
        if pt.descending:
            payload["descending"] = pt.descending
        if pt.n_spikes is not None:
            payload["n_spikes"] = pt.n_spikes
        if pt.eye_preview:
            payload["eye_preview"] = pt.eye_preview
        payload["yaw_rate"] = pt.yaw_rate
        payload["turning"] = pt.turning
    return payload


@app.websocket("/ws/stream")
async def stream_trajectory(ws: WebSocket) -> None:
    await ws.accept()

    try:
        while True:
            msg = await ws.receive_text()
            req = json.loads(msg)

            if req.get("action") == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if req.get("action") != "start":
                await ws.send_json({"type": "error", "message": "unknown action"})
                continue

            video_id = req["video_id"]
            try:
                video_path = _resolve_video(video_id)
            except HTTPException:
                await ws.send_json({"type": "error", "message": "video not found"})
                continue

            start = float(req.get("start_seconds", 0.0))
            end = req.get("end_seconds")
            if end is not None:
                end = float(end)
            else:
                duration, _, _, _ = get_video_info(str(video_path))
                end = min(duration, start + STREAM_CONFIG.chunk_seconds)
            x0 = float(req.get("x0", 0.0))
            y0 = float(req.get("y0", 0.0))
            heading = float(req.get("heading_deg", 90.0))
            display_mode = req.get("display_mode", "live")
            # Batch: full brain/retina, but skip embedded JPEG (client loads frames by t).
            include_viz = True
            include_frame = display_mode != "batch"

            started = False
            try:
                async for kind, data in _iter_malecns_stream(
                    str(video_path),
                    start,
                    end,
                    x0,
                    y0,
                    heading,
                    include_viz=include_viz,
                    include_frame=include_frame,
                ):
                    if kind == "ready":
                        await ws.send_json(
                            {
                                "type": "started",
                                "device": data,
                                "sim_hz": STREAM_CONFIG.sim_hz,
                                "start_seconds": start,
                                "end_seconds": end,
                                "x0": x0,
                                "y0": y0,
                                "heading_deg": heading,
                                "n_epg": STREAM_CONFIG.n_epg_neurons,
                                "connectome": "MaleCNS v1.0",
                            }
                        )
                        started = True
                    elif kind == "point":
                        if not started:
                            await ws.send_json(
                                {
                                    "type": "started",
                                    "device": STREAM_CONFIG.device,
                                    "sim_hz": STREAM_CONFIG.sim_hz,
                                    "start_seconds": start,
                                    "x0": x0,
                                    "y0": y0,
                                    "heading_deg": heading,
                                    "n_epg": STREAM_CONFIG.n_epg_neurons,
                                    "connectome": "MaleCNS v1.0",
                                }
                            )
                            started = True
                        await ws.send_json(
                            _point_payload(
                                data,
                                include_viz=include_viz,
                                include_frame=include_frame,
                            )
                        )
                    elif kind == "error":
                        raise data  # type: ignore[misc]
                    elif kind == "done":
                        await ws.send_json({"type": "done"})
            except Exception as exc:
                logger.exception("stream failed")
                await ws.send_json({"type": "error", "message": str(exc)})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


def main() -> None:
    import uvicorn

    uvicorn.run(
        "web.server:app",
        host="127.0.0.1",
        port=8770,
        reload=False,
    )


if __name__ == "__main__":
    main()
