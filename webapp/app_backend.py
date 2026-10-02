"""Desktop app backend: camera import and the per-clip preparation chain.

Everything a fresh machine needs to go from a camera card to "Старт трекера":

    camera     find a mounted card with DCIM/**/*.AVI, copy files without touching the card,
               verify the copy by sha256, give each a new VID id (camera names repeat
               across sessions, so the card's own name is kept only as a label)
    pipeline   frames -> browser mp4 -> brain (DN) -> registry -> brain (early) -> yaw -> warm-up

The camera AVI header overstates the frame count (45000 declared, 36837 real on VID00010),
so duration always comes from counted packets, never from the header.

State lives in data/app/clips.json; every step is skipped when its output already exists,
so an interrupted chain resumes where it stopped after the app is restarted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
P01R = PROJECT / "data" / "p01r"
MEDIA = ROOT / "media"
APP_DATA = PROJECT / "data" / "app"
CLIPS_JSON = APP_DATA / "clips.json"
FRAMES_CACHE = PROJECT / "output" / "final_tracker" / "cache"
REGISTRY = PROJECT / "data" / "p07_videos.json"
FIRST_ID = 10001
MIN_AVI_BYTES = 1_000_000
FINGER_BYTES = 1 << 20

STEPS = [
    ("frames", "Кадры и длительность"),
    ("mp4", "Видео для плеера"),
    ("brain_dn", "Мозг мухи: нисходящие нейроны"),
    ("register", "Регистрация ролика"),
    ("brain_early", "Мозг мухи: ранние зрительные клетки"),
    ("yaw", "Сигнал поворота"),
    ("warm", "Движение кадра и ритм шагов"),
]

_LOCK = threading.RLock()
_WAKE = threading.Event()
_PROCS: set[subprocess.Popen] = set()
_IMPORT = {"state": "idle", "file": "", "done_bytes": 0, "total_bytes": 0,
           "files_done": 0, "files_total": 0, "error": "", "imported": []}
_worker_started = False
# set by server.py: track_hook(cid, start) -> (code, error); runs V1..V5 from a plan start
track_hook = None


# --------------------------------------------------------------------------- helpers

def python_exe() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        exe = exe.with_name("python.exe")
    return str(exe)


def child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")
    if sys.platform == "darwin":
        parts = env.get("PATH", "").split(os.pathsep)
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"):
            if prefix not in parts:
                parts.insert(0, prefix)
        env["PATH"] = os.pathsep.join(parts)
    return env


def popen_kwargs() -> dict:
    kw = {"cwd": str(PROJECT), "env": child_env(), "stdout": subprocess.PIPE,
          "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace"}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return kw


def stop_children() -> None:
    for p in list(_PROCS):
        try:
            p.terminate()
        except Exception:
            pass


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _clips() -> dict:
    return _read(CLIPS_JSON, {"clips": {}})


def _save_clip(cid: str, **fields) -> dict:
    with _LOCK:
        doc = _clips()
        rec = doc["clips"].setdefault(cid, {"id": cid})
        rec.update(fields)
        _write(CLIPS_JSON, doc)
        return dict(rec)


def clip_record(cid: str) -> dict | None:
    return _clips()["clips"].get(cid)


def clip_title(cid: str) -> str:
    rec = clip_record(cid) or {}
    src = rec.get("source_name")
    when = rec.get("source_mtime_text")
    if src:
        return f"{cid} — {src}" + (f", снято {when}" if when else "")
    return cid


# --------------------------------------------------------------------------- camera

def _drive_roots() -> list[Path]:
    extra = [Path(p) for p in os.environ.get("FLY_CAMERA_ROOTS", "").split(os.pathsep) if p]
    return extra + _system_roots()


def _system_roots() -> list[Path]:
    if os.name == "nt":
        import ctypes
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        system = os.environ.get("SystemDrive", "C:").upper()
        roots = []
        for i, letter in enumerate(string.ascii_uppercase):
            if not mask & (1 << i) or f"{letter}:" == system:
                continue
            kind = ctypes.windll.kernel32.GetDriveTypeW(f"{letter}:\\")
            if kind in (2, 3):  # removable, fixed (some card readers report fixed)
                roots.append(Path(f"{letter}:\\"))
        return roots
    bases = [Path("/Volumes")] if sys.platform == "darwin" else [Path("/media"), Path("/run/media")]
    roots = []
    for b in bases:
        if not b.exists():
            continue
        for d in b.iterdir():
            roots.append(d)
            if sys.platform != "darwin":
                roots.extend(x for x in d.iterdir() if x.is_dir())
    return roots


def _volume_label(root: Path) -> str:
    if os.name == "nt":
        import ctypes
        buf = ctypes.create_unicode_buffer(261)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(str(root), buf, 261, None, None, None, None, 0)
        return buf.value if ok else ""
    return root.name


def fingerprint(path: Path) -> str:
    """size + first and last MiB: cheap enough to run over USB on every scan."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(FINGER_BYTES))
        if size > FINGER_BYTES:
            fh.seek(max(0, size - FINGER_BYTES))
            h.update(fh.read(FINGER_BYTES))
    return h.hexdigest()


def _imported_fingerprints() -> dict[str, str]:
    return {r.get("fingerprint"): cid for cid, r in _clips()["clips"].items() if r.get("fingerprint")}


def scan_camera() -> dict:
    known = _imported_fingerprints()
    cameras = []
    for root in _drive_roots():
        dcim = next((root / n for n in ("DCIM", "dcim") if (root / n).is_dir()), None)
        if dcim is None:
            continue
        files = []
        try:
            found = [p for p in dcim.rglob("*") if p.suffix.lower() == ".avi"]
        except OSError:
            continue
        for p in found:
            try:
                if p.name.startswith("._") or not p.is_file():
                    continue
                st = p.stat()
                if st.st_size < MIN_AVI_BYTES:
                    continue
                fp = fingerprint(p)
            except OSError:
                continue
            files.append({
                "path": str(p), "name": p.name, "size": st.st_size,
                "mtime": st.st_mtime,
                "mtime_text": time.strftime("%d.%m.%Y %H:%M", time.localtime(st.st_mtime)),
                "fingerprint": fp, "imported_as": known.get(fp),
            })
        files.sort(key=lambda f: f["mtime"])
        cameras.append({"root": str(root), "label": _volume_label(root), "dcim": str(dcim), "files": files})
    return {"ok": True, "cameras": cameras}


def _next_id() -> str:
    used = set(_clips()["clips"])
    if P01R.exists():
        used |= {p.stem for p in P01R.glob("VID*.AVI")}
    nums = [int(m.group(1)) for u in used if (m := re.fullmatch(r"VID(\d{5})", u))]
    n = max([FIRST_ID - 1] + [x for x in nums if x >= FIRST_ID]) + 1
    return f"VID{n:05d}"


def _copy_verified(src: Path, dst: Path) -> str:
    part = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    with src.open("rb") as fi, part.open("wb") as fo:
        while True:
            buf = fi.read(8 << 20)
            if not buf:
                break
            fo.write(buf)
            h.update(buf)
            with _LOCK:
                _IMPORT["done_bytes"] += len(buf)
        fo.flush()
        os.fsync(fo.fileno())
    src_hash = h.hexdigest()
    if part.stat().st_size != src.stat().st_size:
        part.unlink(missing_ok=True)
        raise IOError(f"{src.name}: размер копии не совпал с камерой")
    h2 = hashlib.sha256()
    with part.open("rb") as fh:
        for buf in iter(lambda: fh.read(8 << 20), b""):
            h2.update(buf)
    if h2.hexdigest() != src_hash:
        part.unlink(missing_ok=True)
        raise IOError(f"{src.name}: копия на диске отличается от прочитанного с камеры")
    shutil.copystat(src, part)
    os.replace(part, dst)
    return src_hash


def _track_fields(start: dict | None) -> dict:
    if not start:
        return {}
    return {"start": start, "track_status": "queued", "track_error": ""}


def _import_worker(paths: list[str], start: dict | None = None) -> None:
    known = _imported_fingerprints()
    try:
        for p in map(Path, paths):
            with _LOCK:
                _IMPORT["file"] = p.name
            fp = fingerprint(p)
            if fp in known:
                if start:
                    _save_clip(known[fp], **_track_fields(start))
                    _WAKE.set()
                with _LOCK:
                    _IMPORT["done_bytes"] += p.stat().st_size
                    _IMPORT["files_done"] += 1
                    _IMPORT["imported"].append(known[fp])
                continue
            free = shutil.disk_usage(P01R).free
            if free < p.stat().st_size * 1.3 + 500_000_000:
                raise IOError(f"мало места на диске для {p.name}: свободно {free / 1e9:.1f} ГБ")
            cid = _next_id()
            st = p.stat()
            sha = _copy_verified(p, P01R / f"{cid}.AVI")
            _save_clip(cid, source_name=p.name, source_path=str(p), size=st.st_size,
                       source_mtime=st.st_mtime,
                       source_mtime_text=time.strftime("%d.%m.%Y %H:%M", time.localtime(st.st_mtime)),
                       fingerprint=fp, sha256=sha, imported_at=time.time(),
                       status="queued", step=None, error="", log="", **_track_fields(start))
            known[fp] = cid
            with _LOCK:
                _IMPORT["files_done"] += 1
                _IMPORT["imported"].append(cid)
            _WAKE.set()
        with _LOCK:
            _IMPORT["state"] = "done"
    except Exception as e:
        with _LOCK:
            _IMPORT["state"] = "error"
            _IMPORT["error"] = str(e)


def start_import(paths: list[str], start: dict | None = None) -> dict:
    with _LOCK:
        if _IMPORT["state"] == "running":
            return {"ok": False, "error": "копирование уже идёт"}
        good = [p for p in paths if Path(p).is_file() and Path(p).suffix.lower() == ".avi"]
        if not good:
            return {"ok": False, "error": "не выбрано ни одного AVI"}
        P01R.mkdir(parents=True, exist_ok=True)
        _IMPORT.update(state="running", file="", done_bytes=0,
                       total_bytes=sum(Path(p).stat().st_size for p in good),
                       files_done=0, files_total=len(good), error="", imported=[])
    threading.Thread(target=_import_worker, args=(good, start), daemon=True).start()
    ensure_worker()
    return {"ok": True}


def import_status() -> dict:
    with _LOCK:
        return dict(_IMPORT)


# --------------------------------------------------------------------------- pipeline

def _tag(cid: str) -> str:
    return cid.lower()


def _frames(cid: str) -> dict:
    return _read(FRAMES_CACHE / f"frames_{cid}.json", {})


def _duration(cid: str) -> float:
    f = _frames(cid)
    return f["frames"] / f["fps"] if f.get("frames") and f.get("fps") else 0.0


def _chunks_expected(cid: str) -> int:
    return max(1, math.ceil(_duration(cid) / 60.0))


def _registered(cid: str) -> bool:
    return cid in _read(REGISTRY, {})


def _done(cid: str, step: str) -> bool:
    if step == "frames":
        return bool(_frames(cid).get("frames"))
    if step == "mp4":
        return (MEDIA / f"{cid}_fixed.mp4").exists()
    if step == "brain_dn":
        return (PROJECT / "output" / f"p06_neurons_{_tag(cid)}" / "spike_trace.npz").exists()
    if step == "register":
        return _registered(cid)
    if step == "brain_early":
        return (PROJECT / "output" / "p10" / f"brain_{cid}.npz").exists()
    if step == "yaw":
        return (PROJECT / "output" / "p07" / f"yaw_signal_{cid}.csv").exists()
    if step == "warm":
        return (APP_DATA / f"warm_{cid}.done").exists()
    return False


def _step_progress(cid: str, step: str) -> float | None:
    if step == "brain_dn":
        d = PROJECT / "output" / f"p06_neurons_{_tag(cid)}" / "chunks"
        return min(1.0, len(list(d.glob("chunk_*.npz"))) / _chunks_expected(cid)) if d.exists() else 0.0
    if step == "brain_early":
        d = PROJECT / "output" / "p10" / f"chunks_brain_{cid}"
        return min(1.0, len(list(d.glob("chunk_*.npz"))) / _chunks_expected(cid)) if d.exists() else 0.0
    if step == "mp4":
        return (clip_record(cid) or {}).get("mp4_progress")
    return None


def _run(cid: str, cmd: list[str], on_line=None) -> None:
    lines: list[str] = []
    proc = subprocess.Popen(cmd, **popen_kwargs())
    _PROCS.add(proc)
    try:
        assert proc.stdout is not None
        last_save = 0.0
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if on_line:
                on_line(line)
            if time.time() - last_save > 1.0:
                _save_clip(cid, log="\n".join(lines[-30:]))
                last_save = time.time()
        code = proc.wait()
    finally:
        _PROCS.discard(proc)
    _save_clip(cid, log="\n".join(lines[-30:]))
    if code != 0:
        tail = [x for x in lines if x.strip()][-1:] or [f"код {code}"]
        raise RuntimeError(tail[0])


def _ffprobe_stream(avi: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_packets,nb_frames,r_frame_rate,width,height,codec_name",
                        "-of", "json", str(avi)], capture_output=True, text=True,
                       **({"creationflags": 0x08000000} if os.name == "nt" else {}))
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe не открыл видео: {r.stderr.strip()[:200]}")
    s = json.loads(r.stdout)["streams"][0]
    num, den = (s.get("r_frame_rate") or "30/1").split("/")
    return {"frames": int(s["nb_read_packets"]), "declared": int(s.get("nb_frames") or 0),
            "resolution": f"{s['width']}x{s['height']}", "fps": float(num) / float(den or 1),
            "codec": s.get("codec_name")}


def _do_step(cid: str, step: str) -> None:
    avi = P01R / f"{cid}.AVI"
    py = python_exe()
    if step == "frames":
        info = _ffprobe_stream(avi)
        if info["frames"] < info["fps"] * 2:
            raise RuntimeError(f"в ролике только {info['frames']} кадров — это обрывок, не обрабатываю")
        FRAMES_CACHE.mkdir(parents=True, exist_ok=True)
        _write(FRAMES_CACHE / f"frames_{cid}.json",
               {k: info[k] for k in ("frames", "declared", "resolution", "fps")})
        _save_clip(cid, frames=info["frames"], declared_frames=info["declared"], fps=info["fps"],
                   resolution=info["resolution"], duration_s=round(info["frames"] / info["fps"], 1))
    elif step == "mp4":
        MEDIA.mkdir(parents=True, exist_ok=True)
        dur = _duration(cid)
        tmp = MEDIA / f"{cid}_fixed.tmp.mp4"
        fps = _frames(cid).get("fps") or 30.0

        def on_line(line: str) -> None:
            if line.startswith("out_time_us=") and dur > 0:
                try:
                    _save_clip(cid, mp4_progress=min(1.0, int(line.split("=")[1]) / 1e6 / dur))
                except ValueError:
                    pass

        # -r before -i: read the AVI as contiguous frames, so browser time = real time
        _run(cid, ["ffmpeg", "-y", "-v", "error", "-progress", "pipe:1", "-nostats",
                   "-r", f"{fps:g}", "-i", str(avi), "-t", f"{dur:.3f}", "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(tmp)], on_line)
        os.replace(tmp, MEDIA / f"{cid}_fixed.mp4")
    elif step == "brain_dn":
        _run(cid, [py, "scripts/p062_neuron_run.py", "--video", avi.name, "--tag", _tag(cid)])
    elif step == "register":
        out = f"output/p06_neurons_{_tag(cid)}"
        _run(cid, [py, "scripts/p07_videos.py", "--add", cid,
                   "--trace", f"{out}/spike_trace.npz", "--targets", f"{out}/targets.csv"])
        if not _registered(cid):
            raise RuntimeError("ролик не попал в реестр p07_videos.json")
    elif step == "brain_early":
        _run(cid, [py, "scripts/p10_record.py", "--video", cid])
    elif step == "yaw":
        _run(cid, [py, "scripts/p07_fly_readout.py"])
        if not _done(cid, "yaw"):
            raise RuntimeError("сигнал поворота не записан")
    elif step == "warm":
        try:
            _run(cid, [py, "scripts/final_tracker.py", "--video", cid, "--quiet",
                       "--out", str(PROJECT / "output" / "app_warm" / cid)])
            _run(cid, [py, "-c", "import sys; from pathlib import Path; sys.path.insert(0, 'scripts'); "
                                 "import final_tracker_v3 as V3; "
                                 f"V3.image_shifts(Path('data/p01r/{cid}.AVI'))"])
        except RuntimeError as e:
            # the trackers compute the same caches on the first «Старт»; only slower
            _save_clip(cid, warm_warning=str(e))
        (APP_DATA / f"warm_{cid}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def _process(cid: str) -> None:
    _save_clip(cid, status="running", error="", started_at=time.time())
    for i, (step, _title) in enumerate(STEPS):
        if _done(cid, step):
            continue
        _save_clip(cid, step=step, step_index=i, step_started=time.time(), mp4_progress=0.0)
        _do_step(cid, step)
    _save_clip(cid, status="done", step=None, finished_at=time.time())


def _track(cid: str) -> None:
    rec = clip_record(cid) or {}
    if track_hook is None or not rec.get("start") or rec.get("track_status") not in ("queued", "running"):
        return
    _save_clip(cid, track_status="running", track_error="", track_started=time.time())
    try:
        code, err = track_hook(cid, rec["start"])
    except Exception as e:
        code, err = 1, str(e)
    _save_clip(cid, track_status="done" if code == 0 else "error",
               track_error="" if code == 0 else (err or f"код {code}"), track_finished=time.time())


def _worker() -> None:
    while True:
        clips = sorted(_clips()["clips"].items())
        todo = [cid for cid, r in clips if r.get("status") in ("queued", "running")]
        to_track = [cid for cid, r in clips
                    if r.get("status") == "done" and r.get("track_status") in ("queued", "running")]
        if not todo and not to_track:
            _WAKE.wait(5.0)
            _WAKE.clear()
            continue
        if todo:
            cid = todo[0]
            try:
                if not (P01R / f"{cid}.AVI").exists():
                    raise RuntimeError(f"нет файла data/p01r/{cid}.AVI")
                _process(cid)
            except Exception as e:
                _save_clip(cid, status="error", error=str(e))
                continue
        else:
            cid = to_track[0]
        _track(cid)


def request_track(cid: str, start: dict) -> dict:
    if not clip_record(cid):
        _save_clip(cid, status="done", external=True)
    _save_clip(cid, **_track_fields(start))
    _WAKE.set()
    ensure_worker()
    return {"ok": True}


def ensure_worker() -> None:
    global _worker_started
    with _LOCK:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_worker, daemon=True, name="clip-pipeline").start()


def retry(cid: str) -> dict:
    rec = clip_record(cid)
    if not rec:
        return {"ok": False, "error": f"нет ролика {cid}"}
    _save_clip(cid, status="queued", error="")
    _WAKE.set()
    ensure_worker()
    return {"ok": True}


def clips_overview(tracker_ready) -> list[dict]:
    doc = _clips()["clips"]
    ids = set(doc)
    brain = PROJECT / "output" / "p10"
    if brain.exists():
        ids |= {p.stem.removeprefix("brain_") for p in brain.glob("brain_VID*.npz")}
    out = []
    for cid in sorted(ids, reverse=True):
        rec = dict(doc.get(cid) or {"id": cid, "status": "done" if tracker_ready(cid) else "external"})
        if rec.get("status") == "done":
            rec.pop("log", None)
        step = rec.get("step")
        rec["steps"] = [{"id": s, "title": t, "done": _done(cid, s)} for s, t in STEPS]
        rec["step_title"] = dict(STEPS).get(step, "") if step else ""
        rec["step_progress"] = _step_progress(cid, step) if step else None
        rec["tracker_ready"] = bool(tracker_ready(cid))
        rec["has_video"] = (MEDIA / f"{cid}_fixed.mp4").exists()
        rec["title"] = clip_title(cid)
        if not rec.get("duration_s") and _duration(cid):
            rec["duration_s"] = round(_duration(cid), 1)
        out.append(rec)
    return out
