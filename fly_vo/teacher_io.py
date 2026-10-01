"""Load/save user-drawn teacher trajectories per video."""

from __future__ import annotations

import json
from pathlib import Path

from .dopamine import video_key_from_path


def teachers_dir(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    d = root / "data" / "teachers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def teacher_path_for(video: str | Path, root: Path | None = None) -> Path:
    vk = video_key_from_path(video)
    return teachers_dir(root) / f"{vk}.json"


def load_teacher(
    video: str | Path,
    root: Path | None = None,
) -> list[tuple[float, float]] | None:
    rec = load_teacher_record(video, root)
    if rec is None:
        return None
    pts, _src = rec
    return pts


def load_teacher_record(
    video: str | Path,
    root: Path | None = None,
) -> tuple[list[tuple[float, float]], str] | None:
    path = teacher_path_for(video, root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pts = data.get("points") or data.get("correction") or []
        out = [(float(p["x"]), float(p["y"])) for p in pts if "x" in p and "y" in p]
        if len(out) < 2:
            return None
        return out, str(data.get("source") or "unknown")
    except Exception:
        return None


def save_teacher(
    video: str | Path,
    points: list[tuple[float, float]],
    *,
    source: str = "user",
    root: Path | None = None,
    meta: dict | None = None,
) -> Path:
    path = teacher_path_for(video, root)
    payload = {
        "video_key": video_key_from_path(video),
        "source": source,
        "points": [{"x": float(x), "y": float(y)} for x, y in points],
    }
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
