"""Dopaminergic learning loop — adapt egomotion from user-drawn teacher trajectories."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .streaming import SegmentNeuralCache, StreamingFlyVO


@dataclass
class AdaptiveParams:
    """Learned corrections applied on top of FlyEgoMotionDecoder."""

    lateral_bias: float = 0.0
    speed_scale: float = 1.0
    yaw_gain_scale: float = 1.0
    turn_ratio_boost: float = 0.0
    heading_offset_deg: float = 0.0
    yaw_rate_bias: float = 0.0
    sessions: int = 0
    last_loss: float | None = None

    def clamp(self) -> AdaptiveParams:
        self.lateral_bias = float(np.clip(self.lateral_bias, -0.5, 0.5))
        self.speed_scale = float(np.clip(self.speed_scale, 0.35, 2.8))
        self.yaw_gain_scale = float(np.clip(self.yaw_gain_scale, 0.15, 2.5))
        self.turn_ratio_boost = float(np.clip(self.turn_ratio_boost, -0.35, 1.0))
        self.heading_offset_deg = float(np.clip(self.heading_offset_deg, -45.0, 45.0))
        self.yaw_rate_bias = float(np.clip(self.yaw_rate_bias, -0.06, 0.06))
        return self


@dataclass
class DopamineResult:
    loss_before: float
    loss_after: float
    params: AdaptiveParams
    fly_path: list[dict[str, float]]
    teacher_path: list[dict[str, float]]
    message: str
    trials: int = 0


def default_state_path(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    return root / "data" / "dopamine_state.json"


def video_key_from_path(path: str | Path) -> str:
    """Stable key for per-video learned params (filename stem)."""
    return Path(path).resolve().stem


def _params_from_dict(data: dict) -> AdaptiveParams:
    return AdaptiveParams(
        **{k: data[k] for k in asdict(AdaptiveParams()) if k in data}
    ).clamp()


def load_adaptive(path: Path | None = None, video_key: str | None = None) -> AdaptiveParams:
    path = path or default_state_path()
    if not path.exists():
        return AdaptiveParams()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if video_key:
            by_video = data.get("by_video")
            if isinstance(by_video, dict) and video_key in by_video:
                return _params_from_dict(by_video[video_key])
        return _params_from_dict(data)
    except Exception:
        return AdaptiveParams()


def save_adaptive(
    params: AdaptiveParams,
    path: Path | None = None,
    video_key: str | None = None,
) -> None:
    path = path or default_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    clamped = params.clamp()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if video_key:
        by_video = data.setdefault("by_video", {})
        if not isinstance(by_video, dict):
            by_video = {}
            data["by_video"] = by_video
        by_video[video_key] = asdict(clamped)
    else:
        data.update(asdict(clamped))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resample_polyline(points: np.ndarray, n: int = 64) -> np.ndarray:
    """Arc-length resample (N, 2) polyline to n vertices."""
    if len(points) < 2:
        return points.copy()
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-8:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        w = 0.0 if seg_len < 1e-8 else (t - cum[j]) / seg_len
        out[i] = (1 - w) * points[j] + w * points[j + 1]
    return out


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _curvature(points: np.ndarray) -> float:
    """Mean absolute turning angle per step (radians)."""
    if len(points) < 3:
        return 0.0
    v0 = np.diff(points, axis=0)
    angles = []
    for i in range(len(v0) - 1):
        a, b = v0[i], v0[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            continue
        cos = np.clip(float(a @ b) / (na * nb), -1.0, 1.0)
        angles.append(abs(np.arccos(cos)))
    return float(np.mean(angles)) if angles else 0.0


def _align_similarity(fly: np.ndarray, teach: np.ndarray) -> tuple[np.ndarray, float]:
    """Translate to origin, optimal rotation + uniform scale: fly -> teach frame."""
    fly = fly - fly[0]
    teach = teach - teach[0]
    n = min(len(fly), len(teach))
    fly = _resample_polyline(fly, n)
    teach = _resample_polyline(teach, n)

    fly_c = fly - fly.mean(axis=0)
    teach_c = teach - teach.mean(axis=0)
    fly_scale = np.sqrt((fly_c**2).sum()) + 1e-8
    teach_scale = np.sqrt((teach_c**2).sum()) + 1e-8
    fly_n = fly_c / fly_scale
    teach_n = teach_c / teach_scale

    H = fly_n.T @ teach_n
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    fly_aligned = ((fly_n @ R) * teach_scale) + teach.mean(axis=0)
    scale_ratio = teach_scale / fly_scale
    return fly_aligned, scale_ratio


def split_polyline(
    points: np.ndarray | list[tuple[float, float]],
    frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a polyline by arc length. Returns (head, tail) sharing the cut vertex."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        return pts, pts
    dense = _resample_polyline(pts, max(16, len(pts) * 4))
    n = len(dense)
    k = int(round(n * float(np.clip(frac, 0.5, 0.85))))
    k = min(max(k, 4), n - 3)
    return dense[: k + 1], dense[k:]


def compare_trajectories(
    fly_xy: np.ndarray | list,
    teach_xy: np.ndarray | list,
    n: int = 64,
) -> dict:
    """Score how well the fly path matches the teacher in world coordinates."""
    fly = np.asarray(fly_xy, dtype=np.float64)
    teach = np.asarray(teach_xy, dtype=np.float64)
    empty = {
        "rmse": float("inf"),
        "rmse_aligned": float("inf"),
        "endpoint_m": float("inf"),
        "heading_err_deg": 0.0,
        "length_ratio": 0.0,
        "overlap_pct": 0.0,
        "teach_length": 0.0,
        "fly_length": 0.0,
        "pairs": [],
    }
    if len(fly) < 2 or len(teach) < 2:
        return empty

    fly_r = _resample_polyline(fly, n)
    teach_r = _resample_polyline(teach, n)
    err = fly_r - teach_r
    rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    teach_len = _path_length(teach)
    fly_len = _path_length(fly)
    endpoint = float(np.linalg.norm(fly[-1] - teach[-1]))
    heading_err = float(np.degrees(_heading_change(fly_r) - _heading_change(teach_r)))
    overlap = float(100.0 * np.exp(-rmse / (0.18 * max(teach_len, 1e-6))))
    pairs = [
        {"fx": float(a[0]), "fy": float(a[1]), "tx": float(b[0]), "ty": float(b[1])}
        for a, b in zip(fly_r[::8], teach_r[::8])
    ]
    return {
        "rmse": round(rmse, 4),
        "rmse_aligned": round(path_loss(fly, teach), 4),
        "endpoint_m": round(endpoint, 4),
        "heading_err_deg": round(heading_err, 2),
        "length_ratio": round(fly_len / (teach_len + 1e-8), 3),
        "overlap_pct": round(float(np.clip(overlap, 0.0, 100.0)), 1),
        "teach_length": round(teach_len, 3),
        "fly_length": round(fly_len, 3),
        "pairs": pairs,
    }


def path_loss(fly_xy: np.ndarray, teach_xy: np.ndarray) -> float:
    """RMSE after similarity alignment (scale + rotation + translation)."""
    fly = np.asarray(fly_xy, dtype=np.float64)
    teach = np.asarray(teach_xy, dtype=np.float64)
    if len(fly) < 2 or len(teach) < 2:
        return float("inf")
    fly_a, _ = _align_similarity(fly, teach)
    teach_r = _resample_polyline(teach - teach[0], len(fly_a))
    teach_r = teach_r - teach_r[0] + fly_a[0]
    n = min(len(fly_a), len(teach_r))
    err = fly_a[:n] - teach_r[:n]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def _heading_change(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    v0 = points[1] - points[0]
    v1 = points[-1] - points[-2]
    a0 = np.arctan2(v0[1], v0[0])
    a1 = np.arctan2(v1[1], v1[0])
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    return float(d)


def _heuristic_params(
    params: AdaptiveParams,
    fly: np.ndarray,
    teach: np.ndarray,
    lr: float = 0.75,
) -> AdaptiveParams:
    """Strong error-driven update from aligned paths."""
    fly_a, scale_ratio = _align_similarity(fly, teach)
    teach_r = _resample_polyline(teach - teach[0], len(fly_a))
    teach_r = teach_r - teach_r[0]
    err = fly_a - teach_r
    mean_lat = float(np.mean(err[:, 0]))
    end_lat = float(err[-1, 0])
    heading_err = _heading_change(fly_a) - _heading_change(teach_r)
    teach_curv = _curvature(teach_r)
    fly_curv = _curvature(fly_a)

    new = AdaptiveParams(**asdict(params))
    new.lateral_bias = (1 - lr) * new.lateral_bias + lr * (
        new.lateral_bias + np.clip(mean_lat * 0.18 + end_lat * 0.08, -0.25, 0.25)
    )
    new.speed_scale = (1 - lr) * new.speed_scale + lr * (new.speed_scale * scale_ratio)
    yaw_target = max(0.2, 1.0 - abs(heading_err) * 2.5)
    new.yaw_gain_scale = (1 - lr) * new.yaw_gain_scale + lr * (new.yaw_gain_scale * yaw_target)
    new.heading_offset_deg = (1 - lr) * new.heading_offset_deg + lr * (
        new.heading_offset_deg - np.degrees(heading_err) * 0.45
    )
    new.yaw_rate_bias = (1 - lr) * new.yaw_rate_bias + lr * (
        new.yaw_rate_bias - heading_err * 0.04
    )
    if teach_curv < 0.07 and fly_curv > teach_curv + 0.03:
        new.turn_ratio_boost = (1 - lr) * new.turn_ratio_boost + lr * (new.turn_ratio_boost + 0.15)
        new.yaw_gain_scale *= 0.88
    return new.clamp()


def _perturb_params(base: AdaptiveParams, scale: float, rng: random.Random) -> AdaptiveParams:
    p = AdaptiveParams(**asdict(base))
    p.lateral_bias += rng.uniform(-scale, scale) * 0.35
    p.speed_scale *= 1.0 + rng.uniform(-scale, scale) * 0.45
    p.yaw_gain_scale *= 1.0 + rng.uniform(-scale, scale) * 0.55
    p.turn_ratio_boost += rng.uniform(-scale, scale) * 0.25
    p.heading_offset_deg += rng.uniform(-scale, scale) * 18.0
    p.yaw_rate_bias += rng.uniform(-scale, scale) * 0.025
    return p.clamp()


def dopamine_optimize(
    params: AdaptiveParams,
    cache: "SegmentNeuralCache",
    processor: "StreamingFlyVO",
    teacher_points: list[tuple[float, float]],
    x0: float,
    y0: float,
    heading_deg: float,
    n_random: int = 96,
) -> tuple[AdaptiveParams, float, float, list, int]:
    """
    Search adaptive params using cached MaleCNS steps (no re-simulation).

    Returns best params, loss_before, loss_after, best path (t,x,y,heading), trials.
    """
    teach = np.asarray(teacher_points, dtype=np.float64)

    def eval_params(p: AdaptiveParams) -> tuple[float, list[tuple[float, float, float, float]]]:
        path = processor.integrate_from_cache(
            cache, p, x0=x0, y0=y0, heading_deg=heading_deg, emit_stride=4
        )
        xy = np.asarray([(pt.x, pt.y) for pt in path], dtype=np.float64)
        return path_loss(xy, teach), path

    loss_before, path_before = eval_params(params)
    best_params = params
    best_loss = loss_before
    best_path = path_before

    heuristic = _heuristic_params(params, np.asarray([(p.x, p.y) for p in path_before]), teach)
    candidates: list[AdaptiveParams] = [params, heuristic]

    rng = random.Random(42 + params.sessions)
    for scale in (0.35, 0.55, 0.85):
        for _ in range(n_random // 3):
            candidates.append(_perturb_params(heuristic, scale, rng))

    # Coarse grid around heuristic for straight-corridor tuning
    for lat in (-0.15, 0.0, 0.15):
        for spd in (0.75, 0.9, 1.05, 1.2):
            for yaw in (0.4, 0.7, 1.0):
                for hd in (-12.0, 0.0, 12.0):
                    c = AdaptiveParams(**asdict(heuristic))
                    c.lateral_bias += lat
                    c.speed_scale *= spd
                    c.yaw_gain_scale *= yaw
                    c.heading_offset_deg += hd
                    candidates.append(c.clamp())

    trials = 0
    for cand in candidates:
        loss, path = eval_params(cand)
        trials += 1
        if loss < best_loss:
            best_loss, best_params, best_path = loss, cand, path

    # When random search stalls, blend toward heuristic (teacher-aligned nudge).
    if best_loss >= loss_before - 1e-4:
        h_loss, h_path = eval_params(heuristic)
        trials += 1
        if h_loss < best_loss:
            best_loss, best_params, best_path = h_loss, heuristic, h_path
        else:
            blended = AdaptiveParams(**asdict(params))
            for key in (
                "lateral_bias",
                "speed_scale",
                "yaw_gain_scale",
                "turn_ratio_boost",
                "heading_offset_deg",
                "yaw_rate_bias",
            ):
                cur = getattr(params, key)
                tgt = getattr(heuristic, key)
                setattr(blended, key, 0.55 * cur + 0.45 * tgt)
            blended = blended.clamp()
            b_loss, b_path = eval_params(blended)
            trials += 1
            if b_loss < best_loss:
                best_loss, best_params, best_path = b_loss, blended, b_path

    best_params.sessions = params.sessions + 1
    best_params.last_loss = best_loss
    return best_params, loss_before, best_loss, best_path, trials


def dopamine_learn(
    params: AdaptiveParams,
    fly_points: list[tuple[float, float]],
    teacher_points: list[tuple[float, float]],
    lr: float = 0.75,
) -> tuple[AdaptiveParams, DopamineResult]:
    """Legacy single-step heuristic (used when no neural cache available)."""
    fly = np.asarray(fly_points, dtype=np.float64)
    teach = np.asarray(teacher_points, dtype=np.float64)
    if len(fly) < 2 or len(teach) < 2:
        raise ValueError("Need at least 2 points in fly and teacher paths")

    loss_before = path_loss(fly, teach)
    new = _heuristic_params(params, fly, teach, lr=lr)
    new.sessions = params.sessions + 1
    fly_a, scale_ratio = _align_similarity(fly, teach)
    heading_err = _heading_change(fly_a) - _heading_change(
        _resample_polyline(teach - teach[0], len(fly_a))
    )
    msg = (
        f"сессия #{new.sessions} · loss {loss_before:.3f} · "
        f"Δheading {np.degrees(heading_err):+.1f}° · scale×{scale_ratio:.2f}"
    )
    return new, DopamineResult(
        loss_before=loss_before,
        loss_after=loss_before,
        params=new,
        fly_path=[{"x": float(x), "y": float(y)} for x, y in fly],
        teacher_path=[{"x": float(x), "y": float(y)} for x, y in teach],
        message=msg,
    )
