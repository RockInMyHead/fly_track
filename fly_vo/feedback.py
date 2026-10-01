"""Interactive feedback loop — fly draws, human judges, params adapt."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .dopamine import (
    AdaptiveParams,
    dopamine_optimize,
    save_adaptive,
    video_key_from_path,
)
from .teacher_io import save_teacher

if TYPE_CHECKING:
    from .streaming import StreamingMaleCNS

Verdict = Literal["correct", "wrong"]

# Quick fixes when user picks issue tags without drawing a full correction.
ISSUE_DELTAS: dict[str, dict[str, float]] = {
    "no_turn": {"yaw_gain_scale": 1.28, "turn_ratio_boost": 0.12},
    "weak_turn": {"yaw_gain_scale": 1.18, "turn_ratio_boost": 0.08},
    "wrong_direction": {"heading_offset_deg": 12.0, "yaw_rate_bias": 0.012},
    "too_slow": {"speed_scale": 1.18},
    "too_fast": {"speed_scale": 0.82},
    "drift_left": {"lateral_bias": -0.12},
    "drift_right": {"lateral_bias": 0.12},
}


@dataclass
class FeedbackRecord:
    ts: str
    verdict: Verdict
    issues: list[str] = field(default_factory=list)
    note: str = ""
    loss_before: float | None = None
    loss_after: float | None = None
    video_id: str = ""
    fly_points: int = 0
    had_correction: bool = False


@dataclass
class FeedbackState:
    correct: int = 0
    wrong: int = 0
    records: list[FeedbackRecord] = field(default_factory=list)


def default_log_path(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    return root / "data" / "feedback_log.json"


def load_feedback_state(path: Path | None = None) -> FeedbackState:
    path = path or default_log_path()
    if not path.exists():
        return FeedbackState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = [FeedbackRecord(**r) for r in raw.get("records", [])]
        return FeedbackState(
            correct=int(raw.get("correct", 0)),
            wrong=int(raw.get("wrong", 0)),
            records=records[-100:],
        )
    except Exception:
        return FeedbackState()


def save_feedback_state(state: FeedbackState, path: Path | None = None) -> None:
    path = path or default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "correct": state.correct,
        "wrong": state.wrong,
        "records": [asdict(r) for r in state.records[-100:]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_issue_tags(params: AdaptiveParams, issues: list[str]) -> AdaptiveParams:
    new = AdaptiveParams(**asdict(params))
    for issue in issues:
        deltas = ISSUE_DELTAS.get(issue, {})
        for key, delta in deltas.items():
            val = getattr(new, key)
            if key.endswith("_scale") and isinstance(delta, float) and delta > 0 and delta < 2:
                setattr(new, key, val * delta)
            else:
                setattr(new, key, val + delta)
    return new.clamp()


def reinforce_correct(params: AdaptiveParams) -> AdaptiveParams:
    """Small consolidation when the fly got it right."""
    new = AdaptiveParams(**asdict(params))
    new.sessions += 1
    new.last_loss = 0.0
    return new.clamp()


def apply_feedback(
    params: AdaptiveParams,
    verdict: Verdict,
    processor: "StreamingMaleCNS",
    video_path: str,
    fly_path: list[tuple[float, float]],
    *,
    correction: list[tuple[float, float]] | None = None,
    issues: list[str] | None = None,
    note: str = "",
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    x0: float = 0.0,
    y0: float = 0.0,
    heading_deg: float = 90.0,
    state_path: Path | None = None,
    adaptive_path: Path | None = None,
) -> dict:
    """
    Apply human feedback after the fly drew a trajectory.

    - correct → reinforce current params
    - wrong + correction path → dopamine optimize against correction
    - wrong + issue tags only → heuristic param nudges
    """
    issues = issues or []
    state = load_feedback_state(state_path)
    loss_before = loss_after = None
    fly_path_out: list[dict] | None = None
    message = ""

    if verdict == "correct":
        new_params = reinforce_correct(params)
        message = f"✓ Запомнила — сессия #{new_params.sessions}. Параметры закреплены."
        state.correct += 1
    else:
        new_params = AdaptiveParams(**asdict(params))
        if correction and len(correction) >= 2:
            cache = processor.collect_neural_cache(
                video_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            new_params, loss_before, loss_after, best_path, trials = dopamine_optimize(
                params,
                cache,
                processor,
                correction,
                x0=x0,
                y0=y0,
                heading_deg=heading_deg,
            )
            fly_path_out = [
                {"t": p.timestamp, "x": p.x, "y": p.y, "heading": p.heading_deg}
                for p in best_path
            ]
            message = (
                f"Исправила по твоему рисунку · {trials} вариантов · "
                f"loss {loss_before:.3f} → {loss_after:.3f} · "
                f"yaw×{new_params.yaw_gain_scale:.2f} speed×{new_params.speed_scale:.2f}"
            )
        elif issues:
            new_params = _apply_issue_tags(params, issues)
            new_params.sessions = params.sessions + 1
            labels = ", ".join(issues)
            message = f"Подстроила по меткам ({labels}) · сессия #{new_params.sessions}"
        else:
            raise ValueError("Для «неправильно» нарисуй исправление или выбери метки")

        state.wrong += 1

    vk = video_key_from_path(video_path)
    save_adaptive(new_params, adaptive_path, video_key=vk)
    if correction and len(correction) >= 2:
        save_teacher(video_path, correction, source="user")

    record = FeedbackRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        verdict=verdict,
        issues=issues,
        note=note or "",
        loss_before=loss_before,
        loss_after=loss_after,
        video_id=vk,
        fly_points=len(fly_path),
        had_correction=bool(correction and len(correction) >= 2),
    )
    state.records.append(record)
    save_feedback_state(state, state_path)

    old = asdict(params)
    new_d = asdict(new_params)
    param_delta = {
        k: round(new_d[k] - old[k], 4)
        for k in old
        if k in new_d and isinstance(old[k], (int, float)) and k != "sessions"
    }

    return {
        "message": message,
        "params": new_d,
        "param_delta": param_delta,
        "stats": {"correct": state.correct, "wrong": state.wrong},
        "fly_path_learned": fly_path_out,
        "loss_before": loss_before,
        "loss_after": loss_after,
    }
