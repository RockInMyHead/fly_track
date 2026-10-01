"""P0.1R+ gate metrics — shared by controlled gate and regression fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SNR_THRESHOLD = 3.0
YAW_NEAR_ZERO = 0.15


def robust_snr(pre: np.ndarray, turn: np.ndarray) -> float:
    med_pre = float(np.median(pre))
    med_turn = float(np.median(turn))
    mad_pre = float(np.median(np.abs(pre - med_pre)))
    scale = 1.4826 * mad_pre + 1e-9
    return abs(med_turn - med_pre) / scale


def segment_median(rows: list[dict], t0: float, t1: float, key: str = "yaw_signal") -> float | None:
    vals = [float(r[key]) for r in rows if t0 <= r["time"] < t1]
    if not vals:
        return None
    return float(np.median(vals))


@dataclass(frozen=True)
class TurnEval:
    label: str
    turn_sign: float
    pre_window: tuple[float, float]
    turn_window: tuple[float, float]
    post_window: tuple[float, float]
    snr: float
    yaw_pre: float
    yaw_turn: float
    yaw_post: float
    sign_ok: bool
    pre_near_zero: bool
    post_near_zero: bool
    fwd_cohesion_ok: bool
    pass_gate: bool

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "pre": self.pre_window,
            "turn": self.turn_window,
            "post": self.post_window,
            "snr_robust": round(self.snr, 3),
            "yaw_pre": self.yaw_pre,
            "yaw_turn": self.yaw_turn,
            "yaw_post": self.yaw_post,
            "sign_ok": self.sign_ok,
            "pre_near_zero": self.pre_near_zero,
            "post_near_zero": self.post_near_zero,
            "fwd_cohesion_ok": self.fwd_cohesion_ok,
            "pass": self.pass_gate,
        }


def evaluate_turn(
    rows: list[dict],
    *,
    label: str,
    pre: tuple[float, float],
    turn: tuple[float, float],
    post: tuple[float, float],
    turn_sign: float,
    fwd_pre: float,
    fwd_post: float,
) -> TurnEval | None:
    pre_vals = np.array([float(r["yaw_signal"]) for r in rows if pre[0] <= r["time"] < pre[1]])
    turn_vals = np.array([float(r["yaw_signal"]) for r in rows if turn[0] <= r["time"] < turn[1]])
    post_vals = np.array([float(r["yaw_signal"]) for r in rows if post[0] <= r["time"] < post[1]])
    if len(pre_vals) < 5 or len(turn_vals) < 5 or len(post_vals) < 5:
        return None

    snr = robust_snr(pre_vals, turn_vals)
    yaw_pre = float(np.median(pre_vals))
    yaw_turn = float(np.median(turn_vals))
    yaw_post = float(np.median(post_vals))

    sign_ok = yaw_turn * turn_sign > 0
    pre_near_zero = abs(yaw_pre) < YAW_NEAR_ZERO
    post_near_zero = abs(yaw_post) < YAW_NEAR_ZERO
    snr_ok = snr >= SNR_THRESHOLD

    # Adjacent forward segments closer to each other than to this turn.
    fwd_cohesion_ok = (
        abs(fwd_pre - fwd_post) < abs(fwd_pre - yaw_turn)
        and abs(fwd_pre - fwd_post) < abs(fwd_post - yaw_turn)
    )

    pass_gate = sign_ok and snr_ok and pre_near_zero and post_near_zero and fwd_cohesion_ok

    return TurnEval(
        label=label,
        turn_sign=turn_sign,
        pre_window=pre,
        turn_window=turn,
        post_window=post,
        snr=snr,
        yaw_pre=yaw_pre,
        yaw_turn=yaw_turn,
        yaw_post=yaw_post,
        sign_ok=sign_ok,
        pre_near_zero=pre_near_zero,
        post_near_zero=post_near_zero,
        fwd_cohesion_ok=fwd_cohesion_ok,
        pass_gate=pass_gate,
    )


def load_controlled_segments(path: Path | None = None) -> dict:
    p = path or Path(__file__).resolve().parents[1] / "data/p01r/controlled_segments.json"
    return json.loads(p.read_text(encoding="utf-8"))


def event_stats(rows: list[dict], start: float, end: float) -> dict | None:
    seg = [r for r in rows if start <= r["time"] < end]
    if len(seg) < 5:
        return None
    yaw = np.array([float(r.get("yaw_frozen", r.get("yaw_signal", 0))) for r in seg])
    return {
        "n": len(seg),
        "yaw_frozen_median": float(np.median(yaw)),
        "yaw_frozen_mad": float(np.median(np.abs(yaw - np.median(yaw)))),
        "yaw_frozen_mean": float(np.mean(yaw)),
        "coherence_L_median": float(np.median([float(r.get("coherence_L") or 0) for r in seg])),
        "coherence_R_median": float(np.median([float(r.get("coherence_R") or 0) for r in seg])),
        "expansion_median": float(np.median([float(r.get("expansion_signal") or 0) for r in seg])),
        "R_dx4_dt2_yaw_median": float(np.median([float(r.get("R_dx4_dt2_yaw") or 0) for r in seg])),
        "R_dx8_dt3_yaw_median": float(np.median([float(r.get("R_dx8_dt3_yaw") or 0) for r in seg])),
        "R_dx4_dt3_yaw_median": float(np.median([float(r.get("R_dx4_dt3_yaw") or 0) for r in seg])),
        **{f"oracle_dt{i}_median": float(np.median([float(r.get(f'oracle_dt{i}') or 0) for r in seg])) for i in range(1, 6)},
    }


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_event_benchmark(events_doc: dict, rows: list[dict]) -> dict:
    """Evaluate blind-labeled events against frozen yaw_frozen readout."""
    eval_events: list[dict] = []
    for ev in events_doc["events"]:
        stats = event_stats(rows, ev["start"], ev["end"])
        item = {**ev, "stats": stats}
        if stats and ev["label"] in ("LEFT_YAW", "RIGHT_YAW", "FWD"):
            med = stats["yaw_frozen_median"]
            if ev["label"] == "LEFT_YAW":
                item["sign_ok"] = med < 0
            elif ev["label"] == "RIGHT_YAW":
                item["sign_ok"] = med > 0
            else:
                item["sign_ok"] = abs(med) < YAW_NEAR_ZERO
        eval_events.append(item)

    left = [e for e in eval_events if e["label"] == "LEFT_YAW" and e.get("stats")]
    right = [e for e in eval_events if e["label"] == "RIGHT_YAW" and e.get("stats")]
    fwd = [e for e in eval_events if e["label"] == "FWD" and e.get("stats")]

    left_acc = sum(1 for e in left if e.get("sign_ok")) / len(left) if left else 0.0
    right_acc = sum(1 for e in right if e.get("sign_ok")) / len(right) if right else 0.0

    fwd_meds = [e["stats"]["yaw_frozen_median"] for e in fwd]
    left_meds = [e["stats"]["yaw_frozen_median"] for e in left]
    right_meds = [e["stats"]["yaw_frozen_median"] for e in right]

    def _pool(evlist: list[dict]) -> np.ndarray:
        vals: list[float] = []
        for e in evlist:
            vals.extend(
                float(r.get("yaw_frozen", r.get("yaw_signal", 0)))
                for r in rows
                if e["start"] <= r["time"] < e["end"]
            )
        return np.array(vals, dtype=np.float64)

    fwd_pool = _pool(fwd)
    left_pool = _pool(left)
    right_pool = _pool(right)

    snr_left = robust_snr(fwd_pool, left_pool) if len(fwd_pool) and len(left_pool) else 0.0
    snr_right = robust_snr(fwd_pool, right_pool) if len(fwd_pool) and len(right_pool) else 0.0

    med_fwd = float(np.median(fwd_meds)) if fwd_meds else 0.0
    med_left = float(np.median(left_meds)) if left_meds else 0.0
    med_right = float(np.median(right_meds)) if right_meds else 0.0
    ordering_ok = med_left < med_fwd < med_right

    fwd_abs_median = float(np.median(np.abs(fwd_meds))) if fwd_meds else 999.0

    # Diagnostic 3-class classifier — thresholds from FWD spread only (not frontend retune).
    fwd_scale = max(fwd_abs_median, 0.05)
    y_true, y_pred = [], []
    class_map = {"LEFT_YAW": "LEFT", "RIGHT_YAW": "RIGHT", "FWD": "FWD"}
    for e in eval_events:
        if e["label"] not in class_map or not e.get("stats"):
            continue
        y = class_map[e["label"]]
        med = e["stats"]["yaw_frozen_median"]
        if med < -fwd_scale:
            pred = "LEFT"
        elif med > fwd_scale:
            pred = "RIGHT"
        else:
            pred = "FWD"
        y_true.append(y)
        y_pred.append(pred)

    macro_f1 = _macro_f1(y_true, y_pred, ["LEFT", "FWD", "RIGHT"])

    gate = {
        "n_left": len(left),
        "n_right": len(right),
        "n_fwd": len(fwd),
        "left_sign_accuracy": round(left_acc, 3),
        "right_sign_accuracy": round(right_acc, 3),
        "fwd_abs_median": round(fwd_abs_median, 4),
        "snr_left_vs_fwd": round(snr_left, 3),
        "snr_right_vs_fwd": round(snr_right, 3),
        "median_ordering": {"left": med_left, "fwd": med_fwd, "right": med_right},
        "ordering_ok": ordering_ok,
        "macro_f1": round(macro_f1, 3),
        "classifier_threshold": fwd_scale,
    }

    min_events = 3
    pass_gate = (
        len(left) >= min_events
        and len(right) >= min_events
        and len(fwd) >= min_events
        and left_acc >= 0.80
        and right_acc >= 0.80
        and fwd_abs_median < YAW_NEAR_ZERO
        and snr_left >= SNR_THRESHOLD
        and snr_right >= SNR_THRESHOLD
        and ordering_ok
        and macro_f1 >= 0.80
    )
    gate["pass"] = pass_gate

    return {"events": eval_events, "gate": gate}


def evaluate_controlled_clip(rows: list[dict], seg: dict) -> dict:
    s = seg["segments"]
    fwd1 = segment_median(rows, *s["fwd_1"])
    fwd2 = segment_median(rows, *s["fwd_2"])
    fwd3 = segment_median(rows, *s["fwd_3"])
    if fwd1 is None or fwd2 is None or fwd3 is None:
        return {"pass": False, "reason": "empty forward segment"}

    left = evaluate_turn(
        rows,
        label="LEFT",
        pre=tuple(s["fwd_1"]),
        turn=tuple(s["left_turn"]),
        post=tuple(s["fwd_2"]),
        turn_sign=-1.0,
        fwd_pre=fwd1,
        fwd_post=fwd2,
    )
    right = evaluate_turn(
        rows,
        label="RIGHT",
        pre=tuple(s["fwd_2"]),
        turn=tuple(s["right_turn"]),
        post=tuple(s["fwd_3"]),
        turn_sign=+1.0,
        fwd_pre=fwd2,
        fwd_post=fwd3,
    )

    fwd_spread = max(abs(fwd1 - fwd2), abs(fwd2 - fwd3), abs(fwd1 - fwd3))
    turn_vals = [left.yaw_turn if left else 999, right.yaw_turn if right else -999]
    global_fwd_ok = all(
        abs(f - t) > fwd_spread for f in (fwd1, fwd2, fwd3) for t in turn_vals if left and right
    )

    left_pass = left.pass_gate if left else False
    right_pass = right.pass_gate if right else False

    return {
        "fwd_medians": {"fwd_1": fwd1, "fwd_2": fwd2, "fwd_3": fwd3},
        "fwd_spread": round(fwd_spread, 4),
        "global_fwd_separated": global_fwd_ok,
        "left": left.as_dict() if left else {"pass": False, "reason": "empty left segment"},
        "right": right.as_dict() if right else {"pass": False, "reason": "empty right segment"},
        "pass": left_pass and right_pass and global_fwd_ok,
    }
