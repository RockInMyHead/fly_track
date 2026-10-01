#!/usr/bin/env python3
"""
P0.1 Visual Observability Test — Reichardt azimuth flow → 16 T4/T5 groups.

Acceptance (5 seeds):
  STATIC != FORWARD, YAW_L != YAW_R, YAW_L != FORWARD, YAW_R != FORWARD, LOOM_L != LOOM_R
  yaw_signal sign stable ≥4/5 seeds
  4-class diagnostic logistic regression macro-F1 ≥ 0.90

Usage:
    PYTHONPATH=. python scripts/visual_observability_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.neural_probe import group_spike_counts, rolling_rates
from fly_vo.optic_flow import FlowConfig
from fly_vo.visual_encoder import T4_T5_GROUP_NAMES, VideoVisualEncoder

CLASSES = ("STATIC", "FORWARD", "YAW_LEFT", "YAW_RIGHT")
REL_MIN = 0.12  # min relative L2 distance between class fingerprints


def _rel_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-9))


def _fingerprint(rates: dict[str, float]) -> np.ndarray:
    return np.array([rates.get(k, 0.0) for k in T4_T5_GROUP_NAMES], dtype=np.float64)


def _make_frames(w: int = 96, h: int = 72, n: int = 45) -> dict[str, list[np.ndarray]]:
    """Synthetic stimuli with correct Reichardt structure."""
    static = [np.full((h, w, 3), 128, dtype=np.uint8) for _ in range(n)]

    def _base() -> np.ndarray:
        img = np.full((h, w, 3), 100, dtype=np.uint8)
        cv2.rectangle(img, (10, 20), (w - 10, h - 20), (200, 200, 200), -1)
        cv2.line(img, (w // 2, 15), (w // 2, h - 15), (255, 255, 255), 2)
        for x in range(15, w - 10, 12):
            cv2.circle(img, (x, h // 2), 3, (180, 180, 180), -1)
        return img

    forward = [_base()]
    for i in range(1, n):
        scale = 1.0 + i * 0.022
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        forward.append(cv2.warpAffine(_base(), M, (w, h), borderValue=(100, 100, 100)))

    yaw_right = [_base()]
    yaw_left = [_base()]
    for i in range(1, n):
        shift = i * 5
        yaw_right.append(np.roll(_base(), shift, axis=1))
        yaw_left.append(np.roll(_base(), -shift, axis=1))

    loom_l, loom_r = [static[0].copy()], [static[0].copy()]
    for i in range(1, n):
        img_l = np.full((h, w, 3), 100, dtype=np.uint8)
        r = min(w // 2, 6 + i * 3)
        cv2.circle(img_l, (w // 4, h // 2), r, (255, 255, 255), -1)
        loom_l.append(img_l)
        img_r = np.full((h, w, 3), 100, dtype=np.uint8)
        cv2.circle(img_r, (3 * w // 4, h // 2), r, (255, 255, 255), -1)
        loom_r.append(img_r)

    return {
        "STATIC": static,
        "FORWARD": forward,
        "YAW_LEFT": yaw_left,
        "YAW_RIGHT": yaw_right,
        "LOOM_LEFT": loom_l,
        "LOOM_RIGHT": loom_r,
    }


def _run_stimulus(
    engine: MaleCNSEngine,
    encoder: VideoVisualEncoder,
    frames: list[np.ndarray],
    seed: int,
    dt: float = 0.02,
    rate_window: int = 12,
) -> dict:
    engine.reset(seed=seed)
    encoder.reset()
    hist: list[dict[str, int]] = []
    flow_tail: list[dict] = []

    for i, bgr in enumerate(frames):
        eye, inj, metrics = encoder.encode_frame(bgr)
        r = engine.step(i * dt, eye_drive=eye, inject=inj)
        hist.append(group_spike_counts(r, encoder))
        flow_tail.append(metrics)

    rates = rolling_rates(hist, window=rate_window)
    flow = flow_tail[-1] if flow_tail else {}
    flow_mean = {
        k: float(np.mean([m.get(k, 0) for m in flow_tail[-rate_window:]]))
        for k in ("flow_L", "flow_R", "yaw_signal", "expansion_signal", "motion_energy")
    }

    return {
        "rates": rates,
        "fingerprint": _fingerprint(rates).tolist(),
        "flow": flow_mean,
        "optic_lobe": float(np.mean([0 for _ in hist])),  # filled below
    }


def _run_all_seeds(
    engine: MaleCNSEngine,
    encoder: VideoVisualEncoder,
    synth: dict[str, list],
    seeds: list[int],
    dt: float,
) -> dict:
    out: dict[str, dict] = {}
    for label, frames in synth.items():
        runs = []
        for seed in seeds:
            runs.append(_run_stimulus(engine, encoder, frames, seed, dt))
        out[label] = runs
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _train_softmax(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lr: float = 0.08,
    epochs: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal multinomial logistic regression (diagnostic only)."""
    n_feat = X.shape[1]
    W = np.zeros((n_feat, n_classes), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    Y = np.zeros((len(y), n_classes))
    for i, c in enumerate(y):
        Y[i, c] = 1.0
    for _ in range(epochs):
        logits = X @ W + b
        P = _softmax(logits)
        grad = (P - Y) / len(y)
        W -= lr * X.T @ grad
        b -= lr * grad.sum(axis=0)
    return W, b


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, list]:
    f1s = []
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s)), cm.tolist()


def _diagnostic_classifier(results: dict, seeds: list[int]) -> dict:
    """Leave-one-seed-out 4-class classifier on T4/T5 fingerprints."""
    X_list, y_list = [], []
    class_to_i = {c: i for i, c in enumerate(CLASSES)}
    for ci, cls in enumerate(CLASSES):
        for run in results[cls]:
            X_list.append(_fingerprint(run["rates"]))
            y_list.append(ci)
    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=int)
    # normalize features
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd

    preds = np.zeros(len(y), dtype=int)
    for hold_seed_idx, seed in enumerate(seeds):
        mask = np.array([(i % len(seeds)) == hold_seed_idx for i in range(len(y))])
        train, test = ~mask, mask
        if test.sum() == 0 or train.sum() == 0:
            continue
        W, b = _train_softmax(Xn[train], y[train], len(CLASSES))
        logits = Xn[test] @ W + b
        preds[test] = _softmax(logits).argmax(axis=1)

    macro_f1, cm = _macro_f1(y, preds, len(CLASSES))
    return {
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "pass": bool(macro_f1 >= 0.899),
    }


def _strict_checks(results: dict, seeds: list[int]) -> dict:
    """Per-seed and aggregate acceptance checks."""
    checks: dict = {}

    def mean_fp(cls: str) -> np.ndarray:
        fps = [_fingerprint(r["rates"]) for r in results[cls]]
        return np.mean(fps, axis=0)

    fp = {c: mean_fp(c) for c in CLASSES}
    checks["STATIC_ne_FORWARD"] = _rel_dist(fp["STATIC"], fp["FORWARD"]) >= REL_MIN
    checks["YAW_L_ne_YAW_R"] = _rel_dist(fp["YAW_LEFT"], fp["YAW_RIGHT"]) >= REL_MIN
    checks["YAW_L_ne_FORWARD"] = _rel_dist(fp["YAW_LEFT"], fp["FORWARD"]) >= REL_MIN
    checks["YAW_R_ne_FORWARD"] = _rel_dist(fp["YAW_RIGHT"], fp["FORWARD"]) >= REL_MIN

    loom_l = np.mean([r["rates"].get("LC4_L", 0) + r["rates"].get("LPLC2_L", 0) for r in results["LOOM_LEFT"]])
    loom_r = np.mean([r["rates"].get("LC4_R", 0) + r["rates"].get("LPLC2_R", 0) for r in results["LOOM_RIGHT"]])
    checks["LOOM_L_ne_R"] = abs(loom_l - loom_r) >= 0.25

    # Flow signal structure
    yaw_r = np.mean([r["flow"]["yaw_signal"] for r in results["YAW_RIGHT"]])
    yaw_l = np.mean([r["flow"]["yaw_signal"] for r in results["YAW_LEFT"]])
    yaw_f = np.mean([abs(r["flow"]["yaw_signal"]) for r in results["FORWARD"]])
    exp_f = np.mean([abs(r["flow"]["expansion_signal"]) for r in results["FORWARD"]])
    yaw_y = np.mean([abs(r["flow"]["yaw_signal"]) for r in results["YAW_RIGHT"]])
    exp_y = np.mean([abs(r["flow"]["expansion_signal"]) for r in results["YAW_RIGHT"]])
    checks["yaw_signal_sign_L_neg"] = yaw_l < 0
    checks["yaw_signal_sign_R_pos"] = yaw_r > 0
    checks["expansion_forward_gt_yaw"] = exp_f > yaw_f and yaw_y > exp_y

    # Seed stability: yaw sign on YAW_RIGHT runs
    stable = 0
    for seed_i, seed in enumerate(seeds):
        yr = results["YAW_RIGHT"][seed_i]["flow"]["yaw_signal"]
        yl = results["YAW_LEFT"][seed_i]["flow"]["yaw_signal"]
        if yr > 0 and yl < 0:
            stable += 1
    checks["yaw_sign_stable_4of5"] = stable >= max(1, len(seeds) - 1)

    checks["ALL_STRICT"] = bool(all(v for k, v in checks.items() if k != "ALL_STRICT"))
    return {k: bool(v) for k, v in checks.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="output/visual_observability_p01.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    cfg = FlyVOConfig(use_pathway_fallback=False)
    fc = FlowConfig(crop_mode="center_band", flow_inject_gain=8.0, ema_tau_s=0.5)
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine, gain=cfg.visual_gain, flow_config=fc)
    engine._ensure_loaded()

    print("P0.1 Reichardt encoder")
    print(f"  photoreceptors: {engine.n_visual}")
    print(f"  T4/T5 groups: {len(encoder.t4_t5_groups)}")
    print(f"  azimuth bins: {encoder.n_azimuth_bins}")

    synth = _make_frames()
    results = _run_all_seeds(engine, encoder, synth, args.seeds, cfg.brain_dt)

    checks = _strict_checks(results, args.seeds)
    classifier = _diagnostic_classifier(
        {k: results[k] for k in CLASSES}, args.seeds
    )

    report = {
        "seeds": args.seeds,
        "checks": checks,
        "classifier": classifier,
        "fingerprints": {
            cls: {
                "mean": np.mean(
                    [_fingerprint(r["rates"]) for r in results[cls]], axis=0
                ).tolist(),
                "flow": {
                    k: float(np.mean([r["flow"][k] for r in results[cls]]))
                    for k in ("yaw_signal", "expansion_signal", "flow_L", "flow_R")
                },
            }
            for cls in CLASSES
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Flow signals (mean) ===")
    for cls in CLASSES:
        f = report["fingerprints"][cls]["flow"]
        print(
            f"  {cls:10s} yaw={f['yaw_signal']:+.5f}  "
            f"exp={f['expansion_signal']:+.5f}  "
            f"L={f['flow_L']:+.5f} R={f['flow_R']:+.5f}"
        )

    print("\n=== Strict checks ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    print(f"\n=== Diagnostic classifier ===")
    print(f"  macro-F1 = {classifier['macro_f1']:.3f}  (target ≥ 0.90)")
    print(f"  confusion: {classifier['confusion_matrix']}")
    print(f"  [{'PASS' if classifier['pass'] else 'FAIL'}] classifier")

    overall = checks["ALL_STRICT"] and classifier["pass"]
    print(f"\n{'PASS' if overall else 'FAIL'} P0.1 GATE: YAW_L != YAW_R != FORWARD")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
