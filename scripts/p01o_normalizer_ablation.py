#!/usr/bin/env python3
"""
P0.1O — does the per-bin division turn a usable signal into noise?

One change only: the temporal normaliser. Everything else is untouched —
same videos, same windows, same 128 azimuth bins, same detector bank, same
dx/dt set, same frozen readout, same crop. `optic_flow.py` is NOT modified:
the normaliser is swapped in at runtime on the frontend instance.

Variants
--------
A  current:   (pan - mu) / (dev + eps), clip +-3      dev is per-bin
B  mean only: (pan - mu)                              no division at all
C  global:    (pan - mu) / (median(dev) + eps)        one scalar for the frame

mu and dev are computed identically in all three, exactly as TemporalNormalizer
does it, so the only thing that differs is the divisor.

Reported per variant per event
------------------------------
  1. sign of the pooled readout, vs the blind label
  2. per-bin temporal sign stability (with the coin-flip chance level)
  3. signal-to-noise of the field and of the readout
  4. adjacent-bin spatial coherence

Usage:
    PYTHONPATH=. python scripts/p01o_normalizer_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import EPS, DirectionalMotionFrontend, FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

EVENTS = {
    "left_65":   (63.5, -1),
    "left_81":   (78.5, -1),
    "left_108":  (106.0, -1),
    "right_67":  (66.0, +1),
    "right_210": (209.0, +1),
    "right_216": (214.0, +1),
}
# label here is the sign the pooled yaw_frozen must take (-1 = LEFT, +1 = RIGHT)

PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
VARIANTS = ("A_per_bin", "B_mean_only", "C_global")


class _Base:
    """Shared EMA bookkeeping, identical to TemporalNormalizer."""

    def __init__(self, n_bins: int, alpha: float):
        self.n_bins = n_bins
        self.alpha = alpha
        self._mu: np.ndarray | None = None
        self._dev: np.ndarray | None = None

    def reset(self) -> None:
        self._mu = None
        self._dev = None

    def _update(self, pan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pan = pan.astype(np.float32)
        a = self.alpha
        if self._mu is None:
            self._mu = pan.copy()
            self._dev = np.full_like(pan, 0.05, dtype=np.float32)
        else:
            self._mu = (1.0 - a) * self._mu + a * pan
            self._dev = (1.0 - a) * self._dev + a * np.abs(pan - self._mu)
        return pan, self._dev


class PerBinNormalizer(_Base):
    """Variant A — exactly the current implementation."""

    def normalize(self, pan: np.ndarray) -> np.ndarray:
        pan, dev = self._update(pan)
        out = (pan - self._mu) / (dev + EPS)
        return np.clip(out, -3.0, 3.0).astype(np.float32)


class MeanOnlyNormalizer(_Base):
    """Variant B — subtract the running mean, no division."""

    def normalize(self, pan: np.ndarray) -> np.ndarray:
        pan, _dev = self._update(pan)
        return (pan - self._mu).astype(np.float32)


class GlobalScaleNormalizer(_Base):
    """Variant C — one scalar divisor shared by all bins."""

    def normalize(self, pan: np.ndarray) -> np.ndarray:
        pan, dev = self._update(pan)
        g = float(np.median(dev))
        out = (pan - self._mu) / (g + EPS)
        return np.clip(out, -3.0, 3.0).astype(np.float32)


NORMALIZERS = {
    "A_per_bin": PerBinNormalizer,
    "B_mean_only": MeanOnlyNormalizer,
    "C_global": GlobalScaleNormalizer,
}


def chance_consistency(n_mat: np.ndarray, n_pos: np.ndarray, n_neg: np.ndarray, trials: int = 200) -> float:
    """Median per-bin sign consistency expected if each bin's sign were a fair coin."""
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(trials):
        flip = rng.random(n_mat.shape[1]) < 0.5
        p = np.where(flip, n_pos, n_neg)
        q = np.where(flip, n_neg, n_pos)
        t = np.maximum(p + q, 1)
        vals.append(float(np.nanmedian(np.maximum(p, q) / t)))
    return float(np.mean(vals))


def run_variant(video: Path, cfg: FlyVOConfig, variant: str) -> dict:
    fc = FlowConfig(crop_mode="center_band", ema_tau_s=cfg.ema_tau_s, brain_dt=cfg.brain_dt)
    n_pre = int(round(PRE_ROLL_S / cfg.brain_dt))
    out: dict[str, dict] = {}

    for eid, (start, lab) in EVENTS.items():
        front = DirectionalMotionFrontend(fc)
        alpha = float(np.clip(cfg.brain_dt / max(fc.ema_tau_s, 1e-3), 0.01, 0.5))
        front._normalizer = NORMALIZERS[variant](fc.n_azimuth_bins, alpha)

        fields, yaws = [], []
        for i, (_t, frame) in enumerate(
            iter_video_at_brain_hz(str(video), max(0.0, start - PRE_ROLL_S), start + EVENT_DUR_S, cfg.brain_dt)
        ):
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(
                gray, (cfg.encode_width, cfg.encode_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32) / 255.0
            d = front.process(img)
            if i < n_pre:
                continue
            fields.append(np.asarray(d.signed_flow_field, dtype=np.float64))
            yaws.append(float(d.yaw_frozen))

        n_bins = min(len(f) for f in fields)
        mat = np.stack([f[:n_bins] for f in fields])          # [time, bins]
        times = len(yaws)

        # ---- 1. sign of the pooled readout ----
        y_med = float(np.median(yaws))
        sign_ok = bool(np.sign(y_med) == lab)

        # ---- 2. per-bin temporal sign stability ----
        n_pos = np.nansum(mat > 0, axis=0)
        n_neg = np.nansum(mat < 0, axis=0)
        total = np.maximum(n_pos + n_neg, 1)
        cons = np.maximum(n_pos, n_neg) / total
        cons_med = float(np.nanmedian(cons))
        chance = chance_consistency(mat, n_pos, n_neg)
        above = int(np.nansum(cons > chance + 0.15))

        # ---- 3. signal to noise ----
        mu_t = np.nanmean(mat, axis=0)
        sd_t = np.nanstd(mat, axis=0)
        pooled_sd = float(np.nanmean(sd_t))
        field_snr = float(np.nanmean(np.abs(mu_t)) / (pooled_sd + 1e-12))
        yaw_mad = float(np.median(np.abs(np.array(yaws) - y_med)))
        readout_snr = abs(y_med) / (1.4826 * yaw_mad + 1e-12)

        # ---- 4. adjacent-bin spatial coherence ----
        cohs = []
        for row in mat:
            a, b = row[:-1], row[1:]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 4 and a[m].std() > 1e-12 and b[m].std() > 1e-12:
                cohs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
        coh = float(np.nanmedian(cohs)) if cohs else float("nan")

        out[eid] = {
            "label": lab,
            "variant": variant,
            "yaw_frozen_median": y_med,
            "sign_ok": sign_ok,
            "field_scale": float(np.nanmean(np.abs(mat))),
            "bin_consistency_median": cons_med,
            "bin_consistency_chance": chance,
            "bins_above_chance": above,
            "bins_total": int(n_bins),
            "field_snr": field_snr,
            "readout_snr": readout_snr,
            "adjacent_bin_coherence": coh,
            "time_steps": times,
        }
        del fields, mat
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/p01r/VID00001.AVI")
    ap.add_argument("-o", "--output", default="output/p01o_normalizer")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5

    print("P0.1O — normaliser ablation (A per-bin / B mean-only / C global)")
    print("optic_flow.py unchanged; only the normaliser instance is swapped\n")

    results = {v: run_variant(video, cfg, v) for v in VARIANTS}

    # ---- per-event comparison ----
    print("=" * 100)
    print(f"{'event':10s} {'lbl':>4s} | " + " | ".join(f"{v:^28s}" for v in VARIANTS))
    print(f"{'':10s} {'':>4s} | " + " | ".join(f"{'yaw':>8s} {'sign':>4s} {'bins>ch':>8s} {'snr':>5s}" for _ in VARIANTS))
    print("-" * 100)
    for eid in EVENTS:
        lab = "L" if EVENTS[eid][1] < 0 else "R"
        cells = []
        for v in VARIANTS:
            r = results[v][eid]
            cells.append(
                f"{r['yaw_frozen_median']:+8.4f} "
                f"{('yes' if r['sign_ok'] else 'NO'):>4s} "
                f"{r['bins_above_chance']:>3d}/{r['bins_total']:<4d} "
                f"{r['readout_snr']:5.2f}"
            )
        print(f"{eid:10s} {lab:>4s} | " + " | ".join(cells))

    print()
    print("=" * 100)
    for v in VARIANTS:
        rs = results[v]
        ok = sum(1 for r in rs.values() if r["sign_ok"])
        coh = np.mean([r["adjacent_bin_coherence"] for r in rs.values()])
        snr = np.mean([r["readout_snr"] for r in rs.values()])
        bins = sum(r["bins_above_chance"] for r in rs.values())
        tot = sum(r["bins_total"] for r in rs.values())
        print(
            f"{v:14s} signs {ok}/{len(EVENTS)}   "
            f"bins above chance {bins}/{tot}   "
            f"mean adjacent-bin coherence {coh:+.3f}   mean readout SNR {snr:.2f}"
        )

    # ---- the specific question: left_108 ----
    print("\n" + "=" * 100)
    print("left_108 detail")
    for v in VARIANTS:
        r = results[v]["left_108"]
        print(
            f"  {v:14s} yaw={r['yaw_frozen_median']:+.4f}  sign_ok={r['sign_ok']}  "
            f"bin consistency={r['bin_consistency_median']:.3f} (chance {r['bin_consistency_chance']:.3f})  "
            f"bins above chance={r['bins_above_chance']}/{r['bins_total']}  "
            f"field SNR={r['field_snr']:.3f}  coherence={r['adjacent_bin_coherence']:+.3f}"
        )

    # ---- verdict ----
    a_ok = sum(1 for r in results["A_per_bin"].values() if r["sign_ok"])
    b_ok = sum(1 for r in results["B_mean_only"].values() if r["sign_ok"])
    c_ok = sum(1 for r in results["C_global"].values() if r["sign_ok"])
    a_108 = results["A_per_bin"]["left_108"]
    f108 = {v: results[v]["left_108"] for v in VARIANTS}

    recovers = [v for v in ("B_mean_only", "C_global") if f108[v]["sign_ok"] and not a_108["sign_ok"]]
    best_ok = max(b_ok, c_ok)
    keeps_controls = all(
        results[v][e]["sign_ok"]
        for v in ("B_mean_only", "C_global")
        for e in ("left_65", "right_216")
    )

    if recovers and best_ok > a_ok and keeps_controls:
        verdict = (
            f"CONFIRMED: removing the per-bin division recovers left_108 "
            f"({'/'.join(recovers)}) and improves overall sign accuracy from {a_ok}/6 to {best_ok}/6 "
            "without breaking left_65 or right_216. The per-bin divisor was amplifying noise."
        )
    elif recovers:
        verdict = (
            f"PARTIAL: {recovers} recovers left_108, but overall accuracy does not clearly improve "
            f"(A {a_ok}/6, B {b_ok}/6, C {c_ok}/6) or the controls degrade."
        )
    else:
        verdict = (
            f"NOT CONFIRMED: no variant recovers left_108 (A {a_ok}/6, B {b_ok}/6, C {c_ok}/6). "
            "The per-bin divisor is not the cause."
        )

    report = {
        "probe": "P0.1O — normaliser ablation",
        "note": "optic_flow.py unmodified; normaliser instance swapped at runtime",
        "variants": {
            "A_per_bin": "(pan - mu) / (dev + eps), clipped +-3, dev per bin",
            "B_mean_only": "(pan - mu), no division",
            "C_global": "(pan - mu) / (median(dev) + eps), clipped +-3",
        },
        "events": {k: v[0] for k, v in EVENTS.items()},
        "results": results,
        "sign_accuracy": {"A_per_bin": a_ok, "B_mean_only": b_ok, "C_global": c_ok, "n_events": len(EVENTS)},
        "verdict": verdict,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows = [r for v in VARIANTS for r in results[v].values()]
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
