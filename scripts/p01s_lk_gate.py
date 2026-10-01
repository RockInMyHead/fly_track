#!/usr/bin/env python3
"""
P0.1S — local motion by Lucas–Kanade tracking, read out by the fly's own cells.

The Reichardt correlator was measured at chance on real turns (P0.1Q: 1% of grid
cells held the correct sign for 70% of a turn). This gate replaces the estimator
and nothing else:

    video -> local displacement per patch (Lucas-Kanade) -> T4/T5 -> MaleCNS

Still no pooled LEFT/RIGHT decision anywhere: `yaw_frozen` is logged for
comparison only and is never injected.

Stages
------
A. synthetic   recovering known pixel shifts, including sign and a radial control
B. frontend    LK vs the 2-D correlator vs the 1-D panorama on the seven
               blind-labelled events: sign accuracy, per-cell stability, coverage,
               and whether the two estimators agree with each other
C. viz         displacement field and per-frame trace for four events
D. brain       the 20 visual types frozen in P0.1L, 5 seeds, original + mirrored,
               scored exactly as P0.1M scored them so the numbers compare

Frozen and untouched: the 20 cell types and their polarities (P0.1L), the blind
event labels, MaleCNS dynamics, seeds, detector scale, injection gains.

Usage:
    PYTHONPATH=. python scripts/p01s_lk_gate.py
    PYTHONPATH=. python scripts/p01s_lk_gate.py --stages synth,frontend
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.config import FlyVOConfig
from fly_vo.lk_motion_field import LKMotionField
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import DirectionalMotionFrontend, FlowConfig
from fly_vo.visual_encoder import VideoVisualEncoder

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
P01L_REPORT = ROOT / "output/p01l_real_video/report.json"
P01M_REPORT = ROOT / "output/p01m_all_events/report.json"
P01Q_REPORT = ROOT / "output/p01q_grid2d/report.json"
P01Q_STAB = ROOT / "output/p01q_stability/report.json"

EVENTS = {
    "left_65":   (63.5, +1),
    "left_81":   (78.5, +1),
    "left_108":  (106.0, +1),
    "right_67":  (66.0, -1),
    "right_210": (209.0, -1),
    "right_216": (214.0, -1),
    "fwd_140":   (137.5, 0),
}
LEFT_EVENTS = [k for k, v in EVENTS.items() if v[1] == +1]
RIGHT_EVENTS = [k for k, v in EVENTS.items() if v[1] == -1]
FWD_EVENTS = [k for k, v in EVENTS.items() if v[1] == 0]
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DT = 0.02
FWD_NEUTRAL_Z = 0.50
SEEDS = 5
ENCODE = (192, 108)
LNAME = {+1: "LEFT", -1: "RIGHT", 0: "FWD"}
T4T5_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

MODES = {
    "lk": ("lk", 192, 108),
    "grid2d": ("grid2d", 192, 108),
    "legacy96": ("panorama", 96, 72),
}


def flow_config(mode: str) -> FlowConfig:
    return FlowConfig(
        n_azimuth_bins=128, ema_tau_s=0.5, brain_dt=DT, crop_mode="center_band",
        field_mode=mode, grid_w=16, grid_h=8,
    )


def prep(frame_bgr: np.ndarray, ew: int, eh: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (ew, eh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def capture_frontend(mode: str, ew: int, eh: int, t0: float, t1: float, mirror: bool = False) -> list[dict]:
    front = DirectionalMotionFrontend(flow_config(mode))
    rows = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, DT):
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        d = front.process(prep(frame, ew, eh))
        rows.append({
            "t": t,
            "yaw": float(d.yaw_frozen),
            "field": np.asarray(d.motion_field_2d, dtype=np.float64) if d.motion_field_2d else None,
            "coverage": float(d.grid_row_agreement),
            "held": bool(d.detector_bank.get("lk_held", 0.0)),
            "abs_dx": float(d.detector_bank.get("LK_abs_mean", 0.0)),
        })
    return rows


def cell_stability(fields: np.ndarray, want: int) -> dict:
    """fields: [frames, cells] raw signed values for one event."""
    if fields.size == 0 or want == 0:
        return {}
    correct = np.sign(fields) == want
    frac = correct.mean(axis=0)
    pooled = fields.mean(axis=1)
    best = cur = 0
    for v in np.sign(pooled) == want:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {
        "per_cell_frac_correct_median": float(np.median(frac)),
        "cells_above_70pct": float(np.mean(frac >= 0.7)),
        "cells_above_90pct": float(np.mean(frac >= 0.9)),
        "pooled_frac_correct": float(np.mean(np.sign(pooled) == want)),
        "pooled_signed_mean": float(pooled.mean()),
        "pooled_sign_correct": bool(np.sign(pooled.mean()) == want),
        "longest_pooled_correct_s": best * DT,
    }


# -------------------------------------------------------------- stage A: synth
def stage_synth(out: Path) -> dict:
    h, w = 144, 256

    def tex(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.zeros((h, w), np.float32)
        for _ in range(240):
            y, x = rng.integers(6, h - 6), rng.integers(6, w - 6)
            r = int(rng.integers(2, 5))
            img[max(0, y - r):y + r, max(0, x - r):x + r] += float(rng.uniform(0.3, 1.0))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img -= img.min()
        return (img / max(img.max(), 1e-6) * 0.85 + 0.05).astype(np.float32)

    def shift_img(img, dx=0, dy=0):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    print("\n=== A. synthetic shift recovery ===")
    rows = []
    base = tex(11)
    for s in (1, 2, 3, 5, 8):
        lk = LKMotionField(grid_w=16, grid_h=8)
        lk.process(base)
        dxf, dyf, diag = lk.process(shift_img(base, dx=s))
        rows.append({
            "shift_px": s, "recovered_median_px": float(np.median(dxf)),
            "recovered_mean_dy_px": float(np.mean(dyf)),
            "coverage": diag.coverage, "all_positive": bool(np.all(dxf > 0)),
        })
        print(f"  shift {s:2d} px -> median dx {np.median(dxf):+6.2f} px  "
              f"dy {np.mean(dyf):+5.2f}  coverage {diag.coverage:.0%}  "
              f"all cells positive: {bool(np.all(dxf > 0))}")

    lk = LKMotionField(grid_w=16, grid_h=8)
    lk.process(base)
    dxl, _, _ = lk.process(shift_img(base, dx=-4))

    # A single texture can give a sizeable random dx under a vertical shift, so
    # the claim tested is that the *systematic* leak is negligible.
    hv, vv, dv = [], [], []
    for seed in range(16):
        b = tex(100 + seed)
        lk1 = LKMotionField(grid_w=16, grid_h=8)
        lk1.process(b)
        dxh, _, _ = lk1.process(shift_img(b, dx=4))
        hv.append(float(np.mean(dxh)))
        lk2 = LKMotionField(grid_w=16, grid_h=8)
        lk2.process(b)
        dxv, dyv, _ = lk2.process(shift_img(b, dy=4))
        vv.append(float(np.mean(dxv)))
        dv.append(float(np.mean(dyv)))
    horiz, vert_bias, vert_track = float(np.mean(hv)), float(np.mean(vv)), float(np.mean(dv))
    print(f"  horizontal shift +4 px -> dx {horiz:+.3f} px (mean over 16 textures)")
    print(f"  vertical   shift +4 px -> dx {vert_bias:+.3f} px (systematic leak), dy {vert_track:+.3f} px")

    zoom = cv2.resize(base, (int(w * 1.06), int(h * 1.06)), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (zoom.shape[0] - h) // 2, (zoom.shape[1] - w) // 2
    lkz = LKMotionField(grid_w=16, grid_h=8)
    lkz.process(base)
    dz, _, _ = lkz.process(zoom[y0:y0 + h, x0:x0 + w])

    checks = {
        "left_shift_negative": bool(np.all(dxl < 0)),
        "vertical_leak_vs_horizontal": abs(vert_bias) < 0.05 * horiz,
        "vertical_tracked_in_dy": abs(vert_track - 4.0) < 0.6,
        "radial_splits_left_right": float(dz[:, :8].mean()) < 0 < float(dz[:, 8:].mean()),
        "shift_recovery_error_max_px": float(
            max(abs(r["recovered_median_px"] - r["shift_px"]) for r in rows)
        ),
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")
    (out / "synth.json").write_text(json.dumps({"rows": rows, "checks": checks}, indent=2), encoding="utf-8")
    return {"rows": rows, "checks": checks}


# ----------------------------------------------------------- stage B: frontend
def stage_frontend(out: Path, do_viz: bool = False) -> dict:
    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    windows = {e["id"]: (e["start"], e["end"]) for e in doc["events"]}

    print("\n=== B. frontend comparison on the seven blind-labelled events ===")
    print(f"  yaw_frozen median (frozen convention: yaw<0 = LEFT, yaw>0 = RIGHT)")
    print(f"  {'event':10s} {'label':5s} {'legacy96':>10s} {'grid2d':>10s} {'lk':>10s} "
          f"{'|dx|px':>7s} {'cover':>6s} {'held':>6s}")

    report: dict = {"events": {}}
    table = []
    for eid, (start, lab) in EVENTS.items():
        t0, t1 = max(0.0, start - PRE_ROLL_S), start + EVENT_DUR_S
        want = -1 if lab == +1 else (+1 if lab == -1 else 0)
        rec: dict = {"event": eid, "label": lab, "want_sign": want}
        for tag in ("legacy96", "grid2d", "lk"):
            mode, ew, eh = MODES[tag]
            rows = capture_frontend(mode, ew, eh, t0, t1)
            n_pre = int(round(PRE_ROLL_S / DT))
            ev = rows[n_pre:]
            yaw = np.array([r["yaw"] for r in ev])
            rec[tag] = {
                "yaw_median": float(np.median(yaw)),
                "yaw_sign_correct": bool(np.sign(np.median(yaw)) == want) if want else None,
                "mean_coverage": float(np.mean([r["coverage"] for r in ev])),
                "held_fraction": float(np.mean([r["held"] for r in ev])),
                "mean_abs_dx_px": float(np.mean([r["abs_dx"] for r in ev])),
            }
            if tag == "lk" and ev:
                fields = np.stack([r["field"] for r in ev if r["field"] is not None], axis=0)
                fields = fields.reshape(len(fields), -1)
                rec["lk_stability"] = cell_stability(fields, want) if want else {}
                if do_viz:
                    _save_lk_figures(out, eid, ev, want, start)
        table.append(rec)
        print(f"  {eid:10s} {LNAME[lab]:5s} {rec['legacy96']['yaw_median']:+10.4f} "
              f"{rec['grid2d']['yaw_median']:+10.4f} {rec['lk']['yaw_median']:+10.4f} "
              f"{rec['lk']['mean_abs_dx_px']:7.2f} {rec['lk']['mean_coverage']:6.0%} "
              f"{rec['lk']['held_fraction']:6.0%}")
    report["events"] = table

    print("\n  turn sign accuracy (yaw<0 = LEFT):")
    for tag in ("legacy96", "grid2d", "lk"):
        ok = tot = 0
        for r in table:
            if r["want_sign"] == 0:
                continue
            tot += 1
            ok += int(r[tag]["yaw_sign_correct"])
        report[f"{tag}_turn_sign_accuracy"] = [ok, tot]
        print(f"    {tag:10s} {ok}/{tot}")

    # do the two estimators agree with each other, independent of the labels?
    agree = sum(
        1 for r in table
        if np.sign(r["lk"]["yaw_median"]) == np.sign(r["legacy96"]["yaw_median"])
    )
    report["lk_vs_legacy_sign_agreement"] = [agree, len(table)]
    print(f"\n  LK vs 1-D panorama agree on the sign in {agree}/{len(table)} events")

    print(f"\n  per-cell stability on the LK field (event frames only)")
    print(f"  {'event':10s} {'label':5s} {'per-cell med':>13s} {'cells>=70%':>11s} "
          f"{'pooled':>7s} {'longest':>8s}")
    for r in table:
        s = r.get("lk_stability") or {}
        if not s:
            continue
        print(f"  {r['event']:10s} {LNAME[r['label']]:5s} {s['per_cell_frac_correct_median']:13.0%} "
              f"{s['cells_above_70pct']:11.0%} {s['pooled_frac_correct']:7.0%} "
              f"{s['longest_pooled_correct_s']:7.2f}s")

    turns = [r for r in table if r["want_sign"] != 0]
    lk_med = float(np.mean([r["lk_stability"]["per_cell_frac_correct_median"] for r in turns]))
    lk_70 = float(np.mean([r["lk_stability"]["cells_above_70pct"] for r in turns]))
    report["lk_turn_cell_stability"] = {
        "per_cell_frac_correct_median_mean": lk_med,
        "cells_above_70pct_mean": lk_70,
    }
    print(f"\n  LK summary over the six turns: per-cell median {lk_med:.0%}, "
          f"cells >=70% correct {lk_70:.0%}")

    if P01Q_STAB.exists():
        q = json.loads(P01Q_STAB.read_text(encoding="utf-8"))
        g = [r["grid2d"] for r in q["rows"] if r["grid2d"]]
        if g:
            ref = float(np.mean([r["unit_frac_correct_median"] for r in g]))
            ref70 = float(np.mean([r["units_above_70pct"] for r in g]))
            report["grid2d_turn_cell_stability_baseline"] = {
                "per_cell_frac_correct_median_mean": ref, "cells_above_70pct_mean": ref70
            }
            print(f"  2-D correlator baseline (P0.1Q):  per-cell median {ref:.0%}, "
                  f"cells >=70% correct {ref70:.0%}")
            print(f"  chance level is 50%")

    with (out / "frontend_events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "label", "want_sign",
                    "legacy96_yaw", "grid2d_yaw", "lk_yaw", "lk_abs_dx_px",
                    "lk_coverage", "lk_held", "lk_sign_ok",
                    "lk_per_cell_median", "lk_cells_70", "lk_pooled", "lk_longest_s"])
        for r in table:
            s = r.get("lk_stability") or {}
            w.writerow([
                r["event"], r["label"], r["want_sign"],
                f"{r['legacy96']['yaw_median']:.6f}", f"{r['grid2d']['yaw_median']:.6f}",
                f"{r['lk']['yaw_median']:.6f}", f"{r['lk']['mean_abs_dx_px']:.4f}",
                f"{r['lk']['mean_coverage']:.4f}", f"{r['lk']['held_fraction']:.4f}",
                r["lk"]["yaw_sign_correct"],
                f"{s.get('per_cell_frac_correct_median', float('nan')):.4f}",
                f"{s.get('cells_above_70pct', float('nan')):.4f}",
                f"{s.get('pooled_frac_correct', float('nan')):.4f}",
                f"{s.get('longest_pooled_correct_s', float('nan')):.3f}",
            ])
    return report


# ---------------------------------------------------------------- stage C: viz
def _heat(field: np.ndarray, lim: float, cell: int = 30) -> np.ndarray:
    h, w = field.shape
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            v = float(np.clip(field[y, x] / max(lim, 1e-9), -1, 1))
            if v >= 0:
                img[y, x] = (int(255 * v), int(255 * (1 - v * 0.1)), int(255 * (1 - v)))
            else:
                img[y, x] = (int(255 * (1 + v)), int(255 * (1 + v * 0.1)), int(255 * (1 + v)))
    return cv2.resize(img, (w * cell, h * cell), interpolation=cv2.INTER_NEAREST)


def _save_lk_figures(out: Path, eid: str, ev: list[dict], want: int, start: float) -> None:
    vdir = out / "field_lk"
    vdir.mkdir(parents=True, exist_ok=True)
    fields = [r["field"] for r in ev if r["field"] is not None]
    if not fields:
        return
    mean_field = np.mean(fields, axis=0)
    lim = float(np.percentile(np.abs(mean_field), 95)) or 1e-9
    canvas = _heat(mean_field, lim)
    cv2.putText(canvas, f"{eid} mean LK displacement (px)  blue = rightward, red = leftward",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(vdir / f"{eid}_field.png"), canvas)

    means = [float(f.mean()) for f in fields]
    img = np.full((240, 1200, 3), 255, np.uint8)
    pad_l, pad_t, pad_b = 70, 42, 24
    xw, yh = 1200 - pad_l - 20, 240 - pad_t - pad_b
    mid = pad_t + yh // 2
    lim2 = max(max(np.abs(means), default=1.0) * 0.2, 1e-6)

    def px(i: int) -> int:
        return pad_l + int(i / max(len(means) - 1, 1) * xw)

    def py(v: float) -> int:
        return int(mid - np.clip(v / lim2, -1, 1) * (yh / 2 - 6))

    cv2.line(img, (pad_l, mid), (pad_l + xw, mid), (170, 170, 170), 1)
    for i in range(len(means) - 1):
        c = (70, 160, 70) if np.sign(means[i]) == want else (60, 60, 220)
        cv2.line(img, (px(i), py(means[i])), (px(i + 1), py(means[i + 1])), c, 1)
    n_ok = sum(1 for v in means if np.sign(v) == want)
    best = cur = 0
    for v in means:
        cur = cur + 1 if np.sign(v) == want else 0
        best = max(best, cur)
    cv2.putText(img, f"{eid}  per-frame mean LK displacement   "
                     f"correct {n_ok}/{len(means)} = {n_ok / max(len(means), 1):.0%}   "
                     f"longest correct run {best * DT:.2f}s",
                (pad_l - 62, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(vdir / f"{eid}_trace.png"), img)


# --------------------------------------------------------------- stage D: brain
def load_frozen_cells() -> list[dict]:
    rep = json.loads(P01L_REPORT.read_text(encoding="utf-8"))
    frozen = [r for r in rep["all_types"] if r["passes"] and r["mirror_equivalence"]]
    if not frozen:
        raise SystemExit("P0.1L report has no cells passing both criteria")
    return frozen


def capture_encoder(start: float, cfg: FlyVOConfig, mode: str, ew: int, eh: int, mirror: bool) -> dict:
    t0 = max(0.0, start - PRE_ROLL_S)
    t1 = start + EVENT_DUR_S
    encoder = VideoVisualEncoder(
        MaleCNSEngine(cfg), gain=cfg.visual_gain, n_azimuth_bins=128,
        flow_config=flow_config(mode),
    )
    frames = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, cfg.brain_dt):
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        eye, inject, _m = encoder.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject})
    n_pre = min(int(round(PRE_ROLL_S / cfg.brain_dt)), max(0, len(frames) - 1))
    return {"frames": frames, "n_pre": n_pre}


def run_readout(out: Path, tag: str = "lk") -> dict:
    mode, ew, eh = MODES[tag]
    frozen = load_frozen_cells()
    keys = [(r["cell_type"], r["side"]) for r in frozen]
    polarity = {(r["cell_type"], r["side"]): (1 if r["orig_dir_index"] > 0 else -1) for r in frozen}

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    cfg.encode_width, cfg.encode_height = ew, eh
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain

    groups = {k: np.asarray(brain.cells([k[0]], side=k[1]), dtype=np.int64) for k in keys}
    all_idx = np.unique(np.concatenate([v for v in groups.values() if len(v)]))
    union_slot = np.full(int(brain.n), -1, dtype=np.int64)
    union_slot[all_idx] = np.arange(len(all_idx), dtype=np.int64)
    local_pos = {k: union_slot[v] for k, v in groups.items()}

    print(f"\n=== D. the 20 frozen visual cells on the {tag} field ===")
    print(f"  {sum(1 for k in keys if k[0] in T4T5_TYPES)} of the 20 are T4/T5 themselves "
          f"(injected directly); the rest test the connectome")
    raw: dict[tuple[str, str], dict] = {}
    for run in ("orig", "mirror"):
        for eid in EVENTS:
            cap = capture_encoder(EVENTS[eid][0], cfg, mode, ew, eh, run == "mirror")
            lo, hi = cap["n_pre"], len(cap["frames"])
            span = max(hi - lo, 1) * dt
            counts = np.zeros((SEEDS, len(all_idx)), dtype=np.float64)
            for si, seed in enumerate(range(1, SEEDS + 1)):
                engine.reset(seed=seed)
                c = np.zeros(len(all_idx), dtype=np.int64)
                for i, fr in enumerate(cap["frames"]):
                    res = engine.step(fr["t"], eye_drive=fr["eye"], inject=fr["inject"])
                    if i < lo:
                        continue
                    f = np.asarray(res.fired)
                    if not len(f):
                        continue
                    s = union_slot[f]
                    s = s[s >= 0]
                    if len(s):
                        np.add.at(c, s, 1)
                counts[si] = c / span
            raw[(run, eid)] = {k: counts[:, local_pos[k]].sum(axis=1) for k in keys}
            print(f"  {run:6s} {eid:10s} span={span:.2f}s")

    event_index = {e: i for i, e in enumerate(EVENTS)}

    def zscores(run: str) -> dict[tuple[str, str], np.ndarray]:
        out_z = {}
        for k in keys:
            mat = np.stack([raw[(run, eid)][k] for eid in EVENTS], axis=1)
            med = np.median(mat, axis=1, keepdims=True)
            mad = np.median(np.abs(mat - med), axis=1, keepdims=True)
            scale = 1.4826 * mad
            scale[scale < 1e-9] = 1e-9
            out_z[k] = (mat - med) / scale
        return out_z

    z = {run: zscores(run) for run in ("orig", "mirror")}

    type_rows, cell_events = [], []
    for k in keys:
        ctype, side = k
        pol = polarity[k]
        scores = {}
        for run in ("orig", "mirror"):
            for eid in EVENTS:
                scores[(run, eid)] = float(np.median(pol * z[run][k][:, event_index[eid]]))

        def correct(run: str, eid: str) -> bool:
            lab = EVENTS[eid][1]
            exp = lab if run == "orig" else -lab
            if exp == 0:
                return abs(scores[(run, eid)]) < FWD_NEUTRAL_Z
            return np.sign(scores[(run, eid)]) == exp

        left_ok = sum(correct("orig", e) for e in LEFT_EVENTS)
        right_ok = sum(correct("orig", e) for e in RIGHT_EVENTS)
        for eid in EVENTS:
            for run in ("orig", "mirror"):
                lab = EVENTS[eid][1]
                cell_events.append({
                    "cell_type": ctype, "side": side, "polarity": pol, "run": run,
                    "event": eid, "label": lab, "expected": lab if run == "orig" else -lab,
                    "score": scores[(run, eid)], "correct": bool(correct(run, eid)),
                    "rate": float(np.median(raw[(run, eid)][k])),
                    "is_injected": ctype in T4T5_TYPES,
                })
        type_rows.append({
            "cell_type": ctype, "side": side, "polarity": pol,
            "is_injected": ctype in T4T5_TYPES,
            "left_correct": left_ok, "left_n": len(LEFT_EVENTS),
            "right_correct": right_ok, "right_n": len(RIGHT_EVENTS),
            "fwd_neutral_ok": sum(correct("orig", e) for e in FWD_EVENTS),
            "turn_correct": left_ok + right_ok,
            "turn_n": len(LEFT_EVENTS) + len(RIGHT_EVENTS),
            "mirror_correct": sum(correct("mirror", e) for e in LEFT_EVENTS + RIGHT_EVENTS),
            "mirror_n": len(LEFT_EVENTS) + len(RIGHT_EVENTS),
            "scores": {f"{run}:{e}": scores[(run, e)] for run in ("orig", "mirror") for e in EVENTS},
        })

    type_rows.sort(key=lambda r: (-r["turn_correct"], -r["mirror_correct"]))
    downstream = [r for r in type_rows if not r["is_injected"]]
    n_types = len(type_rows)

    print(f"\n  {'type':10s} {'sd':2s} {'inj':3s} {'pol':>4s} "
          f"{'LEFT':>6s} {'RIGHT':>6s} {'FWD':>5s} {'turn':>6s} {'mirror':>7s}")
    for r in type_rows:
        print(f"  {r['cell_type'][:10]:10s} {r['side']:2s} "
              f"{'yes' if r['is_injected'] else '-':3s} {r['polarity']:+4d} "
              f"{r['left_correct']}/{r['left_n']:<4d} {r['right_correct']}/{r['right_n']:<4d} "
              f"{'OK' if r['fwd_neutral_ok'] else 'no':>5s} "
              f"{r['turn_correct']}/{r['turn_n']:<4d} {r['mirror_correct']}/{r['mirror_n']:<4d}")

    agg = []
    for eid in EVENTS:
        lab = EVENTS[eid][1]
        if lab == 0:
            continue
        agg.append({
            "event": eid, "label": lab,
            "types_correct": sum(1 for r in type_rows if (r["scores"][f"orig:{eid}"] > 0) == (lab > 0)),
            "downstream_correct": sum(
                1 for r in downstream if (r["scores"][f"orig:{eid}"] > 0) == (lab > 0)
            ),
            "n_types": n_types, "downstream_n": len(downstream),
        })
    agg.sort(key=lambda r: r["event"])
    pooled = sum(r["types_correct"] for r in agg)
    pooled_n = sum(r["n_types"] for r in agg)
    pooled_dn = sum(r["downstream_correct"] for r in agg)
    pooled_dn_n = sum(r["downstream_n"] for r in agg)
    all6 = sum(1 for r in type_rows if r["turn_correct"] == r["turn_n"])
    all6_dn = sum(1 for r in downstream if r["turn_correct"] == r["turn_n"])

    rng = np.random.default_rng(0)
    n_perm = 2000
    null = []
    for _ in range(n_perm):
        cnt = 0
        for k in keys:
            pol = polarity[k]
            ok = True
            for eid in EVENTS:
                lab = EVENTS[eid][1]
                if lab == 0:
                    continue
                s = float(np.median(pol * z["orig"][k][:, event_index[eid]]))
                if (s > 0) != (lab > 0):
                    ok = False
                    break
            if ok:
                cnt += 1
        null.append(cnt)
    null = np.asarray(null)
    p_all6 = float(np.mean(null >= all6))

    print(f"\n  pooled vote across the {n_types} frozen types:")
    for r in agg:
        print(f"    {r['event']:10s} label={'L' if r['label'] > 0 else 'R'}  "
              f"{r['types_correct']}/{r['n_types']} ({r['types_correct'] / r['n_types']:.0%})   "
              f"downstream-only {r['downstream_correct']}/{r['downstream_n']}")
    print(f"\n  pooled {pooled}/{pooled_n} = {pooled / pooled_n:.1%}   "
          f"downstream-only {pooled_dn}/{pooled_dn_n} = {pooled_dn / pooled_dn_n:.1%}")
    print(f"  types correct on ALL six turns: {all6}/{n_types} (downstream {all6_dn}/{len(downstream)}), "
          f"permutation p={p_all6:.4f}")

    baselines = {}
    for name, path in (("p01m_1d_legacy", P01M_REPORT), ("p01q_grid2d", P01Q_REPORT)):
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        key = "readout_grid2d" if "readout_grid2d" in d else None
        if key:
            baselines[name] = {
                "pooled_correct": d[key]["pooled_correct"], "pooled_total": d[key]["pooled_total"],
                "types_all_turns_correct": d[key]["types_all_turns_correct"], "n_types": d[key]["n_types"],
                "pooled_vote": d[key]["pooled_vote"],
            }
        else:
            baselines[name] = {
                "pooled_correct": d["pooled_correct"], "pooled_total": d["pooled_total"],
                "types_all_turns_correct": d["types_all_turns_correct"], "n_types": d["n_types"],
                "pooled_vote": d["pooled_vote"],
            }
    for name, b in baselines.items():
        print(f"  baseline {name:16s}: pooled {b['pooled_correct']}/{b['pooled_total']} "
              f"= {b['pooled_correct'] / b['pooled_total']:.0%}, "
              f"all-six {b['types_all_turns_correct']}/{b['n_types']}")

    with (out / f"type_results_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[c for c in type_rows[0] if c != "scores"]
            + [f"scores_{k}" for k in type_rows[0]["scores"]],
        )
        w.writeheader()
        for r in type_rows:
            row = {k: v for k, v in r.items() if k != "scores"}
            row.update({f"scores_{k}": v for k, v in r["scores"].items()})
            w.writerow(row)
    with (out / f"cell_events_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_events[0].keys()))
        w.writeheader()
        w.writerows(cell_events)

    return {
        "tag": tag, "mode": mode, "encode": [ew, eh], "n_types": n_types,
        "n_downstream_types": len(downstream),
        "pooled_correct": pooled, "pooled_total": pooled_n,
        "pooled_downstream_correct": pooled_dn, "pooled_downstream_total": pooled_dn_n,
        "types_all_turns_correct": all6, "types_all_turns_correct_downstream": all6_dn,
        "permutation": {"n_perm": n_perm, "mean": float(null.mean()),
                        "p95": float(np.percentile(null, 95)), "observed": all6, "p_value": p_all6},
        "per_type": type_rows, "pooled_vote": agg, "baselines": baselines,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="synth,frontend,brain")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("-o", "--output", default="output/p01s_lk")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.stages.split(",") if s.strip()}
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1S — local motion by Lucas–Kanade tracking")
    print(f"  video: {VIDEO.name}   frozen cells: {P01L_REPORT.name}")

    report: dict = {
        "probe": "P0.1S — Lucas-Kanade local motion read by the fly's visual cells",
        "frozen": ["20 visual types and their polarities (P0.1L)", "blind event labels",
                   "MaleCNS dynamics", "seeds", "injection gains", "azimuth mapping"],
        "not_used": ["yaw_frozen (logged only, never injected)", "descending neurons",
                     "trajectory", "dopamine", "any learned readout"],
        "estimator": {
            "method": "cv2.calcOpticalFlowPyrLK, forward-backward filtered",
            "seeds": "3x3 per grid cell, fixed image-space grid",
            "aggregation": "per-cell median displacement",
            "units": "displacement in pixels (ref_px = 1.0)",
            "hold_on_duplicate": True,
        },
    }

    def save() -> None:
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if "synth" in wanted:
        report["synth"] = stage_synth(out)
        save()
    if "frontend" in wanted:
        report["frontend"] = stage_frontend(out, do_viz=args.viz)
        save()
    elif args.viz:
        stage_frontend(out, do_viz=True)
    if "brain" in wanted:
        report["readout_lk"] = run_readout(out, "lk")
        save()

    f = report.get("frontend")
    r = report.get("readout_lk")
    parts = []
    if f:
        acc = f["lk_turn_sign_accuracy"]
        st = f.get("lk_turn_cell_stability", {})
        base = f.get("grid2d_turn_cell_stability_baseline", {})
        parts.append(
            f"estimator: LK turn sign accuracy {acc[0]}/{acc[1]}, per-cell stability "
            f"{st.get('per_cell_frac_correct_median_mean', float('nan')):.0%} "
            f"(correlator baseline {base.get('per_cell_frac_correct_median_mean', float('nan')):.0%}, "
            f"chance 50%), cells holding the sign for 70% of a turn "
            f"{st.get('cells_above_70pct_mean', float('nan')):.0%} "
            f"(baseline {base.get('cells_above_70pct_mean', float('nan')):.0%})"
        )
    if r:
        parts.append(
            f"fly readout: pooled {r['pooled_correct']}/{r['pooled_total']} = "
            f"{r['pooled_correct'] / r['pooled_total']:.0%}, all-six types "
            f"{r['types_all_turns_correct']}/{r['n_types']} (p={r['permutation']['p_value']:.4f})"
        )
    if parts:
        report["verdict"] = " | ".join(parts)
        print(f"\n=== VERDICT ===\n{report['verdict']}")
        save()

    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
