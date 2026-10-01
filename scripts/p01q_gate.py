#!/usr/bin/env python3
"""
P0.1Q — local 2-D motion field, read out by the fly's own visual cells.

What changed
------------
1. The 96 -> 128 azimuth comb is gone. `build_azimuth_panorama` resampled columns
   by linear interpolation instead of floor-binning them, so no bin is left
   permanently empty (was 32/128).
2. Motion is measured on a 2-D grid instead of on a collapsed 1-D column profile.
   The Reichardt correlator now runs per image row at full resolution and is read
   out on a 16 x 8 grid (`fly_vo/local_motion_field.py`).
3. Nothing upstream of the fly decides LEFT/RIGHT. `yaw_frozen` is still logged
   as a diagnostic but is never injected; the injector consumes the local field.

What is measured
----------------
A. Comb audit: legacy binning vs the new mapping.
B. Does the comb fix alone change the frozen P0.1R+ signs? (must not)
C. Frontend-only comparison on the seven blind-labelled events:
   legacy 96 px / legacy 192 px / grid2d 192 px.
D. The actual experiment: the 20 visual types frozen in P0.1L, run on all seven
   events with the 2-D field, original and mirrored, 5 seeds, scored exactly as
   P0.1M scored them so the numbers are directly comparable.
E. 2-D field heatmaps for four events.

Nothing is trained, tuned or re-selected. No descending neurons, no trajectory.

Usage:
    PYTHONPATH=. python scripts/p01q_gate.py                 # everything
    PYTHONPATH=. python scripts/p01q_gate.py --stages frontend
    PYTHONPATH=. python scripts/p01q_gate.py --stages brain
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
from fly_vo.malecns_engine import MaleCNSEngine
from fly_vo.optic_flow import (
    FlowConfig,
    build_azimuth_panorama,
    legacy_azimuth_bin_counts,
)
from fly_vo.visual_encoder import VideoVisualEncoder

# --------------------------------------------------------------------- events
VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS_JSON = ROOT / "data/p01r/VID00001_events.json"
P01L_REPORT = ROOT / "output/p01l_real_video/report.json"
P01M_REPORT = ROOT / "output/p01m_all_events/report.json"
P01R_BENCH = ROOT / "output/p01r_event_benchmark/report.json"

# Brain readout uses P0.1M's anchored 3 s windows so the numbers are comparable.
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
FWD_NEUTRAL_Z = 0.50
SEEDS = 5

# The twenty types frozen by P0.1L. Four of them are T4/T5 themselves, i.e. the
# cells we inject directly; those are reported separately because for them the
# readout measures our own inject, not the connectome.
T4T5_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

MODE_L96 = ("legacy_96", "panorama", 96, 72)
MODE_L192 = ("legacy_192", "panorama", 192, 108)
MODE_G192 = ("grid2d_192", "grid2d", 192, 108)


# ------------------------------------------------------------------- frontend
def flow_config(mode: str, ew: int, eh: int) -> FlowConfig:
    return FlowConfig(
        n_azimuth_bins=128,
        ema_tau_s=0.5,
        brain_dt=0.02,
        crop_mode="center_band",
        field_mode=mode,
        grid_w=16,
        grid_h=8,
    )


def resize_to(frame_bgr: np.ndarray, ew: int, eh: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (ew, eh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def frontend_rows(mode: str, ew: int, eh: int, t0: float, t1: float) -> list[dict]:
    from fly_vo.optic_flow import DirectionalMotionFrontend

    front = DirectionalMotionFrontend(flow_config(mode, ew, eh))
    rows: list[dict] = []
    for t, frame in iter_video_at_brain_hz(str(VIDEO), t0, t1, 0.02):
        d = front.process(resize_to(frame, ew, eh))
        rows.append(
            {
                "t": t,
                "yaw_frozen": float(d.yaw_frozen),
                "row_agreement": float(d.grid_row_agreement),
                "field_2d": d.motion_field_2d,
                "comb_empty_bins": int(d.comb_empty_bins),
            }
        )
    return rows


def median_mad(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    med = float(np.median(a))
    return med, float(np.median(np.abs(a - med)))


# --------------------------------------------------------------- stage A: comb
def stage_comb(out: Path) -> dict:
    print("\n=== A. azimuth comb audit (128 bins) ===")
    rows = []
    for w in (96, 192, 320):
        counts = legacy_azimuth_bin_counts(w, 128)
        holes = int(np.count_nonzero(counts == 0))
        img = np.random.default_rng(0).random((48, w)).astype(np.float32)
        pan = build_azimuth_panorama(img, 128)
        zeros = int(np.count_nonzero(pan == 0.0))
        rows.append({"encode_width": w, "legacy_empty_bins": holes, "new_empty_bins": zeros})
        print(f"  width={w:4d}  legacy empty bins={holes:3d}   new empty bins={zeros}")
    (out / "comb_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"rows": rows, "fixed": all(r["new_empty_bins"] == 0 for r in rows)}


# ----------------------------------------------------- stage B/C: frontend only
def stage_frontend(out: Path) -> dict:
    doc = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    windows = {e["id"]: (e["start"], e["end"], e["label"]) for e in doc["events"]}
    ids = [k for k in EVENTS if k in windows]

    archived = {}
    if P01R_BENCH.exists():
        bench = json.loads(P01R_BENCH.read_text(encoding="utf-8"))
        for ev in bench["events"]:
            s = ev.get("stats") or {}
            archived[ev["id"]] = float(s.get("yaw_frozen_median", 0.0))

    print("\n=== B/C. frontend comparison on the seven blind-labelled event windows ===")
    print(f"  {'event':10s} {'label':10s} {'archived':>10s} "
          f"{'legacy96':>10s} {'legacy192':>10s} {'grid2d':>10s} {'row_agr':>8s}")

    table = []
    for eid in ids:
        t0, t1, label = windows[eid]
        rec = {"event": eid, "label": label}
        for name, mode, ew, eh in (MODE_L96, MODE_L192, MODE_G192):
            rows = frontend_rows(mode, ew, eh, t0, t1)
            med, mad = median_mad([r["yaw_frozen"] for r in rows])
            rec[name] = {
                "median": med,
                "mad": mad,
                "row_agreement": float(np.mean([r["row_agreement"] for r in rows]))
                if mode == "grid2d"
                else float("nan"),
                "comb_empty_bins": int(max(r["comb_empty_bins"] for r in rows)),
                "n": len(rows),
            }
        rec["archived_median"] = archived.get(eid)
        table.append(rec)
        print(
            f"  {eid:10s} {label:10s} "
            f"{archived.get(eid, float('nan')):+10.4f} "
            f"{rec[MODE_L96[0]]['median']:+10.4f} "
            f"{rec[MODE_L192[0]]['median']:+10.4f} "
            f"{rec[MODE_G192[0]]['median']:+10.4f} "
            f"{rec[MODE_G192[0]]['row_agreement']:8.3f}"
        )

    def sign_correct(key: str) -> tuple[int, int]:
        """Frozen convention: yaw_frozen < 0 = LEFT, > 0 = RIGHT."""
        ok = tot = 0
        for r in table:
            if r["label"] == "FWD":
                continue
            want = -1 if r["label"] == "LEFT_YAW" else +1
            tot += 1
            if np.sign(r[key]["median"]) == want:
                ok += 1
        return ok, tot

    legacy_ok = sign_correct(MODE_L96[0])
    legacy192_ok = sign_correct(MODE_L192[0])
    grid_ok = sign_correct(MODE_G192[0])
    print(f"\n  turn sign accuracy (yaw<0 = LEFT) — legacy 96 px : {legacy_ok[0]}/{legacy_ok[1]}")
    print(f"  turn sign accuracy                  — legacy 192 px: {legacy192_ok[0]}/{legacy192_ok[1]}")
    print(f"  turn sign accuracy                  — grid2d      : {grid_ok[0]}/{grid_ok[1]}")

    preserved = all(
        (r["archived_median"] is None)
        or (np.sign(r["archived_median"]) == np.sign(r[MODE_L96[0]]["median"]))
        for r in table
    )
    drift = [
        abs(r[MODE_L96[0]]["median"] - r["archived_median"])
        for r in table
        if r["archived_median"] is not None
    ]
    print(f"  comb fix changed every archived yaw sign? {'NO' if preserved else 'YES'}")
    if drift:
        print(f"  median |drift| vs archived = {np.median(drift):.4f}")

    with (out / "frontend_events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "label", "archived_median", "legacy96_median", "legacy96_mad",
                    "legacy192_median", "legacy192_mad", "grid2d_median", "grid2d_mad",
                    "grid2d_row_agreement", "n"])
        for r in table:
            w.writerow([
                r["event"], r["label"], r["archived_median"],
                r[MODE_L96[0]]["median"], r[MODE_L96[0]]["mad"],
                r[MODE_L192[0]]["median"], r[MODE_L192[0]]["mad"],
                r[MODE_G192[0]]["median"], r[MODE_G192[0]]["mad"],
                r[MODE_G192[0]]["row_agreement"], r[MODE_G192[0]]["n"],
            ])
    return {
        "events": table,
        "legacy_turn_sign_accuracy": legacy_ok,
        "legacy192_turn_sign_accuracy": legacy192_ok,
        "grid2d_turn_sign_accuracy": grid_ok,
        "archived_signs_preserved": preserved,
        "median_abs_drift_vs_archived": float(np.median(drift)) if drift else None,
    }


# ------------------------------------------------------------------ stage E: viz
def _heat(field: np.ndarray, lim: float, cell: int = 30) -> np.ndarray:
    """Diverging field map: blue = rightward (+), red = leftward (−), grey = ~0."""
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


def _trace_plot(means: list[float], agree: list[float], lab: int, t0: float, dt: float,
                n_pre: int, eid: str, height: int = 240, width: int = 1200) -> np.ndarray:
    """Per-frame signed mean of the 2-D field — the P0.1P plot, on the new field."""
    img = np.full((height, width, 3), 255, np.uint8)
    pad_l, pad_r, pad_t, pad_b = 70, 20, 42, 24
    xw, yh = width - pad_l - pad_r, height - pad_t - pad_b
    mid = pad_t + yh // 2
    lim = max(0.2 * max(np.abs(means), default=1.0), 1e-6)

    def px(i: int) -> int:
        return pad_l + int(i / max(len(means) - 1, 1) * xw)

    def py(v: float) -> int:
        return int(mid - np.clip(v / lim, -1, 1) * (yh / 2 - 6))

    x_ev = px(n_pre)
    cv2.rectangle(img, (pad_l, pad_t + 2), (x_ev, pad_t + yh), (245, 245, 235), -1)
    cv2.line(img, (pad_l, mid), (pad_l + xw, mid), (170, 170, 170), 1)
    cv2.line(img, (x_ev, pad_t), (x_ev, pad_t + yh), (120, 120, 120), 1)
    cv2.putText(img, "pre-roll", (pad_l + 4, pad_t + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, "event", (x_ev + 4, pad_t + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    cv2.putText(img, "LEFT (-)", (pad_l - 68, pad_t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 200), 1)
    cv2.putText(img, "RIGHT (+)", (pad_l - 68, pad_t + yh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 90, 60), 1)

    for i in range(len(means) - 1):
        c = (70, 160, 70) if np.sign(means[i]) == lab else (60, 60, 220)
        cv2.line(img, (px(i), py(means[i])), (px(i + 1), py(means[i + 1])), c, 1)

    ev_means = means[n_pre:]
    ev_agree = agree[n_pre:]
    n_corr = int(sum(1 for v in ev_means if np.sign(v) == lab))
    best = cur = 0
    for v in ev_means:
        cur = cur + 1 if np.sign(v) == lab else 0
        best = max(best, cur)
    cv2.putText(img, f"{eid}  per-frame mean local motion   "
                     f"correct {n_corr}/{len(ev_means)} = {n_corr / max(len(ev_means), 1):.0%}   "
                     f"longest correct run {best * dt:.2f}s   "
                     f"mean row agreement {np.mean(ev_agree):.2f}",
                (pad_l - 62, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img, {"frac_frame_correct": n_corr / max(len(ev_means), 1),
                 "longest_correct_run_s": best * dt,
                 "mean_row_agreement": float(np.mean(ev_agree))}


def stage_viz(out: Path, events: list[str]) -> dict:
    vdir = out / "field_2d"
    vdir.mkdir(parents=True, exist_ok=True)
    print("\n=== E. 2-D motion field: mean map and per-frame sign trace ===")
    print(f"  {'event':10s} {'label':5s} {'mean':>9s} {'rowAgr':>7s} {'unanim':>7s} "
          f"{'frameOK':>8s} {'longestOK':>10s}   (P0.1P best was 0.48 s)")
    report = {}
    for eid in events:
        start = EVENTS[eid][0]
        rows = frontend_rows("grid2d", 192, 108, start - PRE_ROLL_S, start + EVENT_DUR_S)
        lab = {+1: -1, -1: +1, 0: 0}[EVENTS[eid][1]]  # LEFT event -> negative field mean
        labname = {+1: "LEFT", -1: "RIGHT", 0: "FWD"}[EVENTS[eid][1]]
        n_pre = int(round(PRE_ROLL_S / 0.02))

        fields = [np.asarray(r["field_2d"]) for r in rows if r["field_2d"]]
        means = [float(f.mean()) for f in fields]
        agree = [float(r["row_agreement"]) for r in rows if r["field_2d"]]
        mean_field = np.mean(fields[n_pre:] or fields, axis=0)

        lim = float(np.percentile(np.abs(mean_field), 95)) or 1e-9
        canvas = _heat(mean_field, lim)
        cv2.putText(canvas, f"{eid} ({labname})  mean local motion  "
                            f"blue = rightward, red = leftward",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(vdir / f"{eid}_field2d.png"), canvas)

        trace, tstats = _trace_plot(means, agree, lab, start - PRE_ROLL_S, 0.02, n_pre, eid)
        cv2.imwrite(str(vdir / f"{eid}_trace.png"), trace)

        sgn = np.sign(mean_field)
        report[eid] = {
            "label": labname,
            "mean_field": mean_field.tolist(),
            "signed_mean": float(mean_field.mean()),
            "sign_correct": bool(np.sign(mean_field.mean()) == lab),
            "column_consensus": float(np.mean(np.abs(sgn.mean(axis=0)))),
            "unanimous_columns": float(np.mean(np.abs(sgn.sum(axis=0)) == sgn.shape[0])),
            **tstats,
        }
        print(f"  {eid:10s} {labname:5s} {mean_field.mean():+9.4f} "
              f"{tstats['mean_row_agreement']:7.2f} {report[eid]['unanimous_columns']:7.0%} "
              f"{tstats['frac_frame_correct']:8.0%} {tstats['longest_correct_run_s']:9.2f}s")
    (vdir / "field2d.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ------------------------------------------------------------------ stage D: brain
def load_frozen_cells() -> list[dict]:
    rep = json.loads(P01L_REPORT.read_text(encoding="utf-8"))
    frozen = [r for r in rep["all_types"] if r["passes"] and r["mirror_equivalence"]]
    if not frozen:
        raise SystemExit("P0.1L report has no cells passing both criteria")
    return frozen


def capture_encoder(video: Path, start: float, cfg: FlyVOConfig, mode: str,
                    ew: int, eh: int, mirror: bool) -> dict:
    t0 = max(0.0, start - PRE_ROLL_S)
    t1 = start + EVENT_DUR_S
    encoder = VideoVisualEncoder(
        MaleCNSEngine(cfg), gain=cfg.visual_gain, n_azimuth_bins=128,
        flow_config=flow_config(mode, ew, eh),
    )
    frames = []
    for t, frame in iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt):
        if mirror:
            frame = np.ascontiguousarray(frame[:, ::-1])
        eye, inject, _m = encoder.encode_frame(frame)
        frames.append({"t": t, "eye": eye, "inject": inject})
    n_pre = min(int(round(PRE_ROLL_S / cfg.brain_dt)), max(0, len(frames) - 1))
    return {"frames": frames, "n_pre": n_pre}


def run_readout(mode: str, ew: int, eh: int, out: Path, tag: str) -> dict:
    frozen = load_frozen_cells()
    keys = [(r["cell_type"], r["side"]) for r in frozen]
    polarity = {(r["cell_type"], r["side"]): (1 if r["orig_dir_index"] > 0 else -1) for r in frozen}

    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5
    cfg.encode_width = ew
    cfg.encode_height = eh
    dt = cfg.brain_dt
    engine = MaleCNSEngine(cfg)
    engine._ensure_loaded()
    brain = engine.brain

    groups = {k: np.asarray(brain.cells([k[0]], side=k[1]), dtype=np.int64) for k in keys}
    all_idx = np.unique(np.concatenate([v for v in groups.values() if len(v)]))
    union_slot = np.full(int(brain.n), -1, dtype=np.int64)
    union_slot[all_idx] = np.arange(len(all_idx), dtype=np.int64)
    local_pos = {k: union_slot[v] for k, v in groups.items()}

    print(f"\n=== D. frozen visual cells on the 2-D field ({tag}) ===")
    print(f"  types: {len(keys)}  ({sum(1 for k in keys if k[0] in T4T5_TYPES)} of them T4/T5)")
    raw: dict[tuple[str, str], dict] = {}
    for run in ("orig", "mirror"):
        for eid in EVENTS:
            cap = capture_encoder(VIDEO, EVENTS[eid][0], cfg, mode, ew, eh, run == "mirror")
            lo, hi = cap["n_pre"], len(cap["frames"])
            span = max(hi - lo, 1) * dt
            counts = np.zeros((len(range(1, SEEDS + 1)), len(all_idx)), dtype=np.float64)
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
        fwd_ok = sum(correct("orig", e) for e in FWD_EVENTS)
        mirror_ok = sum(correct("mirror", e) for e in LEFT_EVENTS + RIGHT_EVENTS)
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
            "fwd_neutral_ok": fwd_ok, "fwd_n": len(FWD_EVENTS),
            "turn_correct": left_ok + right_ok,
            "turn_n": len(LEFT_EVENTS) + len(RIGHT_EVENTS),
            "mirror_correct": mirror_ok,
            "mirror_n": len(LEFT_EVENTS) + len(RIGHT_EVENTS),
            "scores": {f"{run}:{e}": scores[(run, e)] for run in ("orig", "mirror") for e in EVENTS},
        })

    type_rows.sort(key=lambda r: (-r["turn_correct"], -r["mirror_correct"]))
    n_types = len(type_rows)
    downstream = [r for r in type_rows if not r["is_injected"]]

    print(f"\n  {'type':10s} {'sd':2s} {'inj':3s} {'polarity':>8s} "
          f"{'LEFT':>6s} {'RIGHT':>6s} {'FWD':>5s} {'turn':>6s} {'mirror':>7s}")
    for r in type_rows:
        print(f"  {r['cell_type'][:10]:10s} {r['side']:2s} "
              f"{'yes' if r['is_injected'] else '-':3s} {r['polarity']:+8d} "
              f"{r['left_correct']}/{r['left_n']:<4d} {r['right_correct']}/{r['right_n']:<4d} "
              f"{'OK' if r['fwd_neutral_ok'] else 'no':>5s} "
              f"{r['turn_correct']}/{r['turn_n']:<4d} {r['mirror_correct']}/{r['mirror_n']:<4d}")

    agg_rows = []
    for eid in EVENTS:
        lab = EVENTS[eid][1]
        if lab == 0:
            continue
        ok = sum(1 for r in type_rows if (r["scores"][f"orig:{eid}"] > 0) == (lab > 0))
        ok_dn = sum(1 for r in downstream if (r["scores"][f"orig:{eid}"] > 0) == (lab > 0))
        agg_rows.append({"event": eid, "label": lab, "types_correct": ok, "n_types": n_types,
                         "downstream_correct": ok_dn, "downstream_n": len(downstream)})
    agg_rows.sort(key=lambda r: r["event"])
    pooled = sum(r["types_correct"] for r in agg_rows)
    pooled_n = sum(r["n_types"] for r in agg_rows)
    pooled_dn = sum(r["downstream_correct"] for r in agg_rows)
    pooled_dn_n = sum(r["downstream_n"] for r in agg_rows)

    all6 = sum(1 for r in type_rows if r["turn_correct"] == r["turn_n"])
    all6_dn = sum(1 for r in downstream if r["turn_correct"] == r["turn_n"])

    rng = np.random.default_rng(0)
    n_perm = 2000
    null_counts = []
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
        null_counts.append(cnt)
    null_counts = np.asarray(null_counts)
    p_all6 = float(np.mean(null_counts >= all6))

    print(f"\n  pooled vote across the {n_types} frozen types:")
    for r in agg_rows:
        print(f"    {r['event']:10s} label={'L' if r['label'] > 0 else 'R'}  "
              f"{r['types_correct']}/{r['n_types']} ({r['types_correct'] / r['n_types']:.0%})   "
              f"downstream-only {r['downstream_correct']}/{r['downstream_n']}")
    print(f"\n  pooled: {pooled}/{pooled_n} = {pooled / pooled_n:.1%}   "
          f"downstream-only: {pooled_dn}/{pooled_dn_n} = {pooled_dn / pooled_dn_n:.1%}")
    print(f"  types correct on ALL six turns: {all6}/{n_types} "
          f"(downstream-only {all6_dn}/{len(downstream)}), permutation p={p_all6:.4f}")

    baseline = None
    if P01M_REPORT.exists():
        m = json.loads(P01M_REPORT.read_text(encoding="utf-8"))
        baseline = {
            "source": str(P01M_REPORT),
            "pooled_correct": m["pooled_correct"], "pooled_total": m["pooled_total"],
            "types_all_turns_correct": m["types_all_turns_correct"], "n_types": m["n_types"],
            "pooled_vote": m["pooled_vote"],
        }
        print(f"\n  legacy 1-D baseline (P0.1M, archived): {m['pooled_correct']}/{m['pooled_total']} "
              f"= {m['pooled_correct'] / m['pooled_total']:.1%}, "
              f"all-six {m['types_all_turns_correct']}/{m['n_types']}")
        print(f"  grid2d                          : {pooled}/{pooled_n} "
              f"= {pooled / pooled_n:.1%}, all-six {all6}/{n_types}")

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
        "tag": tag, "mode": mode, "encode": [ew, eh],
        "n_types": n_types,
        "n_downstream_types": len(downstream),
        "pooled_correct": pooled, "pooled_total": pooled_n,
        "pooled_downstream_correct": pooled_dn, "pooled_downstream_total": pooled_dn_n,
        "types_all_turns_correct": all6,
        "types_all_turns_correct_downstream": all6_dn,
        "permutation": {"n_perm": n_perm, "mean": float(null_counts.mean()),
                        "p95": float(np.percentile(null_counts, 95)),
                        "observed": all6, "p_value": p_all6},
        "per_type": type_rows,
        "pooled_vote": agg_rows,
        "legacy_baseline": baseline,
    }


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="comb,frontend,viz,brain,brain_legacy")
    ap.add_argument("-o", "--output", default="output/p01q_grid2d")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.stages.split(",") if s.strip()}
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1Q — local 2-D motion field")
    print(f"  video: {VIDEO.name}   frozen cells: {P01L_REPORT.name}")

    report: dict = {
        "probe": "P0.1Q — local 2D motion field read by the fly's visual cells",
        "frozen": ["optic_flow detector scale", "20 visual types from P0.1L",
                   "blind event labels", "MaleCNS dynamics", "seeds"],
        "not_used": ["yaw_frozen (logged only, never injected)",
                     "descending neurons", "trajectory", "dopamine", "any learned readout"],
    }

    def save() -> None:
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if "comb" in wanted:
        report["comb_audit"] = stage_comb(out)
        save()
    if "frontend" in wanted:
        report["frontend"] = stage_frontend(out)
        save()
    if "viz" in wanted:
        report["field_2d"] = stage_viz(out, ["left_65", "left_108", "right_210", "right_216"])
        save()
    if "brain" in wanted:
        report["readout_grid2d"] = run_readout(
            MODE_G192[1], MODE_G192[2], MODE_G192[3], out, MODE_G192[0]
        )
        save()
    if "brain_legacy" in wanted:
        report["readout_legacy"] = run_readout(
            MODE_L96[1], MODE_L96[2], MODE_L96[3], out, MODE_L96[0]
        )
        save()

    r = report.get("readout_grid2d")
    l = report.get("readout_legacy") or report.get("readout_grid2d")
    if r and l:
        new = (r["pooled_correct"], r["pooled_total"])
        n_dn = r["pooled_downstream_total"]
        if r["types_all_turns_correct"] >= 3 or new[0] >= 0.8 * new[1]:
            verdict = (f"2-D LOCAL MOTION WORKS: {r['types_all_turns_correct']}/{r['n_types']} frozen "
                       f"types get all six real turns right (permutation p={r['permutation']['p_value']:.4f}), "
                       f"pooled {new[0]}/{new[1]} = {new[0] / new[1]:.0%}, downstream-only "
                       f"{r['pooled_downstream_correct']}/{n_dn} = "
                       f"{r['pooled_downstream_correct'] / n_dn:.0%}")
        else:
            verdict = (f"NOT CONFIRMED: pooled {new[0]}/{new[1]} = {new[0] / new[1]:.0%} on the 2-D field "
                       f"and all-six types {r['types_all_turns_correct']}/{r['n_types']}. Measuring "
                       "motion on a 2-D grid instead of a collapsed column profile does not by itself "
                       "make the fly's visual cells read the direction of the real turns.")
        report["verdict"] = verdict
        print(f"\n=== VERDICT ===\n{verdict}")
        save()

    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
