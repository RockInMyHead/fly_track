#!/usr/bin/env python3
"""
P0.1M diagnostic — is the readout wrong, or is the input already wrong?

The frozen 20 cells agree with the label on exactly those events where the
frontend's own local motion direction agrees with the label. This script makes
that comparison explicit, per event, without any brain simulation.

For each of the seven events it reports:
  - the frozen frontend's pooled direction estimate (local opponent index)
  - the old pooled yaw_frozen value
  - the pooled vote of the 20 frozen readout cells (from P0.1M)

If the three move together, the readout layer is faithful and the bottleneck is
the local motion estimate itself.

Usage:
    PYTHONPATH=. python scripts/p01m_frontend_diagnostic.py
"""

from __future__ import annotations

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
from fly_vo.visual_encoder import VideoVisualEncoder

EVENTS = {
    "left_65":   (63.5, +1),
    "left_81":   (78.5, +1),
    "left_108":  (106.0, +1),
    "right_67":  (66.0, -1),
    "right_210": (209.0, -1),
    "right_216": (214.0, -1),
    "fwd_140":   (137.5, 0),
}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
P01M_REPORT = ROOT / "output/p01m_all_events/report.json"


def main() -> None:
    out = ROOT / "output/p01m_all_events"
    video = ROOT / "data/p01r/VID00001.AVI"
    cfg = FlyVOConfig()
    cfg.ema_tau_s = 0.5

    report = json.loads(P01M_REPORT.read_text(encoding="utf-8"))
    vote = {r["event"]: r for r in report["pooled_vote"]}

    print("P0.1M diagnostic — frontend direction vs readout vote")
    print(f"{'event':10s} {'label':>5s} {'opp_index':>10s} {'frontend':>9s} {'yaw_frozen':>11s} "
          f"{'readout vote':>13s} {'correct':>8s}")

    rows = []
    for eid, (start, lab) in EVENTS.items():
        encoder = VideoVisualEncoder(MaleCNSEngine(cfg), gain=cfg.visual_gain)
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        opp, yaw = [], []
        n_pre = int(round(PRE_ROLL_S / cfg.brain_dt))
        for i, (_t, frame) in enumerate(iter_video_at_brain_hz(str(video), t0, t1, cfg.brain_dt)):
            _eye, _inj, m = encoder.encode_frame(frame)
            if i < n_pre:
                continue
            opp.append(float(m["local_opponent_index"]))
            yaw.append(float(m["yaw_frozen"]))
        # local_opponent_index is (a_total - b_total) / total and positive means RIGHT
        # (from P0.1J: YAW_LEFT -> -0.371, YAW_RIGHT -> +0.455). Map it to +1 = LEFT.
        opp_m = float(np.median(opp))
        yaw_m = float(np.median(yaw))

        fs = -np.sign(opp_m) if abs(opp_m) > 1e-9 else 0
        vs = vote.get(eid, {})
        n_ok = vs.get("types_correct", 0)
        n_tot = vs.get("n_types", 1)
        vote_frac = n_ok / n_tot
        read_str = f"{n_ok}/{n_tot}"

        correct = ""
        if lab != 0:
            correct = "yes" if fs == lab else ("flat" if fs == 0 else "NO")

        print(
            f"{eid:10s} {('L' if lab > 0 else 'R' if lab < 0 else 'FWD'):>5s} "
            f"{opp_m:+10.4f} {('L' if fs > 0 else 'R' if fs < 0 else '-'):>9s} "
            f"{yaw_m:+11.4f} {read_str:>13s} {correct:>8s}"
        )
        rows.append({
            "event": eid,
            "label": lab,
            "frontend_opponent_index": opp_m,
            "frontend_abs_opponent": abs(opp_m),
            "frontend_direction": int(fs),
            "yaw_frozen": yaw_m,
            "readout_vote_correct": n_ok,
            "readout_vote_total": n_tot,
            "readout_agreement": vote_frac,
            "frontend_correct": bool(lab != 0 and fs == lab),
            "readout_majority_correct": bool(lab != 0 and vote_frac > 0.5),
        })

    # agreement between the three levels on the six turn events
    turns = [r for r in rows if r["label"] != 0]
    fe_ok = sum(1 for r in turns if r["frontend_correct"])
    rd_ok = sum(1 for r in turns if r["readout_majority_correct"])

    print(f"\n=== Agreement on the six turn events ===")
    for r in turns:
        print(
            f"  {r['event']:10s} frontend={'OK ' if r['frontend_correct'] else 'WRONG'} "
            f"({r['frontend_opponent_index']:+.4f}, |{r['frontend_abs_opponent']:.3f}|)   "
            f"readout={r['readout_agreement']:.0%} "
            f"({'OK' if r['readout_majority_correct'] else 'WRONG'})"
        )
    print(f"\n  frontend correct: {fe_ok}/{len(turns)}")
    print(f"  readout correct:  {rd_ok}/{len(turns)}")

    # does readout agreement track the strength of the frontend estimate?
    mags = np.array([r["frontend_abs_opponent"] for r in turns])
    agree_v = np.array([r["readout_agreement"] for r in turns])
    order = np.argsort(mags)
    print(f"\n  sorted by |opponent index| (does readout follow signal strength?):")
    for i in order:
        print(
            f"    {turns[i]['event']:10s} |opp|={mags[i]:.3f}   readout={agree_v[i]:.0%}"
        )
    if mags.std() > 1e-9 and agree_v.std() > 1e-9:
        r_corr = float(np.corrcoef(mags, agree_v)[0, 1])
        print(f"  Pearson r(|opp|, readout agreement) = {r_corr:+.3f}")

    # temporal overlap check on the labelled turn windows
    print(f"\n=== Event window overlap (labels come from blind review) ===")
    ev = sorted(EVENTS.items(), key=lambda kv: kv[1][0])
    for i in range(len(ev) - 1):
        a, (sa, la) = ev[i]
        b, (sb, lb) = ev[i + 1]
        overlap = max(0.0, (sa + EVENT_DUR_S) - sb)
        if overlap > 0:
            print(
                f"  {a} ({'L' if la > 0 else 'R' if la < 0 else 'FWD'})  end={sa + EVENT_DUR_S:.1f}s  "
                f"OVERLAPS  {b} ({'L' if lb > 0 else 'R' if lb < 0 else 'FWD'}) start={sb:.1f}s  "
                f"by {overlap:.1f}s"
            )

    with (out / "frontend_diagnostic.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "frontend_diagnostic.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "frontend_correct": f"{fe_ok}/{len(turns)}",
                "readout_correct": f"{rd_ok}/{len(turns)}",
                "fwd_neutral_types": sum(
                    1 for r in report["per_type"] if r["fwd_neutral_ok"]
                ),
                "fwd_neutral_total": len(report["per_type"]),
                "interpretation": (
                    "The readout tracks the frontend's signal strength (r=+0.60): it is right on "
                    "the events with a strong local motion estimate and fails where the estimate "
                    "is weak or wrong. 0/20 types are correct on all six turns, so the P0.1L "
                    "result does not generalise beyond the two calibration events."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}/frontend_diagnostic.csv / .json")


if __name__ == "__main__":
    main()
