#!/usr/bin/env python3
"""P13.1 — stop the walker when the camera is not moving.

THE DEFECT THIS FIXES, AND ONLY THIS ONE
----------------------------------------
The route advances on a speed signal that is not a measurement of movement. Over VID00002 the
correlation between that signal and the actual frame-to-frame change of the video is -0.01, and
during a two-minute standstill it is higher than during the walking either side. So the walker
walks at roughly half pace always, reaches junctions that were never visited, and takes decisions
there. Measured: 318 of 3364 metres over seven chunks were covered while the person stood still,
and 184 of those metres are in VID00002 alone.

This adds one condition and changes nothing else:

    video_motion(t) < threshold   ->  progress speed 0
    otherwise                     ->  the speed as it was

It does NOT make the speed correct. Whether the person walks at 0.7 or 1.3 metres per second is a
question this does not touch, and the phase that owns it is a different one.

WHY THE VIDEO AND NOT THE BRAIN
-------------------------------
`net_displacement` from P12 answers a different question and would be harmful here: its
NO_NET_DISPLACEMENT class includes a person who walked and came back, so gating on it would freeze
the walker during real movement. The video, by contrast, answers "is the camera still?" well, and
"how fast is the person going?" poorly — which is exactly the split needed.

THE INDICATOR IS NOT NEW
------------------------
The motion measure is imported from `scripts/p13_stillness.py`, the script that performed the
audit, so there is one implementation rather than two that agree today. The threshold is 12, fixed
before any of the held-out chunks were looked at. 14 is reported alongside as a robustness check
and is not available for choosing between.

Usage:
    PYTHONPATH=. python scripts/p13_1_gate.py --chunks VID00002,VID00003
    PYTHONPATH=. python scripts/p13_1_gate.py --all-available
    PYTHONPATH=. python scripts/p13_1_gate.py --all-available --frozen
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
sys.path.insert(0, str(ROOT / "scripts"))

import p08_graph as G  # noqa: E402
from p08_graph_tracker import load_signals, relative_speed, track  # noqa: E402
from scripts.p13_stillness import THRESHOLDS, frame_motion  # noqa: E402

OUT = ROOT / "output/p13"
MEDIA = ROOT / "webapp/media"
MOTION_CACHE = OUT / "motion"
THRESHOLD = 12.0          # the audit's value, frozen before the held-out chunks were seen
PIX_PER_M = 0.049629166698546515


def motion(video: str) -> tuple[np.ndarray, float] | None:
    """Frame-to-frame change of the video, from the audit's own function, cached."""
    MOTION_CACHE.mkdir(parents=True, exist_ok=True)
    cache = MOTION_CACHE / f"{video}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["mot"], float(d["dt"])
    prev = MEDIA / f"{video}_fixed.mp4"
    if not prev.exists():
        return None
    mot, dt = frame_motion(prev)
    np.savez_compressed(cache, mot=mot, dt=np.float32(dt))
    return mot, dt


def path_of(video: str) -> tuple[np.ndarray, np.ndarray]:
    p = OUT / f"runs/{video}/graph_trajectory.csv"
    if not p.exists():
        return np.zeros(0), np.zeros(0)
    t, xy = [], []
    with p.open() as fh:
        for r in csv.DictReader(fh):
            t.append(float(r["t"]))
            xy.append((float(r["x_px"]), float(r["y_px"])))
    return np.asarray(t), np.asarray(xy)


def metres_in_stop(t: np.ndarray, xy: np.ndarray, mot_t: np.ndarray, mot: np.ndarray,
                   th: float) -> float:
    """Metres of this route covered while the video was not moving."""
    if len(t) < 2:
        return 0.0
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    mid = (t[:-1] + t[1:]) / 2.0
    m = np.interp(mid, mot_t, mot)
    return float((d * (m < th)).sum() * PIX_PER_M)


def walk(video: str, rel: np.ndarray, t: np.ndarray, yaw: np.ndarray,
         g: G.Graph) -> dict | None:
    """Re-walk with the speed that was handed in, using the chunk's own start."""
    rj = OUT / f"runs/{video}/run_start.json"
    rep = OUT / f"runs/{video}/report.json"
    if not rj.exists() or not rep.exists():
        return None
    st = json.loads(rj.read_text(encoding="utf-8"))
    pps = float(json.loads(rep.read_text(encoding="utf-8"))["params"]["pix_per_sec"])
    start = {"type": "edge", "edge": st["edge"], "from": st["from"],
             "to": g.other(st["edge"], st["from"]), "hypothesis": True}
    return track(g, t, yaw, rel, pix_per_sec=pps, start=start)


def analyse(video: str, g: G.Graph, th: float, verbose: bool = False) -> dict | None:
    mo = motion(video)
    if mo is None:
        return None
    mot, dt = mo
    mot_t = np.arange(len(mot)) * dt
    t, yaw, tf, speed, _alt = load_signals(video, "yaw_signal_deadband")
    rel = relative_speed(speed)

    # the gate, and nothing else
    m_at = np.interp(t, mot_t, mot)
    rel_gated = np.where(m_at < th, 0.0, rel)

    t_old, xy_old = path_of(video)
    if len(t_old) < 2:
        return None

    res_new = walk(video, rel_gated, t, yaw, g)
    if res_new is None:
        return None

    seq = res_new["edge_sequence"]
    # the walk returns a per-frame path with pixel coordinates, which is what the metres have to be
    # measured from: the edge sequence names passages, not positions along them
    pn = res_new.get("path") or []
    if len(pn) > 1:
        t_new = np.asarray([float(q["t"]) for q in pn])
        xy_new = np.asarray([[float(q["x_px"]), float(q["y_px"])] for q in pn])
    else:
        t_new, xy_new = np.zeros(0), np.zeros((0, 2))

    old_m = float(np.hypot(np.diff(xy_old[:, 0]), np.diff(xy_old[:, 1])).sum() * PIX_PER_M)
    old_stop = metres_in_stop(t_old, xy_old, mot_t, mot, th)
    new_m = float(np.hypot(np.diff(xy_new[:, 0]), np.diff(xy_new[:, 1])).sum() * PIX_PER_M) \
        if len(xy_new) > 1 else 0.0
    new_stop = metres_in_stop(t_new, xy_new, mot_t, mot, th) if len(xy_new) > 1 else 0.0

    dec_old = list(csv.DictReader((OUT / f"runs/{video}/decisions.csv").open()))
    dec_new = res_new["decisions"]

    def in_stop(ts):
        return sum(1 for x in ts if np.interp(x, mot_t, mot) < th)

    row = {
        "video": video,
        "duration_s": float(t[-1]),
        "stop_fraction": float((mot < th).mean()),
        "old_route_m": old_m,
        "old_m_in_stop": old_stop,
        "new_route_m": new_m,
        "new_m_in_stop": new_stop,
        "false_m_removed": old_stop - new_stop,
        "old_decisions": len(dec_old),
        "old_decisions_in_stop": in_stop([float(d["time"]) for d in dec_old]),
        "new_decisions": len(dec_new),
        "new_decisions_in_stop": in_stop([float(d["time"]) for d in dec_new]),
        "new_edges": len(seq),
        "old_edges": len(t_old) and len(list(csv.DictReader(
            (OUT / f"runs/{video}/edge_sequence.csv").open()))),
    }

    # how far the times of reaching a node moved, on the nodes both walks visited
    old_seq = list(csv.DictReader((OUT / f"runs/{video}/edge_sequence.csv").open()))
    out_old = {r["edge"]: float(r["time_end"]) for r in old_seq}
    out_new = {s["edge"]: float(s["time_end"]) for s in seq}
    shared = set(out_old) & set(out_new)
    diffs = [out_new[k] - out_old[k] for k in shared]
    row["shared_edges"] = len(shared)
    row["time_shift_median_s"] = float(np.median(diffs)) if diffs else None
    row["time_shift_max_s"] = float(max(diffs, key=abs)) if diffs else None

    # Where the new route ends relative to the clip. Gating removes the time spent standing, so the
    # walker covers fewer passages within the same recording and may reach the end of its route
    # before the recording does. That is a consequence to be stated, not hidden: the route no longer
    # spans the whole clip, and the tail of the clip is where the person stood still.
    row["new_last_t"] = float(t_new[-1]) if len(t_new) else None
    row["old_last_t"] = float(t_old[-1]) if len(t_old) else None
    row["new_final_edge"] = res_new.get("final_edge")
    row["old_final_edge"] = list(csv.DictReader(
        (OUT / f"runs/{video}/edge_sequence.csv").open()))[-1]["edge"]

    if verbose:
        row["_new_sequence"] = [{"edge": s["edge"], "t": s["time_end"]} for s in seq]
        row["_old_sequence"] = [{"edge": r["edge"], "t": float(r["time_end"])} for r in old_seq]
    row["_new_path"] = [[round(float(x), 1), round(float(y), 1)] for x, y in xy_new] \
        if len(xy_new) > 1 else []
    row["_new_t"] = [round(float(x), 2) for x in t_new] if len(t_new) else []
    return row


def window_compare(video: str, row: dict, t0: float, t1: float) -> dict | None:
    """Before and after the gate inside one time window, which is what makes the fix checkable."""
    mo = motion(video)
    if mo is None:
        return None
    mot, dt = mo
    mot_t = np.arange(len(mot)) * dt
    th = THRESHOLD

    t_old, xy_old = path_of(video)
    d_old = np.hypot(np.diff(xy_old[:, 0]), np.diff(xy_old[:, 1]))
    mid_old = (t_old[:-1] + t_old[1:]) / 2
    in_win_old = (mid_old >= t0) & (mid_old <= t1)
    still_old = in_win_old & (np.interp(mid_old, mot_t, mot) < th)

    t_new = np.asarray(row.get("_new_t") or [])
    xy_new = np.asarray(row.get("_new_path") or [])
    out = {
        "window": [t0, t1],
        "still_fraction": float((np.interp(mid_old, mot_t, mot)[in_win_old] < th).mean()),
        "old_m": float(d_old[in_win_old].sum() * PIX_PER_M),
        "old_m_while_still": float(d_old[still_old].sum() * PIX_PER_M),
    }
    if len(xy_new) > 1:
        d_new = np.hypot(np.diff(xy_new[:, 0]), np.diff(xy_new[:, 1]))
        mid_new = (t_new[:-1] + t_new[1:]) / 2
        in_win_new = (mid_new >= t0) & (mid_new <= t1)
        still_new = in_win_new & (np.interp(mid_new, mot_t, mot) < th)
        out["new_m"] = float(d_new[in_win_new].sum() * PIX_PER_M)
        out["new_m_while_still"] = float(d_new[still_new].sum() * PIX_PER_M)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default="")
    ap.add_argument("--all-available", action="store_true")
    ap.add_argument("--frozen", action="store_true",
                    help="отметить, что порог уже заморожен (см. FROZEN_STILLNESS_GATE.json)")
    ap.add_argument("--json", default=str(OUT / "gate_report.json"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    order = json.loads((OUT / "CHAIN_ANCHOR.json").read_text(encoding="utf-8"))["order"]
    if a.chunks:
        todo = [v.strip() for v in a.chunks.split(",") if v.strip()]
    elif a.all_available:
        todo = [v for v in order if (MEDIA / f"{v}_fixed.mp4").exists()]
    else:
        print("укажите --chunks или --all-available")
        return 1

    g = G.Graph.load(ROOT / "data/p08/graph.json")
    print("=" * 108)
    print("P13.1 — запрет шагания, когда камера не движется")
    print("=" * 108)
    print(f"  признак движения: тот же, что в аудите P13 (кадровая разница, 2 Гц, 200 px)")
    print(f"  порог {THRESHOLD} (заморожен до просмотра VID00007/00008/00011-00017); "
          f"{THRESHOLDS[1]} — только проверка устойчивости")
    print(f"  кусков к разбору: {len(todo)}")
    print()

    rows = []
    for v in todo:
        r = analyse(v, g, THRESHOLD, a.verbose)
        if r is None:
            print(f"  {v}: нет видео или прогона — пропуск")
            continue
        rows.append(r)

    print(f"  {'кусок':<10}{'стоп,%':>7}{'старый':>8}{'новый':>8}{'убрано':>8}"
          f"{'разв.в стоп':>12}{'разв.всего':>11}{'сдвиг t':>9}{'конец маршр.':>13}")
    print("  " + "-" * 96)
    for r in rows:
        shift = r["time_shift_median_s"]
        shift_s = f"{shift:+.0f}с" if shift is not None else "—"
        last = r.get("new_last_t")
        last_s = f"{last:.0f}с" if last else "—"
        print(f"  {r['video']:<10}{100*r['stop_fraction']:>6.0f}%"
              f"{r['old_m_in_stop']:>8.0f}{r['new_m_in_stop']:>8.0f}"
              f"{r['false_m_removed']:>8.0f}"
              f"{str(r['old_decisions_in_stop'])+'→'+str(r['new_decisions_in_stop']):>12}"
              f"{str(r['old_decisions'])+'→'+str(r['new_decisions']):>11}"
              f"{shift_s:>9}{last_s:>13}")
    print()
    if rows:
        o = sum(r["old_m_in_stop"] for r in rows)
        n = sum(r["new_m_in_stop"] for r in rows)
        print(f"  ИТОГО: было пройдено в стоп {o:.0f} м, стало {n:.0f} м, "
              f"убрано {o-n:.0f} м ({100*(o-n)/max(o,1):.0f}%)")
        print()

    # the named check
    r2 = next((x for x in rows if x["video"] == "VID00002"), None)
    win = window_compare("VID00002", r2, 840.0, 1020.0) if r2 else None
    if win and r2:
        print("  ОСОБО VID00002, окно 840–1020 с (минуты 14–17):")
        print(f"    человек стоял: {100*win['still_fraction']:.0f}% времени")
        print(f"    ДО гейта:   {win['old_m']:.0f} м, из них в стоянии "
              f"{win['old_m_while_still']:.0f} м")
        if "new_m" in win:
            print(f"    ПОСЛЕ гейта: {win['new_m']:.0f} м, из них в стоянии "
                  f"{win['new_m_while_still']:.0f} м")
            print(f"    ложное продвижение в окне убрано: "
                  f"{win['old_m_while_still']-win['new_m_while_still']:.0f} м")
        print(f"    по всему куску: маршрут {r2['old_route_m']:.0f} → "
              f"{r2['new_route_m']:.0f} м, конец на {r2.get('old_last_t') or 0:.0f} → "
              f"{r2.get('new_last_t') or 0:.0f} с")
        print()

    # robustness: the same run at the second threshold, reported not chosen
    if rows:
        alts = []
        for v in [r["video"] for r in rows]:
            rr = analyse(v, g, THRESHOLDS[1])
            if rr:
                alts.append(rr)
        if alts:
            o = sum(r["old_m_in_stop"] for r in rows)
            n1 = sum(r["new_m_in_stop"] for r in rows)
            n2 = sum(r["new_m_in_stop"] for r in alts)
            print(f"  УСТОЙЧИВОСТЬ: при пороге {THRESHOLD} остаётся {n1:.0f} м стоя, "
                  f"при пороге {THRESHOLDS[1]} — {n2:.0f} м. Из {o:.0f} м ложных.")

    dest = Path(a.json)
    keep = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    dest.write_text(json.dumps({
        "phase": "P13.1 — STOP-гейт", "threshold": THRESHOLD,
        "robustness_threshold": THRESHOLDS[1],
        "indicator": "кадровая разница: ffmpeg -vf fps=2,scale=200:-2 -q:v 5, "
                     "MAD между соседними кадрами в сером",
        "logic": "video_motion < threshold -> progress_speed = 0, иначе speed без изменений",
        "not_claimed": "скорость движения НЕ исправлена; вопрос «0.7 или 1.3 м/с» не решается",
        "window_check_VID00002": win,
        "chunks": keep,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n  записано: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
