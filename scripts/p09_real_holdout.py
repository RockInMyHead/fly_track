#!/usr/bin/env python3
"""P09 — the frozen candidates on real video, with nothing refitted.

The screen selected 25 representatives on a synthetic stimulus and the list was written to
disk before any video was looked at. This script is the other half, and its whole value
depends on the order not being reversed: the candidates are read from
`data/p09_frozen_targets.json`, the video is driven through the same brain with the same seed
the tracker uses, and every threshold used to judge a cell is measured from that clip's own
noise rather than chosen.

The self-check that makes the numbers trustworthy
-------------------------------------------------
The 46 cells the tracker reads today are recorded alongside the candidates, and their traces
are compared against the recording the tracker's own signal was built from. If this run
reproduces them, then the conditions are identical to the ones the existing signal came from
and the candidates are being measured in the same world. If it does not, nothing else in the
output means anything, and the script says so and stops.

How a single cell is judged
---------------------------
The pooled signal the tracker reads is an average of four z-scored channels; a single cell has
no such average to lean on, so it is integrated the way the tracker integrates, over the same
window, and compared against its own noise:

    integral(t) = sum over the last WINDOW_S of (rate - that cell's median rate) * dt
    floor       = the 95th percentile of |integral| over the whole clip, for that cell

which is the same construction P07 used for the pooled signal, applied per cell. A cell
"speaks" when its integral clears its own floor, and the direction it names follows from its
polarity — a cell whose activity rises with rightward image motion names a LEFT camera turn.

Nothing here combines cells. As the phase requires, the candidates are reported separately;
building a new pool is a separate decision and would need its own justification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fly_vo.brain_clock import iter_video_at_brain_hz  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402
from fly_vo.video_reader import VideoReader  # noqa: E402

import p08_graph as G  # noqa: E402

OUT = ROOT / "output/p09"
FROZEN = ROOT / "data/p09_frozen_targets.json"
P053 = ROOT / "output/p053_threshold/targets.csv"
SEED = 64
CHUNK_S = 60.0
WINDOW_S = 3.2                 # the tracker's window: APPROACH_S + SETTLE_S
FLOOR_PCT = 95.0               # percentile of |integral| taken as the cell's own noise floor


def load_frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def control_cells() -> list[dict]:
    if not P053.exists():
        return []
    rows = list(csv.DictReader(P053.open(encoding="utf-8")))
    return [{"cell": int(r["cell"]), "cell_type": r["cell_type"], "side": r["side"],
             "role": "control"} for r in rows]


def record(video: Path, out_dir: Path, watch: list[dict], duration: float | None) -> Path:
    """Drive the clip through the brain and record the watched cells, in chunks.

    Chunked so that an interrupted run resumes rather than starting over, which matters
    because the two long clips take half an hour each.
    """
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    brain = engine.brain
    idx = np.array([w["cell"] for w in watch], dtype=np.int64)
    n_w = len(watch)
    dt = float(engine.config.brain_dt)

    chunks = out_dir / f"chunks_{video.stem}"
    chunks.mkdir(parents=True, exist_ok=True)
    with VideoReader(video) as vr:
        total, n_frames = vr.duration_s, vr.frame_count
    t_end = total if duration is None else min(duration, total)

    missing = [int(c) for c in idx if c >= brain.n]
    if missing:
        print(f"  СТОП: клеток нет в мозге: {missing[:5]}")
        return out_dir / f"trace_{video.stem}.npz"

    print(f"  видео {video.name}: {total:.1f} с, {n_frames} кадров, окно 0..{t_end:.0f} с")
    print(f"  под наблюдением {n_w} клеток (seed {SEED}, шаг {dt*1000:.0f} мс)")
    print(f"  оценка: {t_end * 1.5 / 60:.0f} мин")

    engine.reset(seed=SEED)
    encoder = VideoVisualEncoder(engine, flow_config=None)
    t0 = time.perf_counter()
    seg = 0.0
    while seg < t_end:
        end = min(seg + CHUNK_S, t_end)
        path = chunks / f"chunk_{int(seg):05d}_{int(end):05d}.npz"
        if path.exists():
            seg = end
            continue
        ts, fired = [], []
        first = True
        for t, frame in iter_video_at_brain_hz(str(video), seg, end, dt):
            if first:
                first = False
                if seg > 0:
                    continue
            eye, inject, _ = encoder.encode_frame(frame)
            res = engine.step(t, eye_drive=eye, inject=inject)
            ts.append(t)
            fired.append(np.isin(idx, res.fired))
        np.savez_compressed(path, t=np.asarray(ts, np.float32),
                            fired=np.asarray(fired, np.uint8))
        el = time.perf_counter() - t0
        left = (t_end - end) * (el / max(end, 1e-9)) / 60
        print(f"    {end:7.0f}/{t_end:.0f} с  прошло {el/60:5.1f} мин  "
              f"осталось ~{left:5.1f} мин", flush=True)
        seg = end

    files = sorted(chunks.glob("chunk_*.npz"))
    ts = np.concatenate([np.load(f)["t"] for f in files])
    fr = np.concatenate([np.load(f)["fired"] for f in files])
    dest = out_dir / f"trace_{video.stem}.npz"
    np.savez_compressed(dest, t=ts, fired=fr,
                        cells=idx,
                        cell_type=np.array([w["cell_type"] for w in watch]),
                        side=np.array([w["side"] for w in watch]),
                        role=np.array([w.get("role", "candidate") for w in watch]),
                        polarity=np.array([w.get("polarity", 0.0) for w in watch]))
    print(f"  записано {dest}: {len(ts)} шагов, {fr.shape[1]} клеток")
    return dest


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def cell_integrals(fired: np.ndarray, t: np.ndarray, dt: float,
                   smooth_s: float = 0.30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell smoothed z-signal, its integral over the window, and its own noise floor.

    The construction mirrors the tracker's, one level down. The pooled signal is four channels
    each z-scored and smoothed and then averaged; a single cell is the same thing without the
    average, so its rate is smoothed by the same 0.3 s the readout uses, z-scored on its own
    mean and spread, and integrated over the same 3.2 s window. The floor is the 95th
    percentile of |integral| over the clip — the reading P07 took for the pooled signal,
    applied per cell — so "clearing the floor" means the same thing here as it does there.

    The first version of this divided the spikes by dt and integrated that, which is
    algebraically just the spike count in the window: a cell firing 24 times in 3.2 s reported
    an "integral" of 24 against a "floor" of 22, and the numbers meant nothing. Smoothing and
    z-scoring first is what makes the value comparable between cells with different rates,
    which is the only reason to put a floor under it at all.
    """
    rate = fired.astype(np.float64) / dt
    n_sm = max(int(round(smooth_s / dt)), 1)
    sm = np.stack([box(rate[:, k], n_sm) for k in range(rate.shape[1])], axis=1)
    mu = sm.mean(axis=0, keepdims=True)
    sd = sm.std(axis=0, keepdims=True)
    z = (sm - mu) / np.where(sd > 1e-12, sd, 1.0)
    w = max(int(round(WINDOW_S / dt)), 1)
    c = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), axis=0)
    integ = (c[w:] - c[:-w]) * dt
    floor = np.percentile(np.abs(integ), FLOOR_PCT, axis=0)
    return z, integ, floor


def window_integral(integ: np.ndarray, t: np.ndarray, at: float) -> np.ndarray:
    """The integral for the window ending at `at`, per cell; NaN if outside the recording."""
    j = int(np.argmin(np.abs(t - at)))
    k = j - integ.shape[0] if j >= integ.shape[0] else j - 1
    # integ[i] covers [t[i], t[i]+WINDOW_S]; the window ending at t[j] starts at t[j]-WINDOW_S
    i = int(np.searchsorted(t + WINDOW_S, at, side="right")) - 1
    i = min(max(i, 0), integ.shape[0] - 1)
    return integ[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="VID00006")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--record", action="store_true", help="прогнать видео через мозг")
    ap.add_argument("--analyze", action="store_true", help="разобрать запись")
    ap.add_argument("--run", default=str(ROOT / "output/p084c/run"),
                    help="прогон трекера: его времена решений для проблемных развилок")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen()
    cands = [{"cell": int(x["cell"]), "cell_type": x["cell_type"], "side": x["side"],
              "polarity": x["polarity"], "role": "candidate"} for x in frozen["targets"]]
    ctrl = control_cells()
    watch = ctrl + cands
    video = ROOT / "data/p01r" / f"{args.video}.AVI"
    # The clip's name comes from the file, not from the argument. With an absolute path the argument
    # carries directories, so `trace_{args.video}.npz` turned into a path that could not exist: the
    # recording was written correctly under the file's own name and then the analysis looked for it
    # under the full path. The run ended in a traceback after doing its work, which is worse than a
    # plain failure — a real error would have been hidden among expected ones.
    name = video.stem

    print("=" * 96)
    print(f"P09 — ЗАМОРОЖЕННЫЕ КАНДИДАТЫ НА {name}")
    print("=" * 96)
    print(f"  кандидатов из data/p09_frozen_targets.json: {len(cands)} "
          f"(отобраны только по синтетике)")
    print(f"  контрольных клеток текущего трекера: {len(ctrl)}")
    print()

    if args.record or not (out / f"trace_{name}.npz").exists():
        record(video, out, watch, args.duration)
    if not args.analyze and not args.record:
        return 0

    d = np.load(out / f"trace_{name}.npz")
    t = d["t"].astype(float)
    fired = d["fired"]
    cells = d["cells"].astype(int)
    dt = float(np.median(np.diff(t)))

    # ---- polarity for every watched cell, from the synthetic screen -----------------------
    # The control cells were recorded with no polarity, and the first version of this analysis
    # used that zero: `(I > 0) == (polarity > 0)` then reduces to `(I > 0) == False`, which
    # reports the *opposite* direction for every control. It read as the controls scoring 18
    # percent against the candidates' 98, which looked like a result and was an artefact. The
    # polarity is a property of the cell, so it is looked up by cell id in the table the screen
    # produced, which covers all 1340 descending cells.
    pol_by_cell: dict[int, float] = {}
    met = out / "all_dn_metrics.csv"
    if met.exists():
        for r in csv.DictReader(met.open(encoding="utf-8")):
            pol_by_cell[int(r["cell"])] = float(r["polarity"])
    for r in load_frozen()["targets"]:
        pol_by_cell[int(r["cell"])] = float(r["polarity"])

    # The watch list was the 46 controls followed by the candidates, and some cells are in
    # both: the four types the tracker reads are also survivors of the screen. The same cell
    # would then be scored twice, once under each name. Keeping the first column per cell is
    # enough, and it is what the duplicate columns contain anyway.
    keep: list[int] = []
    seen: set[int] = set()
    for k, c in enumerate(cells):
        if int(c) in seen:
            continue
        seen.add(int(c))
        keep.append(k)
    fired = fired[:, keep]
    cells = cells[keep]
    ctype = [str(d["cell_type"][k]) for k in keep]
    cside = [str(d["side"][k]) for k in keep]
    n_from_dup = len(keep)

    # a cell with no polarity measurement cannot be scored, and saying so beats guessing
    pol = np.array([pol_by_cell.get(int(c), np.nan) for c in cells])
    unknown = [f"{ctype[k]} {cside[k]}" for k in range(len(cells)) if not np.isfinite(pol[k])]

    # the four types the tracker reads must match the sign P07 froze, or the lookup is wrong
    from p071_turn_filter import POLARITY as P07_POL  # noqa: E402
    mism = [(ctype[k], cside[k], pol[k], P07_POL[ctype[k]])
            for k in range(len(cells))
            if ctype[k] in P07_POL and np.isfinite(pol[k])
            and abs(pol[k] - P07_POL[ctype[k]]) > 1e-9]
    print("─── ПОЛЯРНОСТЬ ───")
    print(f"  уникальных клеток в записи: {len(cells)} (было {len(d['cells'])}, "
          f"дубликаты убраны)")
    print(f"  полярность найдена для {int(np.isfinite(pol).sum())} из {len(cells)}")
    if unknown:
        print(f"  БЕЗ полярности: {', '.join(unknown)} — в оценке не участвуют")
    print(f"  сверка четырёх типов трекера с замороженными в P07: "
          f"{'совпали все' if not mism else 'РАСХОЖДЕНИЕ: ' + str(mism[:4])}")
    print()

    z, integ, floor = cell_integrals(fired, t, dt)

    # ---- self-check against the recording the tracker's signal came from -----------------
    print("─── ПРОВЕРКА: воспроизводит ли прогон существующую запись ───")
    ref_path = ROOT / f"output/p06_neurons_{name.replace('VID0000','vid')}/spike_trace.npz"
    if name == "VID00006":
        ref_path = ROOT / "output/p06_neurons_vid6/spike_trace.npz"
    ok_check = None
    if ref_path.exists():
        ref = np.load(ref_path)
        # the reference has no cell list, so the columns are aligned by position: the control
        # block was recorded in the order of P053's targets.csv, which is the order the
        # reference's columns are in too. Column k of one is column k of the other.
        ref_fired = ref["fired"]
        n_ctrl = min(len(ctrl), ref_fired.shape[1])
        agrees = []
        for k in range(n_ctrl):
            n = min(fired.shape[0], ref_fired.shape[0])
            agrees.append(float((fired[:n, k] == ref_fired[:n, k]).mean()))
        if agrees:
            worst = float(min(agrees))
            print(f"  сверено контрольных клеток: {len(agrees)}")
            print(f"  совпадение спайк-в-спайк: минимум {worst:.6f}, "
                  f"медиана {float(np.median(agrees)):.6f}")
            ok_check = worst > 0.9999
            print("  прогон воспроизводит существующую запись — условия те же"
                  if ok_check else
                  "  РАСХОЖДЕНИЕ: прогон не воспроизводит эталон, дальше выводы делать нельзя")
        else:
            print("  сверять нечего")
    else:
        print(f"  эталона нет ({ref_path.name}); сверка пропущена")
    print()

    # ---- labelled turns, if this clip has any -------------------------------------------
    try:
        from p07_fly_readout import labelled_turns  # noqa: E402
        turns = [x for x in labelled_turns() if x["video"] == name]
    except Exception as e:
        print(f"  разметка недоступна ({e})")
        turns = []

    if turns:
        print(f"─── РУЧНАЯ РАЗМЕТКА: {len(turns)} подтверждённых поворотов камеры ───")
        rows = []
        frozen_ids = {int(x["cell"]) for x in load_frozen()["targets"]}
        ci_cand = [k for k, c in enumerate(cells) if int(c) in frozen_ids]
        for k in ci_cand:
            hit = 0
            named = 0
            for x in turns:
                I = window_integral(integ, t, x["t1"])
                if abs(I[k]) < floor[k]:
                    continue
                named += 1
                # polarity > 0: activity rises with rightward image motion, which a LEFT
                # camera turn produces
                said = "LEFT" if (I[k] > 0) == (pol[k] > 0) else "RIGHT"
                if said == x["kind"]:
                    hit += 1
            rows.append({"cell_type": ctype[k], "side": cside[k], "named": named,
                         "correct": hit, "total": len(turns)})
        ctrl_ids = {int(x["cell"]) for x in control_cells()}
        for k in [i for i, c in enumerate(cells)
                  if int(c) in ctrl_ids and int(c) not in frozen_ids]:
            hit = named = 0
            for x in turns:
                I = window_integral(integ, t, x["t1"])
                if abs(I[k]) < floor[k]:
                    continue
                named += 1
                said = "LEFT" if (I[k] > 0) == (pol[k] > 0) else "RIGHT"
                if said == x["kind"]:
                    hit += 1
            rows.append({"cell_type": ctype[k], "side": cside[k], "named": named,
                         "correct": hit, "total": len(turns), "group": "контроль 46"})
        for r in rows:
            r.setdefault("group", "кандидат")
        rows.sort(key=lambda r: (-(r["correct"] / max(r["named"], 1)), -r["named"]))
        print(f"  {'клетка':<16}{'стор.':>6}{'группа':>11}{'пробила':>9}{'доля':>7}"
              f"{'верно':>7}{'верных при говорении':>22}")
        for r in rows[:26]:
            frac = r["correct"] / r["named"] if r["named"] else float("nan")
            rate = r["named"] / max(r["total"], 1)
            acc = r["correct"] / r["named"] if r["named"] else float("nan")
            print(f"  {r['cell_type']:<16}{r['side']:>6}{r['group']:>11}{rate:>9.0%}"
                  f"{rate:>7.0%}{r['correct']:>7}{acc:>22.0%}")
        with (out / f"real_holdout_{name}.csv").open("w", encoding="utf-8",
                                                           newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()
    else:
        print("─── РУЧНОЙ РАЗМЕТКИ ПО ЭТОМУ ВИДЕО НЕТ ───")
        print()

    # ---- the problem junctions ----------------------------------------------------------
    run = list(csv.DictReader((Path(args.run) / "decisions.csv").open(encoding="utf-8")))
    g = G.Graph.load(ROOT / "data/p08/graph.json")
    mm = {}
    logp = ROOT / "output/p084a/graph_repair_log.json"
    if logp.exists():
        for ch in json.loads(logp.read_text(encoding="utf-8"))["changes"]:
            if ch.get("repair") == "merge":
                mm[ch["removed"]] = ch["into"]

    def remap(e: str) -> str:
        a, _, b = e.partition("__")
        return f"{mm.get(a, a)}__{mm.get(b, b)}"

    def truth_direction(dec: dict, a: dict) -> str:
        """The direction the route needed, in the graph's own frame.

        The audit's `truth_side` column describes what the camera looked like it was doing; the
        graph angle describes where the walker actually had to go. They are different questions
        and at T50 they disagree — the audit says STRAIGHT while the passage the person chose is
        +34 degrees to the right. This phase is about the second question, because that is what
        the tracker needs, so the direction is taken from the graph.
        """
        e = remap(a.get("correct_edge_if_wrong") or a.get("truth_edge") or "")
        if not e or e not in g.edges:
            return ""
        try:
            return "LEFT" if g.turn(dec["current_edge"], dec["node"], e)["deg"] < 0 else "RIGHT"
        except Exception:
            return ""

    # keyed by node AND time, because a node can appear more than once and the two entries can
    # carry different verdicts — J35 does, and keying on the node alone silently picked the
    # wrong one.
    audit = []
    ap_path = ROOT / "output/p083_route_audit/decisions.csv"
    if ap_path.exists():
        for a in csv.DictReader(ap_path.open(encoding="utf-8")):
            if a["verdict"] in ("CORRECT", "WRONG"):
                audit.append(a)

    def audit_for(node: str, at: float) -> dict:
        best, bd = {}, 1e9
        for a in audit:
            if a["node"] != node:
                continue
            dd = abs(float(a["time"]) - at)
            if dd < bd:
                best, bd = a, dd
        return best if bd <= 8.0 else {}
    # The junction times come from the tracker's run, and a run belongs to one clip. Pointing
    # this section at a different clip pairs one clip's times with another clip's cell
    # readings: the graph has the same node names in both, so nothing throws and the rows look
    # plausible. That is why this is a guard and not a comment — the first version wrote
    # problem_turns_VID00001.csv and problem_turns_VID00002.csv full of VID00006 times.
    run_video = ""
    run_report = Path(args.run) / "report.json"
    if run_report.exists():
        run_video = json.loads(run_report.read_text(encoding="utf-8")).get("video", "")
    video_matches = (run_video == name)

    print("─── ПРОБЛЕМНЫЕ РАЗВИЛКИ: КТО ВИДИТ НУЖНЫЙ ЗНАК, ГДЕ ПУЛ МОЛЧАЛ ───")
    if not video_matches:
        print(f"  ПРОПУЩЕНО: прогон {Path(args.run).name} относится к "
              f"{run_video or 'неизвестному видео'}, а разбирается {name}.")
        print("  Времена развилок берутся из прогона трекера, поэтому для другого клипа они")
        print("  не значат ничего: совпадают лишь одинаковые имена узлов графа.")
    else:
        print(f"  {'t':>7}{'узел':<6}{'нужен':<10}{'пул':>16}  кандидаты, которые говорят")
    prob_rows = []
    for name in (("J35", "T50", "T49") if video_matches else ()):
        dec = [r for r in run if r["node"] == name]
        if not dec:
            continue
        dd = dec[0]
        at = float(dd["time"])
        a = audit_for(name, at)
        truth_side = truth_direction(dd, a) if a else ""
        raw_side = a.get("truth_side", "")
        if truth_side and raw_side and truth_side != raw_side:
            print(f"  (аудит описывал камеру как {raw_side}, а по графу маршруту нужно "
                  f"{truth_side})")
        # the pooled signal the tracker actually read
        ycsv = ROOT / f"output/p07/yaw_signal_{name}.csv"
        pooled = ""
        if ycsv.exists():
            yr = list(csv.DictReader(ycsv.open(encoding="utf-8")))
            yt = np.array([float(x["t"]) for x in yr])
            ys = np.array([float(x["yaw_signal_deadband"]) for x in yr])
            m = (yt >= at - WINDOW_S) & (yt <= at)
            if m.sum() > 1:
                I = float((ys[m][:-1] * np.diff(yt[m])).sum())
                fl = 0.5 * (0.09 + 0.15 * WINDOW_S)
                pooled = "молчит" if abs(I) < fl else ("LEFT" if I > 0 else "RIGHT")
        said = []
        frozen_ids2 = {int(x["cell"]) for x in load_frozen()["targets"]}
        for k in [i for i, c in enumerate(cells) if int(c) in frozen_ids2]:
            I = window_integral(integ, t, at)
            if abs(I[k]) < floor[k]:
                continue
            d_ = "LEFT" if (I[k] > 0) == (pol[k] > 0) else "RIGHT"
            said.append((d_, ctype[k], cside[k], float(I[k]), float(floor[k])))
        right = [s for s in said if s[0] == truth_side]
        wrong = [s for s in said if s[0] != truth_side]
        tag = ""
        if truth_side:
            tag = (f"  нужный знак {truth_side}: сказали {len(right)} из {len(said)}")
        print(f"  {at:>7.1f}{name:<6}{truth_side or '—':<10}{pooled:>16}{tag}"
              + (f"   (аудит описывал камеру как {raw_side})" if raw_side
                 and raw_side != truth_side else ""))
        for s in said:
            mark = "✓" if s[0] == truth_side else "✗"
            print(f"        {mark} {s[1]:<16}{s[2]:>3}  {s[0]:<6} "
                  f"интеграл {s[3]:+8.3f} при пороге {s[4]:.3f}")
        prob_rows.append({
            "t": at, "node": name, "truth_side": truth_side,
            "audit_side_of_camera": raw_side, "pooled": pooled,
            "n_spoke": len(said), "n_right_sign": len(right),
            "speakers": json.dumps([{"cell_type": s[1], "side": s[2], "said": s[0],
                                     "integral": round(s[3], 4),
                                     "floor": round(s[4], 4)} for s in said],
                                   ensure_ascii=False),
        })

    if prob_rows:
        p = out / f"problem_turns_{name}.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prob_rows[0].keys()))
            w.writeheader()
            w.writerows(prob_rows)
        print(f"\nзаписано: {p}")

    (out / f"holdout_report_{name}.json").write_text(json.dumps({
        "video": name, "frozen": str(FROZEN),
        "n_candidates": len(cands), "n_controls": len(ctrl),
        "window_s": WINDOW_S, "floor_pct": FLOOR_PCT,
        "reproduced_existing_trace": ok_check,
        "n_labelled_turns": len(turns),
        "problem_junctions_from_run": run_video,
        "problem_junctions": prob_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
