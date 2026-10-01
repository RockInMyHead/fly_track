#!/usr/bin/env python3
"""P13 — the chain: run the chunks in order and carry the route across every seam.

The queue records the visual drive for each chunk, which is the expensive part and does not depend
on order. This does the cheap part that does: the tracker, walked from the first chunk to the last,
each one starting where the previous one stopped.

    start of chunk 1   given by hand: T12 → J6, from output/p13/CHAIN_ANCHOR.json
    start of chunk N+1 the last edge chunk N was on, along the same direction

It waits for the recordings rather than assuming them, so it can run alongside the queues. After
each chunk it rewrites output/p13/live_trajectory.csv, which is what the dashboard draws, so the
route appears on the plan as the chain advances.

The handover is exact in edge terms and approximate in position: a file boundary falls wherever the
two gigabytes ran out, usually partway along a passage, while the tracker begins an edge at its far
end. Each seam therefore replays up to one passage, about four metres. That is recorded per seam in
the output rather than smoothed over.

Usage:
    PYTHONPATH=. python scripts/p13_chain.py
    PYTHONPATH=. python scripts/p13_chain.py --dry-run     # show the order, touch nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "output/p13"
ANCHOR = OUT / "CHAIN_ANCHOR.json"
CHAIN = OUT / "chain_state.json"
LIVE = OUT / "live_trajectory.csv"
PY = sys.executable


CAMERA = Path("/Volumes/NO NAME/DCIM")
SOURCES = OUT / "sources"


def source_is_camera(video: str) -> bool:
    """Is the local file the same bytes as the camera's?

    The project and the camera disagree about what a number means. The file the project calls
    VID00006 is a 250-second clip while the camera's VID00006 is a 1228-second chunk; VID00002 and
    VID00004 differ in size as well. A recording named after a video is therefore no evidence that
    it was made from the file now carrying that name.
    """
    src = ROOT / f"data/p01r/{video}.AVI"
    cam = CAMERA / f"{video}.AVI"
    if not src.exists() or not cam.exists():
        return False
    return src.stat().st_size == cam.stat().st_size


def fingerprint(video: str) -> dict | None:
    """What the queue wrote when it recorded this chunk, including the source's byte count."""
    p = SOURCES / f"{video}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _last_t(path: Path) -> float | None:
    """Last time stamp of a signal file, without loading it into memory."""
    try:
        with path.open() as fh:
            header = fh.readline()
            if "t" not in header.split(",")[0]:
                return None
            last = None
            for line in fh:
                if line.strip():
                    last = line
            return float(last.split(",")[0]) if last else None
    except Exception:
        return None


def ready(video: str) -> bool:
    """Are this chunk's recordings present, and derived from the right trace?

    The test is a length agreement, and it is the cheap version of the one that matters. A yaw is
    built from a trace; if it was built from the trace that is there now, the two end at the same
    moment. VID00006 is the case that taught this: its yaw ended at 250 seconds while its trace
    ended at 1228, because the yaw had been made from a different, shorter recording that happened
    to share its name — and the walk through it produced 24 edges instead of 111.

    An earlier version measured each chunk by counting frames on the camera, which is accurate but
    takes forty seconds a chunk and made the chain wait on its own bookkeeping. Comparing the files
    to each other costs a few milliseconds and catches the same fault, because the fault is a
    disagreement between two files that are supposed to describe the same chunk.
    """
    tag = "vid" + str(int(video.replace("VID", "")))
    trace = ROOT / f"output/p06_neurons_{tag}/spike_trace.npz"
    yaw = ROOT / f"output/p07/yaw_signal_{video}.csv"
    fwd = ROOT / f"output/p07/forward_signal_{video}.csv"
    traj = ROOT / f"output/p071/trajectory_{video}.csv"
    if not all(p.exists() for p in (trace, yaw, fwd, traj)):
        return False
    try:
        t_trace = round(float(np.load(trace)["t"][-1]), 1)
    except Exception:
        return False
    t_yaw = _last_t(yaw)
    if t_yaw is None or abs(t_yaw - t_trace) > 2.0:
        return False
    for p in (fwd, traj):
        t = _last_t(p)
        if t is None or abs(t - t_yaw) > 2.0:
            return False
    return True


def render(state: dict) -> None:
    order = state["order"]
    done = state["done"]
    print("\n" + "═" * 96)
    print(f"  P13 цепочка   {len(done)}/{len(order)}   старт сейчас: "
          f"{state['start']['from']} → {state['start']['to']} по {state['start']['edge']}")
    print("─" * 96)
    for v in order:
        if v in done:
            d = done[v]
            print(f"   ✔ {v:<10}{d['edges']:>4} рёбер   конец {d['end_edge']:<16}"
                  f"{d['end_from']}→{d['end_to']}")
        elif v == state.get("current"):
            print(f"   ▶ {v:<10}ждём записи…")
        else:
            print(f"     {v:<10}—")
    if state.get("errors"):
        print("─" * 96)
        for e in state["errors"][-3:]:
            print(f"   ✖ {e}")
    print(f"   обновлено {state['updated_at']}")
    print("═" * 96, flush=True)


def write_live() -> None:
    """Every chained run in one file, in order, each tagged with the chunk it came from."""
    runs = []
    for name in ORDER:
        p = OUT / f"runs/{name}/graph_trajectory.csv"
        if p.exists():
            runs.append((name, p))
    OUT.mkdir(parents=True, exist_ok=True)
    with LIVE.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "t_global", "x_px", "y_px", "edge"])
        off = 0.0
        for name, p in runs:
            rows = list(csv.DictReader(p.open()))
            last = 0.0
            for i, r in enumerate(rows):
                last = max(last, float(r["t"]))
                if i % 5:
                    continue
                w.writerow([name, f"{off + float(r['t']):.2f}", r["x_px"], r["y_px"], r["edge"]])
            off += last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="начать заново, не возобновлять")
    ap.add_argument("--wait-min", type=float, default=180.0,
                    help="сколько минут ждать записи одного куска, прежде чем сдаться")
    ap.add_argument("--poll-s", type=float, default=20.0)
    a = ap.parse_args()

    global ORDER
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    ORDER = anchor["order"]
    state = {
        "phase": "P13 — цепочка кусков",
        "order": ORDER,
        "start": dict(anchor["start"]),
        "done": {},
        "current": None,
        "errors": [],
        "handovers": [],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_rendered_wait": False,
    }

    # Resume rather than restart. A chain of sixteen chunks takes hours and will be interrupted —
    # by a bug, by a reboot, by someone stopping it. Starting again from the first chunk would
    # throw away every seam already decided and, worse, would re-walk chunks whose start was handed
    # to them by a chunk that is no longer being reprocessed, so the second run would not even be
    # the same route. The saved state carries the finished chunks and the start the last one handed
    # over, and that is where this picks up.
    if CHAIN.exists() and not a.fresh:
        try:
            prev = json.loads(CHAIN.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("done"):
            state["done"] = prev["done"]
            state["handovers"] = prev.get("handovers", [])
            state["start"] = prev.get("start", state["start"])
            print(f"  возобновляю: уже пройдено {len(state['done'])} кусков, "
                  f"старт {state['start']['from']} → {state['start']['to']} "
                  f"по {state['start']['edge']}")

    print(f"  порядок: {' → '.join(ORDER)}")
    print(f"  старт: {anchor['start']['from']} → {anchor['start']['to']} "
          f"по {anchor['start']['edge']}")
    if a.dry_run:
        for v in ORDER:
            print(f"    {v:<10}{'записи готовы' if ready(v) else 'записей нет'}")
        return 0

    from p08_graph import Graph
    g = Graph.load(ROOT / "data/p08/graph.json")

    for v in ORDER:
        state["current"] = v
        if v in state["done"]:
            continue
        t0 = time.time()
        while not ready(v):
            if time.time() - t0 > a.wait_min * 60:
                state["errors"].append(f"{v}: записей нет за {a.wait_min:.0f} мин — пропуск")
                break
            if not state["_rendered_wait"]:
                state["_rendered_wait"] = True
                render(state)
            time.sleep(a.poll_s)
        state["_rendered_wait"] = False
        if not ready(v):
            continue

        run_dir = OUT / f"runs/{v}"
        rep_path = run_dir / "report.json"
        start_path = run_dir / "run_start.json"
        start = state["start"]

        # A run directory that exists is not the same as a run directory made with the start now
        # required. VID00010's was the blind run, walked from a start given by hand before the
        # chain existed; the chain found report.json, took the chunk as done, and handed that run's
        # ending to the next chunk, so the route jumped at that seam and everything after it
        # inherited the jump. Existence is not correctness — the same mistake that let a stale yaw
        # through, in a different place. The start is therefore written beside each run and
        # compared, and a run that cannot prove it used the right start is discarded and redone.
        want = {"edge": start["edge"], "from": start["from"]}
        was = None
        if rep_path.exists():
            try:
                was = json.loads(start_path.read_text(encoding="utf-8"))
            except Exception:
                was = None
            if not was or {k: was.get(k) for k in ("edge", "from")} != want:
                for f in run_dir.glob("*"):
                    try:
                        f.unlink()
                    except Exception:  # noqa: BLE001
                        pass
                state.setdefault("redone", []).append(
                    {"video": v, "had_start": was, "required_start": want})
                rep_path = run_dir / "report.json"

        log = OUT / f"chain_{v}.txt"
        if not rep_path.exists():
            cmd = [PY, "scripts/p08_graph_tracker.py", "--video", v,
                   "--start-edge", start["edge"], "--start-from", start["from"],
                   "--out", str(run_dir.relative_to(ROOT))]
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                         f"$ {' '.join(cmd)}\n")
                fh.flush()
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
            if rc != 0 or not rep_path.exists():
                state["errors"].append(f"{v}: трекер вернул {rc}")
                render(state)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            start_path.write_text(json.dumps(want, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        end_edge, end_node = rep.get("final_edge"), rep.get("final_node")
        if not end_edge or not end_node:
            state["errors"].append(f"{v}: трекер не дал конечного ребра")
            continue
        end_from = g.other(end_edge, end_node)
        n_edges = len(list(csv.DictReader((run_dir / "edge_sequence.csv").open())))

        # the handover, stated per seam: the end of one chunk becomes the start of the next
        hand = {"after": v, "end_edge": end_edge, "end_node": end_node,
                "next_start_edge": end_edge, "next_start_from": end_from,
                "replays_one_passage": True}
        state["handovers"].append(hand)
        state["done"][v] = {"edges": n_edges, "end_edge": end_edge,
                            "end_from": end_from, "end_to": end_node}
        state["start"] = {"edge": end_edge, "from": end_from, "to": end_node}
        write_live()
        render(state)
        CHAIN.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")

    CHAIN.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")
    write_live()
    print(f"\n  цепочка пройдена: {len(state['done'])}/{len(ORDER)}, "
          f"ошибок {len(state['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
