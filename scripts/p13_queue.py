#!/usr/bin/env python3
"""P13 — walk the camera's recordings one after another, carrying the route across the seam.

WHAT THIS IS FOR
----------------
The camera splits one long recording into chunks of about two gigabytes, roughly twenty minutes
each. Each chunk begins where the previous one stopped, so the walk is one continuous route spread
over many files. Running each file on its own throws that away: every chunk would be started from
an assumed beginning and the tracker would have to rediscover where it is. This runs them in order
and hands the end of one file to the next as its start, with the direction of travel preserved.

    end of chunk N     the last edge the walker was on, and which way along it
    start of chunk N+1 the same edge, same direction

The handover is exact in edge terms. It is not exact in position: the chunk boundary falls wherever
the file limit fell, usually partway along a passage, and the tracker begins an edge at its far end.
So each seam replays up to one passage, about four metres, roughly nine seconds of walking. That is
stated rather than hidden, and it is why the seam times are written into the queue state.

TIME AND SPACE
--------------
One chunk costs about seventy minutes of compute: the visual drive for the tracker's own yaw, the
sixty-eight descending cells for route_change, and the twenty early populations for
net_displacement. Eighteen chunks would be twenty-one hours end to end.

The recordings do not depend on each other or on the route, so they are done first, two at a time,
and only the tracker steps need the order. The disk holds nine gigabytes and each source file is
two, so a source is copied in, used, and removed again before the next is fetched.

Usage:
    PYTHONPATH=. python scripts/p13_queue.py --videos VID00011,VID00012 --start-edge M30__M31 \
        --start-from M31
    PYTHONPATH=. python scripts/p13_queue.py --plan
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CAMERA = Path("/Volumes/NO NAME/DCIM")
OUT = ROOT / "output/p13"
STATE = OUT / "state.json"
LIVE = OUT / "live_trajectory.csv"
PY = sys.executable

# per chunk, in seconds, from the runs done by hand; used only for the estimate in the dashboard
COST = {"source": 120, "visual": 1850, "p09": 1880, "p10": 480, "tracker": 60}


def tag_of(video: str) -> str:
    """VID00011 -> vid11, VID00003 -> vid3. The leading zeros are dropped, as elsewhere."""
    return "vid" + str(int(video.replace("VID", "")))


def run(cmd: list[str], log: Path, label: str) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                 f"{label}\n$ {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return r.returncode


def has(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


class Queue:
    def __init__(self, args):
        self.args = args
        self.videos = args.videos
        self.state = {
            "phase": "P13 — проход по кускам одной записи со склейкой маршрута",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "videos": [],
            "current": None,
            "chain": [],
            "errors": [],
        }
        self.start_edge = args.start_edge
        self.start_from = args.start_from
        self.t0 = time.time()
        for v in self.videos:
            self.state["videos"].append({
                "video": v, "status": "ожидает", "seconds": 0,
                "steps": {}, "start": None, "end": None,
                "segment": None,
            })

    # ------------------------------------------------------------------ state
    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        done = sum(1 for v in self.state["videos"] if v["status"] == "готово")
        total = len(self.state["videos"])
        el = time.time() - self.t0
        self.state.update({
            "done": done, "total": total, "elapsed_s": round(el),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if done:
            per = el / done
            self.state["eta_s"] = round(per * (total - done))
        STATE.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
        self.render()

    def render(self) -> None:
        s = self.state
        done, total = s.get("done", 0), s.get("total", 0)
        bar_n = 36
        filled = int(round(bar_n * done / max(total, 1)))
        bar = "█" * filled + "░" * (bar_n - filled)
        el = s.get("elapsed_s", 0)
        eta = s.get("eta_s")
        print("\n" + "═" * 96)
        print(f"  P13  {bar}  {done}/{total}")
        print(f"  прошло {fmt_hms(el)}" + (f"   осталось ~{fmt_hms(eta)}" if eta else ""))
        print("─" * 96)
        for v in s["videos"]:
            mark = {"готово": "✔", "ошибка": "✖", "в работе": "▶"}.get(v["status"], " ")
            st = v.get("start")
            en = v.get("end")
            chain = ""
            if st:
                chain = f"старт {st.get('from')}→{st.get('to')}"
            if en:
                chain += f"  конец {en.get('from')}→{en.get('to')}"
            print(f"   {mark} {v['video']:<10}{v['status']:<10}{v.get('seconds', 0):>6.0f}с   "
                  f"{chain[:74]}")
        if s["errors"]:
            print("─" * 96)
            print(f"   ошибок: {len(s['errors'])}")
        print("═" * 96, flush=True)

    def step(self, v: dict, name: str, ok: bool, note: str = "") -> None:
        v["steps"][name] = {"ok": bool(ok), "note": note,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.save()

    # ------------------------------------------------------------------ work
    def ensure_source(self, v: str) -> tuple[Path, bool]:
        """Copy the chunk in from the camera. Returns (path, replaced_from_a_different_file).

        The project and the camera do not agree on what a number means. VID00005, VID00009 and
        VID00010 are byte for byte the same file in both places, while the file the project calls
        VID00006 is a 250-second clip and the camera's VID00006 is a 1228-second chunk, and VID00002
        and VID00004 differ in size as well. So a local file with the right name is not evidence
        that it holds the right footage, and everything derived from it — the visual drive, the
        descending cells, the early populations — would quietly belong to a different walk.

        The byte count decides. A local file whose size differs from the camera's is replaced, and
        the flag it returns makes every recording built on it be redone rather than reused.
        """
        dest = ROOT / f"data/p01r/{v}.AVI"
        src = CAMERA / f"{v}.AVI"
        if not src.exists():
            if dest.exists():
                return dest, False
            raise RuntimeError(f"нет источника {src}")
        cam_bytes = src.stat().st_size
        replaced = False
        if dest.exists():
            if dest.stat().st_size == cam_bytes:
                return dest, False
            replaced = True
            dest.unlink()
        free = shutil.disk_usage(ROOT).free
        need = cam_bytes + (1 << 30)
        if free < need:
            raise RuntimeError(f"мало места: свободно {free/1e9:.1f} ГБ, нужно {need/1e9:.1f} ГБ")
        shutil.copy2(src, dest)
        return dest, replaced

    def fp_path(self, video: str) -> Path:
        return OUT / f"sources/{video}.json"

    def fp_ok(self, video: str) -> bool | None:
        """Does an existing recording belong to the file now in data/p01r?

        The clip numbering is not consistent between the project and the camera: VID00005,
        VID00009 and VID00010 match byte for byte, while the project's VID00006 is a 250-second
        clip and the camera's VID00006 is a 1228-second chunk, and VID00004 differs in size too. So
        an output named `trace_VID00006.npz` is not evidence that it was made from the file now
        called VID00006. Every recording writes the byte count of its source next to it, and a
        mismatch forces a re-record rather than quietly reusing the wrong brain activity.

        Returns True when it matches, False when it does not, and None when there is nothing to
        compare against — an older output with no fingerprint. None is treated as usable but is
        reported, so the gap is visible instead of assumed away.
        """
        p = self.fp_path(video)
        src = ROOT / f"data/p01r/{video}.AVI"
        if not src.exists():
            return None
        if not p.exists():
            return None
        try:
            return int(json.loads(p.read_text(encoding="utf-8"))["bytes"]) == src.stat().st_size
        except Exception:  # noqa: BLE001
            return None

    def fp_write(self, video: str) -> None:
        src = ROOT / f"data/p01r/{video}.AVI"
        if not src.exists():
            return
        self.fp_path(video).parent.mkdir(parents=True, exist_ok=True)
        self.fp_path(video).write_text(json.dumps({
            "video": video, "bytes": src.stat().st_size,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def process(self, v: dict) -> None:
        video = v["video"]
        tag = tag_of(video)
        t_start = time.time()
        v["status"] = "в работе"
        self.state["current"] = video
        self.save()

        log = OUT / f"log_{video}.txt"

        # ---- source, always: the recordings read the file, so it has to be local
        try:
            src_path, replaced = self.ensure_source(video)
            src_note = str(src_path)
            if replaced:
                src_note += "  (локальный файл отличался от камерного — заменён)"
                v["source_replaced"] = True
        except Exception as exc:  # noqa: BLE001
            v["status"] = "ошибка"
            self.state["errors"].append({"video": video, "step": "источник", "error": str(exc)})
            self.step(v, "источник", False, str(exc))
            return
        self.step(v, "источник", True, src_note)

        # ---- visual drive: the tracker's own yaw
        trace = ROOT / f"output/p06_neurons_{tag}/spike_trace.npz"
        tar = ROOT / f"output/p06_neurons_{tag}/targets.csv"
        stale = replaced or (self.fp_ok(video) is False)
        if stale:
            self.state.setdefault("notes", []).append(
                f"{video}: {'источник заменён' if replaced else 'источник не совпадает'} — "
                f"нейроданные перезаписываю")

        def do_visual():
            if has(trace) and has(tar) and not stale:
                return "есть"
            rc = run([PY, "scripts/p062_neuron_run.py", "--video", f"{video}.AVI",
                      "--tag", tag], log, f"p062 {video}")
            if rc != 0 or not has(trace):
                raise RuntimeError(f"p062 вернул {rc}")
            self.fp_write(video)
            return "записано"

        # ---- route_change: sixty-eight descending cells
        p09 = ROOT / f"output/p09/trace_{video}.npz"

        def do_p09():
            if has(p09) and not stale:
                return "есть"
            rc = run([PY, "scripts/p09_real_holdout.py", "--video", video, "--record"],
                     log, f"p09 {video}")
            if rc != 0 or not has(p09):
                raise RuntimeError(f"p09 вернул {rc}")
            self.fp_write(video)
            return "записано"

        # ---- net_displacement: twenty early populations
        p10 = ROOT / f"output/p10/brain_{video}.npz"

        def do_p10():
            if has(p10) and not stale:
                return "есть"
            rc = run([PY, "scripts/p10_record.py", "--video", video], log, f"p10 {video}")
            if rc != 0 or not has(p10):
                raise RuntimeError(f"p10 вернул {rc}")
            self.fp_write(video)
            return "записано"

        # ---- register, then the P07 chain (fast: seconds for the whole registry)
        def do_register():
            reg = json.loads((ROOT / "data/p07_videos.json").read_text(encoding="utf-8"))
            if video in reg:
                return "есть"
            run([PY, "scripts/p07_videos.py", "--add", video,
                 "--trace", f"output/p06_neurons_{tag}/spike_trace.npz",
                 "--targets", f"output/p06_neurons_{tag}/targets.csv"],
                log, f"register {video}")
            return "добавлено"

        yaw = ROOT / f"output/p07/yaw_signal_{video}.csv"
        trace_file = ROOT / f"output/p06_neurons_{tag}/spike_trace.npz"

        def do_p07():
            # Regenerate the visual drive whenever it is older than the trace it must come from.
            # The guard here used to be "if the file does not exist", and that quietly kept a stale
            # yaw in place: VID00006's had been built from the 250-second clip that shared its name,
            # so after its trace was correctly re-recorded the yaw still covered a fifth of the
            # chunk, and the walk through it produced 24 edges instead of a hundred. Existence is
            # not correctness.
            need_yaw = True
            if has(yaw) and has(trace_file):
                need_yaw = yaw.stat().st_mtime < trace_file.stat().st_mtime
            if need_yaw:
                run([PY, "scripts/p07_fly_readout.py"], log, f"p07_fly_readout {video}")
            run([PY, "scripts/p07_trajectory.py"], log, f"p07_trajectory {video}")
            run([PY, "scripts/p071_turn_filter.py"], log, f"p071_turn_filter {video}")
            if not has(ROOT / f"output/p071/trajectory_{video}.csv"):
                raise RuntimeError("нет p071/trajectory")
            return "готово"

        # ---- the tracker, carrying the route across the seam
        run_dir = OUT / f"runs/{video}"

        def do_tracker():
            report = run_dir / "report.json"
            if has(report):
                return "есть"
            if not (self.start_edge and self.start_from):
                raise RuntimeError("нет старта: нечего передавать по цепочке")
            rc = run([PY, "scripts/p08_graph_tracker.py", "--video", video,
                      "--start-edge", self.start_edge, "--start-from", self.start_from,
                      "--out", str(run_dir.relative_to(ROOT))], log, f"tracker {video}")
            if rc != 0 or not has(report):
                raise RuntimeError(f"трекер вернул {rc}")
            return "готово"

        plan = [("визуальный вход", do_visual), ("68 клеток", do_p09),
                ("ранние клетки", do_p10), ("регистрация", do_register),
                ("yaw/forward", do_p07), ("трекер", do_tracker)]

        if self.args.lean:
            # The route needs only the visual drive: the tracker reads yaw and speed and nothing
            # else. The sixty-eight descending cells and the twenty early populations feed the P12
            # channels, which the trajectory does not use, and they cost six times as much as the
            # visual drive does. Dropping them takes a chunk from 85 minutes to 35.
            plan = [p for p in plan if p[0] not in ("68 клеток", "ранние клетки")]
        if self.args.record_only:
            plan = [p for p in plan if p[0] != "трекер"]

        for name, fn in plan:
            try:
                note = fn()
            except Exception as exc:  # noqa: BLE001
                v["status"] = "ошибка"
                v["seconds"] = time.time() - t_start
                self.state["errors"].append({"video": video, "step": name, "error": str(exc)})
                self.step(v, name, False, str(exc))
                return
            self.step(v, name, True, note or "")
            v["seconds"] = time.time() - t_start
            self.save()

        # ---- hand the end to the next chunk
        report = run_dir / "report.json"
        if has(report):
            rep = json.loads(report.read_text(encoding="utf-8"))
            from p08_graph import Graph
            g = Graph.load(ROOT / "data/p08/graph.json")
            end_edge = rep.get("final_edge")
            end_node = rep.get("final_node")
            v["end"] = {"edge": end_edge, "node": end_node}
            if end_edge and end_node:
                frm = g.other(end_edge, end_node)
                v["end"].update({"from": frm, "to": end_node})
                self.start_edge, self.start_from = end_edge, frm
                self.state["chain"].append({"after": video, "next_start_edge": end_edge,
                                            "next_start_from": frm})
        v["segment"] = self.segment_of(video)
        v["status"] = "готово"
        v["seconds"] = time.time() - t_start

        # ---- the source is no longer needed: the recordings read it once and that is done
        if self.args.delete_source:
            src = ROOT / f"data/p01r/{video}.AVI"
            try:
                if src.exists() and not src.is_symlink():
                    src.unlink()
            except Exception:  # noqa: BLE001
                pass
        self.save()

    @staticmethod
    def segment_of(video: str) -> dict | None:
        """Where this chunk's route went on the plan, for the live picture."""
        p = OUT / f"runs/{video}/graph_trajectory.csv"
        if not p.exists():
            return None
        rows = list(csv.DictReader(p.open()))
        if not rows:
            return None
        step = max(1, len(rows) // 400)
        pts = rows[::step]
        return {"t": [round(float(r["t"]), 2) for r in pts],
                "x": [round(float(r["x_px"]), 1) for r in pts],
                "y": [round(float(r["y_px"]), 1) for r in pts]}

    def write_live(self) -> None:
        """Every run present under output/p13/runs, each with the offset that puts it on the clock.

        Driven by what is on disk rather than by this queue's own list, so a chunk that was run
        earlier — VID00010, walked by hand before the queue existed — still appears as the first
        segment of the picture instead of leaving a gap at the start of the route.
        """
        OUT.mkdir(parents=True, exist_ok=True)
        runs = []
        for d in sorted((OUT / "runs").glob("*")):
            p = d / "graph_trajectory.csv"
            if p.exists():
                runs.append((d.name, p))
        # keep the camera's order: VID00010, VID00011, ...
        runs.sort(key=lambda kv: kv[0])
        with LIVE.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "t_global", "x_px", "y_px", "edge"])
            off = 0.0
            for name, p in runs:
                rows = list(csv.DictReader(p.open()))
                last_t = 0.0
                for i, r in enumerate(rows):
                    last_t = max(last_t, float(r["t"]))
                    if i % 5:
                        continue
                    w.writerow([name, f"{off + float(r['t']):.2f}",
                                r["x_px"], r["y_px"], r["edge"]])
                off += last_t

    def run_all(self) -> int:
        OUT.mkdir(parents=True, exist_ok=True)
        self.render()
        for v in self.state["videos"]:
            if v["status"] == "готово":
                continue
            self.process(v)
            if v["status"] == "ошибка" and not self.args.keep_going:
                break
        self.write_live()
        self.save()
        s = self.state
        print(f"\n  готово {s.get('done')}/{s.get('total')}, "
              f"ошибок {len(s['errors'])}, время {fmt_hms(time.time() - self.t0)}")
        return 0 if not s["errors"] else 1


def fmt_hms(s: float) -> str:
    s = int(s)
    return f"{s//3600}ч {s%3600//60:02d}м {s%60:02d}с"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="",
                    help="через запятую, в порядке следования; пусто — взять все с камеры")
    ap.add_argument("--start-edge", default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--record-only", action="store_true",
                    help="только записи нейроданных, без трекера (склейка не нужна)")
    ap.add_argument("--lean", action="store_true",
                    help="только то, что нужно маршруту: визуальный вход + yaw. Без 68 клеток "
                         "и ранних популяций (они нужны только каналам P12)")
    ap.add_argument("--delete-source", action="store_true", default=True)
    ap.add_argument("--keep-source", dest="delete_source", action="store_false")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args()

    if args.plan or not args.videos:
        files = sorted(CAMERA.glob("VID*.AVI")) if CAMERA.exists() else []
        print(f"на камере: {len(files)} файлов")
        for f in files:
            mb = f.stat().st_size / 1e6
            done = has(ROOT / f"output/p09/trace_{f.stem}.npz")
            print(f"  {f.stem:<12}{mb:>8.0f} МБ   {'нейроданные есть' if done else '—'}")
        if not args.videos:
            return 0

    args.videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    if not args.videos:
        print("СТОП: не указано ни одного видео")
        return 1
    if not args.record_only and not (args.start_edge and args.start_from):
        print("СТОП: для трекера нужен старт: --start-edge и --start-from")
        print("  продолжаем существующую цепочку: её конец известен из прогона VID00010")
        return 1

    q = Queue(args)
    return q.run_all()


if __name__ == "__main__":
    raise SystemExit(main())
