#!/usr/bin/env python3
"""P10 levels 2 and 3 — the brain input, and the first cells the brain fires.

Two recordings, in one pass over each source video, using the same encoder, the same seed and the
same full-video span as the P09 recording of the descending cells. That last point is the reason
this runs over whole videos instead of over the thirty-second review clips: if the early cells
were driven from a cold start in a clip while the descending cells were driven from a warm start
twenty minutes into the film, any difference between those two levels could be nothing but the
warm-up. Same film, same reset, same clock.

    level 2 — the drive handed to the visual system: how much motion was pushed into T4a/b, T5a/b
              per side, and how much loom into LC4 and LPLC2. This is the encoder's output, before
              a single cell has integrated anything.

    level 3 — what T4a-d, T5a-d, LC4 and LPLC2 actually did, as population rate per type and side.
              Twenty numbers per step. Not the whole retina — the four vertical detectors are the
              pitch pathway and are expected to carry little yaw — but every horizontal type the
              connectome has, which is what the question asks for.

Recorded in one-minute chunks so an interrupted run resumes instead of starting over.

Usage:
    PYTHONPATH=. python scripts/p10_record.py [--video VID00001] [--out output/p10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

SEED = 64          # the seed P09 recorded the descending cells with
CHUNK_S = 60.0

INJECT_GROUPS = (
    "T4a_L", "T4a_R", "T4b_L", "T4b_R",
    "T5a_L", "T5a_R", "T5b_L", "T5b_R",
    "LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R",
)
VIS_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "LC4", "LPLC2")
VIS_GROUPS = tuple(f"{t}_{s}" for t in VIS_TYPES for s in ("L", "R"))


def video_of(name: str) -> Path:
    return ROOT / "data/p01r" / f"{name}.AVI"


def build_maps(encoder: VideoVisualEncoder, engine: MaleCNSEngine):
    """group -> cell indices, for both the injected drive and the cells themselves."""
    inject_idx: dict[str, np.ndarray] = {}
    for key, (_fam, _prefers, bins) in encoder.motion_injector.groups.items():
        cells = np.concatenate([np.asarray(c, dtype=np.int64) for c in bins])
        if len(cells):
            inject_idx[key] = np.unique(cells)
    for key, cells in encoder.loom_groups.items():
        if len(cells):
            inject_idx[key] = np.asarray(cells, dtype=np.int64)

    vis_idx: dict[str, np.ndarray] = {}
    for key in VIS_GROUPS:
        t, s = key.rsplit("_", 1)
        try:
            idx = engine.brain.cells([t], side=s)
        except Exception:
            idx = np.array([], dtype=np.int64)
        if len(idx):
            vis_idx[key] = np.asarray(idx, dtype=np.int64)
    return inject_idx, vis_idx


def VideoReader_len(name: str, dt: float) -> float:
    """Length of the recording to walk, taken from the P09 descending-cell trace.

    Reading it from the video instead costs a full pass over a file that is three and a half
    gigabytes here, and it is also the wrong thing to want: what matters is that the visual cells
    are driven over exactly the same span, with the same reset and the same clock, as the
    descending cells they will be compared against. The trace already knows that span.
    """
    p = ROOT / f"output/p09/trace_{name}.npz"
    if not p.exists():
        from fly_vo.video_reader import VideoReader
        with VideoReader(ROOT / "data/p01r" / f"{name}.AVI") as vr:
            return vr.duration_s
    t = np.load(p)["t"]
    return float(t[-1]) + dt


def record(video: Path, name: str, out: Path) -> None:
    # The name comes from the file, not from the string that was passed in. With an absolute path
    # the raw argument contains slashes, and `brain_{name}.npz` turned into nested directories —
    # so the recording could not be made from a file outside the project at all.
    name = video.stem
    chunks = out / f"chunks_brain_{name}"
    chunks.mkdir(parents=True, exist_ok=True)
    dest = out / f"brain_{name}.npz"
    # Provenance guard. Three cases, and conflating the second with the third destroys work:
    #   fingerprint present and matching   -> reuse
    #   fingerprint present and different  -> the chunks belong to another file; rebuild
    #   no fingerprint, artifact present   -> unknown provenance, so verify what can be verified
    #                                         (the recorded span against the source's own length)
    #                                         and adopt, recording that it was adopted
    # The third case is why this exists: treating a missing fingerprint as "foreign" deleted a
    # valid fourteen-minute recording and started rebuilding it, because recordings made before the
    # fingerprint was introduced carry none.
    def _fp_state(video: Path, chunks: Path) -> str:
        fp = chunks / "source.json"
        if not fp.exists():
            return "none"
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return "none"
        try:
            st = video.stat()
        except OSError:
            return "differs"
        return ("same" if (rec.get("size") == st.st_size
                           and abs(rec.get("mtime", 0) - st.st_mtime) < 1.0) else "differs")

    def _span_matches(dest: Path, video: Path) -> bool:
        try:
            have = float(np.load(dest)["t"][-1])
            from fly_vo.video_reader import VideoReader
            with VideoReader(video) as vr:
                want = vr.duration_s
        except Exception:
            return False
        return abs(have - want) <= 4.0

    state = _fp_state(video, chunks)
    if dest.exists():
        if state == "same":
            print(f"  {name}: уже записано ({dest.name}), пропуск")
            return
        if state == "none":
            if _span_matches(dest, video):
                st = video.stat()
                fp.write_text(json.dumps({"source": str(video), "size": st.st_size,
                                          "mtime": st.st_mtime,
                                          "adopted": True,
                                          "note": "отпечатка не было; запись принята по совпадению "
                                                  "длительности с источником, происхождение не "
                                                  "подтверждено сличением"},
                                         ensure_ascii=False), encoding="utf-8")
                print(f"  {name}: принято как есть — отпечатка не было, длительность совпала")
                return
            print(f"  {name}: запись есть, но длительность не сходится с источником — перезаписываю")
        else:
            print(f"  {name}: запись от другого файла — перезаписываю")
    old = list(chunks.glob("chunk_*.npz"))
    if old and not (dest.exists() and state == "same"):
        print(f"  {name}: куски собраны из другого файла — перезаписываю ({len(old)} шт)")
    for f in old:
        f.unlink()
    dest.unlink(missing_ok=True)
    st = video.stat()
    (chunks / "source.json").write_text(
        json.dumps({"source": str(video), "size": st.st_size, "mtime": st.st_mtime},
                   ensure_ascii=False), encoding="utf-8")

    engine = MaleCNSEngine(FlyVOConfig())
    encoder = VideoVisualEncoder(engine, flow_config=None)
    inject_idx, vis_idx = build_maps(encoder, engine)
    order_i = [g for g in INJECT_GROUPS if g in inject_idx]
    order_v = [g for g in VIS_GROUPS if g in vis_idx]
    lookup = np.full(engine.brain.n + 1, -1, dtype=np.int64)
    for j, g in enumerate(order_v):
        lookup[vis_idx[g]] = j

    dt = float(engine.config.brain_dt)
    total = VideoReader_len(name, dt)
    print(f"  {name}: {total:.0f} с, записываю вход ({len(order_i)} групп) "
          f"и ранние клетки ({len(order_v)} групп)")
    if not order_i:
        print(f"  СТОП: не нашёл ни одной группы входа")
        return

    # The injector hands back one entry per azimuth bin per group, so a group receives many
    # entries every step. Summing them is the only correct reading: that total is what the cells
    # actually integrate. Taking any single bin would have recorded a twelfth of the drive and
    # called it the group's input, which is a mistake worth spelling out because it looks fine
    # until the numbers are compared with the brain.
    gid = np.full(engine.brain.n + 1, -1, dtype=np.int64)
    for j, g in enumerate(order_i):
        gid[inject_idx[g]] = j

    engine.reset(seed=SEED)
    t0 = time.perf_counter()
    seg = 0.0
    while seg < total:
        end = min(seg + CHUNK_S, total)
        path = chunks / f"chunk_{int(seg):05d}_{int(end):05d}.npz"
        if path.exists():
            seg = end
            continue
        ts, inj, vis = [], [], []
        for t, frame in iter_video_at_brain_hz(str(video), seg, end, dt):
            eye, inject, _ = encoder.encode_frame(frame)
            res = engine.step(t, eye_drive=eye, inject=inject)
            ts.append(t)
            row = np.zeros(len(order_i), dtype=np.float64)
            for cells, amt in inject:
                if len(cells) == 0:
                    continue
                j = gid[int(cells[0])]
                if j >= 0:
                    row[j] += float(amt)
            inj.append(row)
            mask = np.zeros(engine.brain.n + 1, dtype=bool)
            if len(res.fired):
                mask[res.fired] = True
            hit = lookup[res.fired] if len(res.fired) else np.array([], dtype=np.int64)
            counts = np.bincount(hit[hit >= 0], minlength=len(order_v)) if len(hit) else \
                np.zeros(len(order_v), dtype=np.int64)
            vis.append(counts)
        np.savez_compressed(path, t=np.asarray(ts, np.float32),
                            inject=np.asarray(inj, np.float32),
                            vis_counts=np.asarray(vis, np.int32))
        el = time.perf_counter() - t0
        left = (total - end) * (el / max(end, 1e-9)) / 60
        print(f"    {end:7.0f}/{total:.0f} с  прошло {el/60:5.1f} мин  "
              f"осталось ~{left:4.1f} мин", flush=True)
        seg = end

    files = sorted(chunks.glob("chunk_*.npz"))
    ts = np.concatenate([np.load(f)["t"] for f in files])
    inj = np.concatenate([np.load(f)["inject"] for f in files])
    vis = np.concatenate([np.load(f)["vis_counts"] for f in files])
    sizes = np.array([len(vis_idx[g]) for g in order_v], dtype=np.float32)
    np.savez_compressed(dest, t=ts, inject=inj, inject_groups=np.array(order_i),
                        vis_rate=(vis / np.maximum(sizes, 1.0)).astype(np.float32),
                        vis_groups=np.array(order_v), vis_pop=sizes)
    print(f"  записано {dest}: {len(ts)} шагов, вход {inj.shape}, клетки {vis.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "output/p10"))
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import json
    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    names = sorted({l["clip"] for l in frozen["labels"]})
    todo = [args.video] if args.video else names

    print("=" * 96)
    print("P10 уровни 2-3 — вход в мозг и ранние зрительные клетки")
    print("=" * 96)
    print(f"  seed {SEED} (тот же, что у записи DN), шаг {CHUNK_S:.0f} с")
    print()
    for n in todo:
        v = video_of(n)
        if not v.exists():
            print(f"  {n}: НЕТ ФАЙЛА {v}")
            continue
        record(v, n, out)
    print()
    print("  готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
