#!/usr/bin/env python3
"""P14 — one manifest that says which file every number came from.

WHY THIS EXISTS
---------------
A whole phase was built on a video that was not the video. `VID00002_fixed.mp4` held a different
recording than the camera's `VID00002.AVI`, because the check that should have caught it compared
durations and the durations matched. The same name then meant two different things in two
independent pipelines: the route was built from the camera, the page and the neural recordings from
a downloaded file. Nothing in the project could answer "which file produced this number", so the
disagreement was invisible until a person watched the camera and noticed.

This manifest is that answer. For every clip it records the file, how that file was identified, its
real geometry, and — for every neural recording — the clip, the script, the seed and the fingerprint
of the encoder code that made it. Then a claim about a level can be traced back to a file, and two
claims that used different files cannot be quietly averaged together.

WHAT THE FINGERPRINT DOES AND DOES NOT GUARANTEE
-----------------------------------------------
A full SHA-256 is the honest identity, and `--full` computes it. It is not the default because the
card reads at about 3 MB/s: hashing 26 GB takes roughly two and a half hours, and a manifest nobody
waits for is a manifest nobody has. The default is a sampled fingerprint over three windows (head,
middle, tail) plus size and mtime. It is strong enough for the failures that actually happen here —
a file replaced by another recording, a truncated copy, a stale preview — because those differ at
once in size or in the first window. It is not strong enough to prove two files are identical, and
the record says so rather than implying otherwise.

Usage:
    PYTHONPATH=. python scripts/p14_manifest.py                  # sampled fingerprints (minutes)
    PYTHONPATH=. python scripts/p14_manifest.py --frames         # also count real frames (hours)
    PYTHONPATH=. python scripts/p14_manifest.py --full           # true sha256 (hours)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMERA = Path("/Volumes/NO NAME/DCIM")
DOWNLOADS = Path("/Users/artem/Downloads")
DATA = ROOT / "data/p14"
CACHE = DATA / "cache"
OUT = DATA / "MANIFEST.json"

SAMPLE_WINDOW = 8 * 1024 * 1024     # bytes read at each of head, middle and tail

# The code that decides how a frame becomes something the brain sees. A change to any of these
# changes what every recording means, so the manifest carries their fingerprint as the encoder's
# version rather than a number someone has to remember to bump.
ENCODER_SOURCES = (
    "fly_vo/visual_encoder.py",
    "fly_vo/content_motion.py",
    "fly_vo/config.py",
    "fly_vo/brain_clock.py",
    "fly_vo/malecns_engine.py",
)

# Which script produced which family of recordings, and from where it read the video. Stated rather
# than inferred: inference from directory names is what produced the original mix-up.
CHAINS = {
    "p053": {"script": "scripts/p053_threshold_run.py", "seed": 64, "reads": "data/p01r/{v}.AVI"},
    "p06": {"script": "scripts/p062_neuron_run.py", "seed": 64, "reads": "data/p01r/{v}.AVI"},
    "p09": {"script": "scripts/p09_real_holdout.py", "seed": 64, "reads": "data/p01r/{v}.AVI"},
    "p10": {"script": "scripts/p10_record.py", "seed": 64, "reads": "data/p01r/{v}.AVI"},
    "p11b": {"script": "scripts/p11b_record.py", "seed": 64, "reads": "data/p01r/{v}.AVI"},
    "camera": {"script": "scripts/p09_real_holdout.py", "seed": 64,
               "reads": "/Volumes/NO NAME/DCIM/{v}.AVI"},
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path: Path, full: bool = False) -> dict:
    """Identity of a file: full digest when asked, otherwise three sampled windows."""
    st = path.stat()
    rec = {"size": st.st_size, "mtime": round(st.st_mtime, 3)}
    if full:
        rec["sha256"] = sha256_of(path)
        rec["method"] = "full sha256"
        rec["guarantee"] = "identical files give equal digests; different files give different ones"
        return rec
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with path.open("rb") as fh:
        spots = [0, max(st.st_size // 2 - SAMPLE_WINDOW // 2, 0),
                 max(st.st_size - SAMPLE_WINDOW, 0)]
        for pos in spots:
            fh.seek(pos)
            h.update(fh.read(SAMPLE_WINDOW))
    rec["sampled_sha256"] = h.hexdigest()
    rec["sampled_windows"] = spots
    rec["window_bytes"] = SAMPLE_WINDOW
    rec["method"] = "sampled sha256 over head/middle/tail plus size"
    rec["guarantee"] = (
        "catches replacement by another recording, truncation and corruption in the sampled "
        "windows; does NOT prove two files are identical. Use --full for that."
    )
    return rec


def probe(path: Path) -> dict:
    """Geometry and the container's declared timeline, which is not the real one."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,nb_frames,codec_name",
           "-show_entries", "format=duration,size", "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    try:
        j = json.loads(r.stdout)
    except Exception:
        return {}
    s = (j.get("streams") or [{}])[0]
    f = j.get("format") or {}
    rate = s.get("r_frame_rate") or "0/1"
    try:
        num, den = (int(x) for x in rate.split("/"))
        fps = num / den if den else None
    except Exception:
        fps = None
    return {
        "resolution": f"{s.get('width')}x{s.get('height')}" if s.get("width") else None,
        "width": s.get("width"), "height": s.get("height"),
        "codec": s.get("codec_name"),
        "declared_fps": fps,
        "declared_frames": int(s["nb_frames"]) if s.get("nb_frames") else None,
        "declared_duration_s": float(f["duration"]) if f.get("duration") else None,
        "declared_note": (
            "the container overstates these: 45000 frames and 1500 s for files that hold about "
            "36840 frames and 1228 s. Never pace anything by the declared timeline."
        ),
    }


def real_frames(path: Path, enabled: bool) -> dict:
    """Frames actually decodable. Costs a full decode, so it is cached and opt-in."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"frames_{path.parent.name}_{path.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if not enabled:
        return {"real_frames": None,
                "why_missing": "не посчитано; запустите с --frames (полный декод, минуты на файл)"}
    t0 = time.time()
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        n = int(r.stdout.strip())
    except Exception:
        n = None
    fps_path = probe(path).get("declared_fps") or 30.0
    rec = {"real_frames": n,
           "real_duration_s": round(n / fps_path, 2) if n else None,
           "measured_in_s": round(time.time() - t0, 1)}
    cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def encoder_fingerprint() -> dict:
    h = hashlib.sha256()
    files = {}
    for rel in ENCODER_SOURCES:
        p = ROOT / rel
        if p.exists():
            d = sha256_of(p)[:16]
            files[rel] = d
            h.update(rel.encode())
            h.update(d.encode())
    try:
        from fly_vo.config import FlyVOConfig
        c = FlyVOConfig()
        enc = f"{c.encode_width}x{c.encode_height}"
    except Exception:
        enc = None
    return {"sources": files, "fingerprint": h.hexdigest()[:16],
            "encode_size": enc, "brain_dt": 0.02,
            "why": ("these files decide how a frame becomes a brain input; a change in any of them "
                    "changes what every recording means")}


def video_record(path: Path, origin: str, a) -> dict:
    rec = {"path": str(path), "origin": origin}
    if not path.exists():
        rec["missing"] = True
        return rec
    rec.update(probe(path))
    rec.update(real_frames(path, a.frames))
    rec["fingerprint"] = fingerprint(path, a.full)
    return rec


def traces(a) -> dict:
    """Every neural recording, with the clip it came from and the chain that made it."""
    out = {}

    def add(trace_id: str, path: Path, video: str, chain: str, source: str, note: str = ""):
        if not path.exists():
            return
        d = {"video": video, "chain": chain, "script": CHAINS[chain]["script"],
             "seed": CHAINS[chain]["seed"], "read_from": source, "path": str(path.relative_to(ROOT)),
             "size": path.stat().st_size}
        if note:
            d["note"] = note
        try:
            import numpy as np
            z = np.load(path)
            if "fired" in z:
                arr = np.asarray(z["fired"])
                d["cells"] = int(arr.shape[1]) if arr.ndim > 1 else 1
                d["samples"] = int(arr.shape[0])
                d["spikes_total"] = int(arr.sum())
                d["seconds"] = float(np.asarray(z["t"]).astype(float)[-1]) if "t" in z else None
            elif "inject" in z:
                d["samples"] = int(np.asarray(z["inject"]).shape[0])
                d["seconds"] = float(np.asarray(z["t"]).astype(float)[-1])
                d["holds"] = "input (inject) and early visual (vis_rate)"
        except Exception as e:
            d["read_error"] = str(e)
        out[trace_id] = d

    # The yaw chain does not read output/p09 at all: it reads whatever `data/p07_videos.json`
    # resolves for the clip, and that registry lets an entry override the built-in default. The
    # overrides were added on 24 September from the camera chunks, so for every clip except the
    # first this chain is already camera-based — the opposite of what a name-based reading of the
    # directory `output/p06_neurons` suggests. Resolving it the way the code does is the only way
    # to know what was actually read.
    resolved = {}
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from p07_videos import videos as _videos
        resolved = {k: v["trace"] for k, v in _videos().items()}
    except Exception as e:
        out["_registry_error"] = {"error": str(e)}
    for clip, path in sorted(resolved.items()):
        known_camera = clip not in ("VID00001",)
        add(f"{clip}_yaw", path, clip, "p06",
            "data/p07_videos.json → " + str(path.relative_to(ROOT)),
            "камерный след (создан 24.09 при обработке кусков камеры)" if known_camera
            else "остался от прежней ветки: реестр для VID00001 не переопределён")

    # traces that exist on disk but that nothing reads any more: kept visible so they cannot be
    # mistaken for the current measurement
    for p in sorted((ROOT / "output").glob("p06_neurons/spike_trace.npz")):
        add("ORPHAN_p06_neurons", p, "VID00002", "p06",
            "~/Downloads (запись от 20.09)",
            "БОЛЬШЕ НЕ ЧИТАЕТСЯ: реестр p07_videos.json переопределён на p06_neurons_vid2. "
            "Остаётся на диске как след прежней ветки.")

    for p in sorted((ROOT / "output/p09").glob("trace_VID*.npz")):
        v = p.stem.replace("trace_", "")
        add(f"{v}_P09", p, v, "p09", f"data/p01r/{v}.AVI")
    for p in sorted((ROOT / "output/p10").glob("brain_VID*.npz")):
        v = p.stem.replace("brain_", "")
        add(f"{v}_P10", p, v, "p10", f"data/p01r/{v}.AVI")
    for p in sorted((ROOT / "output/p11b").glob("cells_VID*.npz")):
        v = p.stem.replace("cells_", "")
        add(f"{v}_P11B", p, v, "p11b", f"data/p01r/{v}.AVI")
    p = ROOT / "output/p13/camera_traces/trace_VID00002_from_camera.npz"
    add("VID00002_CAMERA", p, "VID00002", "camera", "/Volumes/NO NAME/DCIM/VID00002.AVI",
        "посчитано при разборе путаницы файлов; в основные уровни не поставлено")
    p = ROOT / "output/p13/legacy_traces/trace_VID00002_from_downloads.npz"
    add("VID00002_P09_legacy_copy", p, "VID00002", "p09", "ступень копия того же файла")
    return out


def container_defect(videos: dict) -> dict:
    """How far the header overstates the file, measured on every clip.

    Every AVI here declares fifty thousand-odd frames and 1500 seconds while holding about 36,840
    frames and 1228 seconds. The ratio turns out to be the same on all of them, which is worth
    stating because it changes what kind of defect this is: not a per-file corruption to be checked
    case by case, but one constant of the camera. That is also why pacing anything by the declared
    timeline is wrong by a fixed fifth, and why the frame index is the dependable axis.
    """
    ratios, pairs = [], []
    for vid, rec in videos.items():
        for origin, r in rec.items():
            d, q = r.get("declared_frames"), r.get("real_frames")
            if d and q:
                ratios.append(q / d)
                pairs.append({"clip": vid, "origin": origin, "declared": d, "real": q,
                              "ratio": round(q / d, 5)})
    if not ratios:
        return {"measured": False}
    ratios.sort()
    med = ratios[len(ratios) // 2]
    big = [p for p in pairs if p["declared"] > 10000]
    return {
        "measured": True,
        "clips": len(ratios),
        "ratio_real_over_declared_median": round(med, 5),
        "ratio_min": round(min(ratios), 5),
        "ratio_max": round(max(ratios), 5),
        "header_overstates_by": round(1 / med, 4),
        "note_on_spread": (
            "разброс 0.808-0.826 приходит от коротких файлов, где округление велико: у VID00018 "
            "52 объявлено против 42 реальных. На файлах длиннее десяти тысяч кадров отношение "
            "лежит в 0.8186-0.8188, то есть дефект один и тот же для всех."
        ),
        "long_files_ratio_range": [round(min(p["ratio"] for p in big), 5),
                                   round(max(p["ratio"] for p in big), 5)] if big else None,
        "consequence": (
            "заголовок завышает счёт примерно на 22% на каждом клипе: 45000 против 36840 кадров и "
            "1500 против 1228 с. Поэтому по объявленной шкале нельзя ни пейсить, ни искать кадр; "
            "номер кадра — единственная надёжная ось"
        ),
        "pairs": pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", action="store_true", help="посчитать реальные кадры (полный декод)")
    ap.add_argument("--full", action="store_true", help="настоящий sha256 вместо выборки")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("P14 — МАНИФЕСТ ИСТОЧНИКОВ")
    print("=" * 100)

    videos = {}
    cam = sorted(CAMERA.glob("VID*.AVI"))
    print(f"  камера: {len(cam)} файлов")
    t0 = time.time()
    for p in cam:
        videos[p.stem] = {"camera": video_record(p, "camera_card", a)}
        v = videos[p.stem]["camera"]
        print(f"    {p.stem:<10} {v.get('resolution','?'):>9} "
              f"объявлено кадров {v.get('declared_frames')} реально {v.get('real_frames')} "
              f"({v.get('real_duration_s')} с)", flush=True)
    print(f"  камера заняла {time.time()-t0:.0f} с")

    legacy = sorted(p for p in DOWNLOADS.glob("*VID*.AVI") if p.is_file() and not p.is_symlink())
    print(f"\n  старые файлы в ~/Downloads: {len(legacy)}")
    for p in legacy:
        vid = p.stem.split("_")[-1]
        rec = video_record(p, "downloads", a)
        videos.setdefault(vid, {})["legacy"] = rec
        print(f"    {vid:<10} {p.name[:44]:<46} {rec.get('resolution','?'):>9} "
              f"реально {rec.get('real_frames')}", flush=True)

    print("\n  следы:")
    tr = traces(a)
    for k, v in tr.items():
        print(f"    {k:<28} {v.get('seconds')} с  срабатываний {v.get('spikes_total')}")

    print("\n  отпечаток кода энкодера:")
    enc = encoder_fingerprint()
    for k, v in enc["sources"].items():
        print(f"    {k:<34} {v}")

    doc = {
        "phase": "P14 — единый источник правды для экспериментов",
        "why": (
            "фаза была построена на видео, которое не было тем видео: имя VID00002 значило разные "
            "файлы в двух конвейерах, и проект не мог ответить, какой файл дал какое число. "
            "Манифест отвечает."
        ),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint_policy": {
            "default": "sampled sha256 по трём окнам (голова/середина/хвост) + размер + mtime",
            "full": "PYTHONPATH=. python scripts/p14_manifest.py --full",
            "measured_card_read_mb_s": 3.0,
            "why_not_full_by_default": (
                "карта читается около 3 МБ/с; полный sha256 всех 26 ГБ — примерно 2.5 часа. "
                "Отпечаток, которого никто не дождался, — это отсутствие отпечатка."
            ),
        },
        "encoder": enc,
        "chains": CHAINS,
        "videos": videos,
        "traces": tr,
        "levels": {
            "video": {"source": "output/p095/clips/*.mp4 → кадр 320x180 → Farneback",
                      "working_geometry": "320x180",
                      "script": "scripts/p10_video_features.py"},
            "input": {"source": "энкодер, encode_frame → inject", "working_geometry": enc["encode_size"],
                      "script": "scripts/p10_record.py"},
            "early": {"source": "энкодер, encode_frame → vis_rate", "working_geometry": enc["encode_size"],
                      "script": "scripts/p10_record.py"},
            "dn": {"source": "output/p09/trace_*.npz", "script": "scripts/p09_real_holdout.py",
                   "cells": 68},
            "candidates": {"source": "output/p11b/cells_*.npz", "script": "scripts/p11b_record.py",
                           "cells": 38304},
            "yaw_chain": {"source": "output/p06_neurons и output/p053_threshold",
                          "script": "scripts/p07_fly_readout.py"},
            "note": ("геометрия на всех уровнях фиксирована (энкодер приводит к encode_size, признаки "
                     "видео — к 320x180), поэтому от разрешения источника зависит не расположение "
                     "пикселей, а объём сохранённой детали. Это измерено отдельно: p14_geometry.py"),
        },
        "container_defect": container_defect(videos),
        "open_items": [
            "полный sha256 не посчитан: заполняется через --full (около 2.5 часов на 26 ГБ)",
        ],
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
