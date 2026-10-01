#!/usr/bin/env python3
"""
P06 — run the second video through the frozen frontend.

This is a generalisation test that costs only compute. The motion measurement and the
classifier thresholds were tuned on VID00001; nothing about them is touched here. If the
same code produces sensible candidates on a video it has never seen, the frontend
generalises. If it does not, we learn that the thresholds carry the first clip's
particulars.

What is reused, unchanged
-------------------------
    the LK motion field on the same 16x8 grid and the same centre band
    the flow decomposition  dx(x) = a + b * azimuth,  dy(y) = d + c * elevation
    the same thresholds, smoothing, segment length and sign-hold rule

What is new
-----------
    the video, `data/p01r/VID00002.AVI`
    the output paths, so nothing from the first clip is overwritten

The second clip declares 45000 frames and contains 36835, a factor of 0.8186 against
0.8187 for the first. The same container defect, so seeking is wrong here too and
reading is sequential throughout, via `VideoReader`.

Usage:
    PYTHONPATH=. python scripts/p06_scan_new.py --only scan
    PYTHONPATH=. python scripts/p06_scan_new.py --only label
    PYTHONPATH=. python scripts/p06_scan_new.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "data/p01r/VID00002.AVI"
REFERENCE = ROOT / "data/p01r/VID00001.AVI"
CACHE = ROOT / "output/p06_scan/full_signal.npz"
OUT_JSON = ROOT / "data/p06/windows_v1.json"
SCANNER = ROOT / "scripts/p02_scan_full.py"
REGISTRY = ROOT / "data/p07_videos.json"


def load_scanner():
    """Reuse the frozen scanner rather than restating its thresholds here."""
    spec = importlib.util.spec_from_file_location("p02_scan_full", SCANNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["scan", "label", "all"], default="all")
    ap.add_argument("--video", default=None,
                    help="clip to scan; default keeps the previous choice of VID00002")
    ap.add_argument("--tag", default=None,
                    help="suffix for the outputs, so nothing gets overwritten")
    args = ap.parse_args()

    global VIDEO, CACHE, OUT_JSON
    if args.video:
        VIDEO = (Path(args.video) if str(args.video).startswith("/")
                 else ROOT / "data/p01r" / args.video)
    if args.tag:
        CACHE = ROOT / f"output/p06_scan/full_signal_{args.tag}.npz"
        OUT_JSON = ROOT / f"data/p06/windows_{args.tag}.json"
    if not VIDEO.exists():
        print(f"нет видео: {VIDEO}")
        return

    mod = load_scanner()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("P06 — второе видео через замороженный фронтенд")
    print(f"  видео: {VIDEO.name}")
    print(f"  пороги берутся из {SCANNER.name}, не меняются:")
    print(f"    TURN_MIN={mod.TURN_MIN}  FWD_MIN={mod.FWD_MIN}  "
          f"STATIC_MAX={mod.STATIC_MAX}")
    print(f"    DOMINANCE={mod.DOMINANCE}  MIN_SEG_S={mod.MIN_SEG_S}  "
          f"SMOOTH_S={mod.SMOOTH_S}")
    print(f"    сетка {mod.GRID_W}x{mod.GRID_H}, полоса {mod.CROP}")
    print()

    if args.only in ("scan", "all"):
        if CACHE.exists():
            print(f"  кэш уже есть: {CACHE}")
            sig = {k: v for k, v in np.load(CACHE).items()}
        else:
            print("  измеряю движение (последовательно, без перемотки)...")
            sig = mod.scan(VIDEO)
            np.savez_compressed(CACHE, **sig)
            print(f"  записано {CACHE}")
        a = sig["a"]
        print(f"\n  кадров обработано: {len(a):,}")
        print(f"  |a| медиана {np.median(np.abs(a)):.3f}, "
              f"p90 {np.percentile(np.abs(a), 90):.3f}, "
              f"максимум {np.abs(a).max():.2f}")
        # compare with the first clip, same statistic
        ref = ROOT / "output/p02_full/full_signal.npz"
        if ref.exists():
            ra = np.load(ref)["a"]
            print(f"  для сравнения VID00001: |a| медиана "
                  f"{np.median(np.abs(ra)):.3f}, p90 "
                  f"{np.percentile(np.abs(ra), 90):.3f}")
            print(f"  -> {'сопоставимо' if 0.5 < (np.percentile(np.abs(a),90) /
                  max(np.percentile(np.abs(ra),90),1e-9)) < 2.0 else 'РАСХОДИТСЯ'}")

    if args.only in ("label", "all"):
        if not CACHE.exists():
            print("  нет кэша, сначала scan")
            return
        sig = {k: v for k, v in np.load(CACHE).items()}
        windows = mod.label(sig)
        doc = {
            "purpose": "Кандидаты, найденные тем же замороженным классификатором на "
                       "втором видео. Метки классификатора — не истина, их проверяет "
                       "человек.",
            "source_video": str(VIDEO),
            "reference_video": str(REFERENCE),
            "classifier": str(SCANNER),
            "thresholds": {"TURN_MIN": mod.TURN_MIN, "FWD_MIN": mod.FWD_MIN,
                           "STATIC_MAX": mod.STATIC_MAX,
                           "DOMINANCE": mod.DOMINANCE,
                           "MIN_SEG_S": mod.MIN_SEG_S, "SMOOTH_S": mod.SMOOTH_S},
            "counts": dict(Counter(w["kind"] for w in windows)),
            "n": len(windows),
            "windows": windows,
        }
        OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n  окон найдено: {len(windows)}")
        print(f"  состав: {doc['counts']}")
        turns = [w for w in windows if w["kind"] in ("LEFT", "RIGHT")]
        print(f"  поворотов-кандидатов: {len(turns)}")
        if turns:
            durs = [w["duration_s"] for w in turns]
            print(f"  длительность поворотов: медиана {np.median(durs):.1f} с, "
                  f"от {min(durs):.1f} до {max(durs):.1f}")
            print(f"  распределение по времени: "
                  f"{'равномерно' if _spread(turns) else 'скучено'}")
        print(f"\nWrote {OUT_JSON}")
        print(f"  дальше: разметить, затем прогнать замороженный тест нейронов")


def _spread(turns: list[dict], n_bins: int = 5) -> bool:
    """Are the candidates spread across the clip rather than clustered?"""
    t = np.array([w["t0"] for w in turns])
    if len(t) < n_bins:
        return True
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    counts = np.histogram(t, bins=edges)[0]
    return counts.min() > 0


if __name__ == "__main__":
    main()
