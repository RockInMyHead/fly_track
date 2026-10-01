#!/usr/bin/env python3
"""
P07.3 — prepare the newly recorded route clip for the frozen chain.

Three things about this file need handling before anything else can run.

It arrived as `.part`
---------------------
That suffix marks an interrupted transfer, and the contents agree: the AVI has no `idx1`
index at all, which is where the camera writes its frame table when it finishes. The file
is a recording cut off in the middle. The frames that are there decode fine and the last
one ends on a frame boundary, so the clip is usable, but nothing may assume an index
exists. Seeking is already unreliable on this family of files; here it is impossible, and
the existing `VideoReader` reads sequentially, so that part is covered.

The declared frame count is wrong by a different factor
-------------------------------------------------------
Declared 45000 frames, actual 21286, a ratio of 0.473 against 0.8187 for the two earlier
clips. The duration field likewise disagrees with the content, 914 s against 709 s. So the
header cannot be trusted in either direction and the only way to know the length is to
count, which is what the reader does.

The audio track and the container are not what the browser wants
---------------------------------------------------------------
MJPEG in AVI with uncompressed PCM audio plays poorly in a browser and the annotation page
needs to play it. The transcode drops the audio, which nothing in the pipeline uses.

Rotation
--------
Handled by an explicit flag rather than guessed. The frames read as upright: the shelf at
the top of frame has objects resting on it, which only happens if up is up, and there is
no rotation tag in either the stream or the container. If that reading is wrong, pass
`--rotate cw` or `--rotate ccw` and the transcode applies it; the source is left untouched
either way.

Usage:
    PYTHONPATH=. python scripts/p073_prepare_video.py
    PYTHONPATH=. python scripts/p073_prepare_video.py --rotate cw
    PYTHONPATH=. python scripts/p073_prepare_video.py --skip-copy
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = Path("/Users/artem/Downloads/"
              "18fc7465-b8ea-43f9-b4e5-d778d47e5083_VID00004.AVI.part")
DEST = ROOT / "data/p01r/VID00004.AVI"
MEDIA = ROOT / "webapp/media/VID00004_fixed.mp4"
OUT = ROOT / "output/p073_ground_truth"
FPS = 30.0
ROTATE = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
          "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def count_frames(path: Path) -> int:
    """Count by decoding, since the header cannot be trusted on this file."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", default="none", choices=list(ROTATE))
    ap.add_argument("--skip-copy", action="store_true")
    ap.add_argument("--skip-transcode", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"source": str(SOURCE), "dest": str(DEST), "rotate": args.rotate}

    print("P07.3 — подготовка нового ролика")
    if not SOURCE.exists():
        print(f"  исходник не найден: {SOURCE}")
        return
    sz = SOURCE.stat().st_size
    print(f"  исходник: {SOURCE.name}")
    print(f"    {sz / 1e9:.2f} ГБ, расширение .part (незавершённая передача)")

    # --- integrity ----------------------------------------------------------
    with SOURCE.open("rb") as f:
        head = f.read(16)
        f.seek(-64, 2)
        tail = f.read()
    has_idx = b"idx1" in tail
    print(f"  сигнатура RIFF/AVI: {head[:4] == b'RIFF' and head[8:12] == b'AVI '}")
    print(f"  индекс idx1 в конце: {has_idx}  -> "
          f"{'файл завершён' if has_idx else 'файл ОБОРВАН, индекса нет'}")
    report["has_idx1"] = bool(has_idx)

    cap = cv2.VideoCapture(str(SOURCE))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  заявлено контейнером: {declared} кадров ({declared / fps:.0f} с), "
          f"{w}x{h}")

    # --- copy ---------------------------------------------------------------
    if not args.skip_copy:
        if DEST.exists() and DEST.stat().st_size == sz:
            print(f"  копия уже на месте: {DEST.relative_to(ROOT)}")
        else:
            print(f"  копирую в {DEST.relative_to(ROOT)} ...")
            t0 = time.perf_counter()
            shutil.copy2(SOURCE, DEST)
            print(f"    готово за {time.perf_counter() - t0:.0f} с, "
                  f"{DEST.stat().st_size / 1e9:.2f} ГБ")
    report["copied"] = DEST.exists()

    # --- count --------------------------------------------------------------
    print("  считаю кадры (заголовку доверять нельзя) ...")
    t0 = time.perf_counter()
    n = count_frames(DEST if DEST.exists() else SOURCE)
    print(f"    фактически {n} кадров = {n / fps:.1f} с за "
          f"{time.perf_counter() - t0:.0f} с")
    print(f"    множитель к заявленному: {n / max(declared, 1):.4f}   "
          f"(у VID00001 было 0.8187, у VID00002 0.8186)")
    report.update({"declared_frames": declared, "actual_frames": n,
                   "duration_s": n / fps, "fps": fps, "size": [w, h]})

    # --- transcode ----------------------------------------------------------
    if not args.skip_transcode:
        MEDIA.parent.mkdir(parents=True, exist_ok=True)
        vf = "scale=1280:720"
        if args.rotate in ("cw", "ccw"):
            # after a 90 degree turn the frame is portrait, so a fixed 1280x720 would
            # squash it. Keep the proportions and let the height follow, rounded down to
            # an even number.
            t = "transpose=1" if args.rotate == "cw" else "transpose=2"
            vf = f"{t},scale=720:-2"
        elif args.rotate == "180":
            vf = "transpose=1,transpose=1,scale=1280:720"
        print(f"  перекодирую для браузера (поворот: {args.rotate}) ...")
        # `-r 30` before `-i` reinterprets the source as contiguous 30 fps. Without it the
        # declared 45000-frame header stretches the content over 1500 s and every
        # timestamp in the browser lands early by the ratio measured above.
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", "30", "-i", str(DEST),
               "-vf", vf, "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(MEDIA)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ffmpeg не справился: {r.stderr[:300]}")
        else:
            print(f"    готово за {time.perf_counter() - t0:.0f} с, "
                  f"{MEDIA.stat().st_size / 1e6:.0f} МБ")
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames,avg_frame_rate",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1", str(MEDIA)],
                capture_output=True, text=True)
            print(f"    проверка шкалы: {probe.stdout.strip().replace(chr(10), ' ')}")
            report["transcoded"] = True
            report["media"] = str(MEDIA)

    # --- a frame for the record --------------------------------------------
    cap = cv2.VideoCapture(str(DEST))
    for _ in range(3000):
        cap.grab()
    ok, fr = cap.read()
    cap.release()
    if ok:
        if ROTATE[args.rotate] is not None:
            fr = cv2.rotate(fr, ROTATE[args.rotate])
        cv2.imwrite(str(OUT / "prepared_frame.png"),
                    cv2.resize(fr, (768, int(768 * fr.shape[0] / fr.shape[1]))))
        print(f"  кадр для протокола: {OUT.relative_to(ROOT)}/prepared_frame.png")

    (OUT / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  дальше: прогон мозга по этому ролику (как p062 для VID00002)")
    print(f"  перед этим P07.3 требует записать истинный маршрут вручную")


if __name__ == "__main__":
    main()
