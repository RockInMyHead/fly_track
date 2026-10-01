#!/usr/bin/env python3
"""P09.5 — widen the viewing window so the body's response is actually visible.

The first cut showed three seconds before a move and five after. Watching it, the response to
the move was simply not in the frame: the subject walks at about 0.17 m/s and is in motion only a
quarter of the time, so five seconds after a rotation contains almost nothing. Measured over the
sixty selected moves, a window of three seconds before and five after contains more than a metre
of travel in fifteen percent of them; at twenty seconds after, seventy-five percent. The question
being asked — did the body change direction — cannot be answered from a window that does not
contain the body's answer.

That is a defect of the material, not of the labels, and the numbers above are what makes it a
design correction rather than a preference: the window is chosen because the evidence has to be
present, not because a longer window gives the answer anyone wants.

What changes and what does not
------------------------------
The sixty events, their order, the clips they come from, the hidden algorithmic data and the
frozen rule are untouched. Only two numbers per event move — where the default view starts and
stops, and where the wider one does — and the clip files are re-cut to cover them. The event list
is what must not change; the window is a property of how the question is put.

Labels already recorded
-----------------------
If any labels exist they were made under the short window and cannot be compared with labels made
under the long one: the same event, watched for eight seconds and for twenty-three, is not the
same question. They are moved aside rather than deleted, and the events they covered go back into
the queue so that every event is answered once, under one window.

Usage:
    PYTHONPATH=. python scripts/p095_recut.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/p095"
CLIPS = ROOT / "output/p095/clips"
MEDIA = ROOT / "webapp/media"

# frozen material: five seconds before the move is enough to see the approach, twenty-five after
# is where the body's answer has appeared in three quarters of the moves
PRE_S, POST_S = 5.0, 25.0
# what the page plays first, and what the wider button opens
NARROW = (2.0, 25.0)      # 3 s before the centre .. 20 s after
WIDE = (0.0, 30.0)        # the whole file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre", type=float, default=PRE_S)
    ap.add_argument("--post", type=float, default=POST_S)
    args = ap.parse_args()

    setp = DATA / "review_set.json"
    s = json.loads(setp.read_text(encoding="utf-8"))
    events = s["events"]
    print("=" * 96)
    print("P09.5 — ПЕРЕРЕЗКА КЛИПОВ: ОКНО, В КОТОРОМ ВИДНО ОТВЕТ ТЕЛА")
    print("=" * 96)
    print(f"  было: 3 с до, 5 с после (8 с)   ·   стало: {args.pre:.0f} с до, "
          f"{args.post:.0f} с после ({args.pre + args.post:.0f} с)")
    print(f"  событий: {len(events)} (список и порядок НЕ меняются)")
    print()

    # ---- labels made under the old window -------------------------------------------------
    lab = DATA / "human_labels.json"
    if lab.exists():
        old = json.loads(lab.read_text(encoding="utf-8")).get("labels", [])
        if old:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            keep = DATA / f"human_labels_short_window_{stamp}.json"
            shutil.copy(lab, keep)
            lab.unlink()
            covered = sorted({l["event_id"] for l in old if l.get("round", 1) == 1})
            print(f"  меток было: {len(old)}, под коротким окном")
            print(f"  они сохранены в {keep.name}, но в зачёт не идут:")
            print("  одно и то же событие, просмотренное 8 и 23 секунды, — это разные вопросы,")
            print("  и смешивать такие метки нельзя. Все события отвечаются заново.")
            print(f"  события, которые придётся пройти снова: {len(covered)} "
                  f"({', '.join(covered[:8])}{'…' if len(covered) > 8 else ''})")
            print()

    # ---- re-cut ---------------------------------------------------------------------------
    clips_dir = CLIPS
    done = failed = 0
    for i, e in enumerate(events, 1):
        clip = e["clip"]
        # The centre of the move, which is `time` and nothing else. An earlier version of this
        # script tried to recover it from the stored offsets by subtracting the old lead, and cut
        # every clip from the start of the recording instead: `clip_offset_s` is a constant lead,
        # not a position, so the subtraction gave zero for all sixty. The centre was in the file
        # the whole time under its own name.
        t_center = float(e["time"])
        hp = Path(str(CLIPS / f"{e['event_id']}.mp4"))
        src = MEDIA / f"{clip}_fixed.mp4"
        start = max(t_center - args.pre, 0.0)
        dur = (t_center + args.post) - start
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
               "-t", f"{dur:.3f}", "-vf", "scale=854:-2", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "30", "-an", "-movflags", "+faststart",
               str(hp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed += 1
            print(f"  {e['event_id']}: ffmpeg ошибка {r.stderr[:80]}")
            continue
        # the file now starts pre seconds before the centre
        e["clip_offset_s"] = round(args.pre, 3)
        e["narrow"] = [round(args.pre - 3.0, 3), round(args.pre + 20.0, 3)]
        e["wide"] = [0.0, round(args.pre + args.post, 3)]
        done += 1
        if i % 15 == 0 or i == len(events):
            print(f"   перерезано {i} из {len(events)}", flush=True)

    s["window_first"] = {"before_s": 3.0, "after_s": 20.0}
    s["window_wide"] = {"before_s": args.pre, "after_s": args.post}
    s["window_note"] = (
        "окно расширено по измерению: при 5 с после события тело видно лишь в 15% окон "
        "(медиана пути 0.59 м), при 20 с — в 75% (1.44 м). Список событий и порядок не менялись.")
    s["recut_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    setp.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- check by frame: is the moment of the move actually inside the new file? ---------
    # The previous version of this script produced sixty well-formed files of the wrong part of
    # the recording, and nothing about a file's size or duration would have caught that. So one
    # event is checked against the source directly: the frame at the narrow window's start must
    # match the source frame three seconds before the centre.
    print()
    print("  ПРОВЕРКА: кадр из нового клипа против кадра из источника")
    import tempfile
    ok = 0
    for e in events[:3]:
        src = MEDIA / f"{e['clip']}_fixed.mp4"
        t_center = float(e["time"])
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.png"
            b = Path(td) / "b.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss",
                            f"{e['narrow'][0]:.3f}", "-i",
                            str(CLIPS / f"{e['event_id']}.mp4"), "-frames:v", "1",
                            "-vf", "scale=160:-2", str(a)], check=False)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss",
                            f"{t_center - 3.0:.3f}", "-i", str(src), "-frames:v", "1",
                            "-vf", "scale=160:-2", str(b)], check=False)
            if not a.exists() or not b.exists():
                print(f"  {e['event_id']}: кадры не извлечены")
                continue
            from PIL import Image
            import numpy as np
            ga = np.asarray(Image.open(a).convert("L"), dtype=float)
            gb = np.asarray(Image.open(b).convert("L"), dtype=float)
            n = min(ga.shape[0], gb.shape[0]), min(ga.shape[1], gb.shape[1])
            diff = float(np.abs(ga[:n[0], :n[1]] - gb[:n[0], :n[1]]).mean())
            good = diff < 6.0
            ok += int(good)
            print(f"  {e['event_id']}: средняя разница яркости {diff:.2f} — "
                  f"{'совпадает' if good else 'НЕ совпадает'}")
    print(f"  проверено {ok} из 3")
    print()

    print(f"  перерезано: {done}, ошибок: {failed}")
    sizes = sum(f.stat().st_size for f in clips_dir.glob("*.mp4"))
    print(f"  объём клипов: {sizes / 1e6:.0f} МБ")
    print(f"  записано: {setp}")
    print()
    print("  страница подхватит новые окна сама — они читаются из review_set.json")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
