#!/usr/bin/env python3
"""Слепой прогон FINAL TRACKER V1 на новом ролике (VID00018).

Порядок: заморозка → входы → один прогон → LOCK. Маршрут человека не используется.

    PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_freeze.py --verify
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_run.py --check-only \\
        --video VID00018 --start-edge E1__E2 --start-from E1
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_run.py \\
        --video VID00018 --start-edge … --start-from …

Опционально --prep запускает замороженный конвейер входов (долго).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "data/final_tracker/FROZEN_BEFORE_VID00018.json"
PY = sys.executable


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(doc: dict) -> list[str]:
    bad = []
    for rel, m in doc["manifest"].items():
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel} — нет")
        elif sha256(p) != m["sha256"]:
            bad.append(f"{rel} — изменён")
    return bad


def needs(video: str) -> list[tuple[str, str]]:
    v = video
    tag = "vid" + str(int(v.replace("VID", "")))
    return [
        (f"data/p01r/{v}.AVI", "положить ролик (или доступ с камеры через resolve_video)"),
        (f"output/p06_neurons_{tag}/spike_trace.npz", f"p062_neuron_run.py --tag {tag}"),
        (f"output/p09/trace_{v}.npz", f"p09_real_holdout.py --video {v} --record"),
        (f"output/p10/brain_{v}.npz", f"p10_record.py --video {v}"),
        (f"output/p07/yaw_signal_{v}.csv", "p07_fly_readout.py + p07_trajectory.py после регистрации"),
    ]


def run_prep(video: str, log: Path) -> int:
    tag = "vid" + str(int(video.replace("VID", "")))
    steps = [
        [PY, "scripts/p14_manifest.py", "--frames", video],
        [PY, "scripts/p062_neuron_run.py", "--video", f"{video}.AVI", "--tag", tag],
        [PY, "scripts/p09_real_holdout.py", "--video", video, "--record"],
        [PY, "scripts/p10_record.py", "--video", video],
        [PY, "scripts/p07_videos.py", "--add", video,
         "--trace", f"output/p06_neurons_{tag}/spike_trace.npz",
         "--targets", f"output/p06_neurons_{tag}/targets.csv"],
        [PY, "scripts/p07_fly_readout.py"],
        [PY, "scripts/p07_trajectory.py"],
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        for cmd in steps:
            fh.write(f"\n=== {' '.join(cmd)} ===\n")
            fh.flush()
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return r.returncode
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="VID00018")
    ap.add_argument("--start-edge", required=False)
    ap.add_argument("--start-from", required=False)
    ap.add_argument("--prep", action="store_true", help="записать входы замороженным конвейером")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None,
                    help="по умолчанию output/final_tracker/VID00018_BLIND")
    a = ap.parse_args()
    v = a.video
    out_dir = Path(a.out) if a.out else ROOT / f"output/final_tracker/{v}_BLIND"
    lock = out_dir / "LOCK.json"

    print("=" * 100)
    print(f"FINAL TRACKER V1 — СЛЕПОЙ ПРОГОН {v}")
    print("=" * 100)

    if not FREEZE.exists():
        print(f"  СТОП: нет {FREEZE}")
        print("  PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_freeze.py --write")
        return 1
    freeze_doc = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = verify_freeze(freeze_doc)
    if bad:
        print("  ЗАМОРОЗКА НАРУШЕНА:")
        for b in bad:
            print(f"    {b}")
        return 1
    print(f"  заморозка {freeze_doc['frozen_at']}: {len(freeze_doc['manifest'])} файлов OK")

    if lock.exists():
        print(f"  СТОП: {lock} уже есть — результат заблокирован.")
        print("  Повторный прогон не выполняется (даже если результат выглядит странно).")
        return 1

    miss = [(rel, how) for rel, how in needs(v) if not (ROOT / rel).exists()]
    if a.prep and miss:
        print("  --prep: записываю входы…")
        rc = run_prep(v, out_dir / "prep.log")
        if rc != 0:
            print(f"  prep завершился с кодом {rc}, см. {out_dir / 'prep.log'}")
            return rc
        miss = [(rel, how) for rel, how in needs(v) if not (ROOT / rel).exists()]

    print("\n─── ВХОДЫ ───")
    for rel, how in needs(v):
        ok = (ROOT / rel).exists()
        print(f"  {rel:<44}{'есть' if ok else 'НЕТ'}")
    if miss:
        print("\n  СТОП: не хватает входов. Конвейер — в FROZEN_BEFORE_VID00018.json")
        for rel, how in miss:
            print(f"    {rel}: {how}")
        print(f"  или: … final_tracker_blind_run.py --video {v} --prep …")
        return 1

    if not a.start_edge or not a.start_from:
        print("\n  СТОП: укажите --start-edge и --start-from (старт известен до просмотра маршрута).")
        return 1

    if a.check_only:
        print("\n  --check-only: заморозка и входы OK, прогон не выполнен.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "scripts/final_tracker.py",
        "--video", v,
        "--graph", "data/p08/graph.json",
        "--start-edge", a.start_edge,
        "--start-from", a.start_from,
        "--out", str(out_dir),
    ]
    print("\n─── ОДИН ПРОГОН FINAL V1 ───")
    print("  " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        return r.returncode

    rep = out_dir / "report.json"
    if not rep.exists():
        print("  СТОП: нет report.json после прогона")
        return 1
    rep_doc = json.loads(rep.read_text(encoding="utf-8"))
    if rep_doc.get("video") != v:
        print(f"  СТОП: report.json описывает {rep_doc.get('video')}, не {v}")
        return 1

    freeze_copy = {
        "blind_freeze": freeze_doc,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": v,
        "start": {"edge": a.start_edge, "from": a.start_from},
        "constants": freeze_doc["constants"],
        "human_annotation_must_happen_after_this": True,
    }
    (out_dir / "FREEZE.json").write_text(
        json.dumps(freeze_copy, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(FREEZE, out_dir / "FROZEN_BEFORE_VID00018.json")

    lock_doc = {
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": v,
        "rule": "не пересчитывать после просмотра маршрута человеком",
        "artifacts": [
            "trajectory.csv", "edge_sequence.csv", "decisions.csv",
            "trajectory.png", "report.json", "FREEZE.json",
        ],
    }
    lock.write_text(json.dumps(lock_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  записано и заблокировано: {out_dir}")
    print("  Следующий шаг: человек смотрит ролик и кладёт truth в")
    print("  data/final_tracker/VID00018_truth.json")
    print("  затем: final_tracker_blind_compare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
