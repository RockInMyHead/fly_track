#!/usr/bin/env python3
"""Заморозка цепочки FINAL TRACKER V1 перед слепым прогоном на VID00018.

Делать ДО появления полноценного VID00018 (не 1.4-с обрывка из манифеста).

    PYTHONPATH=. .venv/bin/python scripts/final_tracker_freeze.py --verify
    PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_freeze.py --write

Пишет data/final_tracker/FROZEN_BEFORE_VID00018.json — полные sha256 артефактов и
копию правил из FROZEN_V1.json. После записи любое изменение файлов из manifest
запрещает слепой прогон до новой заморозки с причиной.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/final_tracker/FROZEN_BEFORE_VID00018.json"
V1 = ROOT / "data/final_tracker/FROZEN_V1.json"
GRAPH = ROOT / "data/p08/graph.json"
TRACKER = ROOT / "scripts/final_tracker.py"
MANIFEST = ROOT / "data/p14/MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha16(path: Path) -> str:
    return sha256(path)[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="записать заморозку")
    ap.add_argument("--verify", action="store_true", help="проверить manifest против диска")
    a = ap.parse_args()

    if not a.write and not a.verify:
        ap.print_help()
        return 0

    if not OUT.exists() and a.verify:
        print(f"нет {OUT} — сначала --write")
        return 1

    if a.write:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/final_tracker_freeze.py"), "--verify"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            print("  СТОП: FROZEN_V1.json не совпадает с кодом. Сначала приведите V1 в согласие.")
            return 1

        if not V1.exists():
            print(f"нет {V1}")
            return 1
        v1 = json.loads(V1.read_text(encoding="utf-8"))
        enc = {}
        if MANIFEST.exists():
            enc = json.loads(MANIFEST.read_text(encoding="utf-8")).get("encoder") or {}

        manifest: dict[str, dict] = {}
        for rel in [
            "scripts/final_tracker.py",
            "data/final_tracker/FROZEN_V1.json",
            "data/p08/graph.json",
            "scripts/p12_channels.py",
            "scripts/p084b_geometry.py",
            "scripts/p08_graph.py",
            "scripts/p13_stillness.py",
            "scripts/p13_1_gate.py",
            "fly_vo/visual_encoder.py",
            "fly_vo/brain_clock.py",
            "fly_vo/malecns_engine.py",
            "fly_vo/config.py",
            "fly_vo/video_reader.py",
        ]:
            p = ROOT / rel
            if p.exists():
                manifest[rel] = {"sha256": sha256(p), "sha16": sha16(p)}

        doc = {
            "phase": "FINAL TRACKER V1 — слепая проверка",
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frozen_before": "VID00018",
            "also_prepare_untouched": "VID00019 — снять и не прогонять до следующей версии",
            "why": ("V1 больше не улучшаем. Единственная защита — зафиксировать код, граф, "
                    "константы и правила до первого прогона на новом ролике."),
            "constants": {
                "v_walk_m_s": v1["v_walk"],
                "v_walk_kind": v1["v_walk_kind"],
                "stop_threshold": v1["stop"]["threshold"],
                "stop_source": v1["stop"]["source"],
            },
            "channels": {
                "camera_yaw": {
                    "implementation": "scripts/p12_channels.py yaw_channel + p084b_geometry.alignment",
                    "signal_file": "output/p07/yaw_signal_{v}.csv",
                    "column": "yaw_signal_deadband",
                    "classes": ["LEFT", "RIGHT", "SILENT"],
                },
                "stop_gate": {
                    "implementation": "scripts/p13_stillness.frame_motion",
                    "threshold": v1["stop"]["threshold"],
                    "effect": "progress_speed = 0",
                },
                "move_no_net": {
                    "implementation": "scripts/p12_channels.py early features + frozen reader",
                    "reader": v1["reader_net_displacement"],
                    "classes": ["MOVE", "NO_NET"],
                    "not_stop": v1["reader_net_displacement"].get("not_stop"),
                },
                "graph_beam": v1["rules"],
            },
            "encoder": enc,
            "start_parameters": {
                "note": ("старт задаётся при слепом прогоне: --start-edge и --start-from "
                         "(или --start-node / --previous-node). Записывается в "
                         "output/final_tracker/VID00018_BLIND/FREEZE.json вместе с результатом."),
                "example_from_p13": "output/p13/runs/{v}/run_start.json после цепочки P13",
            },
            "prep_chain_frozen": {
                "order": [
                    "положить data/p01r/VID00018.AVI (или камера; не смотреть маршрут)",
                    "PYTHONPATH=. .venv/bin/python scripts/p14_manifest.py --frames VID00018",
                    "PYTHONPATH=. .venv/bin/python scripts/p062_neuron_run.py --video VID00018.AVI --tag vid18",
                    "PYTHONPATH=. .venv/bin/python scripts/p09_real_holdout.py --video VID00018 --record",
                    "PYTHONPATH=. .venv/bin/python scripts/p10_record.py --video VID00018",
                    "PYTHONPATH=. .venv/bin/python scripts/p07_videos.py --add VID00018 "
                    "--trace output/p06_neurons_vid18/spike_trace.npz "
                    "--targets output/p06_neurons_vid18/targets.csv",
                    "PYTHONPATH=. .venv/bin/python scripts/p07_fly_readout.py",
                    "PYTHONPATH=. .venv/bin/python scripts/p07_trajectory.py",
                ],
                "outputs_required_by_tracker": [
                    "output/p10/brain_VID00018.npz",
                    "output/p07/yaw_signal_VID00018.csv",
                ],
            },
            "blind_run": {
                "once": True,
                "out_dir": "output/final_tracker/VID00018_BLIND",
                "command": ("PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_run.py "
                            "--video VID00018 --start-edge … --start-from …"),
                "lock_rule": "повторный прогон запрещён, пока не удалён LOCK.json осознанно",
            },
            "compare_after_human_truth": {
                "script": "scripts/final_tracker_blind_compare.py",
                "truth": "data/final_tracker/VID00018_truth.json",
                "primary_metric": "fraction_time_on_correct_edge",
            },
            "rules": {
                "do_not_tune_on_vid00018": True,
                "do_not_rerun_after_lock": True,
                "human_route_only_after_blind_lock": True,
                "even_strange_result_counts": True,
                "v2_needs_new_video": "если будет V2 — проверять на VID00019, не пересчитывать 18",
            },
            "manifest": manifest,
            "fingerprint_summary": {
                "final_tracker_py_sha256": manifest.get("scripts/final_tracker.py", {}).get("sha256"),
                "frozen_v1_sha256": manifest.get("data/final_tracker/FROZEN_V1.json", {}).get("sha256"),
                "graph_json_sha256": manifest.get("data/p08/graph.json", {}).get("sha256"),
            },
            "verification": "PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_freeze.py --verify",
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  записано: {OUT}")
        print(f"  V_WALK = {doc['constants']['v_walk_m_s']}, STOP = {doc['constants']['stop_threshold']}")
        print(f"  файлов в manifest: {len(manifest)}")
        return 0

    doc = json.loads(OUT.read_text(encoding="utf-8"))
    print("=" * 96)
    print("ПРОВЕРКА FROZEN_BEFORE_VID00018")
    print("=" * 96)
    print(f"  заморожено: {doc['frozen_at']}")
    bad = []
    for rel, m in doc["manifest"].items():
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel} — пропал")
            continue
        now = sha256(p)
        if now != m["sha256"]:
            bad.append(f"{rel} — изменён ({m['sha256'][:16]} → {now[:16]})")
    if bad:
        for b in bad:
            print(f"  {b}")
        print(f"\n  РАСХОЖДЕНИЙ: {len(bad)}. Слепой прогон запрещён.")
        return 1
    print(f"  все {len(doc['manifest'])} файлов совпадают")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
