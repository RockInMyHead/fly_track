#!/usr/bin/env python3
"""
P07.2 — the trajectory once the detector decides when a turn is happening.

The gated signal is built from the events `p072_event_detector.py` accepted, integrated
exactly as before. Nothing else changes: the same heading rule, the same forward signal,
the same units. The only difference from P07.1 is which stretches contribute.

The trade the detector makes is deliberate and stated in the output: sign accuracy stays
at 96 to 100 percent and false events fall to under two per turn, but between half and
three quarters of the real turns are now missed. Fewer, more reliable turns.

Usage:
    PYTHONPATH=. python scripts/p072_trajectory.py
    PYTHONPATH=. python scripts/p072_trajectory.py --video VID00002
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p072"
OUT71 = ROOT / "output/p071"
FPS = 50.0
def _applied() -> list[str]:
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    try:
        from p07_videos import videos as _reg
        out = [v for v in ("VID00001", "VID00002") ]
        for name in _reg():
            if name not in out and (OUT / f"events_{name}.csv").exists():
                out.append(name)
        return out
    except Exception:
        return ["VID00001", "VID00002"]


VIDEOS = _applied()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw-scale", type=float, default=90.0)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    videos = [args.video] if args.video else VIDEOS
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    result = {}

    for video in videos:
        m = np.load(OUT71 / f"turn_mask_{video}.npz")
        t = m["t"]
        combined = m["combined"]
        events = list(csv.DictReader((OUT / f"events_{video}.csv").open()))
        speed_rows = list(csv.DictReader(
            (ROOT / "output/p07" / f"forward_signal_{video}.csv").open()))
        speed = np.array([float(r["speed"]) for r in speed_rows])

        mask = np.zeros(len(t), dtype=bool)
        for e in events:
            a = int(round(float(e["t_start"]) * FPS))
            b = int(round(float(e["t_end"]) * FPS))
            mask[a:b + 1] = True

        gated = np.where(mask, combined, 0.0)
        dt = float(np.median(np.diff(t)))
        n = min(len(t), len(speed))
        theta = np.zeros(n)
        x = np.zeros(n)
        y = np.zeros(n)
        th = 0.0
        for k in range(1, n):
            th += gated[k] * args.yaw_scale * dt
            theta[k] = th
            x[k] = x[k - 1] + speed[k] * np.cos(th) * dt
            y[k] = y[k - 1] + speed[k] * np.sin(th) * dt

        seg = np.hypot(np.diff(x), np.diff(y))
        path_len = float(seg.sum())
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        turning = float(np.sum(np.abs(gated)) * args.yaw_scale * dt)
        result[video] = {
            "n_events": len(events),
            "frac_time_turning": float(np.mean(mask)),
            "path_length": path_len, "net_displacement": net,
            "straightness": net / path_len if path_len > 0 else 0.0,
            "turning_units": turning,
            "net_turn_units": float(theta[-1] - theta[0]),
            "final": [float(x[-1]), float(y[-1])],
            "sign_window_accuracy": report["summary"][video]["sign_window"],
            "false_per_real": report["summary"][video]["false_per_real"],
            "recall": report["summary"][video]["recall"],
        }

        with (OUT / f"trajectory_{video}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "theta", "yaw_gated", "yaw_full", "turning",
                        "speed"])
            for k in range(n):
                w.writerow([f"{t[k]:.3f}", f"{x[k]:.6f}", f"{y[k]:.6f}",
                            f"{theta[k]:.6f}", f"{gated[k]:.5f}",
                            f"{combined[k]:.5f}", int(mask[k]), f"{speed[k]:.5f}"])

        print(f"=== {video} ===")
        print(f"  событий {len(events)}, занимают {np.mean(mask) * 100:.1f}% времени")
        print(f"  путь {path_len:.1f}, смещение {net:.1f}, "
              f"прямизна {net / max(path_len, 1e-9):.3f}")
        print(f"  суммарный поворот {turning:.0f} ед., чистый {theta[-1]:.0f} ед.")

    # ---- compare with everything that came before --------------------------
    old71 = json.loads((OUT71 / "report.json").read_text(encoding="utf-8"))
    print(f"\n=== ПРЯМИЗНА: ВСЕ ВАРИАНТЫ ===")
    print(f"  {'источник':>38s} {'прямизна':>9s} {'событий':>8s}")
    for video in videos:
        f = ROOT / "output/p04_trajectory/trajectory_turns.csv"
        if f.exists():
            r = list(csv.DictReader(f.open()))
            xx = np.array([float(a["x"]) for a in r])
            yy = np.array([float(a["y"]) for a in r])
            L = float(np.hypot(np.diff(xx), np.diff(yy)).sum())
            nt = float(np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]))
            print(f"  {'фронтенд (P0.4, по поворотам)':>38s} {nt / max(L, 1e-9):>9.3f} "
                  f"{'—':>8s}")
            break
    for video in videos:
        r7 = json.loads((ROOT / "output/p07/report.json").read_text(encoding="utf-8"))
        print(f"  {f'P07 непрерывно ({video})':>38s} "
              f"{r7['videos'][video]['straightness']:>9.3f} "
              f"{r7['videos'][video]['n_events'] if 'n_events' in r7['videos'][video] else '—':>8}")
        print(f"  {f'P07.1 гейт ({video})':>38s} "
              f"{old71['videos'][video]['straightness']:>9.3f} "
              f"{old71['videos'][video]['n_events']:>8}")
        print(f"  {f'P07.2 детектор ({video})':>38s} "
              f"{result[video]['straightness']:>9.3f} "
              f"{result[video]['n_events']:>8}")

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, len(videos), figsize=(7 * len(videos), 6.4))
    if len(videos) == 1:
        axes = [axes]
    for ax, video in zip(axes, videos):
        r = list(csv.DictReader((OUT / f"trajectory_{video}.csv").open()))
        xx = np.array([float(a["x"]) for a in r])
        yy = np.array([float(a["y"]) for a in r])
        ax.plot(xx, yy, lw=0.9, color="tab:blue")
        ax.plot(xx[0], yy[0], "o", color="tab:green", ms=11, label="START")
        ax.plot(xx[-1], yy[-1], "s", color="tab:red", ms=11, label="END")
        ax.set_aspect("equal")
        d = result[video]
        ax.set_title(f"{video} — траектория с детектором событий\n"
                     f"прямизна {d['straightness']:.3f}, событий {d['n_events']}, "
                     f"знак {d['sign_window_accuracy'] * 100:.0f}%, "
                     f"ложных/наст. {d['false_per_real']:.2f}, "
                     f"найдено {d['recall'] * 100:.0f}%")
        ax.set_xlabel("x (условные единицы)")
        ax.set_ylabel("y (условные единицы)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("P07.2 — траектория из MaleCNS, поворот только в подтверждённых событиях")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "trajectory.png", dpi=120)
    plt.close(fig)

    (OUT / "trajectory_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}/trajectory.png, trajectory_report.json")


if __name__ == "__main__":
    main()
