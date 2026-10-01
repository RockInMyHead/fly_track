#!/usr/bin/env python3
"""
P07.1 — figures and film for the turn-filtered trajectory.

Three figures and a film, no numbers that are not in the csv files.

    trajectory_<video>.png   the gated route, with the detected events marked
    signals_<video>.png      yaw, its gate, the heading and the forward signal
    events_vs_turns.png      when the filter fired against when a human saw a turn

The last one is the diagnostic that explains the result. The yaw sign is right whenever
the filter fires on a real turn, but it fires several times more often than real turns
occur, so the extra events walk the heading around.

Usage:
    PYTHONPATH=. python scripts/p071_plots.py
    PYTHONPATH=. python scripts/p071_plots.py --video VID00002 --no-film
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p071"
FPS = 50.0


def read_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    out = {}
    for c in rows[0].keys():
        try:
            out[c] = np.array([float(r[c]) for r in rows])
        except (ValueError, TypeError):
            out[c] = np.array([r[c] for r in rows])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="only this one; default is all")
    ap.add_argument("--no-film", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    videos = [args.video] if args.video else list(report["videos"].keys())

    # ---- trajectories ------------------------------------------------------
    fig, axes = plt.subplots(1, len(videos), figsize=(7 * len(videos), 6.4))
    if len(videos) == 1:
        axes = [axes]
    for ax, v in zip(axes, videos):
        d = read_csv(OUT / f"trajectory_{v}.csv")
        ax.plot(d["x"], d["y"], lw=0.8, color="tab:blue", zorder=1)
        ax.plot(d["x"][0], d["y"][0], "o", color="tab:green", ms=11, zorder=3,
                label="START")
        ax.plot(d["x"][-1], d["y"][-1], "s", color="tab:red", ms=11, zorder=3,
                label="END")
        r = report["videos"][v]
        ax.set_aspect("equal")
        ax.set_xlabel("x (условные единицы)")
        ax.set_ylabel("y (условные единицы)")
        ax.set_title(f"{v} — траектория из MaleCNS, повороты только в событиях\n"
                     f"прямизна {r['straightness']:.3f}, поворот осталось "
                     f"{r['turning_kept'] * 100:.0f}%, "
                     f"ложных на настоящий {r['false_events_per_real_turn']:.1f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("P07.1 — траектория после фильтра реальных поворотов. "
                 "Масштаб условный.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "trajectory.png", dpi=120)
    plt.close(fig)

    # ---- signals -----------------------------------------------------------
    for v in videos:
        d = read_csv(OUT / f"trajectory_{v}.csv")
        m = np.load(OUT / f"turn_mask_{v}.npz")
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(d["t"], d["yaw_full"], lw=0.4, color="0.7", label="весь сигнал")
        axes[0].plot(d["t"], d["yaw_gated"], lw=0.6, color="tab:blue",
                     label="после фильтра")
        axes[0].set_ylabel("yaw (>0 = влево)", fontsize=9)
        axes[0].legend(fontsize=8)

        axes[1].fill_between(d["t"], 0, d["turning"].astype(float) * 3.6,
                             color="tab:green", alpha=0.6, step="mid")
        axes[1].set_ylabel("событие поворота", fontsize=9)
        axes[1].set_yticks([0, 3.6])
        axes[1].set_yticklabels(["нет", "да"], fontsize=8)

        axes[2].plot(d["t"], np.rad2deg(d["theta"]), lw=0.8, color="tab:purple")
        axes[2].set_ylabel("курс, град (условные)", fontsize=9)

        axes[3].plot(d["t"], d["speed"], lw=0.5, color="tab:orange")
        axes[3].set_ylabel("speed", fontsize=9)

        for a in axes:
            a.grid(alpha=0.3)
        axes[-1].set_xlabel("время, с")
        r = report["videos"][v]
        fig.suptitle(f"{v} — сигналы после фильтра. "
                     f"Поворот осталось {r['turning_kept'] * 100:.0f}%, "
                     f"настоящих поворотов замечено {r['turns_detected']} "
                     f"из {r['turns_total']}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT / f"signals_{v}.png", dpi=120)
        plt.close(fig)

    # ---- events against human turns ---------------------------------------
    from p071_turn_filter import human_windows  # noqa: E402
    sys_path_ok = True

    fig, axes = plt.subplots(len(videos), 1, figsize=(15, 3.4 * len(videos)))
    if len(videos) == 1:
        axes = [axes]
    for ax, v in zip(axes, videos):
        m = np.load(OUT / f"turn_mask_{v}.npz")
        t = m["t"]
        mask = m["mask"].astype(float)
        # detected events, drawn as a band
        ax.fill_between(t, 1.5, 1.5 + mask * 1.0, color="tab:blue", alpha=0.6,
                        step="mid", label="событие фильтра")
        wins = human_windows(v)
        for a, b, kind in wins:
            ax.fill_between([a, b], 0.2, 1.2,
                            color=("tab:green" if kind == "LEFT" else "tab:red"),
                            alpha=0.6)
        ax.set_ylim(-0.1, 2.9)
        ax.set_yticks([0.7, 2.0])
        ax.set_yticklabels(["разметка\n(зел. LEFT, кр. RIGHT)", "события\nфильтра"],
                           fontsize=8)
        r = report["videos"][v]
        ax.set_title(f"{v}: событий {r['n_events']}, настоящих поворотов "
                     f"{r['turns_total']}, замечено {r['turns_detected']}, "
                     f"знак верен {r['turns_with_correct_sign']} — "
                     f"ложных событий на один настоящий "
                     f"{r['false_events_per_real_turn']:.1f}", fontsize=10)
        ax.set_xlim(t[0], min(t[-1], t[0] + 400))
        ax.set_xlabel("время, с (первые 400 с)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("P07.1 — когда сработал фильтр и когда человек видел поворот. "
                 "Срабатывает верно, но слишком часто.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "events_vs_turns.png", dpi=120)
    plt.close(fig)

    print(f"Wrote {OUT}/trajectory.png, signals_*.png, events_vs_turns.png")

    if args.no_film:
        return

    # ---- film --------------------------------------------------------------
    import cv2
    from fly_vo.video_reader import VideoReader

    for v in videos:
        d = read_csv(OUT / f"trajectory_{v}.csv")
        src = ROOT / f"data/p01r/{v}.AVI"
        x, y, t, turning = d["x"], d["y"], d["t"], d["turning"].astype(float)
        pad = 0.08 * max(x.max() - x.min(), y.max() - y.min(), 1e-9)
        span = max(x.max() - x.min() + 2 * pad, y.max() - y.min() + 2 * pad)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        w, h = 640, 360
        vw = cv2.VideoWriter(str(OUT / f"trajectory_video_{v}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (w + h, h))
        if not vw.isOpened():
            print("  не удалось открыть VideoWriter")
            return

        def to_px(px, py):
            u = (px - cx) / span * (h * 0.9) + h / 2
            vv = (py - cy) / span * (h * 0.9) + h / 2
            return int(u), int(h - vv)

        idx = 0
        n_written = 0
        with VideoReader(src) as vr:
            for fi, frame in vr.read_all():
                while idx + 1 < len(t) and t[idx + 1] <= fi / 30.0:
                    idx += 1
                if fi % 3:
                    continue
                left = cv2.resize(frame, (w, h))
                canvas = np.zeros((h, h, 3), np.uint8)
                canvas[:] = (18, 18, 18)
                end = min(idx + 1, len(x))
                pts = np.array([to_px(x[k], y[k]) for k in range(0, end, 6)],
                               np.int32)
                if len(pts) > 1:
                    cv2.polylines(canvas, [pts], False, (200, 170, 60), 1,
                                  cv2.LINE_AA)
                cv2.circle(canvas, to_px(x[end - 1], y[end - 1]), 5, (80, 80, 235), -1)
                cv2.circle(canvas, to_px(x[0], y[0]), 5, (90, 200, 90), -1)
                if turning[end - 1] > 0:
                    cv2.putText(canvas, "TURN", (10, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 220, 250), 2)
                cv2.putText(canvas, f"t={t[end - 1]:.0f}s", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                cv2.putText(canvas, "P07.1 gated trajectory", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
                vw.write(np.hstack([left, canvas]))
                n_written += 1
        vw.release()
        print(f"  {v}: записано {n_written} кадров")


if __name__ == "__main__":
    main()
