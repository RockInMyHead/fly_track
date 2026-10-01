#!/usr/bin/env python3
"""
P08.3 — checking the junction decisions against the video.

What is being checked and what is not
-------------------------------------
Not the whole route. Only the places where the system had a choice to make, because those are
the only places where the fly was consulted and therefore the only places where a wrong answer
is evidence about the fly rather than about the graph. Everywhere else there was one way on and
nothing was decided.

For each such junction the script produces:

    a video clip from three seconds before to five seconds after the decision
    a picture of the plan around the junction, with the way in, the ways on, and the way taken
    a row in decisions.csv carrying every number that went into the choice

The verdicts are left empty. They have to be filled in by watching, and the point of the stage
is that this judgement is not something the code can make: the code knows what the fly said and
what the geometry said, and neither of those is the answer to whether the walker went that way.

A note on what a wrong verdict means here
-----------------------------------------
If the fly says LEFT and the video shows the walker going right, the fault is in the fly's
reading for that junction. If the fly says nothing and geometry chose, a wrong verdict says
nothing about the fly at all — it says the tie-break rule needs work. The two are kept apart in
the report, because they call for different repairs.

Usage:
    PYTHONPATH=. python scripts/p083_route_audit.py
    PYTHONPATH=. python scripts/p083_route_audit.py --video VID00006 --graph data/p08/graph.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

tile = 380        # width of one frame in the strip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402

OUT = ROOT / "output/p083_route_audit"
P08_OUT = ROOT / "output/p08"
MEDIA = ROOT / "webapp/media"

PRE_S = 3.0        # seconds of video before the decision
POST_S = 5.0       # seconds after it
PAD_PX = 260       # how much plan to show around a junction


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def cut(src: Path, t0: float, t1: float, dest: Path) -> bool:
    """One clip, re-encoded so that the seek lands where asked rather than at a keyframe."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", str(src),
           "-t", f"{t1 - t0:.3f}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-an", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return dest.exists() and dest.stat().st_size > 1000
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"    не удалось нарезать {dest.name}: {exc}")
        return False


def frame_strip(src: Path, t: float, dest: Path, n: int = 7,
                half_s: float = 2.0) -> bool:
    """A row of stills from the decision window, so the motion can be seen at a glance.

    Playing an eight second clip and watching for the turn is slow, and on a dark corridor it
    is easy to miss. A strip of seven frames from four seconds before to four seconds after
    puts the whole movement in front of the eye at once, and the frame nearest the decision is
    marked, so there is something specific to look at.
    """
    if not src.exists():
        return False
    span = np.linspace(t - half_s, t + 2.0, n)
    tiles = []
    for k, ts in enumerate(span):
        tmp = dest.with_name(f".strip_{dest.stem}_{k}.png")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, ts):.3f}",
               "-i", str(src), "-frames:v", "1",
               # the corridor clips are dark, and a dark frame shows nothing at thumbnail
               # size, which is the whole point of the strip
               "-vf", f"eq=brightness=0.12:contrast=1.35:saturation=1.2,scale={tile}:-1",
               str(tmp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if tmp.exists():
                tiles.append((ts, Image.open(tmp).convert("RGB")))
                tmp.unlink()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    if not tiles:
        return False
    w, h = tiles[0][1].size
    bar = 30
    out = Image.new("RGB", (w * len(tiles), h + bar), "white")
    d = ImageDraw.Draw(out)
    for k, (ts, im) in enumerate(tiles):
        out.paste(im, (k * w, bar))
        d.rectangle([k * w, bar, (k + 1) * w - 1, h + bar - 1], outline=(200, 200, 200))
        near = abs(ts - t) < 0.35
        d.rectangle([k * w, 0, (k + 1) * w - 1, bar - 1],
                    fill=(220, 38, 38) if near else (240, 242, 245))
        d.text((k * w + 6, 9), f"{ts:+.1f} с" + ("   ← РЕШЕНИЕ" if near else ""),
               fill="white" if near else (30, 30, 30))
    out.save(dest)
    return True


def junction_picture(g: Graph, plan_path: Path, row: dict, dest: Path,
                     arrival: str | None) -> None:
    """The junction drawn the way the walker saw it, not the way the plan is drawn.

    The first version of this picture put the junction on the plan with north up. That is the
    wrong frame of reference for the question being asked. The question is whether the walker
    went left, right or straight *as they experienced it*, and the person answering it is
    watching the video, where left and right are relative to the direction of travel. On a
    north-up picture the two do not match, so judging becomes a mental rotation, and on a
    plan with corridors running at odd angles it becomes guesswork.

    So the picture is rotated about the junction until the direction of arrival points down,
    which puts the direction of travel up. After that the picture's left is the walker's left.
    The edges are then drawn as arrows from the junction, labelled with the angle off the
    direction of travel, and the one taken is filled.
    """
    img = plt.imread(plan_path)
    node = row["node"]
    cx, cy = g.pos(node)

    def rel(px: float, py: float) -> tuple[float, float]:
        """Plan coordinates relative to the junction, in screen units."""
        return (px - cx), (py - cy)

    # the rotation: bring the arrival direction onto straight up
    ang = 0.0
    if arrival and arrival in g.edges:
        # direction of travel at the junction: from the far end of the arrival edge to the
        # node, so that "up" on the picture becomes the way the walker was moving
        ax_, ay_ = rel(*g.pos(g.edges[arrival]["from"]))
        bx_, by_ = rel(*g.pos(g.edges[arrival]["to"]))
        # arrival may be stored either way round; the far end is the one we came from
        far = (ax_, ay_) if (ax_ ** 2 + ay_ ** 2) > (bx_ ** 2 + by_ ** 2) else (bx_, by_)
        travel = math.atan2(-far[1], -far[0])     # from the far end towards the junction
        # we want travel to point up (-y in image coordinates), which is angle -90
        ang = (-math.pi / 2) - travel

    ca, sa = math.cos(ang), math.sin(ang)

    def rot(px: float, py: float) -> tuple[float, float]:
        x, y = rel(px, py)
        return (x * ca - y * sa, x * sa + y * ca)

    span = 700.0     # metres of plan shown, in pixels
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.imshow(img, extent=(0, g.img_w, g.img_h, 0), alpha=0.75, zorder=0)
    ax.set_aspect("equal")

    # everything nearby, faint, so the junction can be located
    for e in g.edges.values():
        p, q = rot(*g.pos(e["from"])), rot(*g.pos(e["to"]))
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#94a3b8", lw=1.0, alpha=0.45, zorder=1)

    cands = [c for c in row["candidate_edges"].split("|") if c]
    sides = [s for s in row["candidate_sides"].split("|") if s]
    angles = [a for a in row["candidate_angles"].split("|") if a]

    # the way in, drawn as the tail behind the walker
    if arrival and arrival in g.edges:
        p, q = rot(*g.pos(g.edges[arrival]["from"])), rot(*g.pos(g.edges[arrival]["to"]))
        far = p if (p[0] ** 2 + p[1] ** 2) > (q[0] ** 2 + q[1] ** 2) else q
        ax.plot([0, far[0] * 0.9], [0, far[1] * 0.9], color="#f59e0b", lw=6.0,
                alpha=0.9, zorder=3, solid_capstyle="round")
        ax.annotate("", xy=(0, 0), xytext=(far[0] * 0.45, far[1] * 0.45),
                    arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=3), zorder=4)
        ax.text(far[0] * 0.55, far[1] * 0.55, "откуда пришёл", fontsize=11,
                color="#7c2d12", ha="center", va="center", zorder=6,
                bbox=dict(fc="#fef3c7", ec="#b45309", alpha=0.95))

    if os.environ.get("P083_DEBUG"):
        print(f"    [debug] узел {node} cands={cands} sides={sides} angs={angles} "
              f"chosen={row['chosen_edge']!r}")
    for i, eid in enumerate(cands):
        if eid not in g.edges:
            print(f"    [debug] {eid} нет в графе — пропускаю")
            continue
        taken = eid == row["chosen_edge"]
        if os.environ.get("P083_DEBUG"):
            print(f"    [debug]   i={i} {eid} taken={taken}")
        p, q = rot(*g.pos(g.edges[eid]["from"])), rot(*g.pos(g.edges[eid]["to"]))
        far = p if (p[0] ** 2 + p[1] ** 2) > (q[0] ** 2 + q[1] ** 2) else q
        # trim so the arrow leaves the junction cleanly
        L = math.hypot(*far) or 1.0
        tipx, tipy = far[0] * min(1.0, span * 0.42 / L), far[1] * min(1.0, span * 0.42 / L)
        col = "#dc2626" if taken else "#2563eb"
        # the whole passage, not only the arrow head, so there is no doubt which line is
        # which when the two lie close together
        reach = min(3.0, span * 0.40 / (math.hypot(*far) or 1.0))
        ax.plot([0, far[0] * reach], [0, far[1] * reach], color=col,
                lw=8.0 if taken else 5.0, alpha=0.9, zorder=4, solid_capstyle="round")
        ax.annotate("", xy=(tipx, tipy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=8.0 if taken else 5.0,
                                    shrinkA=14, shrinkB=0), zorder=5)
        side = sides[i] if i < len(sides) else "?"
        a = angles[i] if i < len(angles) else "?"
        side_ru = {"STRAIGHT": "ПРЯМО", "LEFT": "ВЛЕВО", "RIGHT": "ВПРАВО",
                   "BACK": "НАЗАД"}.get(side, side)
        to_node = g.other(eid, node)
        lab = (f"{side_ru} {a}°  → {to_node}"
               + ("\n← СИСТЕМА ПОШЛА СЮДА" if taken else "\n(не выбрано)"))
        ax.text(tipx * 1.12, tipy * 1.12, lab, fontsize=13 if taken else 12,
                color="white" if taken else "#1e3a8a", ha="center", va="center", zorder=7,
                weight="bold" if taken else "normal",
                bbox=dict(fc="#dc2626" if taken else "#dbeafe",
                          ec="#7f1d1d" if taken else "#2563eb", alpha=0.97, pad=3))

    ax.plot([0], [0], "o", ms=15, color="#16a34a", mec="k", mew=2, zorder=8)
    ax.text(0, span * 0.30, "ТЫ ЗДЕСЬ ↑", fontsize=11, color="#14532d", ha="center",
            zorder=9, bbox=dict(fc="#dcfce7", ec="#16a34a", alpha=0.95))
    fly = row.get("male_direction") or "молчала"
    fly_ru = {"LEFT": "ВЛЕВО", "RIGHT": "ВПРАВО"}.get(fly, fly)
    ax.set_title(f"t = {row['time']} с · узел {node}\n"
                 f"муха: {fly_ru} · система пошла: "
                 f"{'ВЫБРАННЫЙ КРАСНЫЙ' if row['chosen_edge'] else '—'}\n"
                 f"вид сверху развёрнут так, что движение направлено ВВЕРХ: "
                 f"лево на картинке = лево на видео",
                 fontsize=12)
    # trim to what is actually drawn, so the junction fills the picture instead of floating
    # in the middle of a mostly empty square
    pts = [(0.0, 0.0)]
    if arrival and arrival in g.edges:
        p, q = rot(*g.pos(g.edges[arrival]["from"])), rot(*g.pos(g.edges[arrival]["to"]))
        pts.append(p if math.hypot(*p) > math.hypot(*q) else q)
    for eid in cands:
        if eid in g.edges:
            p, q = rot(*g.pos(g.edges[eid]["from"])), rot(*g.pos(g.edges[eid]["to"]))
            pts.append(p if math.hypot(*p) > math.hypot(*q) else q)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    m = 0.35 * max(max(xs) - min(xs), max(ys) - min(ys), 120)
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(max(ys) + m, min(ys) - m)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(dest, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID00006")
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--seq", default=None, help="route to audit; defaults to output/p08")
    ap.add_argument("--no-clips", action="store_true", help="skip video cutting")
    args = ap.parse_args()

    out = OUT
    clips = out / "clips"
    pics = out / "junctions"
    clips.mkdir(parents=True, exist_ok=True)
    pics.mkdir(parents=True, exist_ok=True)

    g = Graph.load(args.graph)
    seq_dir = Path(args.seq) if args.seq else P08_OUT
    seq = read_csv(seq_dir / "edge_sequence.csv")
    dec = read_csv(seq_dir / "decisions.csv")
    if not seq or not dec:
        print("нет результатов трекера — сначала p08_graph_tracker.py")
        return

    plan = ROOT / "data/p08" / str(g.doc.get("image") or "plan.png")
    src = MEDIA / f"{args.video}_fixed.mp4"
    if not src.exists():
        src = MEDIA / f"{args.video}.mp4"

    # only the junctions where more than one way on existed
    cand = [d for d in dec if len([x for x in d["candidate_edges"].split("|") if x]) >= 2]

    print("P08.3 — проверка решений на развилках по видео")
    print(f"  клип {args.video}, маршрут {seq_dir}")
    print(f"  рёбер в маршруте {len(seq)}, решений {len(dec)}, с выбором {len(cand)}")
    print(f"  вырезаю клипы {PRE_S:.0f} с до и {POST_S:.0f} с после решения")
    print()

    rows = []
    for i, d in enumerate(cand, 1):
        t = float(d["time"])
        t0 = max(0.0, t - PRE_S)
        t1 = t + POST_S
        stem = f"{i:02d}_t{t:07.1f}_{d['node']}"
        clip = clips / f"{stem}.mp4"
        pic = pics / f"{stem}.png"
        made = False
        if not args.no_clips and src.exists():
            made = cut(src, t0, t1, clip)
        junction_picture(g, plan, d, pic, d.get("current_edge"))
        # How visible is the turn in this window? Measured with the frozen frontend, which
        # knows nothing of the fly, so it is an independent reading of the video. Reported
        # with each decision because it tells the person what to expect before they look:
        # where it is near zero there is no turn in the video to judge the choice against.
        rot_coh = rot_dx = None
        if src.exists():
            try:
                from fly_vo.content_motion import measure_window
                mw = measure_window(str(src), max(0.0, t - 3.0), t + 5.0, "p083")
                rot_coh, rot_dx = float(mw.coherence), float(mw.content_dx)
            except Exception:
                pass
        strip = pics / f"{stem}_strip.png"
        if not args.no_clips and src.exists():
            frame_strip(src, t, strip)

        n_cand = len([x for x in d["candidate_edges"].split("|") if x])
        row = {
            "n": i,
            "time": t,
            "clip_start": round(t0, 2),
            "clip_end": round(t1, 2),
            "node": d["node"],
            "arrival_edge": d.get("current_edge", ""),
            "candidates": n_cand,
            "candidate_edges": d["candidate_edges"],
            "candidate_sides": d["candidate_sides"],
            "candidate_angles": d["candidate_angles"],
            "male_direction": d.get("male_direction", ""),
            "yaw_integral": d.get("yaw_integral", ""),
            "yaw_floor": d.get("yaw_floor", ""),
            "chosen_edge": d["chosen_edge"],
            "reason": d["reason"],
            "geometry_scores": d.get("geometry_scores", ""),
            "novelty_scores": d.get("novelty_scores", ""),
            "visit_counts": d.get("visit_counts", ""),
            "final_scores": d.get("final_scores", ""),
            "clip": clip.name if made else "",
            "picture": pic.name,
            "strip": strip.name if strip.exists() else "",
            "rotation_coherence": "" if rot_coh is None else round(rot_coh, 3),
            "rotation_dx": "" if rot_dx is None else round(rot_dx, 3),
            "turn_visible": ("" if rot_coh is None else
                             ("явный поворот" if rot_coh >= 0.4 else
                              "слабый поворот" if rot_coh >= 0.2 else
                              "поворота в видео не видно")),
            "verdict": "",
            "correct_edge_if_wrong": "",
            "note": "",
        }
        rows.append(row)
        fly = d.get("male_direction") or "—"
        print(f"  {i:2d}. t={t:7.1f} с  узел {d['node']:>5s}  муха {fly:>5s}  "
              f"варианты {n_cand}  выбрано {d['chosen_edge']}"
              + (f"  клип {clip.name}" if made else ""))

    # the table, with the verdict columns left for a person to fill
    fields = list(rows[0].keys())
    csv_path = out / "decisions.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- figure: the decisions and what each was decided by ---------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax = axes[0]
    ys = np.arange(len(rows))
    with_fly = [bool(r["male_direction"]) for r in rows]
    ax.barh(ys, [1] * len(rows),
            color=["#2563eb" if w else "#cbd5e1" for w in with_fly])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['n']}. t={r['time']:.0f} {r['node']}" for r in rows],
                       fontsize=10)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_title("синим — решения, где муха назвала сторону\n"
                 "серым — решала геометрия, муха молчала", fontsize=12)
    for i, r in enumerate(rows):
        ax.text(0.5, i, r["chosen_edge"], ha="center", va="center", fontsize=10,
                color="white" if with_fly[i] else "black")

    ax = axes[1]
    ax.axis("off")
    tbl = [[r["n"], f"{r['time']:.0f}", r["node"], r["male_direction"] or "—",
            r["chosen_edge"], r["reason"]] for r in rows]
    t = ax.table(cellText=tbl, colLabels=["№", "t,с", "узел", "муха", "выбрано", "причина"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    ax.set_title("решения, ожидающие вердикта", fontsize=12)

    fig.suptitle(f"P08.3 — {len(rows)} развилок, где P08 делал выбор. "
                 f"Вердикты ставить по видео: /p083", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=105)
    plt.close(fig)

    n_fly = sum(1 for r in rows if r["male_direction"])
    rep = {
        "video": args.video,
        "route": str(seq_dir),
        "n_decisions": len(dec),
        "n_with_choice": len(rows),
        "n_decided_by_fly": n_fly,
        "n_decided_by_geometry": len(rows) - n_fly,
        "clip_window_s": [PRE_S, POST_S],
        "clips_made": sum(1 for r in rows if r["clip"]),
        "decisions": rows,
        "verdicts": {"CORRECT": 0, "WRONG": 0, "UNCLEAR": 0},
        "note": ("Вердикты пусты: их ставит человек по видео. Разделение важно — там, где муха "
                 "молчала, неверный выбор говорит о правиле разрешения ничьих, а не о мухе."),
    }
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    print()
    print(f"  решений, где муха называла сторону: {n_fly}")
    print(f"  решений, где решала геометрия:      {len(rows) - n_fly}")
    print(f"  клипов нарезано: {rep['clips_made']}")
    print()
    print(f"Wrote {out}/")
    print()
    print("  размечать вердикты: http://127.0.0.1:8796/p083")
    print("    клавиши 1 верно / 2 неверно / 3 непонятно, ← → между развилками, "
          "пробел пауза, Home сначала")


if __name__ == "__main__":
    main()
