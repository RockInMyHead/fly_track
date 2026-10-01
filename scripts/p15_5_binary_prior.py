#!/usr/bin/env python3
"""P15.5 — map prior for binary forks only, then camera + map.

The corridor code from P15.4 separates one passage and a pair. It does not
separate the triple. So:

    two non-back passages, and the reading margin is at least 0.05  →  +β or −β
    three passages, or the margin is below 0.05                      →  0

β does not choose an edge. It only says whether this hypothesis's pair of
passages agrees with the picture. FINAL V1 is not modified.

Angles outside the bands already read in P15.3 stay unread:

    ≤ −60° LEFT,  |θ| ≤ 15° STRAIGHT,  ≥ +60° RIGHT,  otherwise no prior.

Camera mix uses the alphas fixed in P15: 0.1 and 0.25. Neither is chosen here.

    PYTHONPATH=. .venv/bin/python scripts/p15_5_binary_prior.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_3_motion2d as P  # noqa: E402
import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402
from p08_graph import Graph  # noqa: E402

OUT = ROOT / "output/p15"
VIDEO = ROOT / "webapp/media/VID00010_fixed.mp4"
# Same weak scale as the geometry term already in the tracker. Not fitted.
BETA = 0.05
MARGIN = 0.05
ALPHAS = (0.1, 0.25)
PER_PAIR = 4
NAMES = ("LEFT", "STRAIGHT", "RIGHT")
HEADING = {"LEFT": -90.0, "STRAIGHT": 0.0, "RIGHT": 90.0}


def passage_class(deg: float) -> str | None:
    """Bands P15.3 could already tell apart. The gaps stay unread."""
    if deg <= -60.0:
        return "LEFT"
    if deg >= 60.0:
        return "RIGHT"
    if abs(deg) <= 15.0:
        return "STRAIGHT"
    return None


def canon(classes) -> list[str]:
    have = set(classes)
    return [name for name in NAMES if name in have]


def map_prior(n_forward: int, decoded: list[str] | None, margin: float,
              hypothesis: list[str], beta: float = BETA) -> float:
    """+β agree, −β contradict, 0 when the fork is not a readable binary pair.

    A triple fork is ignored even if the margin is large. A margin below 0.05
    is ignored even on a binary fork. β never picks one of the two edges.
    """
    if n_forward != 2:
        return 0.0
    if decoded is None or margin < MARGIN:
        return 0.0
    if len(decoded) != 2 or len(hypothesis) != 2:
        return 0.0
    if len(set(hypothesis)) != 2:
        return 0.0
    if canon(decoded) == canon(hypothesis):
        return float(beta)
    return float(-beta)


def eligible(g: Graph) -> list[dict]:
    rows = []
    for node in sorted(g.nodes):
        for in_edge in sorted(g.edges_at(node)):
            exits = M.forward_exits(g, node, in_edge)
            if len(exits) != 2:
                continue
            classes = [passage_class(float(e["deg"])) for e in exits]
            if any(c is None for c in classes) or classes[0] == classes[1]:
                continue
            rows.append({
                "node": node,
                "in_edge": in_edge,
                "degrees": [round(float(e["deg"]), 2) for e in exits],
                "classes": canon(classes),
            })
    return rows


def sample_forks(rows: list[dict], per: int = PER_PAIR) -> list[dict]:
    buckets = {("LEFT", "STRAIGHT"): [], ("STRAIGHT", "RIGHT"): [], ("LEFT", "RIGHT"): []}
    for row in rows:
        key = tuple(row["classes"])
        if key in buckets and len(buckets[key]) < per:
            buckets[key].append(row)
    out = []
    for key in buckets:
        out.extend(buckets[key])
    return out


def wrong_fork(i: int, forks: list[dict]) -> dict:
    mine = tuple(forks[i]["classes"])
    for step in range(1, len(forks)):
        other = forks[(i + step) % len(forks)]
        if tuple(other["classes"]) != mine:
            return other
    raise RuntimeError("no fork with a different pair")


def grab_clip(video: Path, t_s: float, n: int = 8) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"не открывается видео {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"камера не дала кадров на t={t_s}")
    return frames


def play_mix(engine, cam_enc, map_enc, groups, cam_frames, map_frames, alpha: float):
    """Camera drive plus alpha times the map drive. Both encoders use the 2D path."""
    engine.reset(64)
    cam_enc.reset()
    map_enc.reset()
    n_bins = len(groups["a"])
    h = np.zeros(n_bins, dtype=float)
    v = np.zeros(n_bins, dtype=float)
    n = max(len(cam_frames), len(map_frames))
    for i in range(n):
        eye_c, inj_c, _ = cam_enc.encode_frame(cam_frames[i % len(cam_frames)], vertical=True)
        eye_m, inj_m, _ = map_enc.encode_frame(map_frames[min(i, len(map_frames) - 1)], vertical=True)
        eye, inj = M.mix_drives((eye_c, inj_c), (eye_m, inj_m), alpha)
        fired = set(np.asarray(
            engine.step((i + 1) * engine.config.brain_dt, eye_drive=eye, inject=inj).fired
        ).tolist())
        for b in range(n_bins):
            h[b] += len(groups["a"][b] & fired) - len(groups["b"][b] & fired)
            v[b] += len(groups["c"][b] & fired) - len(groups["d"][b] & fired)
    h /= n
    v /= n
    return h, v


def decode(obs: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[list[str], float]:
    ranked = P.rank_subsets(obs, templates)
    best = ranked[0]
    rival = next(r for r in ranked if r["classes"] != best["classes"])
    return list(best["classes"]), round(best["corr"] - rival["corr"], 4)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not VIDEO.exists():
        print(f"нет камеры: {VIDEO}")
        return 1
    g = Graph.load(ROOT / "data/p08/graph.json")
    forks = sample_forks(eligible(g))
    if len(forks) < 6:
        print(f"мало двоичных развилок в читаемых полосах: {len(forks)}")
        return 1

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    enc = VideoVisualEncoder(engine)
    cam_enc = VideoVisualEncoder(engine)
    map_enc = VideoVisualEncoder(engine)
    groups = {
        "a": P.family_bins(enc.motion_injector, "a"),
        "b": P.family_bins(enc.motion_injector, "b"),
        "c": P.family_bins(enc.vertical_injector, "c"),
        "d": P.family_bins(enc.vertical_injector, "d"),
    }
    w, h = cfg.encode_width, cfg.encode_height
    print("P15.5 — бинарный приор, камера только на двоичных узлах")
    print(f"  развилок {len(forks)}, β {BETA}, порог {MARGIN}, α {list(ALPHAS)}")
    print("  FINAL V1 не менялся")

    templates = {}
    for name in NAMES:
        hv = P.play(engine, enc, M.corridor_clip([HEADING[name]], w, h), groups)
        templates[name] = P.pack(*hv)

    rows = []
    n_pass = 0
    n_map_only = 0
    for i, fork in enumerate(forks):
        truth = list(fork["classes"])
        template = np.sum([templates[name] for name in truth], axis=0)
        other = wrong_fork(i, forks)
        pictures = {
            "correct": M.corridor_clip(fork["degrees"], w, h),
            "mirror": M.corridor_clip([-d for d in fork["degrees"]], w, h),
            "wrong": M.corridor_clip(other["degrees"], w, h),
        }
        cam = grab_clip(VIDEO, 12.0 + 7.0 * i)
        obs_map = P.pack(*P.play(engine, enc, pictures["correct"], groups))
        dec_map, margin_map = decode(obs_map, templates)
        map_only_ok = canon(dec_map) == truth and margin_map >= MARGIN
        n_map_only += int(map_only_ok)

        by_alpha = []
        fork_ok = True
        for alpha in ALPHAS:
            scores = {}
            priors = {}
            decoded = {}
            margins = {}
            for kind, frames in pictures.items():
                obs = P.pack(*play_mix(engine, cam_enc, map_enc, groups, cam, frames, alpha))
                dec, margin = decode(obs, templates)
                decoded[kind] = dec
                margins[kind] = margin
                scores[kind] = round(P.corr(obs, template), 4)
                priors[kind] = map_prior(2, dec, margin, truth)
            holds = margins["correct"] >= MARGIN and canon(decoded["correct"]) == truth
            better_mirror = priors["correct"] > priors["mirror"]
            better_wrong = priors["correct"] > priors["wrong"]
            ok = holds and better_mirror and better_wrong
            fork_ok = fork_ok and ok
            by_alpha.append({
                "alpha": alpha,
                "decoded": decoded,
                "margins": margins,
                "corr_with_truth": scores,
                "prior": priors,
                "margin_holds": holds,
                "correct_gt_mirror": better_mirror,
                "correct_gt_wrong": better_wrong,
                "passed": ok,
            })
        n_pass += int(fork_ok)
        rows.append({
            "node": fork["node"],
            "in_edge": fork["in_edge"],
            "degrees": fork["degrees"],
            "classes": truth,
            "wrong_node": other["node"],
            "wrong_classes": other["classes"],
            "map_only_decoded": dec_map,
            "map_only_margin": margin_map,
            "map_only_ok": map_only_ok,
            "alphas": by_alpha,
            "passed_both_alphas": fork_ok,
        })
        flag = "ok" if fork_ok else "мимо"
        print(f"  {fork['node']:6s} {truth}  карта {dec_map} отрыв {margin_map:+.3f}  "
              f"камера {flag}")

    both = n_pass == len(rows)
    washed = sum(1 for r in rows if r["map_only_ok"] and not r["passed_both_alphas"])
    if both:
        verdict = "BINARY_PRIOR_HOLDS_WITH_CAMERA"
    elif washed:
        verdict = "CAMERA_WASHES_BINARY_PRIOR"
    elif n_map_only == 0:
        verdict = "REAL_FORK_NOT_IN_CODEBOOK"
    else:
        verdict = "BINARY_PRIOR_NOT_SEPARATED"
    doc = {
        "phase": "P15.5",
        "final_v1_touched": False,
        "map_prior_enabled_in_tracker": False,
        "camera_used": True,
        "video": str(VIDEO.relative_to(ROOT)),
        "beta": BETA,
        "beta_role": "слабый бонус гипотезе, чей набор проходов совпал; ребро не выбирает",
        "margin": MARGIN,
        "alphas_fixed": list(ALPHAS),
        "bands": "LEFT ≤ −60°, STRAIGHT |θ| ≤ 15°, RIGHT ≥ +60°. Промежуток без приора.",
        "triples_ignored": True,
        "n_forks": len(rows),
        "map_only_readable": n_map_only,
        "readable_then_washed_by_camera": washed,
        "passed_both_alphas": n_pass,
        "forks": rows,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = OUT / "P15_5_binary_prior.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        labels, vals, colors = [], [], []
        for alpha in ALPHAS:
            for key, title, color in (
                ("margin_holds", "отрыв", "#888"),
                ("correct_gt_mirror", "> зеркало", "#2a7"),
                ("correct_gt_wrong", "> чужая", "#246"),
            ):
                labels.append(f"{title}\nα={alpha}")
                vals.append(sum(1 for r in rows for a in r["alphas"]
                                if a["alpha"] == alpha and a[key]) / len(rows))
                colors.append(color)
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("доля развилок")
        ax.set_title(verdict)
        fig.tight_layout()
        fig.savefig(OUT / "p15_5_binary_prior.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  рисунок не построен: {exc}")

    print()
    print(f"  без камеры читаются {n_map_only}/{len(rows)}")
    print(f"  с камерой на обоих α: {n_pass}/{len(rows)}")
    print(f"  вердикт: {verdict}")
    print(f"  записано: {path}")
    return 0 if both else 2


if __name__ == "__main__":
    raise SystemExit(main())
