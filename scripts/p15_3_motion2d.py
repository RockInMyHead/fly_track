#!/usr/bin/env python3
"""P15.3 — read the map as a 2D motion, not as a left/right number.

Horizontal stays T4a/T5a against T4b/T5b. Vertical is T4c/T5c against T4d/T5d.
Each of the 32 azimuth bins keeps both. One passage is shown from −120° to
+120° in steps of 15°. The readout is the measured pair (dx, dy), not an
assumption that +45° lights the +45° bin.

The success bar is three classes, not 15°:

    −90° → LEFT
      0° → STRAIGHT
    +90° → RIGHT

Synthetic forks are then decoded from those three measured templates only.
No fitted weights. No camera. FINAL V1 is not touched.

    PYTHONPATH=. .venv/bin/python scripts/p15_3_motion2d.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

OUT = ROOT / "output/p15"
ANGLES = list(range(-120, 121, 15))
CANON = {"LEFT": -90, "STRAIGHT": 0, "RIGHT": 90}
FORKS = (
    (("LEFT", "STRAIGHT"), (-90, 0)),
    (("STRAIGHT", "RIGHT"), (0, 90)),
    (("LEFT", "RIGHT"), (-90, 90)),
    (("LEFT", "STRAIGHT", "RIGHT"), (-90, 0, 90)),
)
# ±30° and ±45° sit between the three classes. They are measured and not required.
MARGIN = 0.05


def family_bins(injector, family: str) -> list[set[int]]:
    n = injector.n_az_bins
    acc: list[list[int]] = [[] for _ in range(n)]
    for _key, (fam, _pref, bins) in injector.groups.items():
        if fam != family:
            continue
        for i, cells in enumerate(bins):
            acc[i].extend(int(c) for c in cells)
    return [set(cells) for cells in acc]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def play(engine, encoder, frames, groups) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin (a−b) and (c−d), averaged over frames that actually moved."""
    encoder.reset()
    engine.reset(64)
    n = len(groups["a"])
    h = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    used = 0
    for fr in frames:
        eye, inj, met = encoder.encode_frame(fr, vertical=True)
        if float(met.get("frame_diff_mean") or 0.0) <= 0.0 and used == 0:
            continue
        used += 1
        fired = set(np.asarray(
            engine.step(used * engine.config.brain_dt, eye_drive=eye, inject=inj).fired
        ).tolist())
        for i in range(n):
            h[i] += len(groups["a"][i] & fired) - len(groups["b"][i] & fired)
            v[i] += len(groups["c"][i] & fired) - len(groups["d"][i] & fired)
    if used:
        h /= used
        v /= used
    return h, v


def pack(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([h, v])


def expected_class(deg: int) -> str | None:
    if deg <= -60:
        return "LEFT"
    if deg >= 60:
        return "RIGHT"
    if abs(deg) <= 15:
        return "STRAIGHT"
    return None


def rank_subsets(obs: np.ndarray, templates: dict[str, np.ndarray]) -> list[dict]:
    ranked = []
    names = tuple(templates)
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            pred = np.sum([templates[name] for name in combo], axis=0)
            ranked.append({"classes": list(combo), "corr": round(corr(obs, pred), 4)})
    ranked.sort(key=lambda r: -r["corr"])
    return ranked


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    enc = VideoVisualEncoder(engine)
    groups = {
        "a": family_bins(enc.motion_injector, "a"),
        "b": family_bins(enc.motion_injector, "b"),
        "c": family_bins(enc.vertical_injector, "c"),
        "d": family_bins(enc.vertical_injector, "d"),
    }
    w, h = cfg.encode_width, cfg.encode_height
    n_c = sum(len(s) for s in groups["c"])
    n_d = sum(len(s) for s in groups["d"])
    print("P15.3 — 2D-код карты, без камеры")
    print(f"  углов {len(ANGLES)}, T4c/T5c {n_c}, T4d/T5d {n_d}")

    stored: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for deg in ANGLES:
        hv = play(engine, enc, M.stable_sector_clip([deg], w, h, grating=False), groups)
        stored[deg] = hv
        dx, dy = float(hv[0].sum()), float(hv[1].sum())
        print(f"  {deg:+4d}°  dx {dx:+8.2f}  dy {dy:+8.2f}  "
              f"|H| {np.mean(np.abs(hv[0])):.2f}  |V| {np.mean(np.abs(hv[1])):.2f}")

    templates = {name: pack(*stored[deg]) for name, deg in CANON.items()}
    rows = []
    n_required = 0
    n_hit = 0
    for deg in ANGLES:
        obs = pack(*stored[deg])
        scores = {name: round(corr(obs, templates[name]), 4) for name in CANON}
        if float(obs.std()) < 1e-8 or max(scores.values()) < 0.3:
            called = "SILENT"
        else:
            called = max(scores, key=scores.get)
        want = expected_class(deg)
        hit = want is not None and called == want
        if want is not None:
            n_required += 1
            n_hit += int(hit)
        dx, dy = float(stored[deg][0].sum()), float(stored[deg][1].sum())
        rows.append({
            "deg": deg,
            "dx_brain": round(dx, 4),
            "dy_brain": round(dy, 4),
            "horizontal_energy": round(float(np.mean(np.abs(stored[deg][0]))), 4),
            "vertical_energy": round(float(np.mean(np.abs(stored[deg][1]))), 4),
            "class_called": called,
            "class_expected": want,
            "class_hit": hit if want is not None else None,
            "corr": scores,
            "horizontal": [round(float(x), 4) for x in stored[deg][0]],
            "vertical": [round(float(x), 4) for x in stored[deg][1]],
        })
        mark = "" if want is None else ("ok" if hit else "мимо")
        if want is not None:
            print(f"  класс {deg:+4d}° → {called:8s}  ждали {want:8s}  {mark}")

    dx90 = float(stored[90][0].sum())
    dxm = float(stored[-90][0].sum())
    dy0 = float(stored[0][1].sum())
    dx0 = float(stored[0][0].sum())
    dym = float(stored[-90][1].sum())
    dy90 = float(stored[90][1].sum())
    geometry = {
        "left_horizontal": abs(dxm) > abs(dym),
        "right_horizontal": abs(dx90) > abs(dy90),
        "left_opposes_right": bool(dxm * dx90 < 0.0),
        "straight_vertical": abs(dy0) > abs(dx0) and abs(dy0) > abs(dym) and abs(dy0) > abs(dy90),
    }
    geometry["three_directions"] = all(geometry.values())

    forks = []
    n_sep = 0
    for classes, degrees in FORKS:
        obs = pack(*play(engine, enc, M.stable_sector_clip(list(degrees), w, h, grating=False), groups))
        ranked = rank_subsets(obs, templates)
        truth = list(classes)
        rank = next(i for i, r in enumerate(ranked, start=1) if r["classes"] == truth)
        truth_corr = next(r["corr"] for r in ranked if r["classes"] == truth)
        rival = next(r["corr"] for r in ranked if r["classes"] != truth)
        margin = round(truth_corr - rival, 4)
        separated = rank == 1 and margin > MARGIN
        n_sep += int(separated)
        forks.append({
            "shown_classes": list(classes),
            "shown_deg": list(degrees),
            "rank_of_truth": rank,
            "n_candidates": len(ranked),
            "truth_corr": truth_corr,
            "margin_over_next_different": margin,
            "separated": separated,
            "top3": ranked[:3],
        })
        print(f"  развилка {list(classes)}  место {rank}/{len(ranked)}  "
              f"отрыв {margin:+.3f}  {'отделен' if separated else 'слит'}")

    # ±15° and ±30° are still "ahead", but the marker sits inside one eye.
    # Exactly 0° can stay quiet: that column is the gap between the eyes.
    off = [d for d in (-30, -15, 15, 30)
           if abs(float(stored[d][1].sum())) > abs(float(stored[d][0].sum()))]
    geometry["forward_off_axis"] = len(off) == 4
    geometry["forward_off_axis_angles"] = off
    cluster_ok = n_hit == n_required
    forks_ok = n_sep == len(FORKS)
    if geometry["three_directions"] and cluster_ok and forks_ok:
        verdict = "CLASSES_READABLE"
    elif geometry["forward_off_axis"] and not geometry["straight_vertical"]:
        verdict = "FORWARD_OFF_AXIS_ONLY"
    elif not geometry["forward_off_axis"]:
        verdict = "VERTICAL_STILL_SILENT"
    else:
        verdict = "CLASSES_NOT_SEPARATED"

    doc = {
        "phase": "P15.3",
        "final_v1_touched": False,
        "camera_used": False,
        "no_learned_weights": True,
        "horizontal": "T4a/T5a минус T4b/T5b в каждом азимутальном бине",
        "vertical": "T4c/T5c минус T4d/T5d в каждом азимутальном бине",
        "vertical_field_sign": "плюс = картинка едет вниз, минус = картинка едет вверх",
        "texture": "случайное пятно без решётки периода 2: иначе шаг 2 px не меняет картинку по вертикали",
        "dx_dy": "сумма 32 бинов, горизонталь и вертикаль",
        "class_bar": "LEFT / STRAIGHT / RIGHT. ±30° и ±45° не требуются. −75° против −90° не требуется.",
        "angles_deg": ANGLES,
        "calibration": rows,
        "geometry": geometry,
        "cluster_hits": n_hit,
        "cluster_required": n_required,
        "forks": forks,
        "forks_separated": n_sep,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = OUT / "P15_3_motion2d.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        xs = [r["dx_brain"] for r in rows]
        ys = [r["dy_brain"] for r in rows]
        axes[0].axhline(0, color="0.8", lw=0.6)
        axes[0].axvline(0, color="0.8", lw=0.6)
        axes[0].scatter(xs, ys, c=ANGLES, cmap="coolwarm", s=36)
        for r in rows:
            if r["deg"] in (-90, 0, 90):
                axes[0].annotate(f"{r['deg']:+d}°", (r["dx_brain"], r["dy_brain"]),
                                 fontsize=8, textcoords="offset points", xytext=(4, 4))
        axes[0].set_xlabel("dx  T4a/T5a − T4b/T5b")
        axes[0].set_ylabel("dy  T4c/T5c − T4d/T5d")
        axes[0].set_title("угол прохода → (dx, dy)")
        mat = np.stack([pack(*stored[d]) for d in ANGLES], axis=0)
        lim = np.percentile(np.abs(mat), 98) if np.any(mat) else 1.0
        im = axes[1].imshow(mat, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[1].axvline(len(stored[0][0]) - 0.5, color="k", lw=0.6)
        axes[1].set_yticks(range(len(ANGLES)))
        axes[1].set_yticklabels([f"{d:+d}°" for d in ANGLES], fontsize=7)
        axes[1].set_xlabel("32 бина H, затем 32 бина V")
        axes[1].set_title("P15.3")
        fig.colorbar(im, ax=axes[1], fraction=0.04)
        fig.tight_layout()
        fig.savefig(OUT / "p15_3_motion2d.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  рисунок не построен: {exc}")

    print()
    print(f"  геометрия −90/0/+90: {geometry}")
    print(f"  классы {n_hit}/{n_required}, развилки {n_sep}/{len(FORKS)}")
    print(f"  вердикт: {verdict}")
    print(f"  записано: {path}")
    return 0 if verdict == "CLASSES_READABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
