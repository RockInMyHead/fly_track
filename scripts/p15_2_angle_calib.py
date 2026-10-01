#!/usr/bin/env python3
"""P15.2 — measure which azimuth pattern each passage angle actually produces.

One passage at a time, every 15° from −120° to +120°. Each angle stores the full
32-bin T4/T5 vector. Nothing assumes that +45° lights the +45° bin.

Synthetic forks are then shown:

    [−90°, 0°]
    [0°, +90°]
    [−90°, +90°]
    [−90°, 0°, +90°]

The readout has no fitted weights. It asks which set of measured single-angle
patterns, added together, matches the fork. The true set is ranked against every
other set of the same size.

No camera. FINAL V1 is not touched.

    PYTHONPATH=. .venv/bin/python scripts/p15_2_angle_calib.py
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

import p15_1_sectors as S  # noqa: E402
import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

OUT = ROOT / "output/p15"
ANGLES = list(range(-120, 121, 15))
FORKS = (
    (-90, 0),
    (0, 90),
    (-90, 90),
    (-90, 0, 90),
)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_sets(obs: np.ndarray, templates: dict[int, np.ndarray], k: int) -> list[dict]:
    """Every combination of k calibration angles, scored by correlation with their sum."""
    ranked = []
    for combo in itertools.combinations(sorted(templates), k):
        pred = np.sum([templates[a] for a in combo], axis=0)
        ranked.append({"angles": list(combo), "corr": round(corr(obs, pred), 4)})
    ranked.sort(key=lambda r: -r["corr"])
    return ranked


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    enc = VideoVisualEncoder(engine)
    a_bins, b_bins, centres = S.azimuth_bins(enc)
    w, h = cfg.encode_width, cfg.encode_height
    n_bins = len(centres)

    print("P15.2 — калибровка угла, без камеры")
    print(f"  углов {len(ANGLES)}, бинов {n_bins}")

    templates: dict[int, np.ndarray] = {}
    table = []
    for deg in ANGLES:
        vec = S.play(engine, enc, M.stable_sector_clip([deg], w, h), a_bins, b_bins)
        templates[deg] = vec
        peak = int(np.argmax(np.abs(vec)))
        assumed = S.bin_of_angle(deg, n_bins)
        table.append({
            "deg": deg,
            "peak_bin": peak,
            "peak_azimuth": round(float(centres[peak]), 3),
            "peak_value": round(float(vec[peak]), 4),
            "assumed_bin_sin": assumed,
            "bin_error": peak - assumed,
            "energy": round(float(np.mean(np.abs(vec))), 4),
            "vector": [round(float(x), 4) for x in vec],
        })
        print(f"  {deg:+4d}°  пик бин {peak:2d} (азимут {centres[peak]:+.2f})  "
              f"sin-бин {assumed:2d}  энергия {table[-1]['energy']:.2f}")

    forks = []
    n_first = 0
    for angles in FORKS:
        obs = S.play(engine, enc, M.stable_sector_clip(list(angles), w, h), a_bins, b_bins)
        ranked = rank_sets(obs, templates, len(angles))
        truth = sorted(angles)
        rank = next(i for i, r in enumerate(ranked, start=1) if r["angles"] == truth)
        truth_corr = next(r["corr"] for r in ranked if r["angles"] == truth)
        rival = next(r["corr"] for r in ranked if r["angles"] != truth)
        margin = round(truth_corr - rival, 4)
        separated = rank == 1 and margin > 0.05
        if separated:
            n_first += 1
        top = ranked[0]
        forks.append({
            "shown": list(angles),
            "rank_of_truth": rank,
            "n_candidates": len(ranked),
            "truth_corr": truth_corr,
            "margin_over_next_different": margin,
            "separated": separated,
            "best": top,
            "top3": ranked[:3],
        })
        print(f"  развилка {list(angles)}  место {rank}/{len(ranked)}  "
              f"отрыв {margin:+.3f}  {'отделен' if separated else 'слит с соседом'}")

    # How far the measured peak sits from the sine assumption. This is the calibration.
    errors = [abs(r["bin_error"]) for r in table if abs(r["deg"]) <= 90]
    recoverable = n_first == len(FORKS)
    coarse = all(f["rank_of_truth"] == 1 for f in forks) and not recoverable
    doc = {
        "phase": "P15.2",
        "final_v1_touched": False,
        "camera_used": False,
        "no_learned_weights": True,
        "readout": "корреляция наблюдаемого вектора с суммой измеренных одноугловых шаблонов",
        "angles_deg": ANGLES,
        "azimuth_centres": [round(float(x), 4) for x in centres],
        "calibration": [{k: v for k, v in row.items() if k != "vector"} | {"vector": row["vector"]}
                        for row in table],
        "mean_abs_bin_error_within_90": round(float(np.mean(errors)) if errors else 0.0, 2),
        "forks": forks,
        "truth_ranked_first": n_first,
        "n_forks": len(FORKS),
        "verdict": ("ANGLES_READABLE" if recoverable
                    else "COARSE_LEFT_RIGHT" if coarse
                    else "ANGLES_NOT_SEPARATED"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = OUT / "P15_2_calibration.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mat = np.stack([templates[d] for d in ANGLES], axis=0)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm",
                       vmin=-np.percentile(np.abs(mat), 98),
                       vmax=np.percentile(np.abs(mat), 98))
        ax.set_yticks(range(len(ANGLES)))
        ax.set_yticklabels([f"{d:+d}°" for d in ANGLES], fontsize=8)
        ax.set_xlabel("азимутальный бин, слева направо")
        ax.set_title("P15.2 — угол прохода → вектор T4/T5")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(OUT / "p15_2_calibration.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  рисунок не построен: {exc}")

    print()
    print(f"  вердикт: {doc['verdict']}  "
          f"({n_first}/{len(FORKS)} развилок отделены от соседнего угла)")
    print(f"  средний |сдвиг пика от sin(θ)| внутри ±90°: {doc['mean_abs_bin_error_within_90']} бина")
    print(f"  записано: {path}")
    return 0 if recoverable else 2


if __name__ == "__main__":
    raise SystemExit(main())
