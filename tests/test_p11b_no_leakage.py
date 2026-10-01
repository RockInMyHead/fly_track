"""P11.B — доказательство, что отложенный ролик не влияет на отбор.

The phase asked for one property to be guaranteed rather than argued:

    4 recordings -> choose types and features -> freeze -> the fifth is used only to score

The way to know that is true is not to read the code and agree with it, but to change the labels
of the held-out recording and check that the selection does not move. If any part of the screening
reached the fifth recording, some pair's outcome would change when its labels do, and the selection
would change with it.

Two tests:

    test_selection_is_blind_to_heldout   the same selection with the held-out labels intact and
                                         with them permuted beyond recognition
    test_selection_uses_all_training     the opposite failure: a screen that is blind to the
                                         held-out recording but also to some of the training ones
                                         would be useless. Both training recordings must be able
                                         to change the answer.

Run:  PYTHONPATH=. python tests/test_p11b_no_leakage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.p11b_analyze import FEATS, screen_pairs  # noqa: E402


def _dn_features() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Real features from the P09 descending-cell recording, all five recordings present.

    The candidate recording is still running, so the test runs on the only sixty-event dataset
    that has all five videos: the descending cells. What is under test is the screening function,
    which does not care where its columns came from.
    """
    from scripts.p10_levels import build as build_level, Store, to_matrix

    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    ev = frozen["labels"]
    rows, names = build_level("dn", ev, Store())
    X = to_matrix(rows)
    cat = np.array([e["category"] for e in ev])
    clips = np.array([e["clip"] for e in ev])
    y = np.where(cat == "NO_NET_DISPLACEMENT", 0, 1).astype(int)
    n_cells = np.ones(len(names), dtype=float)
    return X, [str(n) for n in names], y, clips, n_cells


def _selected(y, X, clips, uniq, nf, n_cells, train, min_auc=0.65, min_agree=0.6) -> set:
    pairs = screen_pairs(y, X, clips, uniq, nf, n_cells, FEATS, train, min_auc, min_agree)
    return {(p["group"], p["feature"], p["sign"]) for p in pairs}


def test_selection_is_blind_to_heldout() -> None:
    X, uniq, y, clips, n_cells = _dn_features()
    nf = len(FEATS)
    rng = np.random.default_rng(0)
    checked = 0
    for c in sorted(set(clips.tolist())):
        train = clips != c
        base = _selected(y, X, clips, uniq, nf, n_cells, train)
        # the held-out recording's labels are permuted into nonsense; the selection must not care
        y_alt = y.copy()
        m = clips == c
        y_alt[m] = rng.permutation(y_alt[m])
        # invert them as well, which no permutation-based null would ever produce
        alt2 = y.copy()
        alt2[m] = 1 - alt2[m]
        for variant, tag in ((y_alt, "перемешаны"), (alt2, "инвертированы")):
            got = _selected(variant, X, clips, uniq, nf, n_cells, train)
            assert got == base, (
                f"отбор изменился, когда метки отложенного ролика {c} были {tag}: "
                f"добавилось {got - base}, пропало {base - got}"
            )
        checked += 1
    print(f"  отложенный ролик не влияет на отбор: проверено {checked} проходов, "
          f"метки менялись двумя способами")
    print("  (инверсия меняет даже AUC=0.0 на 1.0 — ни один порог этого не пропустит, "
          "если прочитает ролик)")
    assert checked == 5, f"ожидалось 5 проходов, получено {checked}"


def _shuffle_not_identity(rng: np.random.Generator, labels: np.ndarray) -> np.ndarray:
    """A permutation that is guaranteed to differ from the input.

    The first version of the second test drew a permutation and used it as it came. On a recording
    with four events, `rng.permutation` returns the identity often enough that the test quietly
    became a no-op and reported that the training data did not matter. A test that cannot fail is
    worse than no test, so the permutation is rejected until it actually moves something.
    """
    if len(labels) < 2:
        return labels.copy()
    for _ in range(200):
        out = rng.permutation(labels)
        if not np.array_equal(out, labels):
            return out
    return np.roll(labels, 1)     # always differs when len >= 2 and the values are not constant


def test_selection_uses_all_training() -> None:
    """A screen that ignored the training recordings would pass the first test trivially."""
    X, uniq, y, clips, n_cells = _dn_features()
    nf = len(FEATS)
    rng = np.random.default_rng(1)
    moved = 0
    for c in sorted(set(clips.tolist())):
        train = clips != c
        base = _selected(y, X, clips, uniq, nf, n_cells, train)
        # the largest training recording, so the change is not lost among the small ones
        others = [d for d in sorted(set(clips.tolist())) if d != c]
        d = max(others, key=lambda x: int((clips == x).sum()))
        y_alt = y.copy()
        m = clips == d          # a TRAINING recording: its labels must be able to change things
        y_alt[m] = _shuffle_not_identity(rng, y_alt[m])
        got = _selected(y_alt, X, clips, uniq, nf, n_cells, train)
        if got != base:
            moved += 1
    print(f"  обучающие ролики действительно влияют: отбор сдвинулся в {moved} из 5 проходов")
    assert moved >= 4, (
        f"перемешивание обучающего ролика сдвинуло отбор лишь в {moved} проходах — "
        "похоже, отбор не читает обучающие данные"
    )


if __name__ == "__main__":
    print("P11.B — тест отсутствия утечки отложенного ролика")
    test_selection_is_blind_to_heldout()
    test_selection_uses_all_training()
    print("все проверки пройдены")
