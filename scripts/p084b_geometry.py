#!/usr/bin/env python3
"""P08.4B — choose a passage by its real angle, not by which side of 45 degrees it falls.

The rule this replaces read the graph's own label and decided on it:

    |delta| < 45  ->  STRAIGHT, and every such passage was the same passage

so 3 degrees, 12 degrees and 34 degrees were one thing, and once a turn had been labelled
STRAIGHT the number behind the label was gone. At the old J37 that meant two ways on at 2.9
and 11.7 degrees were both "carrying straight on", and the difference between them — the only
thing that could have told them apart — had already been thrown away.

Here the number is the decision and the label is only a caption. The captions come back at
the end, for the interface and the reason column, with narrower and differently-purposed
edges:

    |d| <= 15        STRAIGHT     a caption meaning "close to carrying on"
    15 < |d| < 135   LEFT/RIGHT   a caption meaning "a turn, to that side"
    |d| >= 135       BACK         a caption meaning "close to a reversal"

None of those three numbers is consulted while choosing. The number that is consulted is `d`
itself, and the passage behind is kept off the table by the caller rather than by a threshold
inside the score.

How the fly is used
-------------------
The yaw integral says the camera turned, and which way, and how hard. It does not say by how
much: a neural readout is not a protractor and nothing here pretends otherwise. So the signal
is used as a *direction with a confidence*, never as a target angle:

    alignment(d) = sin(signed angle, in the direction the signal named)

Zero at straight on, 0.14 at 8 degrees, 0.64 at 40 degrees, 1.00 at 90 degrees, and back to
zero at a full reversal. On the other side of straight it goes *negative*, so a passage 20
degrees the wrong way is argued against rather than merely left un-promoted — which is what
"MaleCNS said right" ought to mean. It peaks at a right angle on purpose: a signal that names
a direction is evidence for turning that way, and a reversal is a different thing that the
signal cannot distinguish, so it earns no promotion at all.

Below the floor the ramp is exactly zero and the signal contributes nothing, leaving geometry
to decide — rule 5, and the reason this phase exists.

Rebuild if it is wrong, do not tune
-----------------------------------
No number here was chosen by looking at how the route came out. `geometry_score` is the same
cosine the old rule already used and is strictly decreasing in |angle|, so "smallest change of
heading" and "largest score" are the same statement. `back_penalty` tapers continuously to
the same 0.08 floor the old rule used, for the same reason, minus the step. The alignment is
a projection. What changed is which of these the decision reads, not how they are shaped.

Ambiguity
---------
When the two best ways on differ by less than AMBIGUOUS_DEG, the fly is not asked to choose
between them even when it is loud. Two passages 2.9 and 11.7 degrees apart are one corridor
drawn twice as far as a fly's readout is concerned, and letting a confident LEFT nominate one
of them would invent a distinction the data does not contain. The decision then falls to
geometry and memory and is labelled GEOMETRY_AMBIGUOUS, which is a statement about what is
knowable rather than a failure.
"""

from __future__ import annotations

import math

# Publishing edges for the captions. Deliberately NOT used to choose anything.
STRAIGHT_DISPLAY_DEG = 15.0     # a passage this close to straight is captioned STRAIGHT
BACK_DISPLAY_DEG = 135.0        # a passage this far off is captioned BACK

# Where the smooth reversal suppressor starts and ends. It exists because the cosine alone
# still gives 0.15 at 135 degrees, not far enough below a real turn to be safe; the taper
# takes a reversal from 1.0 down to `back_penalty_min` without ever introducing a step.
BACK_TAPER_FROM_DEG = 90.0
BACK_TAPER_TO_DEG = 180.0

AMBIGUOUS_DEG = 15.0            # two ways on closer than this cannot be told apart


# ------------------------------------------------------------------ pure geometry
def geometry_score(deg: float) -> float:
    """How naturally a passage continues the way we are already going, from the angle alone.

    `0.5 * (1 + cos(deg))`: one at straight on, a half at a right angle, zero at a reversal.
    Strictly decreasing in |deg|, so "the smallest change of heading" and "the largest
    geometry score" are the same rule stated either way.
    """
    return float(0.5 * (1.0 + math.cos(math.radians(deg))))


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def back_penalty(deg: float, min_factor: float = 0.08) -> float:
    """A continuous suppression of near-reversals, one everywhere up to 90 degrees.

    The old rule multiplied by a flat 0.08 once |deg| passed 135, so a passage at 134
    degrees and one at 136 were different kinds of thing. This tapers instead, keeping the
    suppression strong where it matters and removing the step.
    """
    ad = abs(deg)
    if ad <= BACK_TAPER_FROM_DEG:
        return 1.0
    t = (ad - BACK_TAPER_FROM_DEG) / (BACK_TAPER_TO_DEG - BACK_TAPER_FROM_DEG)
    return float(1.0 - (1.0 - min_factor) * _smoothstep(t))


def display_side(deg: float) -> str:
    """The caption. For the interface and the reason column; never for the choice."""
    ad = abs(deg)
    if ad <= STRAIGHT_DISPLAY_DEG:
        return "STRAIGHT"
    if ad >= BACK_DISPLAY_DEG:
        return "BACK"
    return "RIGHT" if deg > 0 else "LEFT"


def seems_straight(deg: float) -> bool:
    """A caption-level convenience, kept separate so it cannot leak into scoring."""
    return abs(deg) <= STRAIGHT_DISPLAY_DEG


def is_reversal(deg: float) -> bool:
    """Structurally a reversal, not a scored category.

    Used by the caller to keep the passage behind off the table, and by the caption. Kept
    here beside `back_penalty` so the two agree on what a reversal is.
    """
    return abs(deg) >= BACK_DISPLAY_DEG


# ------------------------------------------------------------------ the fly's signal
def signal_ramp(integral: float, floor: float, back_factor: float = 3.0) -> float:
    """How far the integral stands above the floor, from 0 at the floor to 1 at 3x it.

    Below the floor this is exactly zero, which is the whole of rule 5: a reading that has
    not cleared the noise does not name a direction, so it cannot promote a passage.
    """
    if floor <= 0:
        return 1.0 if integral else 0.0
    span = max(back_factor * floor - floor, 1e-9)
    return float(min(max((abs(integral) - floor) / span, 0.0), 1.0))


def signal_direction(integral: float, floor: float, yaw_sign_left: int = 1) -> str:
    """LEFT, RIGHT, or empty when the reading is below the floor."""
    if abs(integral) < floor:
        return ""
    return "LEFT" if integral * yaw_sign_left > 0 else "RIGHT"


def alignment(deg: float, direction: str) -> float:
    """How well a passage at `deg` answers a signal that named `direction`.

    `sin` of the angle measured *into* the direction named: zero at straight on, one at a
    right angle, zero again at a reversal, negative on the other side. The graph's sign
    convention is `deg > 0` is RIGHT, so a LEFT signal is answered by negative angles.

    Not clamped at a right angle on purpose. Clamping would give a passage 112 degrees off
    the same maximum as one at 90, and a signal naming a direction is evidence for turning
    that way, not for turning all the way round.
    """
    if not direction:
        return 0.0
    signed = deg if direction == "RIGHT" else -deg
    return float(math.sin(math.radians(signed)))


# ------------------------------------------------------------------ memory
def novelty(edge: str, visits: dict, last_visit: dict, tnow: float,
            decay: float = 0.7, recent_penalty: float = 0.12,
            return_s: float = 60.0) -> float:
    """How attractive a passage is, given what has already been walked.

    Unchanged from P08.2 on purpose. It multiplies the geometry rather than competing with
    it, so novelty can separate two passages geometry cannot, and can never lift a passage
    geometry has already rejected.
    """
    n = visits.get(edge, 0)
    if n == 0:
        return 1.0
    by_count = 1.0 / (1.0 + decay * n)
    if tnow - last_visit.get(edge, -1e9) <= return_s:
        return float(by_count * recent_penalty)
    return float(by_count)


# ------------------------------------------------------------------ scoring
def score_candidates(cands: list[dict], integral: float, floor: float,
                     visits: dict | None = None, last_visit: dict | None = None,
                     tnow: float = 0.0, male_bonus: float = 1.5,
                     back_factor: float = 3.0, back_penalty_min: float = 0.08,
                     decay: float = 0.7, recent_penalty: float = 0.12,
                     return_s: float = 60.0, ambiguous_deg: float = AMBIGUOUS_DEG,
                     defer_margin: float = 0.25, yaw_sign_left: int = 1) -> dict:
    """Score every way on and pick one.

    `cands` carry at least `edge` and `deg`, the deflection from the current heading,
    positive to the right, as the graph defines it. Every term is returned for every
    candidate, so a decision can be read back rather than inferred.

    Two different kinds of uncertainty are reported, and they are not the same thing:

      `ambiguous`   two ways on lie within `ambiguous_deg` of each other, so the *map* does
                    not distinguish them. No signal can be trusted to choose here, because a
                    readout that names a side cannot be finer than the drawing it is choosing
                    from.
      `deferred`    the map does distinguish them but the scores land within `defer_margin`
                    of each other — the classic case of two passages set symmetrically about
                    straight on, where geometry is indifferent and no direction was named.
                    Nothing is wrong with the data; the choice is simply too close to make
                    once and never revisit.

    Both are cases for watching what happens next rather than deciding on the spot, which is
    what the caller's revision window is for.
    """
    visits = visits or {}
    last_visit = last_visit or {}
    ramp = signal_ramp(integral, floor, back_factor)
    direction = signal_direction(integral, floor, yaw_sign_left) if ramp > 0 else ""

    scored = []
    for c in cands:
        deg = float(c["deg"])
        geo = geometry_score(deg)
        pen = back_penalty(deg, back_penalty_min)
        nov = novelty(c["edge"], visits, last_visit, tnow, decay, recent_penalty, return_s)
        align = alignment(deg, direction)
        male = male_bonus * ramp * align
        scored.append({**c, "display": display_side(deg), "geometry_score": geo,
                       "back_penalty": pen, "novelty": nov, "alignment": align,
                       "male_score": male, "visits": int(visits.get(c["edge"], 0)),
                       "final_score": geo * pen * nov + male})

    # Ambiguity is decided on geometry alone, before the signal is let in: the question is
    # whether the map distinguishes these ways on, and a loud signal does not change the
    # map. The two straightest ways on that are actually offered are the pair that matters.
    forward = sorted([s for s in scored if not is_reversal(s["deg"])],
                     key=lambda s: -geometry_score(s["deg"]))
    ambiguous = False
    pair: tuple[str, str] | None = None
    if len(forward) >= 2:
        gap = abs(forward[0]["deg"] - forward[1]["deg"])
        if gap < ambiguous_deg:
            ambiguous = True
            pair = (forward[0]["edge"], forward[1]["edge"])
            # the signal cannot separate two passages this close, so it is not allowed to
            # nominate one of them; the choice falls back to geometry and memory
            for s in scored:
                s["male_score"] = 0.0
                s["final_score"] = s["geometry_score"] * s["back_penalty"] * s["novelty"]

    total = sum(max(s["final_score"], 0.0) for s in scored) or 1.0
    probs = {s["edge"]: max(s["final_score"], 0.0) / total for s in scored}
    ranked = sorted(scored, key=lambda s: (-s["final_score"], s["edge"]))
    chosen = ranked[0]

    # A tie in the scores is a different thing from a tie in the angles, and it gets a
    # different name: here geometry is indifferent by construction, usually because the two
    # ways on are mirror images about straight on.
    scores = sorted((s["final_score"] for s in scored), reverse=True)
    deferred = len(scores) >= 2 and (scores[0] - scores[1]) < defer_margin

    if is_reversal(chosen["deg"]):
        reason = "BACK"
    elif ambiguous:
        reason = "GEOMETRY_AMBIGUOUS"
    elif deferred:
        reason = "DELAYED_DECISION"
    elif direction and chosen["male_score"] > 0:
        reason = f"MALE_{direction}"
    elif seems_straight(chosen["deg"]):
        reason = "STRAIGHT"
    else:
        reason = "GEOMETRY"

    return {"scored": ranked, "chosen": chosen, "probs": probs, "reason": reason,
            "male_direction": direction, "ramp": ramp, "ambiguous": ambiguous,
            "ambiguous_pair": pair, "deferred": deferred,
            "confidence": probs[chosen["edge"]]}


# ------------------------------------------------------------------ the whole decision
def decide(forward: list[dict], back: list[dict], integral: float, floor: float,
           visits: dict | None = None, last_visit: dict | None = None, tnow: float = 0.0,
           back_factor: float = 3.0, back_from_dead_end: bool = False,
           male_bonus: float = 1.5, back_penalty_min: float = 0.08,
           decay: float = 0.7, recent_penalty: float = 0.12, return_s: float = 60.0,
           ambiguous_deg: float = AMBIGUOUS_DEG, defer_margin: float = 0.25,
           yaw_sign_left: int = 1) -> dict:
    """The rule at a junction, from the offered passages to the one taken.

    Split out from the tracker so that the replay and the run cannot disagree: both call
    this, so a difference between them can only be a difference in the inputs.

    `forward` are the ways on other than the passage behind; `back` is that passage, or an
    empty list when the caller does not offer it at all.

    The order of business, which is the part worth arguing about:

      1. nothing ahead — the passage behind is not a choice, it is the only way out.
      2. exactly one way on — taken without asking anything, and the fly is not consulted, so
         no direction is reported either. A corridor is not a junction however loud the
         signal is, and this also means a reversal is never taken where there is somewhere to
         carry on.
      3. otherwise geometry and the signal decide between them, with the passage behind on
         the table only when there is a reason to believe a reversal: a signal that has
         cleared the reversal level.

    A third reason for offering the reversal was written and then removed, and the reason is
    worth keeping. The P08.4B note asks for a reversal also when "this is a return from a dead
    end". At a real dead end there is no way on, so clause 1 has already returned the walker
    and anything more is dead code; and every broader reading — the ways on all lead to tips,
    or the walker has just left one — puts the passage behind back among the candidates at
    junctions that have good ways on. That is exactly the reversal-at-a-junction defect the
    P08.1 rules removed, and the synthetic case in `p08_logic_test.py` catches it immediately.
    So the trigger is a parameter, it is off, and `back_from_dead_end=False` is what the run
    and the replay both pass.
    """
    ramp = signal_ramp(integral, floor, back_factor)
    strong_reversal = ramp > 0 and abs(integral) >= back_factor * floor

    if not forward:
        if not back:
            return {"chosen": None, "reason": "NO_EXIT", "scored": [], "probs": {},
                    "male_direction": "", "ramp": 0.0, "ambiguous": False,
                    "ambiguous_pair": None, "deferred": False, "confidence": 0.0,
                    "back_rule": "выхода нет"}
        res = score_candidates(back, integral, floor, visits, last_visit, tnow,
                               male_bonus, back_factor, back_penalty_min, decay,
                               recent_penalty, return_s, ambiguous_deg, defer_margin,
                               yaw_sign_left)
        # the only way out, so the fly was not asked, and no direction is reported
        res["reason"] = "BACK"
        res["male_direction"] = ""
        res["back_rule"] = "других выходов нет"
        return res

    if len(forward) == 1:
        res = score_candidates([forward[0]], integral, floor, visits, last_visit, tnow,
                               male_bonus, back_factor, back_penalty_min, decay,
                               recent_penalty, return_s, ambiguous_deg, defer_margin,
                               yaw_sign_left)
        res["reason"] = "ONLY_OPTION"
        res["male_direction"] = ""
        res["back_rule"] = "коридор: спрашивать нечего"
        return res

    if strong_reversal:
        back_rule = "сильный сигнал разворота"
    elif back_from_dead_end:
        back_rule = "впереди только тупики"
    else:
        back_rule = "запрещён"

    allowed = list(forward) + (list(back) if back_rule != "запрещён" else [])
    res = score_candidates(allowed, integral, floor, visits, last_visit, tnow,
                           male_bonus, back_factor, back_penalty_min, decay,
                           recent_penalty, return_s, ambiguous_deg, defer_margin,
                           yaw_sign_left)
    res["back_rule"] = back_rule
    return res
