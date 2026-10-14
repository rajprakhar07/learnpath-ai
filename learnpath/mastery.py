
"""Explainable mastery model: a decayed, difficulty-weighted Beta posterior
per topic and per concept.

Prior Beta(2, 2) gives an initial mean of 0.5.
Evidence decays by 3% before each answer update.
Multiple-choice guessing is not modelled; confidence uses attempt count.
"""

import math

PRIOR_A = PRIOR_B = 2.0

CORRECT_W = {1: 0.6, 2: 1.0, 3: 1.4}
INCORRECT_W = {1: 1.4, 2: 1.0, 3: 0.6}

DECAY = 0.97


def update(a: float, b: float, correct: bool, difficulty: int):
    """Update mastery evidence using answer correctness and difficulty."""
    if difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2, or 3")

    if not isinstance(correct, bool):
        raise ValueError("correct must be a boolean")

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("a and b must be finite numbers")

    if a < PRIOR_A or b < PRIOR_B:
        raise ValueError("a and b must not be below the prior")

    # Decay accumulated evidence toward the prior.
    a = PRIOR_A + (a - PRIOR_A) * DECAY
    b = PRIOR_B + (b - PRIOR_B) * DECAY

    if correct:
        a += CORRECT_W[difficulty]
    else:
        b += INCORRECT_W[difficulty]

    return a, b


def mean(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("a and b must be finite numbers")
    if a < PRIOR_A or b < PRIOR_B:
        raise ValueError("a and b must not be below the prior")
    return a / (a + b)


def confidence(n: int) -> str:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")
    return "low" if n < 3 else "medium" if n < 6 else "high"


def snapshot(a: float, b: float, n: int) -> dict:
    return {
        "mean": round(mean(a, b), 3),
        "n": n,
        "confidence": confidence(n),
        "a": round(a, 3),
        "b": round(b, 3),
    }


def target_difficulty(m: float) -> int:
    if not math.isfinite(m) or not 0 <= m <= 1:
        raise ValueError("mastery must be between 0 and 1")
    return 1 if m < 0.45 else 2 if m < 0.70 else 3


def priority(m: float, n: int) -> float:
    """Higher means practise sooner; untested topics get an uncertainty bonus."""
    if not math.isfinite(m) or not 0 <= m <= 1:
        raise ValueError("mastery must be between 0 and 1")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")

    return 0.7 * (1 - m) + 0.3 / (1 + n)
