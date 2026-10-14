"""Pure metric helpers (unit-tested against hand-computed values)."""
import math


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion. Returns (lo, hi) or None when n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"k": k, "n": n, "value": (k / n) if n else None, "ci95": [round(ci[0], 3), round(ci[1], 3)] if ci else None}


def mrr(ranks):
    """Mean reciprocal rank. `ranks` are 1-based ranks or None (not retrieved => 0)."""
    if not ranks:
        return None
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else None
    return {"precision": p, "recall": r, "f1": f1}


def matches(cit: dict, expected: dict) -> bool:
    """Does a citation/hit {doc_name, loc_kind, loc_start, loc_end} satisfy an expected location?"""
    if cit["doc_name"] != expected["doc"]:
        return False
    if "page" in expected:
        return cit["loc_kind"] == "page" and int(cit["loc_start"]) == expected["page"]
    if "slide" in expected:
        return cit["loc_kind"] == "slide" and int(cit["loc_start"]) == expected["slide"]
    if "t" in expected:
        return cit["loc_kind"] == "time" and cit["loc_start"] <= expected["t"] <= cit["loc_end"]
    return False


def any_match(cit, expected_list) -> bool:
    return any(matches(cit, e) for e in expected_list)
