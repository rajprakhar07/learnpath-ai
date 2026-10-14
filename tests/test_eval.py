"""Stage 4b: evaluation pipeline. Metric math is checked against hand-computed values."""
import json
import re
from pathlib import Path

import pytest

from eval.metrics import any_match, matches, mrr, prf, rate, wilson
from eval.run_benchmark import SAMPLES, load_jsonl, summarize_questions
from .conftest import ROOT

DS = ROOT / "eval" / "datasets"


# ---------- metric math (hand-computed) ----------
def test_wilson_matches_hand_calculation():
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-3)
    assert wilson(10, 10)[1] == 1.0 and wilson(5, 0) is None


def test_rate_mrr_prf():
    assert rate(3, 4) == {"k": 3, "n": 4, "value": 0.75, "ci95": [round(wilson(3, 4)[0], 3), round(wilson(3, 4)[1], 3)]}
    assert rate(0, 0)["value"] is None and rate(0, 0)["ci95"] is None
    assert mrr([1, 2, None, 4]) == pytest.approx(0.4375) and mrr([]) is None
    m = prf(3, 1, 2)
    assert m["precision"] == 0.75 and m["recall"] == 0.6 and m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert prf(0, 0, 0) == {"precision": None, "recall": None, "f1": None}
    assert prf(0, 2, 0)["f1"] is None            # recall undefined -> no F1


def test_location_matching_rules():
    page = {"doc_name": "a.pdf", "loc_kind": "page", "loc_start": 4.0, "loc_end": 4.0}
    assert matches(page, {"doc": "a.pdf", "page": 4}) and not matches(page, {"doc": "a.pdf", "page": 5})
    assert not matches(page, {"doc": "b.pdf", "page": 4}) and not matches(page, {"doc": "a.pdf", "slide": 4})
    t = {"doc_name": "v.vtt", "loc_kind": "time", "loc_start": 130.0, "loc_end": 142.0}
    assert matches(t, {"doc": "v.vtt", "t": 130}) and matches(t, {"doc": "v.vtt", "t": 142}) and not matches(t, {"doc": "v.vtt", "t": 143})
    assert any_match(page, [{"doc": "a.pdf", "page": 1}, {"doc": "a.pdf", "page": 4}]) and not any_match(page, [])


def rec(i, ans, status, cited_match, rank=None, n_cited=None):
    n = len(cited_match) if n_cited is None else n_cited
    return {"id": i, "question": i, "answerable": ans, "note": "", "status": status, "mode": "x", "reason": "",
            "cited": ["c"] * n, "cited_match": cited_match, "retrieval_rank": rank, "grounding_overlap": None}


def test_summary_computes_expected_numbers_from_known_records():
    recs = [rec("a1", True, "answered", [True, False], 1), rec("a2", True, "answered", [False], 3), rec("a3", True, "unsupported", [], 2),
            rec("a4", True, "answered", [True], None), rec("u1", False, "unsupported", []), rec("u2", False, "answered", [False])]
    s = summarize_questions(recs)
    assert s["gate"]["confusion"] == {"tp": 3, "fn": 1, "fp": 1, "tn": 1}
    assert s["gate"]["precision"] == 0.75 and s["gate"]["recall"] == 0.75
    assert s["citations"]["first_citation_correct"]["k"] == 2 and s["citations"]["first_citation_correct"]["n"] == 4      # a1, a4
    assert s["citations"]["any_citation_correct"]["k"] == 2
    assert (s["citations"]["citation_precision"]["k"], s["citations"]["citation_precision"]["n"]) == (2, 4)               # 2 good of 4 cited
    r = s["retrieval_independent_of_gate"]
    assert r["hit_at_1"]["k"] == 1 and r["hit_at_k"]["k"] == 3 and r["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2 + 0) / 4)
    assert {f["id"] for f in s["failures"]} == {"a2", "a3", "u2"}      # a4 answered with right citation => not a failure


# ---------- dataset integrity ----------
def test_datasets_are_well_formed_and_disjoint():
    dev, held = load_jsonl(DS / "dev.jsonl"), load_jsonl(DS / "heldout.jsonl")
    ids = [r["id"] for r in dev + held]
    assert len(ids) == len(set(ids))
    norm = lambda q: re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()
    assert not ({norm(r["question"]) for r in dev} & {norm(r["question"]) for r in held}), "dev/heldout contamination"
    for r in dev + held:
        assert bool(r["expected"]) == r["answerable"], r["id"]
        assert len(r["question"].split()) >= 3
    assert len(held) >= 40 and sum(not r["answerable"] for r in held) >= 15


def test_expected_locations_exist_in_the_corpus():
    from pypdf import PdfReader
    from pptx import Presentation
    n_pages = {n: len(PdfReader(str(ROOT / "sample_data" / n)).pages) for n in SAMPLES if n.endswith(".pdf")}
    n_slides = len(Presentation(str(ROOT / "sample_data" / "sample_deck.pptx")).slides)
    for r in load_jsonl(DS / "dev.jsonl") + load_jsonl(DS / "heldout.jsonl"):
        for e in r["expected"]:
            assert e["doc"] in SAMPLES, r["id"]
            if "page" in e:
                assert 1 <= e["page"] <= n_pages[e["doc"]], r["id"]
            if "slide" in e:
                assert 1 <= e["slide"] <= n_slides, r["id"]
            if "t" in e:
                assert 0 <= e["t"] <= 600, r["id"]


def test_adaptive_simulation_reports_consistent_metrics():
    from eval.run_benchmark import simulate_adaptive
    sim = simulate_adaptive(seeds=2, steps=14)
    assert "SYNTHETIC" in sim and len(sim["per_seed"]) == 2
    for r in sim["per_seed"]:
        assert 0 <= r["v2_observed"] <= 1 and 0 < r["v2_random_baseline"] < 1
        assert r["bank_capacity_weak"] <= r["bank_total"] and r["v2_m"] <= r["n_post_coverage"]
