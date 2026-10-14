"""Stage 3: Proof-of-Mastery Loop."""
import json
import re

import pytest

from learnpath import create_app
from learnpath import mastery as M
from learnpath.db import connect
from learnpath.questions import find_occurrences
from .conftest import SAMPLE
from .test_stage2 import answer, ask_next, drain, key_of, rows

HEADINGS_OK = None


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def miss_one(client, app, n_before_miss=0):
    """Answer n correct, then one wrong. Returns the missed question's DB row."""
    for _ in range(n_before_miss):
        answer(client, app, ask_next(client)["question"]["question_id"], correct=True)
    q = ask_next(client)["question"]
    answer(client, app, q["question_id"], correct=False)
    return rows(app, "SELECT * FROM questions WHERE id=?", q["question_id"])[0]


def start(client):
    return client.post("/api/loop/start", json={}).get_json()


# ---------- diagnosis ----------
def test_no_diagnosis_without_recorded_misses(app, loaded):
    assert start(loaded)["status"] == "no_weak_concepts"
    for _ in range(3):
        answer(loaded, app, ask_next(loaded)["question"]["question_id"], correct=True)   # only successes
    assert start(loaded)["status"] == "no_weak_concepts"
    recs = loaded.get("/api/review").get_json()["recommendations"]
    assert all(r["type"] != "start_loop" for r in recs)
    assert rows(app, "SELECT COUNT(*) c FROM mastery_loops")[0]["c"] == 0


def test_diagnoses_the_concept_actually_missed_with_real_evidence(app, loaded):
    missed = miss_one(loaded, app, n_before_miss=2)
    r = start(loaded)
    assert r["status"] == "ok"
    L = r["loop"]
    assert L["concept_key"] == missed["concept_key"] and L["status"] == "revising"
    d = L["diagnosis"]
    assert d["misses"] == 1 and d["n"] == 1 and d["mean"] == round(d["snapshot"]["mean"], 3)
    # the evidence attempt is the real one recorded in the DB
    att = d["attempts"][0]
    assert att["question_id"] == missed["id"] and att["correct"] is False and att["source_quote"] == missed["source_quote"]
    opts = json.loads(missed["options_json"])
    assert att["answer"] == opts[missed["answer_index"]]["text"] and att["chosen"] != att["answer"]
    # numbers in the explanation come from the stored mastery row
    cm = rows(app, "SELECT * FROM concept_mastery WHERE concept_key=?", missed["concept_key"])[0]
    assert f"{M.mean(cm['a'], cm['b']):.2f}" in d["text"]


def test_picks_the_weakest_concept_when_several_are_weak(app, loaded):
    first = miss_one(loaded, app)
    # miss the same concept again if it recurs, otherwise miss another: loop until some concept has 2 misses or 12 tries
    miss_counts = {first["concept_key"]: 1}
    for _ in range(25):
        q = ask_next(loaded)
        if q["status"] != "ok":
            break
        row = rows(app, "SELECT concept_key FROM questions WHERE id=?", q["question"]["question_id"])[0]
        answer(loaded, app, q["question"]["question_id"], correct=False)
        miss_counts[row["concept_key"]] = miss_counts.get(row["concept_key"], 0) + 1
        if len(miss_counts) >= 3:
            break
    cms = rows(app, "SELECT concept_key, a, b FROM concept_mastery")
    expected = min(cms, key=lambda r: (M.mean(r["a"], r["b"]), r["concept_key"]))["concept_key"]
    assert start(loaded)["loop"]["concept_key"] == expected


# ---------- source retrieval + revision plan ----------
def test_revision_plan_uses_only_real_passages_and_sentences(app, loaded):
    miss_one(loaded, app)
    L = start(loaded)["loop"]
    chunks = {c["id"]: c for c in rows(app, "SELECT * FROM chunks")}
    rev = L["revision"]
    assert rev["passages"] and len(rev["passages"]) <= 3
    pages = [p["citation"]["loc_start"] for p in rev["passages"]]
    assert len(pages) == len(set(pages))                     # no repeated page
    for p in rev["passages"]:
        ch = chunks[p["citation"]["chunk_id"]]
        assert p["citation"]["excerpt"] == ch["text"] and p["why"]
        for s in p["key_sentences"]:                         # key sentences are verbatim and contain the concept
            assert norm(s) in norm(ch["text"]) and find_occurrences(s, rev["concept_surface"])
    assert any("defines this concept" in p["why"] for p in rev["passages"])
    for c in rev["contrasts"]:                               # the wrongly chosen option is explained with its real passage
        assert norm(c["quote"]) in norm(chunks[c["citation"]["chunk_id"]]["text"])
    assert rev["steps"][0].startswith("Re-read") and rev["steps"][-1].startswith("When you're done")
    assert rev["home_topic"] in {r["topic"] for r in chunks.values()}


# ---------- retest + comparison ----------
def test_retest_is_new_same_concept_and_not_served_as_practice(app, loaded):
    missed = miss_one(loaded, app)
    L = start(loaded)["loop"]
    r = loaded.post(f"/api/loop/{L['id']}/retest").get_json()
    assert r["status"] == "ok" and r["loop"]["status"] == "retest_pending"
    q = rows(app, "SELECT * FROM questions WHERE id=?", r["loop"]["retest_question"]["question_id"])[0]
    assert q["purpose"] == "retest" and q["concept_key"] == missed["concept_key"]
    assert q["source_quote"] != missed["source_quote"] and q["id"] != missed["id"]
    assert q["fingerprint"] != missed["fingerprint"]
    # asking again resumes the same retest; practice does not pick it up
    again = loaded.post(f"/api/loop/{L['id']}/retest").get_json()
    assert again["resumed"] is True and again["loop"]["retest_question"]["question_id"] == q["id"]
    prac = ask_next(loaded)["question"]
    assert prac["question_id"] != q["id"]


@pytest.mark.parametrize("retest_correct", [True, False])
def test_comparison_matches_actual_mastery_rows(app, loaded, retest_correct):
    missed = miss_one(loaded, app, n_before_miss=1)
    L = start(loaded)["loop"]
    base = rows(app, "SELECT a, b, n FROM concept_mastery WHERE concept_key=?", missed["concept_key"])[0]
    qid = loaded.post(f"/api/loop/{L['id']}/retest").get_json()["loop"]["retest_question"]["question_id"]
    k = key_of(app, qid)
    res = loaded.post(f"/api/loop/{L['id']}/answer", json={"choice": k if retest_correct else (k + 1) % 4}).get_json()
    R = res["loop"]["result"]
    after = rows(app, "SELECT a, b, n FROM concept_mastery WHERE concept_key=?", missed["concept_key"])[0]
    # numbers in the report equal the database
    assert R["baseline"]["mean"] == round(M.mean(base["a"], base["b"]), 3)
    assert R["after"]["mean"] == round(M.mean(after["a"], after["b"]), 3) and R["after"]["n"] == after["n"] == base["n"] + 1
    assert R["delta"] == round(R["after"]["mean"] - R["baseline"]["mean"], 3)
    assert R["retest_correct"] is retest_correct
    assert (R["verdict"] == "improved") == (retest_correct and R["delta"] > 0)
    assert (R["delta"] > 0) == retest_correct
    assert R["retest"]["tests_same_concept"] and R["retest"]["different_source_sentence_than_all_earlier_questions"]
    assert R["caveats"] and "weak evidence" in R["caveats"][0]
    assert res["loop"]["status"] == "complete"
    # the retest attempt is real mastery evidence like any other
    assert rows(app, "SELECT COUNT(*) c FROM attempts")[0]["c"] == 3


def test_loop_state_machine_errors(app, loaded):
    miss_one(loaded, app)
    L = start(loaded)["loop"]
    lid = L["id"]
    assert loaded.post(f"/api/loop/{lid}/answer", json={"choice": 0}).status_code == 409     # no retest yet
    assert loaded.get("/api/loop/999").status_code == 404
    assert loaded.post("/api/loop/999/retest").status_code == 404
    loaded.post(f"/api/loop/{lid}/retest")
    assert loaded.post(f"/api/loop/{lid}/answer", json={"choice": 9}).status_code == 400
    assert loaded.post(f"/api/loop/{lid}/answer", json={"choice": 0}).status_code == 200
    assert loaded.post(f"/api/loop/{lid}/answer", json={"choice": 0}).status_code == 409
    assert loaded.post(f"/api/loop/{lid}/retest").status_code == 409
    assert rows(app, "SELECT COUNT(*) c FROM attempts")[0]["c"] == 2


def test_start_resumes_open_loop_instead_of_duplicating(app, loaded):
    miss_one(loaded, app)
    a = start(loaded)
    b = start(loaded)
    assert b["resumed"] is True and a["loop"]["id"] == b["loop"]["id"]
    assert rows(app, "SELECT COUNT(*) c FROM mastery_loops")[0]["c"] == 1
    recs = loaded.get("/api/review").get_json()["recommendations"]
    assert recs[0]["type"] == "continue_loop"


def test_honest_when_no_distinct_retest_exists(app, loaded):
    drain(loaded, app)                           # every usable sentence now used; alternating answers leave misses behind
    L = start(loaded)
    assert L["status"] == "ok"
    r = loaded.post(f"/api/loop/{L['loop']['id']}/retest").get_json()
    assert r["status"] == "no_retest_available" and "already been used" in r["message"]
    assert rows(app, "SELECT status FROM mastery_loops")[0]["status"] == "revising"
    assert rows(app, "SELECT COUNT(*) c FROM questions WHERE purpose='retest'")[0]["c"] == 0


# ---------- persistence + explainable recommendations ----------
def test_loop_and_learner_state_persist_across_restart(app, loaded):
    miss_one(loaded, app)
    L = start(loaded)["loop"]
    loaded.post(f"/api/loop/{L['id']}/retest")
    app2 = create_app({"DATA_DIR": app.config["DATA_DIR"], "LLM_API_KEY": ""})
    c2 = app2.test_client()
    got = c2.get(f"/api/loop/{L['id']}").get_json()
    assert got["status"] == "retest_pending" and got["retest_question"]["stem"].count("_____") == 1
    assert got["diagnosis"] == L["diagnosis"] and got["revision"] == L["revision"]
    qid = got["retest_question"]["question_id"]
    k = rows(app, "SELECT answer_index FROM questions WHERE id=?", qid)[0]["answer_index"]
    fin = c2.post(f"/api/loop/{L['id']}/answer", json={"choice": k}).get_json()
    assert fin["loop"]["status"] == "complete"
    c3 = create_app({"DATA_DIR": app.config["DATA_DIR"], "LLM_API_KEY": ""}).test_client()
    assert c3.get("/api/review").get_json()["loops"][0]["result"] == fin["loop"]["result"]


def test_recommendations_are_explained_with_numbers_from_the_database(app, loaded):
    r0 = loaded.get("/api/review").get_json()["recommendations"]
    assert [r["type"] for r in r0] == ["practice"] and len(r0[0]["evidence"]["untested_topics"]) == 6
    missed = miss_one(loaded, app)
    recs = loaded.get("/api/review").get_json()["recommendations"]
    rec = next(r for r in recs if r["type"] == "start_loop")
    cm = rows(app, "SELECT * FROM concept_mastery WHERE concept_key=?", missed["concept_key"])[0]
    assert rec["evidence"]["mean"] == round(M.mean(cm["a"], cm["b"]), 3)
    assert rec["evidence"]["misses"] == 1 and rec["evidence"]["answers"] == cm["n"]
    assert f"{M.mean(cm['a'], cm['b']):.2f}" in rec["reason"] and rec["concept_key"] == missed["concept_key"]
    assert all(r["reason"] and r["title"] for r in recs)


def test_deleting_document_removes_loops(app, loaded):
    miss_one(loaded, app); start(loaded)
    loaded.delete(f"/api/documents/{loaded.get('/api/documents').get_json()[0]['id']}")
    assert rows(app, "SELECT COUNT(*) c FROM mastery_loops")[0]["c"] == 0


def test_retest_targets_the_requested_concept_even_when_it_is_not_the_weakest(app, loaded):
    # build >= 2 weak concepts, then loop on one that is NOT the weakest
    for _ in range(30):
        q = ask_next(loaded)
        if q["status"] != "ok":
            break
        answer(loaded, app, q["question"]["question_id"], correct=False)
        recs = [r for r in loaded.get("/api/review").get_json()["recommendations"] if r["type"] == "start_loop"]
        if len(recs) >= 2:
            break
    recs = [r for r in loaded.get("/api/review").get_json()["recommendations"] if r["type"] == "start_loop"]
    assert len(recs) >= 2
    weakest, other = recs[0]["concept_key"], recs[-1]["concept_key"]
    assert weakest != other
    L = loaded.post("/api/loop/start", json={"concept_key": other}).get_json()["loop"]
    assert L["concept_key"] == other
    r = loaded.post(f"/api/loop/{L['id']}/retest").get_json()
    assert r["status"] == "ok"
    q = rows(app, "SELECT concept_key, purpose FROM questions WHERE id=?", r["loop"]["retest_question"]["question_id"])[0]
    assert q["concept_key"] == other and q["purpose"] == "retest"
