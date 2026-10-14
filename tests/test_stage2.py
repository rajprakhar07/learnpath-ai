"""Stage 2: adaptive assessment engine."""
import json
import re
import shutil
import sqlite3
import subprocess

import pytest

from learnpath import create_app
from learnpath import mastery as M
from learnpath.db import connect
from learnpath.questions import BLANK, validate_question
from .conftest import SAMPLE

HEADINGS = {"What Is Machine Learning", "Supervised Learning: Classification and Regression",
            "Loss Functions and Gradient Descent", "Overfitting, Underfitting and Regularization",
            "Evaluating Models", "Neural Networks and Backpropagation"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def rows(app, sql, *args):
    conn = connect(app.config["DB_PATH"])
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def key_of(app, qid):
    return rows(app, "SELECT answer_index FROM questions WHERE id=?", qid)[0]["answer_index"]


def ask_next(client):
    r = client.post("/api/quiz/next", json={})
    return r.get_json()


def answer(client, app, qid, correct=True):
    k = key_of(app, qid)
    return client.post("/api/quiz/answer", json={"question_id": qid, "choice": k if correct else (k + 1) % 4}).get_json()


def drain(client, app, limit=200):
    """Answer every question the engine will give us (alternating right/wrong). Returns number generated."""
    n = 0
    while n < limit:
        r = ask_next(client)
        if r["status"] != "ok":
            return n, r
        answer(client, app, r["question"]["question_id"], correct=(n % 2 == 0))
        n += 1
    raise AssertionError("engine never exhausted")


# ---------- invariants on every generated question ----------
def test_every_generated_question_satisfies_all_invariants(app, loaded):
    n, last = drain(loaded, app)
    assert n >= 15, f"expected a useful number of questions from the sample, got {n}"
    assert last["status"] == "exhausted"

    qs = rows(app, "SELECT * FROM questions")
    chunks = {c["id"]: c for c in rows(app, "SELECT * FROM chunks")}
    for q in qs:
        chunk = chunks[q["chunk_id"]]
        opts = json.loads(q["options_json"])
        # real source location
        assert q["loc_kind"] == "page" and 1 <= q["loc_start"] <= 6
        assert (q["loc_start"], q["doc_id"]) == (chunk["loc_start"], chunk["doc_id"])
        # topic comes from a real heading; difficulty is valid
        assert q["topic"] in HEADINGS and q["topic"] == chunk["topic"]
        assert q["difficulty"] in (1, 2, 3)
        # validated answer key, recomputed from stored data
        dto = {"options": opts, "answer_index": q["answer_index"], "topic": q["topic"], "difficulty": q["difficulty"],
               "loc_kind": q["loc_kind"], "loc_start": q["loc_start"], "chunk_id": q["chunk_id"], "stem": q["stem"],
               "source_quote": q["source_quote"], "qtype": q["qtype"]}
        ok, checks = validate_question(dto, chunk["text"])
        assert ok, (q["stem"], checks)
        assert json.loads(q["validation_json"])["checks"]
        # filling the blank with the key reproduces the source sentence exactly
        assert norm(q["stem"].replace(BLANK, opts[q["answer_index"]]["text"])) == norm(q["source_quote"])
        # exactly one correct option; the other three do not appear in the stem
        assert sum(1 for o in opts if norm(o["text"]) == norm(opts[q["answer_index"]]["text"])) == 1


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_source_quotes_exist_on_claimed_page_per_independent_extractor(app, loaded):
    drain(loaded, app)
    cache = {}
    for q in rows(app, "SELECT source_quote, loc_start FROM questions"):
        p = int(q["loc_start"])
        if p not in cache:
            cache[p] = norm(subprocess.check_output(["pdftotext", "-f", str(p), "-l", str(p), str(SAMPLE), "-"], text=True))
        assert norm(q["source_quote"]) in cache[p], (p, q["source_quote"])


def test_difficulty_is_measured_from_distractor_composition(app, loaded):
    drain(loaded, app)
    chunks = {c["id"]: c for c in rows(app, "SELECT * FROM chunks")}
    for q in rows(app, "SELECT * FROM questions WHERE generator='template'"):
        opts = json.loads(q["options_json"])
        same = sum(1 for i, o in enumerate(opts) if i != q["answer_index"] and chunks[o["chunk_id"]]["topic"] == q["topic"])
        # a distractor's "home" chunk topic is where its defining sentence lives
        assert q["difficulty"] == {0: 1, 1: 2}.get(same, 3), (q["stem"], same, q["difficulty"])


def test_no_duplicates_ever(app, loaded):
    n, last = drain(loaded, app)
    qs = rows(app, "SELECT fingerprint, quote_hash, stem FROM questions")
    assert len(qs) == n
    assert len({q["fingerprint"] for q in qs}) == n
    assert len({q["quote_hash"] for q in qs}) == n
    assert len({norm(q["stem"]) for q in qs}) == n
    # asking again after exhaustion must not invent a repeat
    again = loaded.post("/api/quiz/next", json={})
    assert again.status_code == 409 and again.get_json()["status"] == "exhausted"
    assert rows(app, "SELECT COUNT(*) c FROM questions")[0]["c"] == n


def test_db_rejects_duplicate_source_sentence_and_fingerprint(app, loaded):
    r = ask_next(loaded)
    q = rows(app, "SELECT * FROM questions WHERE id=?", r["question"]["question_id"])[0]
    conn = connect(app.config["DB_PATH"])
    cols = [k for k in q if k != "id"]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(f"INSERT INTO questions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [q[k] for k in cols])
    conn.close()


def test_public_question_does_not_leak_key_topic_or_source(loaded):
    q = ask_next(loaded)["question"]
    assert set(q) == {"question_id", "stem", "options", "difficulty", "qtype", "purpose", "status"}
    assert len(q["options"]) == 4 and q["stem"].count(BLANK) == 1


def test_pending_question_is_resumed_not_regenerated(app, loaded):
    a = ask_next(loaded)
    b = ask_next(loaded)
    assert a["question"]["question_id"] == b["question"]["question_id"] and b["resumed"] is True
    assert rows(app, "SELECT COUNT(*) c FROM questions")[0]["c"] == 1


def test_skip_marks_question_used_without_touching_mastery(app, loaded):
    a = ask_next(loaded)["question"]["question_id"]
    assert loaded.post("/api/quiz/skip", json={"question_id": a}).status_code == 200
    b = ask_next(loaded)["question"]["question_id"]
    assert b != a
    assert rows(app, "SELECT COUNT(*) c FROM attempts")[0]["c"] == 0 and rows(app, "SELECT COUNT(*) c FROM topic_mastery")[0]["c"] == 0
    assert loaded.post("/api/quiz/answer", json={"question_id": a, "choice": 0}).status_code == 409


# ---------- answering, feedback, mastery ----------
def test_correct_answer_feedback_and_exact_mastery_math(app, loaded):
    q = ask_next(loaded)["question"]
    fb = answer(loaded, app, q["question_id"], correct=True)
    d = q["difficulty"]
    a, b = M.PRIOR_A + M.CORRECT_W[d], M.PRIOR_B  # first update: decay of zero evidence is a no-op
    assert fb["correct"] is True
    assert fb["mastery_before"]["topic"]["n"] == 0 and fb["mastery_after"]["topic"]["n"] == 1
    assert fb["mastery_after"]["topic"]["mean"] == round(a / (a + b), 3)
    assert fb["source_quote"] in fb["citation"]["excerpt"]
    t = rows(app, "SELECT * FROM topic_mastery")[0]
    assert (round(t["a"], 6), round(t["b"], 6), t["n"]) == (round(a, 6), round(b, 6), 1)


def test_incorrect_answer_lowers_mastery_and_shows_evidence_for_the_wrong_choice(app, loaded):
    q = ask_next(loaded)["question"]
    fb = answer(loaded, app, q["question_id"], correct=False)
    assert fb["correct"] is False and fb["mastery_after"]["topic"]["mean"] < 0.5
    ev = fb["chosen_option_evidence"]
    assert ev and ev["quote"] in ev["citation"]["excerpt"]  # the wrong option is explained with a real passage
    assert "Not quite" in fb["explanation"] and fb["answer"] in fb["explanation"]
    assert fb["citation"]["label"].startswith("p.") and fb["selection_reason"]


def test_second_update_applies_decay_then_weight(app, loaded):
    # two answers in the same topic: compute expected values independently
    first = ask_next(loaded)["question"]
    f1 = answer(loaded, app, first["question_id"], correct=True)
    topic = f1["topic"]
    # force the same topic again by answering the rest until it recurs
    for _ in range(40):
        q = ask_next(loaded)["question"]
        row = rows(app, "SELECT topic, difficulty FROM questions WHERE id=?", q["question_id"])[0]
        if row["topic"] == topic:
            f2 = answer(loaded, app, q["question_id"], correct=False)
            a1, b1 = M.PRIOR_A + M.CORRECT_W[first["difficulty"]], M.PRIOR_B
            a2 = M.PRIOR_A + (a1 - M.PRIOR_A) * M.DECAY
            b2 = M.PRIOR_B + (b1 - M.PRIOR_B) * M.DECAY + M.INCORRECT_W[row["difficulty"]]
            assert f2["mastery_after"]["topic"]["mean"] == round(a2 / (a2 + b2), 3)
            assert f2["mastery_after"]["topic"]["n"] == 2
            return
        answer(loaded, app, q["question_id"], correct=True)
    pytest.fail("topic never recurred")


def test_answer_validation_errors(app, loaded):
    q = ask_next(loaded)["question"]["question_id"]
    for bad in (4, -1, "1", None, True, 1.5):
        assert loaded.post("/api/quiz/answer", json={"question_id": q, "choice": bad}).status_code == 400, bad
    assert loaded.post("/api/quiz/answer", json={"question_id": 99999, "choice": 0}).status_code == 404
    assert loaded.post("/api/quiz/answer", json={"question_id": q, "choice": 0}).status_code == 200
    assert loaded.post("/api/quiz/answer", json={"question_id": q, "choice": 0}).status_code == 409
    assert rows(app, "SELECT COUNT(*) c FROM attempts")[0]["c"] == 1


def test_mastery_report_has_no_data_before_any_attempt(loaded):
    rep = loaded.get("/api/mastery").get_json()
    assert rep["attempts"] == 0 and rep["concepts"] == []
    assert len(rep["topics"]) == 6 and all(t["assessed"] is False and t["mean"] is None for t in rep["topics"])


def test_mastery_report_matches_recorded_attempts(app, loaded):
    for i in range(6):
        q = ask_next(loaded)["question"]
        answer(loaded, app, q["question_id"], correct=(i < 4))
    rep = loaded.get("/api/mastery").get_json()
    assert rep["attempts"] == 6 and rep["correct"] == 4
    assert sum(t["n"] for t in rep["topics"]) == 6
    assert all(0 < t["mean"] < 1 for t in rep["topics"] if t["assessed"])


def test_mastery_and_history_persist_across_restart(app, loaded, tmp_path):
    for i in range(3):
        answer(loaded, app, ask_next(loaded)["question"]["question_id"], correct=bool(i % 2 == 0))
    before = loaded.get("/api/mastery").get_json()
    app2 = create_app({"DATA_DIR": app.config["DATA_DIR"], "LLM_API_KEY": ""})  # "restart": new app on the same files
    c2 = app2.test_client()
    assert c2.get("/api/mastery").get_json() == before
    assert len(c2.get("/api/quiz/history").get_json()) == 3
    nxt = c2.post("/api/quiz/next", json={}).get_json()
    assert nxt["status"] == "ok"  # and it does not repeat history
    old = {r["stem"] for r in rows(app, "SELECT stem FROM questions WHERE id<=3")}
    assert nxt["question"]["stem"] not in old


# ---------- adaptivity ----------
def test_untested_topics_are_covered_in_document_order_then_weakest_first(app, loaded):
    seen_topics = []
    wrong_topic = None
    for i in range(6):
        q = ask_next(loaded)["question"]
        topic = rows(app, "SELECT topic FROM questions WHERE id=?", q["question_id"])[0]["topic"]
        seen_topics.append(topic)
        if i == 2:
            wrong_topic = topic
        answer(loaded, app, q["question_id"], correct=(i != 2))
    assert seen_topics[0] == "What Is Machine Learning"            # document order
    assert len(set(seen_topics)) == 6                               # coverage before repetition
    nxt = ask_next(loaded)["question"]
    assert rows(app, "SELECT topic FROM questions WHERE id=?", nxt["question_id"])[0]["topic"] == wrong_topic  # weakest
    reason = rows(app, "SELECT selection_reason FROM questions WHERE id=?", nxt["question_id"])[0]["selection_reason"]
    assert "mastery" in reason and wrong_topic in reason


def test_low_mastery_requests_easier_questions(app, loaded):
    # fail repeatedly so a topic's mastery drops below 0.45 -> requested difficulty 1
    topic_state = {}
    for _ in range(40):
        q = ask_next(loaded)
        if q["status"] != "ok":
            break
        row = rows(app, "SELECT topic, requested_difficulty FROM questions WHERE id=?", q["question"]["question_id"])[0]
        m = topic_state.get(row["topic"])
        if m is not None and m < 0.45:
            assert row["requested_difficulty"] == 1
            return
        fb = answer(loaded, app, q["question"]["question_id"], correct=False)
        topic_state[row["topic"]] = fb["mastery_after"]["topic"]["mean"]
    pytest.fail("never observed a low-mastery topic being re-asked")


def test_delete_document_removes_dependent_assessment_data(app, loaded):
    answer(loaded, app, ask_next(loaded)["question"]["question_id"], correct=True)
    doc_id = loaded.get("/api/documents").get_json()[0]["id"]
    loaded.delete(f"/api/documents/{doc_id}")
    for t in ("questions", "attempts", "topic_mastery", "concept_mastery", "chunks"):
        assert rows(app, f"SELECT COUNT(*) c FROM {t}")[0]["c"] == 0, t
    assert loaded.post("/api/quiz/next", json={}).get_json()["status"] == "no_material"


# ---------- LLM question generation (scripted local server; NOT a real model) ----------
GOOD_GEN = {"stem": "Machine learning algorithms improve their performance on a task by learning from what?",
            "options": ["data", "hand-written rules", "random guessing", "hardware upgrades"], "answer_index": 0,
            "quote": "Machine learning is the study of algorithms that improve their performance on a task by learning from data rather than by following explicitly programmed rules.",
            "concept": "machine learning"}


def test_llm_question_accepted_only_when_quote_verbatim_and_solver_agrees(tmp_path, llm_client, fake_llm):
    fake_llm.reply(json.dumps(GOOD_GEN))
    fake_llm.reply(json.dumps({"answer_index": 0}))
    r = llm_client.post("/api/quiz/next", json={}).get_json()
    assert r["status"] == "ok" and r["question"]["qtype"] == "llm_mcq"
    assert len(fake_llm.requests) == 2
    solver_prompt = fake_llm.requests[1]["body"]["messages"][1]["content"]
    assert "answer_index" not in solver_prompt  # solver never sees the key


def _app_of(client):
    return client.application


def test_llm_question_with_fabricated_quote_falls_back_to_template(llm_client, fake_llm):
    bad = dict(GOOD_GEN, quote="Machine learning was invented by aliens in 1847.")
    fake_llm.reply(json.dumps(bad))
    fake_llm.reply(json.dumps(bad))
    r = llm_client.post("/api/quiz/next", json={}).get_json()
    assert r["status"] == "ok" and r["question"]["qtype"] == "cloze_mcq"
    assert rows(_app_of(llm_client), "SELECT generator FROM questions")[0]["generator"] == "template"


def test_llm_question_rejected_when_solver_disagrees(llm_client, fake_llm):
    for _ in range(2):
        fake_llm.reply(json.dumps(GOOD_GEN))
        fake_llm.reply(json.dumps({"answer_index": 2}))
    r = llm_client.post("/api/quiz/next", json={}).get_json()
    assert r["question"]["qtype"] == "cloze_mcq"


def test_llm_duplicate_question_is_rejected(llm_client, fake_llm):
    app = _app_of(llm_client)
    fake_llm.reply(json.dumps(GOOD_GEN)); fake_llm.reply(json.dumps({"answer_index": 0}))
    first = llm_client.post("/api/quiz/next", json={}).get_json()["question"]["question_id"]
    answer(llm_client, app, first, correct=True)
    # same sentence/stem proposed again for the same topic must be refused (falls back to a different template question)
    for _ in range(3):
        fake_llm.reply(json.dumps(GOOD_GEN)); fake_llm.reply(json.dumps({"answer_index": 0}))
    second = llm_client.post("/api/quiz/next", json={}).get_json()["question"]
    stems = [r["stem"] for r in rows(app, "SELECT stem FROM questions")]
    assert len(stems) == len(set(stems)) == 2 and second["question_id"] != first
