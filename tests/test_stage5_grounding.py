"""Stage 5: Source-grounding reliability regression tests.

Covers:
  1. False-refusal regressions -- questions wrongly refused in the baseline.
  2. Unsupported-question protection -- questions that should still be refused.
  3. Paraphrase tolerance -- morphological variants of corpus terms.
  4. Multi-source answers -- questions answered from more than one document.
  5. ReLU definition-trap -- must be refused, not guessed.
  6. Coverage-floor for short queries -- 1-term queries not refused by MIN_COVERAGE.
  7. Citation precision -- extractive sentences must be verbatim from cited chunks.

All tests run in offline (extractive) mode; no LLM key needed.
"""
import re
import pytest
from learnpath.textutil import terms
from learnpath.tutor import _is_definition_trap, _stem_prefix_coverage, assess_support


# ---------------------------------------------------------------------------
# 1. False-refusal regressions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multi_loaded(tmp_path_factory):
    from learnpath import create_app
    tmp = tmp_path_factory.mktemp("multi")
    app = create_app({"DATA_DIR": tmp / "data", "LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""})
    c = app.test_client()
    r = c.post("/api/documents/sample", json={"names": [
        "sample_course.pdf", "sample_deck.pptx", "sample_lecture.vtt", "sample_figure.pdf"
    ]})
    assert r.status_code in (200, 201), r.get_json()
    return c


FALSE_REFUSAL_CASES = [
    "What does the scheduler decide?",           # heldout-17: scheduler vs Scheduling
    "What is the purpose of the learning rate?",  # heldout-19: purpose was formerly stopword
    "How can I stop a model from memorizing the training data?",   # heldout-28
    "Which SQL clause removes rows that don't meet a condition?",  # heldout-02
]


@pytest.mark.parametrize("q", FALSE_REFUSAL_CASES)
def test_formerly_refused_question_is_now_answered(multi_loaded, q):
    r = multi_loaded.post("/api/ask", json={"question": q}).get_json()
    assert r["status"] == "answered", (
        f"FALSE REFUSAL: {q!r}  reason={r.get('reason')}  support={r.get('support')}"
    )
    assert r["citations"]


# ---------------------------------------------------------------------------
# 2. Out-of-scope protection must not regress
# ---------------------------------------------------------------------------

STILL_UNSUPPORTED = [
    "What is the capital of France?",
    "Who won the 2018 World Cup?",
    "What is the boiling point of water?",
    "How does photosynthesis work?",
    "What is the time complexity of quicksort?",
    "What is a NoSQL database?",
    "How does the Adam optimizer work?",
    "What is a convolutional neural network?",
]


@pytest.mark.parametrize("q", STILL_UNSUPPORTED)
def test_out_of_scope_still_refused_after_gate_softening(multi_loaded, q):
    r = multi_loaded.post("/api/ask", json={"question": q}).get_json()
    assert r["status"] == "unsupported", (
        f"SHOULD STILL BE REFUSED: {q!r}  citations={r.get('citations')}"
    )
    assert r["answer"] == "" and r["citations"] == []


# ---------------------------------------------------------------------------
# 3. Paraphrase tolerance unit tests
# ---------------------------------------------------------------------------

def test_stem_prefix_coverage_fuzzy_ge_exact(tmp_path):
    from learnpath import create_app
    from learnpath.store import Retriever
    from learnpath.db import connect
    app = create_app({"DATA_DIR": tmp_path / "data", "LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""})
    c = app.test_client()
    c.post("/api/documents/sample", json={"names": ["sample_lecture.vtt"]})
    conn = connect(app.config["DB_PATH"])
    index = Retriever().index(conn)
    q_terms = terms("What does the scheduler decide")
    hits = index.search("What does the scheduler decide", k=4)
    if not hits:
        pytest.skip("No hits for transcript query")
    exact = index.weighted_coverage(q_terms, [ch for ch, _ in hits[:3]])
    fuzzy = _stem_prefix_coverage(q_terms, [ch for ch, _ in hits[:3]], index)
    assert fuzzy >= exact
    conn.close()


def test_assess_support_passes_paraphrase_via_fuzzy(tmp_path):
    from learnpath import create_app
    from learnpath.store import Retriever
    from learnpath.db import connect
    app = create_app({"DATA_DIR": tmp_path / "data", "LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""})
    c = app.test_client()
    c.post("/api/documents/sample", json={"names": ["sample_lecture.vtt"]})
    conn = connect(app.config["DB_PATH"])
    index = Retriever().index(conn)
    q_terms = terms("What does the scheduler decide")
    hits = index.search("What does the scheduler decide", k=4)
    if not hits:
        pytest.skip("No hits for transcript query")
    supported, _, reason = assess_support(index, q_terms, hits, app.config)
    assert supported, f"Paraphrase falsely refused: {reason}"
    conn.close()


# ---------------------------------------------------------------------------
# 4. Multi-source answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "How does test error change as model complexity increases?",
    "What is the difference between supervised and unsupervised learning?",
])
def test_multi_source_question_is_answered(multi_loaded, q):
    r = multi_loaded.post("/api/ask", json={"question": q}).get_json()
    assert r["status"] == "answered", f"Multi-source refused: {q!r}  reason={r.get('reason')}"


# ---------------------------------------------------------------------------
# 5. ReLU definition-trap regression
# ---------------------------------------------------------------------------

def test_relu_definition_trap_is_refused(multi_loaded):
    r = multi_loaded.post("/api/ask", json={"question": "What does ReLU stand for?"}).get_json()
    assert r["status"] == "unsupported", (
        f"DEFINITION TRAP: ReLU should be refused, got answer={r.get('answer')!r}"
    )
    assert r["answer"] == ""


def test_relu_trap_unit():
    fake_hits = [
        ({"text": "ReLU is a commonly used activation function in deep learning."}, 2.0),
        ({"text": "Activation functions like ReLU introduce non-linearity."}, 1.5),
    ]
    assert _is_definition_trap("What does ReLU stand for?", fake_hits) is True


def test_trap_not_triggered_when_passage_defines_term():
    fake_hits = [
        ({"text": "ReLU stands for Rectified Linear Unit, an activation function."}, 3.0),
    ]
    assert _is_definition_trap("What does ReLU stand for?", fake_hits) is False


def test_trap_not_triggered_for_non_abbreviation_question():
    fake_hits = [
        ({"text": "Gradient descent minimizes a loss function by stepping against the gradient."}, 2.0),
    ]
    assert _is_definition_trap("What is gradient descent?", fake_hits) is False


def test_trap_not_triggered_when_acid_defined():
    fake_hits = [
        ({"text": "ACID stands for Atomicity, Consistency, Isolation, and Durability."}, 3.0),
    ]
    assert _is_definition_trap("What does the D in ACID stand for?", fake_hits) is False


def test_trap_triggered_for_abbreviation_only_mentioned():
    fake_hits = [
        ({"text": "We use SGD for training large models efficiently."}, 2.0),
        ({"text": "SGD is faster than batch gradient descent in practice."}, 1.5),
    ]
    assert _is_definition_trap("What does SGD stand for?", fake_hits) is True


# ---------------------------------------------------------------------------
# 6. Short-query coverage floor
# ---------------------------------------------------------------------------

def test_single_content_term_query_not_refused(multi_loaded):
    r = multi_loaded.post("/api/ask", json={"question": "What is overfitting?"}).get_json()
    assert r["status"] == "answered", f"Single-term query refused: {r.get('reason')}"


def test_two_term_query_answered(multi_loaded):
    r = multi_loaded.post("/api/ask", json={"question": "How does dropout work?"}).get_json()
    assert r["status"] == "answered", f"Two-term query refused: {r.get('reason')}"


# ---------------------------------------------------------------------------
# 7. Citation precision -- extractive sentences verbatim from cited chunks
# ---------------------------------------------------------------------------

def test_extractive_sentences_come_from_cited_chunks(loaded):
    r = loaded.post("/api/ask", json={"question": "What is overfitting?"}).get_json()
    assert r["status"] == "answered"
    cited_texts = {c["chunk_id"]: c["excerpt"] for c in r["citations"]}
    norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()
    for line in r["answer"].split("\n"):
        sentence = re.sub(r"\[S\d+\]", "", line).strip()
        if not sentence:
            continue
        assert any(norm(sentence) in norm(text) for text in cited_texts.values()), (
            f"Extractive sentence not in any cited chunk: {sentence!r}"
        )


def test_citations_have_non_empty_excerpts_and_location_labels(loaded):
    r = loaded.post("/api/ask", json={"question": "What is backpropagation?"}).get_json()
    assert r["status"] == "answered"
    for c in r["citations"]:
        assert c["excerpt"] and len(c["excerpt"]) > 10
        assert c["label"]
        assert c["loc_start"] is not None and c["loc_kind"]
