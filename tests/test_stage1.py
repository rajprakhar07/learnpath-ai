"""Stage 1: PDF upload -> retrieval -> cited answer -> unsupported-question workflow."""
import io
import re
import shutil
import subprocess

import pytest
from pypdf import PdfReader, PdfWriter

from learnpath.ingest_pdf import IngestError, clean_text, extract_pages, ingest_pdf
from learnpath.llm import parse_json_object
from learnpath.textutil import stem, terms
from learnpath.tutor import normalize_markers
from .conftest import SAMPLE


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ---------- text utilities ----------
def test_stemmer_groups_inflections():
    assert stem("overfitting") == stem("overfit") == stem("overfits")
    assert stem("classes") == stem("class")
    # "difference" and "between" are no longer stopped (they are content-bearing query signals).
    result = terms("What is the difference between precision and recall?")
    assert "precis" in result or "precision" in result
    assert "recal" in result or "recall" in result


# ---------- ingestion ----------
def test_sample_pdf_pages_and_chunks_stay_on_one_page():
    pages, chunks, n_empty = ingest_pdf(SAMPLE)
    assert [p["page"] for p in pages] == [1, 2, 3, 4, 5, 6]
    assert n_empty == 0 and len(chunks) >= 6
    assert all(c["loc_start"] == c["loc_end"] and c["loc_kind"] == "page" for c in chunks)
    assert {int(c["loc_start"]) for c in chunks} == {1, 2, 3, 4, 5, 6}


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler pdftotext not installed")
def test_every_chunk_text_appears_on_its_claimed_page_per_independent_extractor():
    """Cross-check with poppler (a different PDF parser) that stored passages really are on the stored page."""
    _, chunks, _ = ingest_pdf(SAMPLE)
    cache = {}
    for c in chunks:
        p = int(c["loc_start"])
        if p not in cache:
            cache[p] = norm(subprocess.check_output(["pdftotext", "-f", str(p), "-l", str(p), str(SAMPLE), "-"], text=True))
        assert norm(c["text"]) in cache[p], f"chunk not found on page {p}: {c['text'][:60]!r}"


def test_heading_is_separated_from_body():
    pages = extract_pages(SAMPLE)
    assert "\nSupervised learning problems" in pages[1]["text"] or "Regression\nSupervised" in pages[1]["text"]
    assert "Gradient Descent\nA loss function" in pages[2]["text"]


def test_clean_text_dehyphenates_and_collapses():
    assert clean_text("regu-\nlarization   is\tgood.") == "regularization is good."


def make_pdf(tmp_path, name, *, blank=False, encrypt=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    p = tmp_path / name
    c = canvas.Canvas(str(p), pagesize=A4)
    if not blank:
        c.drawString(100, 700, "Hello world. This is a short text page about photosynthesis in plants.")
    else:
        c.rect(100, 100, 200, 200, fill=1)  # drawing only, no text
    c.save()
    if encrypt:
        r, w = PdfReader(str(p)), PdfWriter()
        for pg in r.pages:
            w.add_page(pg)
        w.encrypt(encrypt)
        with open(p, "wb") as f:
            w.write(f)
    return p


def test_image_only_pdf_is_rejected_with_clear_message(tmp_path):
    with pytest.raises(IngestError, match="No extractable text"):
        ingest_pdf(make_pdf(tmp_path, "blank.pdf", blank=True))


def test_password_protected_pdf_is_rejected(tmp_path):
    with pytest.raises(IngestError, match="password"):
        ingest_pdf(make_pdf(tmp_path, "enc.pdf", encrypt="hunter2"))


def test_corrupt_pdf_is_rejected(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(IngestError):
        ingest_pdf(p)


# ---------- upload API ----------
def test_upload_sample_duplicate_and_delete(client):
    data = SAMPLE.read_bytes()
    r = client.post("/api/documents", data={"file": (io.BytesIO(data), "notes.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 201 and r.get_json()["document"]["n_units"] == 6
    doc_id = r.get_json()["document"]["id"]
    r2 = client.post("/api/documents", data={"file": (io.BytesIO(data), "renamed.pdf")}, content_type="multipart/form-data")
    assert r2.get_json()["duplicate"] is True and len(client.get("/api/documents").get_json()) == 1
    assert client.get(f"/files/{doc_id}").data == data
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get(f"/files/{doc_id}").status_code == 404
    assert client.get("/api/documents").get_json() == []


@pytest.mark.parametrize("name,payload,code", [
    ("notes.docx", b"hello", 415),         # unsupported type
    ("notes.txt", b"hello", 422),          # .txt is a transcript format now; no timestamps -> clear 422
    ("fake.pdf", b"not a pdf at all", 415),
    ("bad.pdf", b"%PDF-1.4 garbage", 422),
])
def test_bad_uploads_rejected(client, name, payload, code):
    r = client.post("/api/documents", data={"file": (io.BytesIO(payload), name)}, content_type="multipart/form-data")
    assert r.status_code == code and "error" in r.get_json()


def test_upload_without_file(client):
    assert client.post("/api/documents", data={}, content_type="multipart/form-data").status_code == 400


def test_failed_upload_leaves_no_temp_files(client, app):
    client.post("/api/documents", data={"file": (io.BytesIO(b"%PDF-1.4 garbage"), "bad.pdf")}, content_type="multipart/form-data")
    assert list(app.config["UPLOAD_DIR"].glob("*")) == []


# ---------- retrieval + extractive answers (offline mode) ----------
IN_SCOPE = [  # (question, expected page that must be cited)
    ("What is gradient descent?", 3),
    ("What happens if the learning rate is too large?", 3),
    ("What is the difference between precision and recall?", 5),
    ("How does L1 regularization differ from L2?", 4),
    ("What is backpropagation?", 6),
    ("What is overfitting?", 4),
    ("What is k-fold cross-validation?", 5),
    ("What is the difference between classification and regression?", 2),
    ("What is the validation set used for?", 2),
    ("What are the three families of machine learning?", 1),
    ("What does dropout do?", 6),
    ("Why does regularization reduce overfitting?", 4),
]
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the 2018 World Cup?",
    "What is the best way to learn Python?",
    "How do I bake sourdough bread?",
    "Explain the causes of the French Revolution",
    "What is the boiling point of water?",
    "How does photosynthesis work?",
    "Who is the CEO of Tesla?",
    "What is the derivative of sin(x)?",
]


@pytest.mark.parametrize("q,page", IN_SCOPE)
def test_in_scope_question_cites_expected_page(loaded, q, page):
    r = loaded.post("/api/ask", json={"question": q}).get_json()
    assert r["status"] == "answered", r
    assert page in {int(c["loc_start"]) for c in r["citations"]}, (q, [c["loc_start"] for c in r["citations"]])
    assert r["citations"] and "[S" in r["answer"]


@pytest.mark.parametrize("q", OUT_OF_SCOPE)
def test_out_of_scope_question_is_flagged_unsupported(loaded, q):
    r = loaded.post("/api/ask", json={"question": q}).get_json()
    assert r["status"] == "unsupported", r
    assert r["citations"] == [] and r["answer"] == "" and r["reason"]


def test_citations_are_verbatim_stored_chunks_and_answer_sentences_come_from_them(loaded):
    r = loaded.post("/api/ask", json={"question": "What is overfitting?"}).get_json()
    pages = {p["page"]: norm(p["text"]) for p in extract_pages(SAMPLE)}
    for c in r["citations"]:
        assert norm(c["excerpt"]) in pages[int(c["loc_start"])]
    cited_text = norm(" ".join(c["excerpt"] for c in r["citations"]))
    for line in r["answer"].split("\n"):
        assert norm(re.sub(r"\[S\d+\]", "", line)) in cited_text


def test_ask_validation_and_empty_corpus(client):
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 400
    assert client.post("/api/ask", data="not json", content_type="text/plain").status_code == 400
    r = client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["status"] == "unsupported" and "uploaded" in r["reason"]


def test_doc_filter_excludes_other_documents(loaded):
    r = loaded.post("/api/ask", json={"question": "What is gradient descent?", "doc_ids": [999]}).get_json()
    assert r["status"] == "unsupported"


# ---------- LLM path (against a local scripted server; NOT a real model) ----------
def test_marker_normalisation():
    assert normalize_markers("A [S1, S2] b [S9].", 4) == ("A [S1][S2] b.", [1, 2], [9])


def test_parse_json_object_tolerates_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"answerable": true, "answer": "x {y}"} done') == {"answerable": True, "answer": "x {y}"}
    assert parse_json_object("no json here") is None


def test_llm_answer_with_valid_citation(llm_client, fake_llm):
    fake_llm.reply('{"answerable": true, "answer": "Gradient descent minimizes a loss by stepping against the gradient [S1]."}')
    r = llm_client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["status"] == "answered" and r["mode"] == "llm"
    assert [int(c["loc_start"]) for c in r["citations"]] == [3]
    req = fake_llm.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-secret-test-key"
    assert req["body"]["model"] == "fake-model" and "SOURCES:" in req["body"]["messages"][1]["content"]


def test_llm_invented_citation_marker_is_stripped(llm_client, fake_llm):
    fake_llm.reply('{"answerable": true, "answer": "It minimizes loss [S1]. It was invented in 1847 [S9]."}')
    r = llm_client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["status"] == "answered" and "[S9]" not in r["answer"] and r["removed_invalid_citations"] == [9]
    assert all(c["n"] != 9 for c in r["citations"])


def test_llm_answer_without_valid_citation_is_withheld(llm_client, fake_llm):
    fake_llm.reply('{"answerable": true, "answer": "Gradient descent is great [S9]."}')
    r = llm_client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["status"] == "unsupported" and r["answer"] == "" and r["citations"] == []


def test_llm_declares_unanswerable(llm_client, fake_llm):
    fake_llm.reply('{"answerable": false, "answer": ""}')
    r = llm_client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["status"] == "unsupported" and r["mode"] == "llm"


def test_gate_blocks_off_topic_before_calling_llm(llm_client, fake_llm):
    r = llm_client.post("/api/ask", json={"question": "What is the capital of France?"}).get_json()
    assert r["status"] == "unsupported" and fake_llm.requests == []


def test_llm_http_error_falls_back_to_extractive_without_leaking_key(llm_client, fake_llm):
    fake_llm.reply("upstream exploded", status=500)
    r = llm_client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["status"] == "answered" and r["mode"] == "extractive_fallback" and "500" in r["llm_error"]
    assert "sk-secret-test-key" not in str(r)


def test_llm_malformed_json_falls_back(llm_client, fake_llm):
    fake_llm.reply("I think gradient descent is cool")
    r = llm_client.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["mode"] == "extractive_fallback" and r["status"] == "answered"


def test_llm_unreachable_falls_back(tmp_path):
    from learnpath import create_app
    app = create_app({"DATA_DIR": tmp_path / "d", "LLM_BASE_URL": "http://127.0.0.1:9/v1", "LLM_API_KEY": "k",
                      "LLM_MODEL": "m", "LLM_TIMEOUT_SECONDS": 1})
    c = app.test_client(); c.post("/api/documents/sample")
    r = c.post("/api/ask", json={"question": "What is gradient descent?"}).get_json()
    assert r["mode"] == "extractive_fallback" and "failed" in r["llm_error"]


def test_status_endpoint_never_exposes_key(llm_client):
    s = llm_client.get("/api/status").get_json()
    assert s["llm_configured"] is True and "sk-secret" not in str(s)


def test_index_page_and_static_assets(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200
