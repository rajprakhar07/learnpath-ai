"""HTTP routes."""
import secrets
import shutil
import sqlite3
from pathlib import Path

from flask import Blueprint, current_app, g, jsonify, render_template, request, send_file

from . import assessment, loop as loopmod, store
from .db import connect
from .figures import extract_pdf_figures
from .ingest_pdf import IngestError, ingest_pdf
from .ingest_pptx import ingest_pptx, read_slides
from .ingest_transcript import decode, ingest_transcript, parse_cues
from .locfmt import fmt_time
from .tutor import answer_question

bp = Blueprint("web", __name__)
SAMPLE_PDF = Path(__file__).resolve().parent.parent / "sample_data" / "sample_course.pdf"


def db():
    if "db" not in g:
        g.db = connect(current_app.config["DB_PATH"])
    return g.db


@bp.teardown_app_request
def _close(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def err(msg, code=400):
    return jsonify({"error": msg}), code


@bp.errorhandler(413)
def too_large(_e):
    mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return err(f"File too large (limit {mb} MB).", 413)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/status")
def status():
    llm = current_app.extensions["llm"]
    docs = store.list_documents(db())
    return jsonify({
        "llm_configured": llm.configured,
        "llm_model": llm.model if llm.configured else None,
        "mode": "llm" if llm.configured else "extractive",
        "n_documents": len(docs),
    })


@bp.get("/api/documents")
def documents():
    return jsonify(store.list_documents(db()))


SAMPLES_DIR = SAMPLE_PDF.parent
SAMPLE_NAMES = ["sample_course.pdf", "sample_deck.pptx", "sample_lecture.vtt", "sample_figure.pdf"]
KINDS = {".pdf": "pdf", ".pptx": "pptx", ".vtt": "transcript", ".srt": "transcript", ".txt": "transcript"}
FORMAT_HELP = "Supported files: PDF, PPTX, and timestamped transcripts (.vtt, .srt, .txt)."


def _looks_right(ext: str, head: bytes) -> bool:
    if ext == ".pdf":
        return head.startswith(b"%PDF-")
    if ext == ".pptx":
        return head.startswith(b"PK\x03\x04")
    return b"\x00" not in head[:1024]          # transcripts must be text


def _ingest_saved(tmp: Path, name: str, ext: str):
    """Shared by upload and sample loading. `tmp` is a file already on disk inside UPLOAD_DIR."""
    cfg = current_app.config
    kind = KINDS[ext]
    try:
        sha = store.sha256_file(tmp)
        existing = store.find_by_sha(db(), sha)
        if existing:
            doc = store.get_document(db(), existing["id"])
            return jsonify({"document": doc, "duplicate": True})
        try:
            figures = []
            if kind == "pdf":
                pages, chunks, n_empty = ingest_pdf(tmp)
                figures = extract_pdf_figures(tmp, {p["page"]: p["text"] for p in pages})
                n_units = len(pages)
            elif kind == "pptx":
                slides, chunks, figures, n_empty = ingest_pptx(tmp)
                n_units = len(slides)
            else:
                cues, chunks, figures = ingest_transcript(tmp)
                n_units, n_empty = len(cues), 0
        except IngestError as e:
            return err(str(e), 422)
        stored = f"{sha}{ext}"
        tmp.replace(cfg["UPLOAD_DIR"] / stored)
        try:
            doc_id = store.add_document(db(), filename=name, kind=kind, sha256=sha, stored_name=stored,
                                        n_units=n_units, n_empty=n_empty, chunks=chunks)
        except sqlite3.IntegrityError:  # concurrent identical upload
            return jsonify({"document": store.get_document(db(), store.find_by_sha(db(), sha)["id"]), "duplicate": True})
        store.add_figures(db(), doc_id, figures, cfg["FIGURE_DIR"])
        doc = store.get_document(db(), doc_id)
        doc["n_chunks"], doc["n_figures"] = len(chunks), len(figures)
        return jsonify({"document": doc, "duplicate": False}), 201
    finally:
        tmp.unlink(missing_ok=True)


@bp.post("/api/documents")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return err("No file provided.")
    name = Path(f.filename).name
    ext = Path(name).suffix.lower()
    if ext not in KINDS:
        return err(FORMAT_HELP, 415)
    head = f.stream.read(2048)
    f.stream.seek(0)
    if not _looks_right(ext, head):
        return err(f"That file doesn't look like a valid {ext[1:].upper()}.", 415)
    tmp = current_app.config["UPLOAD_DIR"] / f"tmp_{secrets.token_hex(8)}{ext}"
    f.save(tmp)
    return _ingest_saved(tmp, name, ext)


@bp.post("/api/documents/sample")
def load_sample():
    body = request.get_json(silent=True) or {}
    names = body.get("names") or ["sample_course.pdf"]
    if not isinstance(names, list) or any(n not in SAMPLE_NAMES for n in names):
        return err("Unknown sample file.", 400)
    results = []
    for n in names:
        src = SAMPLES_DIR / n
        if not src.exists():
            return err(f"Bundled sample {n} is missing. Run: python sample_data/make_sample_pdf.py && python sample_data/make_samples.py", 404)
        ext = src.suffix.lower()
        tmp = current_app.config["UPLOAD_DIR"] / f"tmp_{secrets.token_hex(8)}{ext}"
        shutil.copyfile(src, tmp)
        resp = _ingest_saved(tmp, n, ext)
        r, code = (resp if isinstance(resp, tuple) else (resp, 200))
        if code >= 400:
            return resp
        results.append(r.get_json())
    last = results[-1]
    out = {"document": last["document"], "duplicate": last["duplicate"], "documents": [r["document"] for r in results]}
    return jsonify(out), (200 if all(r["duplicate"] for r in results) else 201)


@bp.delete("/api/documents/<int:doc_id>")
def delete_doc(doc_id):
    if not store.delete_document(db(), doc_id, current_app.config["UPLOAD_DIR"], current_app.config["FIGURE_DIR"]):
        return err("Document not found.", 404)
    return jsonify({"deleted": doc_id})


@bp.get("/files/<int:doc_id>")
def serve_file(doc_id):
    doc = store.get_document(db(), doc_id)
    if not doc or doc["kind"] != "pdf":
        return err("Document not found.", 404)
    path = Path(current_app.config["UPLOAD_DIR"]) / doc["stored_name"]
    if not path.exists():
        return err("Stored file is missing.", 404)
    return send_file(path, mimetype="application/pdf", download_name=doc["filename"], conditional=True)


@bp.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}
    q = (body.get("question") or "").strip()
    if not q:
        return err("Please enter a question.")
    if len(q) > 1000:
        return err("Question is too long (max 1000 characters).")
    conn = db()
    index = current_app.extensions["retriever"].index(conn)
    doc_ids = body.get("doc_ids")
    doc_ids = set(int(x) for x in doc_ids) if isinstance(doc_ids, list) and doc_ids else None
    result = answer_question(q, index, current_app.extensions["llm"], current_app.config, doc_ids)
    return jsonify(result)


# ------------------------------------------------------------------ Stage 2: adaptive assessment
def _doc_ids(body):
    ids = body.get("doc_ids")
    return set(int(x) for x in ids) if isinstance(ids, list) and ids else None


@bp.post("/api/quiz/next")
def quiz_next():
    body = request.get_json(silent=True) or {}
    conn = db()
    index = current_app.extensions["retriever"].index(conn)
    res = assessment.next_question(conn, assessment.LEARNER, index, current_app.extensions["llm"], _doc_ids(body))
    return jsonify(res), (200 if res["status"] == "ok" else 409 if res["status"] == "exhausted" else 200)


@bp.post("/api/quiz/answer")
def quiz_answer():
    body = request.get_json(silent=True) or {}
    try:
        res = assessment.submit_answer(db(), assessment.LEARNER, body.get("question_id"), body.get("choice"))
    except assessment.AnswerError as e:
        return err(str(e), e.code)
    return jsonify(res)


@bp.post("/api/quiz/skip")
def quiz_skip():
    body = request.get_json(silent=True) or {}
    try:
        assessment.skip_question(db(), assessment.LEARNER, body.get("question_id"))
    except assessment.AnswerError as e:
        return err(str(e), e.code)
    return jsonify({"skipped": body.get("question_id")})


@bp.get("/api/mastery")
def mastery():
    conn = db()
    index = current_app.extensions["retriever"].index(conn)
    return jsonify(assessment.mastery_report(conn, assessment.LEARNER, index))


@bp.get("/api/quiz/history")
def quiz_history():
    return jsonify(assessment.history(db(), assessment.LEARNER))


# ------------------------------------------------------------------ Stage 3: Proof-of-Mastery Loop
@bp.get("/api/review")
def review():
    conn = db()
    index = current_app.extensions["retriever"].index(conn)
    return jsonify({"recommendations": loopmod.recommendations(conn, assessment.LEARNER, index),
                    "loops": loopmod.list_loops(conn, assessment.LEARNER)})


@bp.post("/api/loop/start")
def loop_start():
    body = request.get_json(silent=True) or {}
    conn = db()
    index = current_app.extensions["retriever"].index(conn)
    return jsonify(loopmod.start_loop(conn, assessment.LEARNER, index, body.get("concept_key")))


def _loop_call(fn, *a):
    try:
        return jsonify(fn(*a))
    except loopmod.LoopError as e:
        return err(str(e), e.code)


@bp.get("/api/loop/<int:loop_id>")
def loop_get(loop_id):
    try:
        return jsonify(loopmod.loop_view(db(), loopmod._loop_row(db(), assessment.LEARNER, loop_id)))
    except loopmod.LoopError as e:
        return err(str(e), e.code)


@bp.post("/api/loop/<int:loop_id>/retest")
def loop_retest(loop_id):
    conn = db()
    index = current_app.extensions["retriever"].index(conn)
    return _loop_call(loopmod.request_retest, conn, assessment.LEARNER, index, current_app.extensions["llm"], loop_id)


@bp.post("/api/loop/<int:loop_id>/answer")
def loop_answer(loop_id):
    body = request.get_json(silent=True) or {}
    return _loop_call(loopmod.answer_retest, db(), assessment.LEARNER, loop_id, body.get("choice"))


# ------------------------------------------------------------------ Stage 4: figures + viewers
def attach_figures(conn, obj):
    """Walk a JSON-able result; give every citation-like dict the real figures stored at its location."""
    if isinstance(obj, dict):
        if {"doc_id", "loc_kind", "loc_start"} <= obj.keys() and ("excerpt" in obj or "label" in obj):
            obj["figures"] = store.figures_at(conn, obj["doc_id"], obj["loc_kind"], obj["loc_start"])
        for v in obj.values():
            attach_figures(conn, v)
    elif isinstance(obj, list):
        for v in obj:
            attach_figures(conn, v)
    return obj


@bp.after_request
def _decorate_json(resp):
    """Attach figures to citation objects in the JSON of tutoring / assessment / loop endpoints."""
    path = request.path
    if resp.mimetype == "application/json" and resp.status_code < 400 and (
            path.startswith(("/api/ask", "/api/quiz/", "/api/loop", "/api/review"))):
        import json as _json
        try:
            data = _json.loads(resp.get_data(as_text=True))
        except ValueError:
            return resp
        conn = connect(current_app.config["DB_PATH"])
        try:
            attach_figures(conn, data)
        finally:
            conn.close()
        resp.set_data(_json.dumps(data))
    return resp


@bp.get("/figures/<int:fid>")
def serve_figure(fid):
    r = db().execute("SELECT stored_name, mime FROM figures WHERE id=?", (fid,)).fetchone()
    if not r:
        return err("Figure not found.", 404)
    path = Path(current_app.config["FIGURE_DIR"]) / r["stored_name"]
    if not path.exists():
        return err("Figure file is missing.", 404)
    return send_file(path, mimetype=r["mime"], conditional=True)


@bp.get("/view/<int:doc_id>")
def view_doc(doc_id):
    """Text viewer for slides and transcripts (PDFs use the browser's own viewer via /files/<id>)."""
    doc = store.get_document(db(), doc_id)
    if not doc or doc["kind"] == "pdf":
        return err("Document not found.", 404)
    path = Path(current_app.config["UPLOAD_DIR"]) / doc["stored_name"]
    if not path.exists():
        return err("Stored file is missing.", 404)
    if doc["kind"] == "pptx":
        try:
            target = int(request.args.get("slide", 0))
        except ValueError:
            target = 0
        slides = read_slides(path)
        figs = {}
        for r in db().execute("SELECT id, loc_start, alt_text FROM figures WHERE doc_id=? AND loc_kind='slide' ORDER BY id", (doc_id,)):
            figs.setdefault(int(r["loc_start"]), []).append({"id": r["id"], "alt": r["alt_text"]})
        return render_template("view.html", doc=doc, kind="pptx", slides=slides, target=target, figs=figs)
    try:
        t = float(request.args.get("t", -1))
        te = float(request.args.get("te", t))
    except ValueError:
        t = te = -1
    cues = parse_cues(decode(path.read_bytes()))
    for i, c in enumerate(cues):
        c["hl"] = t >= 0 and c["start"] >= t - 0.001 and c["start"] <= max(te, t) + 0.001
        c["label"] = fmt_time(c["start"])
        c["i"] = i
    return render_template("view.html", doc=doc, kind="transcript", cues=cues)
