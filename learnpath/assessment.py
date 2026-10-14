"""Adaptive assessment: choose what to ask, generate a validated question, record the answer, update mastery."""
import json
import sqlite3
from collections import defaultdict

from . import mastery as M
from .locfmt import loc_label
from .questions import (BLANK, build_pool, build_template_question, concept_key, fingerprint, llm_generate_question,
                        quote_hash, template_candidates, validate_question)
from .textutil import split_sentences, terms

LEARNER = "local"
NEAR_DUP_JACCARD = 0.8
TARGET_LEN = {1: 14, 2: 22, 3: 30}  # preferred sentence length (words) by difficulty


def label_of(c):
    return loc_label(c["loc_kind"], c["loc_start"], c.get("loc_end"))


# ---------------------------------------------------------------- history / dedupe
def _history(conn, learner):
    rows = conn.execute("SELECT fingerprint, quote_hash, stem, concept_key FROM questions WHERE learner_id=?", (learner,)).fetchall()
    return ({r["fingerprint"] for r in rows}, {r["quote_hash"] for r in rows},
            [set(terms(r["stem"].replace(BLANK, " "))) for r in rows])


def is_near_duplicate(stem_text: str, prior_stem_termsets) -> bool:
    s = set(terms(stem_text.replace(BLANK, " ")))
    if not s:
        return False
    return any(p and len(s & p) / len(s | p) >= NEAR_DUP_JACCARD for p in prior_stem_termsets)


def _dup_reason(q, fps, qhs, stems):
    if fingerprint(q["stem"], q["options"][q["answer_index"]]["text"]) in fps:
        return "exact duplicate of an earlier question"
    if quote_hash(q["source_quote"]) in qhs:
        return "this source sentence was already used for a question"
    if is_near_duplicate(q["stem"], stems):
        return "near-duplicate of an earlier question"
    return None


# ---------------------------------------------------------------- mastery access
def topic_state(conn, learner, doc_id, topic):
    r = conn.execute("SELECT a,b,n FROM topic_mastery WHERE learner_id=? AND doc_id=? AND topic=?", (learner, doc_id, topic)).fetchone()
    return (r["a"], r["b"], r["n"]) if r else (M.PRIOR_A, M.PRIOR_B, 0)


def concept_state(conn, learner, doc_id, ckey):
    r = conn.execute("SELECT a,b,n FROM concept_mastery WHERE learner_id=? AND doc_id=? AND concept_key=?", (learner, doc_id, ckey)).fetchone()
    return (r["a"], r["b"], r["n"]) if r else (M.PRIOR_A, M.PRIOR_B, 0)


# ---------------------------------------------------------------- generation
def _row_to_public(r):
    """A question as shown to the student BEFORE answering: no key, no topic, no source."""
    opts = json.loads(r["options_json"])
    return {"question_id": r["id"], "stem": r["stem"], "options": [o["text"] for o in opts], "difficulty": r["difficulty"],
            "qtype": r["qtype"], "purpose": r["purpose"], "status": r["status"]}


def _store(conn, learner, q, validation, reason, purpose):
    try:
        cur = conn.execute(
            """INSERT INTO questions (learner_id, doc_id, chunk_id, loc_kind, loc_start, loc_end, topic, concept, concept_key,
               difficulty, requested_difficulty, qtype, generator, stem, options_json, answer_index, source_quote,
               fingerprint, quote_hash, validation_json, selection_reason, purpose)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (learner, q["doc_id"], q["chunk_id"], q["loc_kind"], q["loc_start"], q["loc_end"], q["topic"], q["concept"],
             q["concept_key"], q["difficulty"], q["requested_difficulty"], q["qtype"], q["generator"], q["stem"],
             json.dumps(q["options"]), q["answer_index"], q["source_quote"],
             fingerprint(q["stem"], q["options"][q["answer_index"]]["text"]), quote_hash(q["source_quote"]),
             json.dumps(validation), reason, purpose))
    except sqlite3.IntegrityError:
        conn.rollback()
        return None
    conn.commit()
    return cur.lastrowid


def generate_question(conn, learner, index, llm, *, doc_ids=None, topic=None, concept_keys=None, requested=2,
                      purpose="practice", reason="", max_llm_tries=2):
    """Create, validate and persist one new question. Returns (question_id | None, notes[])."""
    notes = []
    fps, qhs, stems = _history(conn, learner)
    chunks = [c for c in index.chunks if (doc_ids is None or c["doc_id"] in doc_ids)]
    chunk_by_id = {c["id"]: c for c in chunks}
    all_sentences = [s for c in chunks for s in split_sentences(c["text"])]
    pool = build_pool(chunks)

    def accept(q):
        chunk = chunk_by_id[q["chunk_id"]]
        ok, checks = validate_question(q, chunk["text"])
        if not ok:
            notes.append(f"rejected by validation: {[k for k, v in checks.items() if not v]}")
            return None
        why = _dup_reason(q, fps, qhs, stems)
        if why:
            notes.append(f"rejected duplicate: {why}")
            return None
        return _store(conn, learner, q, {"checks": checks, "generator": q["generator"]}, reason, purpose)

    # 1) optional LLM generator (only for plain practice; retests need a specific concept)
    if llm is not None and llm.configured and concept_keys is None:
        used_chunks = defaultdict(int)
        for r in conn.execute("SELECT chunk_id FROM questions WHERE learner_id=?", (learner,)):
            used_chunks[r["chunk_id"]] += 1
        pool_chunks = sorted([c for c in chunks if topic is None or c["topic"] == topic], key=lambda c: (used_chunks[c["id"]], c["id"]))
        for c in pool_chunks[:max_llm_tries]:
            q, note = llm_generate_question(llm, c, requested, "seed")
            notes.append(f"llm chunk {c['id']}: {note}")
            if q:
                qid = accept(q)
                if qid:
                    return qid, notes

    # 2) deterministic template generator
    cands = list(template_candidates(chunks, pool, topic=topic, doc_ids=doc_ids, concept_keys=concept_keys))
    cstate = {}
    def cmean(doc_id, k):
        if (doc_id, k) not in cstate:
            a, b, _n = concept_state(conn, learner, doc_id, k)
            cstate[(doc_id, k)] = M.mean(a, b)
        return cstate[(doc_id, k)]
    # weakest concepts first, then sentence length closest to the difficulty target, then stable order
    cands.sort(key=lambda t: (cmean(t[0]["doc_id"], t[2].key), abs(len(t[1].split()) - TARGET_LEN[requested]), t[0]["id"]))
    tried = 0
    for chunk, sent, term, span in cands:
        if quote_hash(sent) in qhs:
            continue
        q = build_template_question(chunk, sent, term, span, pool, all_sentences, requested, seed=f"{learner}:{chunk['id']}")
        tried += 1
        if q is None:
            notes.append(f"skipped '{term.surface}': fewer than 3 usable distractors")
            continue
        qid = accept(q)
        if qid:
            return qid, notes
    if not cands:
        notes.append("no sentence can be blanked on a known concept in this scope")
    return None, notes


def next_question(conn, learner, index, llm, doc_ids=None):
    """Return the pending question if there is one, else pick a target adaptively and generate."""
    pend = conn.execute("SELECT * FROM questions WHERE learner_id=? AND status='pending' AND purpose='practice' ORDER BY id LIMIT 1", (learner,)).fetchone()
    if pend:
        return {"status": "ok", "question": _row_to_public(pend), "resumed": True}
    chunks = [c for c in index.chunks if (doc_ids is None or c["doc_id"] in doc_ids)]
    if not chunks:
        return {"status": "no_material", "message": "Upload course material first."}
    topics, seen = [], set()
    for c in chunks:
        if (c["doc_id"], c["topic"]) not in seen:
            seen.add((c["doc_id"], c["topic"]))
            topics.append((c["doc_id"], c["topic"]))
    ranked = []
    for pos, (doc_id, t) in enumerate(topics):  # pos = order of appearance in the material
        a, b, n = topic_state(conn, learner, doc_id, t)
        ranked.append((M.priority(M.mean(a, b), n), pos, doc_id, t, a, b, n))
    ranked.sort(key=lambda r: (-round(r[0], 9), r[1]))
    ranked = [(r[0], r[2], r[3], r[4], r[5], r[6]) for r in ranked]
    all_notes = []
    for pr, doc_id, t, a, b, n in ranked:
        m = M.mean(a, b)
        d = M.target_difficulty(m)
        if n == 0:
            why = f"You haven't been assessed on “{t}” yet, so it comes first for coverage."
        else:
            why = f"“{t}” has your lowest priority score: mastery {m:.2f} after {n} answer{'s' if n != 1 else ''}."
        reason = f"{why} Target difficulty {d} ({'easy' if d == 1 else 'medium' if d == 2 else 'hard'}) from mastery {m:.2f}."
        for dd in dict.fromkeys([d, 2, 1, 3]):  # fall back to other difficulties if this one is exhausted
            qid, notes = generate_question(conn, learner, index, llm, doc_ids={doc_id}, topic=t, requested=dd, reason=reason)
            all_notes += notes
            if qid:
                row = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
                return {"status": "ok", "question": _row_to_public(row), "resumed": False}
    return {"status": "exhausted", "message": "No new, distinct, validated questions can be generated from this material. "
            "Add more material or review your weak topics.", "notes": all_notes[-5:]}


# ---------------------------------------------------------------- answering + feedback
def _upsert_topic(conn, learner, doc_id, topic, a, b, n):
    conn.execute("""INSERT INTO topic_mastery (learner_id, doc_id, topic, a, b, n) VALUES (?,?,?,?,?,?)
                    ON CONFLICT(learner_id, doc_id, topic) DO UPDATE SET a=excluded.a, b=excluded.b, n=excluded.n, updated_at=datetime('now')""",
                 (learner, doc_id, topic, a, b, n))


def _upsert_concept(conn, learner, doc_id, topic, ckey, concept, a, b, n):
    conn.execute("""INSERT INTO concept_mastery (learner_id, doc_id, topic, concept_key, concept, a, b, n) VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(learner_id, doc_id, concept_key) DO UPDATE SET a=excluded.a, b=excluded.b, n=excluded.n, updated_at=datetime('now')""",
                 (learner, doc_id, topic, ckey, concept, a, b, n))


def _chunk_citation(conn, chunk_id):
    r = conn.execute("SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id=?", (chunk_id,)).fetchone()
    if not r:
        return None
    r = dict(r)
    return {"chunk_id": r["id"], "doc_id": r["doc_id"], "doc_name": r["filename"], "loc_kind": r["loc_kind"],
            "loc_start": r["loc_start"], "loc_end": r["loc_end"], "label": label_of(r), "excerpt": r["text"], "topic": r["topic"]}


class AnswerError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


def submit_answer(conn, learner, question_id, choice):
    q = conn.execute("SELECT * FROM questions WHERE id=? AND learner_id=?", (question_id, learner)).fetchone()
    if not q:
        raise AnswerError("Question not found.", 404)
    if q["status"] != "pending":
        raise AnswerError("This question has already been answered or skipped.", 409)
    opts = json.loads(q["options_json"])
    if not isinstance(choice, int) or isinstance(choice, bool) or not (0 <= choice < len(opts)):
        raise AnswerError("Choose one of the listed options.")
    correct = choice == q["answer_index"]
    d = q["difficulty"]

    ta, tb, tn = topic_state(conn, learner, q["doc_id"], q["topic"])
    ca, cb, cn = concept_state(conn, learner, q["doc_id"], q["concept_key"])
    before = {"topic": M.snapshot(ta, tb, tn), "concept": M.snapshot(ca, cb, cn)}
    ta2, tb2 = M.update(ta, tb, correct, d)
    ca2, cb2 = M.update(ca, cb, correct, d)
    after = {"topic": M.snapshot(ta2, tb2, tn + 1), "concept": M.snapshot(ca2, cb2, cn + 1)}

    _upsert_topic(conn, learner, q["doc_id"], q["topic"], ta2, tb2, tn + 1)
    _upsert_concept(conn, learner, q["doc_id"], q["topic"], q["concept_key"], q["concept"], ca2, cb2, cn + 1)
    conn.execute("""INSERT INTO attempts (learner_id, question_id, chosen_index, correct, difficulty, topic, concept_key, before_json, after_json)
                    VALUES (?,?,?,?,?,?,?,?,?)""", (learner, question_id, choice, int(correct), d, q["topic"], q["concept_key"],
                                                   json.dumps(before), json.dumps(after)))
    conn.execute("UPDATE questions SET status='answered' WHERE id=?", (question_id,))
    conn.commit()

    answer_text = opts[q["answer_index"]]["text"]
    chosen = opts[choice]
    filled = q["stem"].replace(BLANK, answer_text) if q["qtype"] == "cloze_mcq" else q["source_quote"]
    filled = filled[:1].upper() + filled[1:]
    if correct:
        explanation = f"Correct. Your material says: “{filled}”"
    else:
        explanation = f"Not quite. Your material says: “{filled}” The answer is “{answer_text}”."
    chosen_ev = None
    if not correct:
        if chosen.get("quote"):
            chosen_ev = {"option": chosen["text"], "quote": chosen["quote"], "citation": _chunk_citation(conn, chosen["chunk_id"])}
            explanation += f" You chose “{chosen['text']}”, which your material discusses elsewhere: “{chosen['quote']}”"
        else:
            explanation += f" You chose “{chosen['text']}”, which is not supported by the source passage."
    return {
        "correct": correct, "chosen_index": choice, "answer_index": q["answer_index"], "answer": answer_text,
        "explanation": explanation, "source_quote": q["source_quote"], "citation": _chunk_citation(conn, q["chunk_id"]),
        "chosen_option_evidence": chosen_ev, "topic": q["topic"], "concept": q["concept"], "difficulty": d,
        "difficulty_note": ("" if q["difficulty"] == q["requested_difficulty"] else
                            f"Requested difficulty {q['requested_difficulty']}, but this material only supported level {q['difficulty']}."),
        "selection_reason": q["selection_reason"], "mastery_before": before, "mastery_after": after,
        "question_id": question_id,
    }


def skip_question(conn, learner, question_id):
    q = conn.execute("SELECT status FROM questions WHERE id=? AND learner_id=?", (question_id, learner)).fetchone()
    if not q:
        raise AnswerError("Question not found.", 404)
    if q["status"] != "pending":
        raise AnswerError("This question has already been answered or skipped.", 409)
    conn.execute("UPDATE questions SET status='skipped' WHERE id=?", (question_id,))
    conn.commit()


# ---------------------------------------------------------------- reporting (all values come from real attempts)
def mastery_report(conn, learner, index):
    topics = []
    seen = set()
    for c in index.chunks:
        if (c["doc_id"], c["topic"]) in seen:
            continue
        seen.add((c["doc_id"], c["topic"]))
        a, b, n = topic_state(conn, learner, c["doc_id"], c["topic"])
        row = {"doc_id": c["doc_id"], "doc_name": c["filename"], "topic": c["topic"], "assessed": n > 0}
        if n:
            s = M.snapshot(a, b, n)
            corr = conn.execute("SELECT COALESCE(SUM(correct),0) FROM attempts WHERE learner_id=? AND topic=? AND question_id IN (SELECT id FROM questions WHERE doc_id=?)",
                                (learner, c["topic"], c["doc_id"])).fetchone()[0]
            row.update(mean=s["mean"], n=n, confidence=s["confidence"], correct=corr)
        else:
            row.update(mean=None, n=0, confidence=None, correct=0)
        topics.append(row)
    concepts = [dict(r) for r in conn.execute(
        "SELECT doc_id, topic, concept_key, concept, a, b, n FROM concept_mastery WHERE learner_id=? ORDER BY topic, concept", (learner,))]
    for r in concepts:
        r["mean"] = round(M.mean(r["a"], r["b"]), 3)
        r["confidence"] = M.confidence(r["n"])
        del r["a"], r["b"]
    total = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(correct),0) c FROM attempts WHERE learner_id=?", (learner,)).fetchone()
    return {"topics": topics, "concepts": concepts, "attempts": total["n"], "correct": total["c"]}


def history(conn, learner, limit=50):
    rows = conn.execute("""SELECT q.id, q.topic, q.concept, q.difficulty, q.requested_difficulty, q.stem, q.status, q.purpose, q.generator,
                                  q.loc_kind, q.loc_start, q.loc_end, a.correct, a.created_at
                           FROM questions q LEFT JOIN attempts a ON a.question_id=q.id
                           WHERE q.learner_id=? ORDER BY q.id DESC LIMIT ?""", (learner, limit)).fetchall()
    out = []
    for r in rows:
        r = dict(r)
        r["label"] = label_of(r)
        r["correct"] = None if r["correct"] is None else bool(r["correct"])
        out.append(r)
    return out
