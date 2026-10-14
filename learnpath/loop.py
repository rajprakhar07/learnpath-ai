"""Proof-of-Mastery Loop:  diagnose -> retrieve source -> targeted revision -> different question, same concept -> compare.

Everything here is derived from recorded attempts and stored passages. Nothing is estimated for concepts the student
has not been tested on, and no source or question is invented.
"""
import json

from . import assessment as A
from . import mastery as M
from .questions import build_pool, find_occurrences
from .textutil import is_full_sentence, split_sentences

WEAK_MEAN = 0.55      # concept mastery below this AND at least one recorded miss => "weak"
MAX_PASSAGES = 3


# ---------------------------------------------------------------- diagnosis
def concept_attempts(conn, learner, doc_id, ckey):
    rows = conn.execute(
        """SELECT a.id, a.correct, a.chosen_index, a.difficulty, a.created_at, q.id AS question_id, q.stem, q.qtype, q.options_json,
                  q.answer_index, q.source_quote, q.chunk_id, q.loc_start, q.purpose
           FROM attempts a JOIN questions q ON q.id=a.question_id
           WHERE a.learner_id=? AND q.doc_id=? AND q.concept_key=? ORDER BY a.id""", (learner, doc_id, ckey)).fetchall()
    out = []
    for r in rows:
        opts = json.loads(r["options_json"])
        out.append({"question_id": r["question_id"], "correct": bool(r["correct"]), "difficulty": r["difficulty"],
                    "page": int(r["loc_start"]), "chunk_id": r["chunk_id"], "purpose": r["purpose"], "source_quote": r["source_quote"],
                    "chosen": opts[r["chosen_index"]]["text"], "answer": opts[r["answer_index"]]["text"],
                    "chosen_quote": opts[r["chosen_index"]].get("quote", ""), "chosen_chunk_id": opts[r["chosen_index"]].get("chunk_id")})
    return out


def find_weak_concepts(conn, learner):
    rows = conn.execute("SELECT * FROM concept_mastery WHERE learner_id=? AND n>=1", (learner,)).fetchall()
    weak = []
    for r in rows:
        att = concept_attempts(conn, learner, r["doc_id"], r["concept_key"])
        misses = [x for x in att if not x["correct"]]
        m = M.mean(r["a"], r["b"])
        if misses and m < WEAK_MEAN:
            weak.append({"doc_id": r["doc_id"], "concept_key": r["concept_key"], "concept": r["concept"], "topic": r["topic"],
                         "mean": m, "n": r["n"], "misses": len(misses), "attempts": att, "a": r["a"], "b": r["b"]})
    weak.sort(key=lambda w: (w["mean"], -w["misses"], w["concept_key"]))
    return weak


def diagnosis_text(w):
    ok = w["n"] - w["misses"]
    confused = [x["chosen"] for x in w["attempts"] if not x["correct"]]
    s = (f"You've answered {w['n']} question{'s' if w['n'] != 1 else ''} on “{w['concept']}” and got {w['misses']} wrong "
         f"({ok} right). Concept mastery is {w['mean']:.2f} ({M.confidence(w['n'])} confidence), below the {WEAK_MEAN:.2f} threshold.")
    if confused:
        s += " When you missed it, you chose: " + ", ".join(f"“{c}”" for c in dict.fromkeys(confused)) + "."
    return s


# ---------------------------------------------------------------- source retrieval + revision plan
def _key_sentences(text, surface, limit=3):
    return [s for s in split_sentences(text) if is_full_sentence(s) and find_occurrences(s, surface)][:limit]


def build_revision(conn, index, w):
    chunks = [c for c in index.chunks if c["doc_id"] == w["doc_id"]]
    by_id = {c["id"]: c for c in chunks}
    pool = build_pool(chunks)
    term = pool.get(w["concept_key"])
    surface = term.surface if term else w["concept"]
    order, reasons = [], {}

    def add(cid, why):
        if cid in by_id and cid not in reasons:
            order.append(cid)
            reasons[cid] = why

    if term:
        add(term.chunk_id, "the passage where your material defines this concept")
    for x in w["attempts"]:
        if not x["correct"]:
            add(x["chunk_id"], "the passage behind a question you missed")
    for c, score in index.search(surface, k=6, doc_ids={w["doc_id"]}):
        if find_occurrences(c["text"], surface):
            add(c["id"], f"keyword search for “{surface}” (BM25 score {score:.2f})")
    passages, seen_pages = [], set()
    for cid in order:
        c = by_id[cid]
        page_key = (c["loc_kind"], c["loc_start"])
        if page_key in seen_pages:   # one passage per page keeps the plan from repeating itself
            continue
        seen_pages.add(page_key)
        if len(passages) >= MAX_PASSAGES:
            break
        passages.append({"citation": A._chunk_citation(conn, cid), "why": reasons[cid],
                         "key_sentences": _key_sentences(c["text"], surface)})
    contrasts = []
    for x in w["attempts"]:
        if not x["correct"] and x["chosen_quote"] and x["chosen_chunk_id"] in by_id:
            contrasts.append({"mistaken_for": x["chosen"], "quote": x["chosen_quote"], "citation": A._chunk_citation(conn, x["chosen_chunk_id"])})
    contrasts = list({c["mistaken_for"]: c for c in contrasts}.values())
    if not passages:
        return None
    steps = []
    p0 = passages[0]
    first_sent = p0["key_sentences"][0] if p0["key_sentences"] else None
    steps.append(f"Re-read {p0['citation']['doc_name']} {p0['citation']['label']} — {p0['why']}."
                 + (f" Focus on: “{first_sent}”" if first_sent else ""))
    for p in passages[1:]:
        steps.append(f"Then read {p['citation']['label']} — {p['why']}.")
    for c in contrasts:
        steps.append(f"You picked “{c['mistaken_for']}” by mistake. See how your material actually uses it on {c['citation']['label']}: “{c['quote']}”")
    steps.append("When you're done, take a new question on this concept (a different sentence from the ones you've seen).")
    return {"concept_surface": surface, "home_topic": term.topic if term else w["topic"], "passages": passages,
            "contrasts": contrasts, "steps": steps}


# ---------------------------------------------------------------- loop lifecycle
class LoopError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


def _loop_row(conn, learner, loop_id):
    r = conn.execute("SELECT * FROM mastery_loops WHERE id=? AND learner_id=?", (loop_id, learner)).fetchone()
    if not r:
        raise LoopError("Loop not found.", 404)
    return r


def loop_view(conn, row):
    d = {"id": row["id"], "status": row["status"], "concept": row["concept"], "concept_key": row["concept_key"], "topic": row["topic"],
         "doc_id": row["doc_id"], "diagnosis": json.loads(row["diagnosis_json"]), "revision": json.loads(row["revision_json"]),
         "result": json.loads(row["result_json"]) if row["result_json"] else None, "created_at": row["created_at"]}
    if row["status"] == "retest_pending" and row["retest_question_id"]:
        q = conn.execute("SELECT * FROM questions WHERE id=?", (row["retest_question_id"],)).fetchone()
        d["retest_question"] = A._row_to_public(q)
    return d


def start_loop(conn, learner, index, concept_key=None):
    open_rows = conn.execute("SELECT * FROM mastery_loops WHERE learner_id=? AND status!='complete' ORDER BY id", (learner,)).fetchall()
    weak = find_weak_concepts(conn, learner)
    if concept_key:
        weak = [w for w in weak if w["concept_key"] == concept_key]
    for r in open_rows:  # resume instead of duplicating
        if not concept_key or r["concept_key"] == concept_key:
            return {"status": "ok", "resumed": True, "loop": loop_view(conn, r)}
    if not weak:
        return {"status": "no_weak_concepts",
                "message": "No weak concept to work on yet. Answer some practice questions first; a concept needs at least one "
                           "recorded miss and mastery below %.2f before a revision loop is suggested." % WEAK_MEAN}
    w = weak[0]
    rev = build_revision(conn, index, w)
    if rev is None:
        return {"status": "no_source", "message": f"Couldn't find a passage in your material that supports “{w['concept']}”, so no revision plan was made."}
    diag = {"text": diagnosis_text(w), "mean": round(w["mean"], 3), "n": w["n"], "misses": w["misses"],
            "confidence": M.confidence(w["n"]), "threshold": WEAK_MEAN, "snapshot": M.snapshot(w["a"], w["b"], w["n"]),
            "attempts": [{k: x[k] for k in ("question_id", "correct", "difficulty", "page", "source_quote", "chosen", "answer")} for x in w["attempts"]]}
    cur = conn.execute("""INSERT INTO mastery_loops (learner_id, doc_id, concept_key, concept, topic, status, diagnosis_json, revision_json)
                          VALUES (?,?,?,?,?,?,?,?)""", (learner, w["doc_id"], w["concept_key"], w["concept"], rev["home_topic"], "revising",
                                                       json.dumps(diag), json.dumps(rev)))
    conn.commit()
    return {"status": "ok", "resumed": False, "loop": loop_view(conn, _loop_row(conn, learner, cur.lastrowid))}


def request_retest(conn, learner, index, llm, loop_id):
    row = _loop_row(conn, learner, loop_id)
    if row["status"] == "retest_pending":
        return {"status": "ok", "resumed": True, "loop": loop_view(conn, row)}
    if row["status"] == "complete":
        raise LoopError("This loop is already complete. Start a new one.", 409)
    diag = json.loads(row["diagnosis_json"])
    missed = [a for a in diag["attempts"] if not a["correct"]]
    want = missed[-1]["difficulty"] if missed else 2
    reason = (f"Retest for the revision loop on “{row['concept']}”: same concept, a source sentence you haven't been asked about. "
              f"Difficulty matched to your last miss (level {want}).")
    qid, notes = None, []
    for d in dict.fromkeys([want, 2, 1, 3]):
        qid, n = A.generate_question(conn, learner, index, llm, doc_ids={row["doc_id"]}, concept_keys={row["concept_key"]}, requested=d,
                                     purpose="retest", reason=reason)
        notes += n
        if qid:
            break
    if not qid:
        return {"status": "no_retest_available", "loop_id": loop_id,
                "message": f"Every usable sentence about “{row['concept']}” has already been used in a question, so no new, distinct "
                           "question can be generated. Re-read the passages above, or add more material.", "notes": notes[-3:]}
    conn.execute("UPDATE mastery_loops SET status='retest_pending', retest_question_id=?, updated_at=datetime('now') WHERE id=?", (qid, loop_id))
    conn.commit()
    return {"status": "ok", "resumed": False, "loop": loop_view(conn, _loop_row(conn, learner, loop_id))}


def answer_retest(conn, learner, loop_id, choice):
    row = _loop_row(conn, learner, loop_id)
    if row["status"] != "retest_pending":
        raise LoopError("Request a retest question first." if row["status"] == "revising" else "This loop is already complete.", 409)
    qid = row["retest_question_id"]
    diag = json.loads(row["diagnosis_json"])
    try:
        fb = A.submit_answer(conn, learner, qid, choice)
    except A.AnswerError as e:
        raise LoopError(str(e), e.code)
    q = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
    cstate = fb["mastery_after"]["concept"]
    before = fb["mastery_before"]["concept"]          # the actual stored state immediately before the retest answer
    prior_quotes = [a["source_quote"] for a in diag["attempts"]]
    prior_pages = sorted({a["page"] for a in diag["attempts"]})
    distinct = q["source_quote"] not in prior_quotes
    delta = round(cstate["mean"] - diag["snapshot"]["mean"], 3)
    improved = fb["correct"] and delta > 0
    result = {
        "verdict": "improved" if improved else "not_yet",
        "retest_correct": fb["correct"],
        "baseline": {"mean": diag["snapshot"]["mean"], "n": diag["snapshot"]["n"], "misses": diag["misses"]},
        "after": {"mean": cstate["mean"], "n": cstate["n"], "confidence": cstate["confidence"]},
        "delta": delta,
        "retest": {"question_id": qid, "difficulty": q["difficulty"], "stem": q["stem"], "answer": fb["answer"], "source_quote": q["source_quote"],
                   "citation": fb["citation"], "tests_same_concept": q["concept_key"] == row["concept_key"],
                   "different_source_sentence_than_all_earlier_questions": distinct, "earlier_pages": prior_pages},
        "caveats": ["A single multiple-choice question is weak evidence (a blind guess is right 25% of the time); "
                    "answer more questions on this concept before treating it as mastered."],
    }
    if improved:
        result["summary"] = (f"You answered a new question on “{row['concept']}” correctly after revising. Concept mastery moved "
                             f"{diag['snapshot']['mean']:.2f} → {cstate['mean']:.2f}.")
    elif fb["correct"]:
        result["summary"] = (f"Correct, but concept mastery is {cstate['mean']:.2f} (was {diag['snapshot']['mean']:.2f}); earlier misses still weigh on it. "
                             "Keep practising this concept.")
    else:
        result["summary"] = (f"Not yet: the new question on “{row['concept']}” was missed. Concept mastery is {cstate['mean']:.2f} "
                             f"(was {diag['snapshot']['mean']:.2f}). Re-read the source below and try another loop.")
    if q["difficulty"] != (max([a["difficulty"] for a in diag["attempts"] if not a["correct"]] or [q["difficulty"]])):
        result["caveats"].append("The retest difficulty differs from the question you originally missed.")
    conn.execute("UPDATE mastery_loops SET status='complete', result_json=?, updated_at=datetime('now') WHERE id=?", (json.dumps(result), loop_id))
    conn.commit()
    return {"feedback": fb, "loop": loop_view(conn, _loop_row(conn, learner, loop_id))}


# ---------------------------------------------------------------- explainable recommendations (from real state only)
def recommendations(conn, learner, index):
    recs = []
    for r in conn.execute("SELECT * FROM mastery_loops WHERE learner_id=? AND status!='complete' ORDER BY id", (learner,)):
        recs.append({"type": "continue_loop", "loop_id": r["id"], "title": f"Finish your revision loop on “{r['concept']}”",
                     "reason": "You started a loop and haven't completed the retest.", "evidence": {"status": r["status"]}})
    for w in find_weak_concepts(conn, learner)[:3]:
        if any(r.get("type") == "continue_loop" for r in recs) and False:
            continue
        recs.append({"type": "start_loop", "concept_key": w["concept_key"], "title": f"Revise “{w['concept']}”",
                     "reason": diagnosis_text(w),
                     "evidence": {"mean": round(w["mean"], 3), "answers": w["n"], "misses": w["misses"], "threshold": WEAK_MEAN}})
    rep = A.mastery_report(conn, learner, index)
    untested = [t for t in rep["topics"] if not t["assessed"]]
    if untested:
        recs.append({"type": "practice", "title": f"Practise {len(untested)} topic{'s' if len(untested) != 1 else ''} you haven't been tested on",
                     "reason": "No answers are recorded for: " + ", ".join(f"“{t['topic']}”" for t in untested[:4]) + (" …" if len(untested) > 4 else "") +
                               ". Mastery can't be estimated without answers.",
                     "evidence": {"untested_topics": [t["topic"] for t in untested]}})
    return recs


def list_loops(conn, learner):
    return [loop_view(conn, r) for r in conn.execute("SELECT * FROM mastery_loops WHERE learner_id=? ORDER BY id DESC", (learner,))]
