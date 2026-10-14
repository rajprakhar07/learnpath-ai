"""Question generation and validation.

Template generator (works offline): finds *defined terms* in the course text (e.g. "...called clustering",
"Gradient descent is ...") and builds fill-in-the-blank multiple-choice questions from real sentences.
Optional LLM generator: writes MCQs from one passage; accepted only if its quote is verbatim in the passage and an
independent "solver" call picks the same answer.

A question exists only if validate_question() passes. See that function for the exact rules.
"""
import hashlib
import json
import random
import re
from dataclasses import dataclass

from .llm import LLMError, parse_json_object
from .textutil import STOPWORDS, is_full_sentence, split_sentences, stem

BLANK = "_____"
CUT_WORDS = set(STOPWORDS) | {"without", "because", "while", "whereas", "within", "after", "before", "during", "such"}
VERBS = {"is", "are", "measures", "predicts", "refers", "means", "happens", "occurs", "reduces", "computes",
         "estimates", "gives", "adds", "minimizes", "maximizes"}
ALIAS_CUES = ("also called", "also known as", "known as", "i.e.", "in other words", "stands for")
_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9\-']*$")


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def concept_key(term: str) -> str:
    return " ".join(stem(w) for w in norm_key(term).split())


def display_form(term: str) -> str:
    """Lower-case a sentence-initial capital ('Gradient descent' -> 'gradient descent'); keep 'L1', 'F1', 'ReLU'."""
    if len(term) > 1 and term[0].isupper() and term[1].islower():
        return term[0].lower() + term[1:]
    return term


def _clean_tok(w: str) -> str:
    return w.strip(",.;:()\"'")


# ---------------------------------------------------------------- term extraction
def extract_terms(sentence: str):
    """Defined terms in a sentence -> [(surface, kind)] with kind in {called, subject}. Patterns: 'called X' and 'X is/are/measures ...'."""
    out = []
    toks = sentence.split()
    low = [_clean_tok(t).lower() for t in toks]
    # pattern 1: "called/termed/named X" and "known as X"
    for i, w in enumerate(low):
        j = None
        if w in ("called", "termed", "named"):
            j = i + 1
        elif w == "known" and i + 1 < len(low) and low[i + 1] == "as":
            j = i + 2
        if j is None:
            continue
        while j < len(toks) and low[j] in ("a", "an", "the"):
            j += 1
        words = []
        while j < len(toks) and len(words) < 3:
            core = _clean_tok(toks[j])
            if core.lower() in CUT_WORDS or not _WORD.match(core):
                break
            words.append(core)
            if toks[j][-1] in ",.;:)":
                break
            j += 1
        term = " ".join(words)
        if len(term) >= 3:
            out.append((term, "called"))
    # pattern 2: definitional subject at sentence start
    k = 0
    if low and low[0] in ("a", "an", "the"):
        k = 1
    subj = []
    while k < len(toks) and k < 6 and len(subj) < 4:
        core = _clean_tok(toks[k])
        if low[k] in VERBS and subj:
            term = " ".join(subj)
            if len(term) >= 3:
                out.append((term, "subject"))
            break
        if low[k] in CUT_WORDS or not _WORD.match(core) or toks[k][-1] in ",;:":
            break
        subj.append(core)
        k += 1
    seen, uniq = set(), []
    for t, kind in out:
        key = concept_key(t)
        if key and key not in seen:
            seen.add(key)
            uniq.append((t, kind))
    return uniq


@dataclass
class PoolTerm:
    surface: str
    key: str
    topic: str
    doc_id: int
    chunk_id: int
    quote: str          # the real sentence this term was found in
    loc_kind: str
    loc_start: float


MIN_SUBJECT_OCCURRENCES = 2   # a sentence subject must recur in the material to count as a concept
MAX_CHUNK_SHARE = 0.4         # terms found in >40% of passages are too generic ("model")
GENERIC_FILTER_MIN_CHUNKS = 10


def build_pool(chunks):
    """Defined terms found in the corpus (one entry per concept key, first occurrence wins), quality-filtered."""
    found = {}  # key -> (PoolTerm, kind)
    for c in chunks:
        for sent in split_sentences(c["text"]):
            if not is_full_sentence(sent):
                continue
            for t, kind in extract_terms(sent):
                k = concept_key(t)
                if k not in found:
                    found[k] = (PoolTerm(display_form(t), k, c["topic"], c["doc_id"], c["id"], sent, c["loc_kind"], c["loc_start"]), kind)
    pool = {}
    n_chunks = len(chunks)
    for k, (t, kind) in found.items():
        n_occ = sum(len(find_occurrences(c["text"], t.surface)) for c in chunks)
        n_chunk = sum(1 for c in chunks if find_occurrences(c["text"], t.surface))
        if kind == "subject" and n_occ < MIN_SUBJECT_OCCURRENCES:
            continue
        if n_chunks >= GENERIC_FILTER_MIN_CHUNKS and n_chunk / n_chunks > MAX_CHUNK_SHARE:
            continue
        pool[k] = t
    return pool


def find_occurrences(sentence: str, surface: str):
    """Whole-word, plural-tolerant, case-insensitive matches of a term inside a sentence -> [(start, end)]."""
    words = surface.split()
    pat = r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(w) for w in words) + r"(?:s|es)?(?![A-Za-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pat, sentence, flags=re.I)]


# ---------------------------------------------------------------- template generation
def _has_alias_link(all_sentences, a_surface: str, b_surface: str) -> bool:
    for s in all_sentences:
        ls = s.lower()
        if any(cue in ls for cue in ALIAS_CUES) and find_occurrences(s, a_surface) and find_occurrences(s, b_surface):
            return True
    return False


def realized_difficulty(n_same_topic: int) -> int:
    return 1 if n_same_topic == 0 else 2 if n_same_topic == 1 else 3


def choose_distractors(answer: PoolTerm, answer_surface: str, sentence: str, pool, all_sentences, requested: int, rng, topic: str, doc_id: int):
    """Pick 3 distractor terms. Easy = other topics; hard = same topic (most confusable). Returns list[PoolTerm] or None."""
    ak = answer.key
    cands = []
    for k, t in pool.items():
        if k == ak or ak in k or k in ak:
            continue
        if find_occurrences(sentence, t.surface):  # distractor visible in the stem -> leak / ambiguity
            continue
        if _has_alias_link(all_sentences, answer_surface, t.surface):  # possible synonym of the answer
            continue
        cands.append(t)
    # "same topic" is relative to the topic of the passage being tested (what the student is studying)
    same = [t for t in cands if t.topic == topic and t.doc_id == doc_id]
    other = [t for t in cands if not (t.topic == topic and t.doc_id == doc_id)]

    def rank(t):  # prefer similar length and same grammatical number as the answer (avoids giveaway cues)
        return (abs(len(t.surface.split()) - len(answer_surface.split())),
                t.surface.endswith("s") != answer_surface.endswith("s"), rng.random())
    same.sort(key=rank)
    other.sort(key=rank)
    want_same = {1: 0, 2: 1, 3: 3}[requested]
    chosen = same[:want_same]
    chosen += other[: 3 - len(chosen)]
    if len(chosen) < 3:  # not enough from other topics: top up from the remaining same-topic terms
        chosen += [t for t in same if t not in chosen][: 3 - len(chosen)]
    if len(chosen) < 3:
        return None
    return chosen[:3]


def fingerprint(stem_text: str, answer: str) -> str:
    return hashlib.sha1((norm_key(stem_text) + "|" + norm_key(answer)).encode()).hexdigest()


def quote_hash(quote: str) -> str:
    return hashlib.sha1(norm_key(quote).encode()).hexdigest()


def template_candidates(chunks, pool, topic=None, doc_ids=None, concept_keys=None):
    """Yield (chunk, sentence, PoolTerm, (start, end)) for every sentence that can be blanked on a pooled term."""
    for c in chunks:
        if topic is not None and c["topic"] != topic:
            continue
        if doc_ids is not None and c["doc_id"] not in doc_ids:
            continue
        for sent in split_sentences(c["text"]):
            n = len(sent.split())
            if not is_full_sentence(sent) or n < 7 or n > 45:
                continue
            for k, t in pool.items():
                if concept_keys is not None and k not in concept_keys:
                    continue
                occ = find_occurrences(sent, t.surface)
                if len(occ) == 1:  # exactly one occurrence: blanking it cannot leave the answer in the stem
                    yield c, sent, t, occ[0]


def build_template_question(chunk, sentence, term: PoolTerm, span, pool, all_sentences, requested: int, seed: str):
    answer_surface = display_form(sentence[span[0]:span[1]])
    stem_text = sentence[: span[0]] + BLANK + sentence[span[1]:]
    rng = random.Random(seed + term.key)
    ds = choose_distractors(term, answer_surface, sentence, pool, all_sentences, requested, rng, chunk["topic"], chunk["doc_id"])
    if ds is None:
        return None
    options = [{"text": answer_surface, "chunk_id": chunk["id"], "quote": sentence}]
    options += [{"text": d.surface, "chunk_id": d.chunk_id, "quote": d.quote} for d in ds]
    order = list(range(4))
    rng.shuffle(order)
    options = [options[i] for i in order]
    answer_index = order.index(0)
    n_same = sum(1 for d in ds if d.topic == chunk["topic"] and d.doc_id == chunk["doc_id"])
    return {
        "doc_id": chunk["doc_id"], "chunk_id": chunk["id"], "loc_kind": chunk["loc_kind"],
        "loc_start": chunk["loc_start"], "loc_end": chunk["loc_end"], "topic": chunk["topic"],
        "concept": answer_surface, "concept_key": term.key, "difficulty": realized_difficulty(n_same),
        "requested_difficulty": requested, "qtype": "cloze_mcq", "generator": "template",
        "stem": stem_text, "options": options, "answer_index": answer_index, "source_quote": sentence,
    }


# ---------------------------------------------------------------- validation
def validate_question(q: dict, chunk_text: str):
    """Return (ok, checks). Every check is recomputed from the stored chunk text, never trusted from the generator."""
    checks, ok = {}, True

    def chk(name, cond):
        nonlocal ok
        checks[name] = bool(cond)
        ok = ok and bool(cond)

    opts = q["options"]
    texts = [norm_key(o["text"]) for o in opts]
    ai = q["answer_index"]
    chk("has_4_options", len(opts) == 4)
    chk("options_unique_nonempty", len(set(texts)) == 4 and all(texts))
    chk("answer_index_valid", isinstance(ai, int) and 0 <= ai < len(opts))
    chk("topic_present", bool(str(q.get("topic", "")).strip()))
    chk("difficulty_valid", q.get("difficulty") in (1, 2, 3))
    chk("location_present", q.get("loc_kind") in ("page", "slide", "time") and q.get("loc_start") is not None and q.get("chunk_id"))
    chk("source_quote_verbatim_in_chunk", norm_ws(q["source_quote"]) and norm_ws(q["source_quote"]) in norm_ws(chunk_text))
    if checks["answer_index_valid"]:
        ans = opts[ai]["text"]
        chk("answer_in_source_quote", bool(find_occurrences(q["source_quote"], ans)) or norm_key(ans) in norm_key(q["source_quote"]))
        others = [o["text"] for i, o in enumerate(opts) if i != ai]
        chk("no_distractor_in_source_quote", not any(find_occurrences(q["source_quote"], d) for d in others))
        if q["qtype"] == "cloze_mcq":
            chk("stem_has_single_blank", q["stem"].count(BLANK) == 1)
            filled = q["stem"].replace(BLANK, ans)
            chk("blank_filled_with_key_reproduces_source_sentence", norm_key(filled) == norm_key(q["source_quote"]))
            rest = q["stem"].replace(BLANK, " ")
            ans_stems = {stem(w) for w in norm_key(ans).split() if w not in STOPWORDS}
            rest_stems = {stem(w) for w in norm_key(rest).split()}
            chk("answer_not_leaked_in_stem", not (ans_stems and ans_stems <= rest_stems))
    return ok, checks


# ---------------------------------------------------------------- optional LLM generator
GEN_SYSTEM = (
    "You write one multiple-choice question from the PASSAGE only. Rules: the question must be answerable from the "
    "passage alone; give exactly 4 distinct options with exactly one correct; copy one sentence from the passage "
    "VERBATIM as \"quote\" that proves the correct option; \"concept\" is the key term being tested and must appear in the "
    "passage. Respond with ONLY JSON: {\"stem\": str, \"options\": [str,str,str,str], \"answer_index\": 0-3, "
    "\"quote\": str, \"concept\": str}."
)
SOLVE_SYSTEM = (
    "Answer the multiple-choice question using ONLY the passage. Respond with ONLY JSON: {\"answer_index\": 0-3} "
    "or {\"answer_index\": null} if the passage does not settle it."
)


def llm_generate_question(llm, chunk, difficulty: int, seed: str):
    """Returns (question_dict | None, notes). Accepts only if quote is verbatim AND an independent solve call agrees."""
    level = {1: "easy (direct recall)", 2: "medium (needs understanding)", 3: "hard (needs applying or distinguishing ideas)"}[difficulty]
    try:
        raw = llm.chat([{"role": "system", "content": GEN_SYSTEM},
                        {"role": "user", "content": f"Difficulty: {level}\nPASSAGE:\n{chunk['text']}"}], temperature=0.4, max_tokens=500)
    except LLMError as e:
        return None, f"llm error: {e}"
    obj = parse_json_object(raw)
    try:
        stem_text = norm_ws(str(obj["stem"]))
        options = [norm_ws(str(o)) for o in obj["options"]]
        ai = int(obj["answer_index"])
        quote = norm_ws(str(obj["quote"]))
        concept = norm_ws(str(obj["concept"]))
    except (TypeError, KeyError, ValueError):
        return None, "llm returned malformed question JSON"
    if len(options) != 4 or not (0 <= ai < 4) or not stem_text or not concept:
        return None, "llm question failed structure checks"
    if quote.lower() not in norm_ws(chunk["text"]).lower():
        return None, "llm quote is not verbatim in the passage"
    if not find_occurrences(norm_ws(chunk["text"]), concept):
        return None, "llm concept does not occur in the passage"
    ans_terms = [w for w in norm_key(options[ai]).split() if w not in STOPWORDS]
    quote_terms = set(norm_key(quote + " " + stem_text).split())
    if ans_terms and sum(1 for w in ans_terms if w in quote_terms) / len(ans_terms) < 0.5:
        return None, "correct option is not supported by the quote"
    order = [{"text": o, "chunk_id": chunk["id"], "quote": quote if i == ai else ""} for i, o in enumerate(options)]
    # independent solver pass: it never sees the key
    try:
        raw2 = llm.chat([{"role": "system", "content": SOLVE_SYSTEM},
                         {"role": "user", "content": f"PASSAGE:\n{chunk['text']}\n\nQUESTION: {stem_text}\n" +
                          "\n".join(f"{i}. {o}" for i, o in enumerate(options))}], temperature=0.0, max_tokens=60)
    except LLMError as e:
        return None, f"llm verifier error: {e}"
    obj2 = parse_json_object(raw2) or {}
    if obj2.get("answer_index") != ai:
        return None, f"verifier disagreed with key (verifier said {obj2.get('answer_index')!r})"
    return {
        "doc_id": chunk["doc_id"], "chunk_id": chunk["id"], "loc_kind": chunk["loc_kind"],
        "loc_start": chunk["loc_start"], "loc_end": chunk["loc_end"], "topic": chunk["topic"],
        "concept": concept.lower(), "concept_key": concept_key(concept), "difficulty": difficulty,
        "requested_difficulty": difficulty, "qtype": "llm_mcq", "generator": "llm",
        "stem": stem_text, "options": order, "answer_index": ai, "source_quote": quote,
    }, "ok (quote verbatim; independent solver agreed)"
