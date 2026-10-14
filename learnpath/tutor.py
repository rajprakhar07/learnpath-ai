"""Source-grounded tutoring.

Pipeline:
retrieve (BM25) -> answerability gate -> answer (LLM or extractive)
-> citation validation.

Guarantees:
- Citations point only to retrieved document chunks.
- Excerpts are copied from stored source text.
- Invalid model-generated citation markers are removed.
- LLM answers without valid citations are withheld.
- Unsupported questions are rejected before an LLM call.
"""

import re
import time

from .llm import LLMError, parse_json_object
from .locfmt import loc_label
from .textutil import is_full_sentence, split_sentences, terms


_MARKER = re.compile(
    r"\[\s*S(\d+(?:\s*,\s*S?\d+)*)\s*\]",
    re.I,
)


def _loc_label(c):
    return loc_label(
        c["loc_kind"],
        c["loc_start"],
        c.get("loc_end"),
    )


def _citation(n, chunk, score):
    return {
        "n": n,
        "chunk_id": chunk["id"],
        "doc_id": chunk["doc_id"],
        "doc_name": chunk["filename"],
        "loc_kind": chunk["loc_kind"],
        "loc_start": chunk["loc_start"],
        "loc_end": chunk["loc_end"],
        "label": _loc_label(chunk),
        "excerpt": chunk["text"],
        "score": round(score, 3),
    }


# Abbreviation-definition safeguards.
_DEF_QUESTION = re.compile(
    r"\b(stand\s+for|stands\s+for|abbreviation|acronym|"
    r"short\s+for|full\s+form|mean[st]?|definition\s+of)\b",
    re.I,
)

# Recognizes uppercase abbreviations and mixed-case forms such as ReLU.
_ABBREV = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{1,6}|"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b"
)

_DEF_EVIDENCE = re.compile(
    r"\b(?:stands?\s+for|is\s+(?:an?\s+)?"
    r"(?:acronym|abbreviation)\s+for|"
    r"means|is\s+defined\s+as)\b",
    re.I,
)


# Conservative word relationships for extractive sentence selection.
# These help match question wording to relevant source wording.
_SENTENCE_ALIASES = {
    "scheduler": {"schedul"},
    "schedul": {"scheduler"},
    "purpose": {"control", "size", "step"},
    "memoriz": {"overfit", "memor"},
    "memor": {"memoriz", "overfit"},
    "stop": {"stopp", "prevent", "reduc", "avoid"},
    "stopp": {"stop", "prevent", "reduc", "avoid"},
    "overfit": {"memoriz", "memor"},
    "remov": {"filter", "delet", "exclud"},
    "filter": {"remov", "exclud"},
    "meet": {"satisf", "match"},
    "condition": {"criter", "qualifi"},
}


def _stem_prefix_coverage(
    q_terms,
    hit_chunks,
    index,
    prefix_len=4,
):
    """Estimate query coverage while tolerating morphological variants."""

    if not q_terms:
        return 0.0

    total = sum(index.idf(t) for t in q_terms)
    if not total:
        return 0.0

    by_id = {
        chunk["id"]: i
        for i, chunk in enumerate(index.chunks)
    }

    corpus_prefixes = set()

    for chunk in hit_chunks:
        idx = by_id.get(chunk["id"])
        if idx is None:
            continue

        for term in index.termsets[idx]:
            if len(term) >= prefix_len:
                corpus_prefixes.add(term[:prefix_len])
            else:
                corpus_prefixes.add(term)

    present_terms = index.union_terms(hit_chunks)
    covered_weight = 0.0

    for term in q_terms:
        prefix = (
            term[:prefix_len]
            if len(term) >= prefix_len
            else term
        )

        if term in present_terms or prefix in corpus_prefixes:
            covered_weight += index.idf(term)

    return covered_weight / total


def _is_definition_trap(question, hits):
    """Reject abbreviation-expansion questions without explicit definitions."""

    if not _DEF_QUESTION.search(question):
        return False

    abbreviations = _ABBREV.findall(question)
    if not abbreviations:
        return False

    for chunk, _score in hits[:3]:
        text = chunk["text"]

        for abbreviation in abbreviations:
            pattern = re.compile(
                rf"\b{re.escape(abbreviation)}\b",
                re.I,
            )

            if not pattern.search(text):
                continue

            # The abbreviation and definition evidence must occur
            # in the same sentence.
            for sentence in split_sentences(text):
                if (
                    pattern.search(sentence)
                    and _DEF_EVIDENCE.search(sentence)
                ):
                    return False

    return True


def _is_specific_definition_unsupported(question, hits):
    """Reject a specific 'What is a/an ...?' question when its
    distinguishing concept is absent from retrieved passages.
    """

    match = re.search(
        r"\bwhat\s+is\s+(?:a|an)\s+(.+?)[?.!]*$",
        question,
        re.I,
    )

    if not match:
        return False

    concept_terms = set(terms(match.group(1)))

    generic_terms = {
        "neural",
        "network",
        "model",
        "algorithm",
        "system",
        "method",
        "technique",
        "process",
        "data",
        "learning",
        "function",
        "type",
    }

    specific_terms = concept_terms - generic_terms

    if not specific_terms:
        return False

    corpus_terms = set()

    for chunk, _score in hits[:3]:
        corpus_terms.update(terms(chunk["text"]))

    # For example, a generic neural-network passage does not
    # establish what a convolutional neural network is.
    return not bool(specific_terms & corpus_terms)


def assess_support(index, q_terms, hits, cfg):
    """Decide whether retrieved material can plausibly answer a question."""

    if index.N == 0:
        return (
            False,
            0.0,
            "No course material has been uploaded yet.",
        )

    if not q_terms:
        return (
            False,
            0.0,
            "The question has no content words to search for.",
        )

    if not hits:
        return (
            False,
            0.0,
            "None of the question's key terms appear in your material.",
        )

    top_score = hits[0][1]

    hit_chunks = [chunk for chunk, _score in hits[:3]]

    coverage = index.weighted_coverage(
        q_terms,
        hit_chunks,
    )

    # For short questions, allow the attainable fraction of query terms.
    effective_min_cov = min(
        cfg["MIN_COVERAGE"],
        1.0 / len(q_terms),
    )

    if coverage < effective_min_cov:
        fuzzy = _stem_prefix_coverage(
            q_terms,
            hit_chunks,
            index,
        )

        if fuzzy > coverage:
            coverage = fuzzy

    if top_score < cfg["MIN_BM25_SCORE"]:
        return (
            False,
            coverage,
            "The best matching passage is only a weak match.",
        )

    if coverage < effective_min_cov:
        present = index.union_terms(hit_chunks)
        missing = [
            term for term in q_terms
            if term not in present
        ]

        reason = "Key parts of the question don't appear in your material"

        if missing:
            reason += f" (not found: {', '.join(missing[:4])})."
        else:
            reason += "."

        return False, coverage, reason

    return True, coverage, "ok"


def extractive_answer(
    index,
    q_terms,
    hits,
    max_sentences=3,
    min_share=0.3,
):
    """Select verbatim, relevant sentences from retrieved passages.

    Sentence selection considers query-term overlap and limited
    aliases for common morphological and wording variations.
    """

    q_mass = sum(index.idf(t) for t in q_terms) or 1.0

    candidates = []
    seen = set()

    for rank, (chunk, _score) in enumerate(hits[:3], start=1):
        for pos, sentence in enumerate(
            split_sentences(chunk["text"])
        ):
            key = sentence.lower()

            if key in seen or not is_full_sentence(sentence):
                continue

            seen.add(key)

            sentence_terms = set(terms(sentence))

            covered = {
                term
                for term in q_terms
                if (
                    term in sentence_terms
                    or bool(
                        _SENTENCE_ALIASES.get(term, set())
                        & sentence_terms
                    )
                )
            }

            weight = sum(
                index.idf(term)
                for term in covered
            )

            if weight / q_mass >= min_share:
                candidates.append({
                    "w": weight,
                    "rank": rank,
                    "pos": pos,
                    "sent": sentence,
                    "chunk": chunk,
                    "covered": covered,
                })

    if not candidates:
        return None, []

    candidates.sort(
        key=lambda c: (
            -c["w"],
            c["rank"],
            c["pos"],
        )
    )

    picked = [candidates[0]]
    covered_all = set(candidates[0]["covered"])
    remaining = candidates[1:]

    while remaining and len(picked) < max_sentences:

        def gain(candidate):
            new_terms = candidate["covered"] - covered_all
            return sum(
                index.idf(term)
                for term in new_terms
            )

        best = max(
            remaining,
            key=lambda c: (
                gain(c),
                c["w"],
                -c["rank"],
                -c["pos"],
            ),
        )

        if gain(best) <= 0:
            # Prefer sentences that add new evidence. If none do,
            # use the strongest remaining candidate.
            best = max(
                remaining,
                key=lambda c: (
                    c["w"],
                    -c["rank"],
                    -c["pos"],
                ),
            )

        picked.append(best)
        covered_all.update(best["covered"])
        remaining.remove(best)

    # Preserve source order within the retrieved chunks.
    picked.sort(
        key=lambda c: (
            c["rank"],
            c["pos"],
        )
    )

    used = {}
    lines = []

    for candidate in picked:
        lines.append(
            f"{candidate['sent']} [S{candidate['rank']}]"
        )
        used[candidate["rank"]] = candidate["chunk"]

    return "\n".join(lines), sorted(used)


SYSTEM_PROMPT = (
    "You are a careful course tutor. Answer the student's question "
    "using ONLY the numbered sources provided. "
    "Rules: (1) Do not use outside knowledge. "
    "(2) After every claim add the supporting source marker like [S1]; "
    "use only the numbers that exist. "
    "(3) If the sources do not contain enough information to answer, "
    'set "answerable" to false and leave "answer" empty. '
    "(4) Be concise (at most 5 sentences) and explain clearly. "
    'Respond with ONLY a JSON object: '
    '{"answerable": true|false, "answer": "..."} .'
)


def _build_prompt(question, hits):
    blocks = []

    for number, (chunk, _score) in enumerate(hits, start=1):
        blocks.append(
            f"[S{number}] ({chunk['filename']}, "
            f"{_loc_label(chunk)}): {chunk['text']}"
        )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "SOURCES:\n"
                + "\n\n".join(blocks)
                + f"\n\nQUESTION: {question}"
            ),
        },
    ]


def normalize_markers(answer: str, n_sources: int):
    """Normalize source markers and remove references to nonexistent sources."""

    removed = []
    valid = []

    def replace_marker(match):
        numbers = [
            int(number)
            for number in re.findall(r"\d+", match.group(1))
        ]

        output = []

        for number in numbers:
            if 1 <= number <= n_sources:
                output.append(f"[S{number}]")
                valid.append(number)
            else:
                removed.append(number)

        return "".join(output)

    text = _MARKER.sub(replace_marker, answer)

    text = re.sub(
        r"[ \t]+([.,;:])",
        r"\1",
        re.sub(r"[ \t]{2,}", " ", text),
    ).strip()

    return text, sorted(set(valid)), sorted(set(removed))

def has_citation_for_each_sentence(answer: str) -> bool:
    """Require every non-empty answer sentence to contain a source marker."""
    if not answer or not answer.strip():
        return False

    sentences = split_sentences(answer)

    if not sentences:
        return False

    for sentence in sentences:
        if not sentence.strip():
            continue

        if not _MARKER.search(sentence):
            return False

    return True



def grounding_overlap(answer_text: str, cited_chunks) -> float:
    """Measure overlap between answer terms and cited source terms."""

    answer_terms = set(
        terms(_MARKER.sub(" ", answer_text))
    )

    if not answer_terms:
        return 0.0

    source_terms = set()

    for chunk in cited_chunks:
        source_terms.update(terms(chunk["text"]))

    return len(answer_terms & source_terms) / len(answer_terms)


def _unsupported(
    question,
    hits,
    reason,
    cfg,
    mode,
    started,
    coverage=0.0,
    extra=None,
):
    """Build a refusal response without presenting nearest passages as citations."""

    result = {
        "status": "unsupported",
        "mode": mode,
        "question": question,
        "answer": "",
        "citations": [],
        "reason": reason,
        "support": {
            "coverage": round(coverage, 3),
            "top_score": round(hits[0][1], 3) if hits else 0.0,
        },
        "nearest": [
            _citation(number, chunk, score)
            for number, (chunk, score) in enumerate(
                hits[:2],
                start=1,
            )
        ],
        "elapsed_ms": int((time.time() - started) * 1000),
    }

    if extra:
        result.update(extra)

    return result


def answer_question(question, index, llm, cfg, doc_ids=None):
    """Answer a question using retrieved course material only."""

    started = time.time()
    question = (question or "").strip()

    q_terms = list(dict.fromkeys(terms(question)))

    hits = index.search(
        question,
        k=cfg["TOP_K"],
        doc_ids=doc_ids,
    )

    supported, coverage, reason = assess_support(
        index,
        q_terms,
        hits,
        cfg,
    )

    if not supported:
        return _unsupported(
            question,
            hits,
            reason,
            cfg,
            "gate",
            started,
            coverage,
        )

    # Reject unsupported abbreviation expansions before calling an LLM.
    if _is_definition_trap(question, hits):
        return _unsupported(
            question,
            hits,
            "Your notes mention this term but don't define or expand it; "
            "the answer would be unreliable.",
            cfg,
            "gate",
            started,
            coverage,
        )

    # Reject specific definitions when distinguishing concept terms
    # are absent from the retrieved passages.
    if _is_specific_definition_unsupported(question, hits):
        return _unsupported(
            question,
            hits,
            "The retrieved passages do not establish the specific "
            "concept asked about.",
            cfg,
            "gate",
            started,
            coverage,
        )

    llm_error = None

    if llm is not None and llm.configured:
        try:
            raw = llm.chat(
                _build_prompt(question, hits),
                max_tokens=700,
            )

            obj = parse_json_object(raw)

            if obj is None:
                raise LLMError(
                    "model did not return valid JSON"
                )

            if (
                not obj.get("answerable", False)
                or not str(obj.get("answer", "")).strip()
            ):
                return _unsupported(
                    question,
                    hits,
                    "The tutor model judged the retrieved passages "
                    "insufficient to answer.",
                    cfg,
                    "llm",
                    started,
                    coverage,
                )

            text, valid, removed = normalize_markers(
                str(obj["answer"]),
                len(hits),
            )
            
            

            if not valid:
                return _unsupported(
                    question,
                    hits,
                    "The model's answer did not cite any retrieved "
                    "source, so it was withheld.",
                    cfg,
                    "llm",
                    started,
                    coverage,
                    {
                        "removed_invalid_citations": removed,
                    },
                )

            cited = [
                hits[number - 1][0]
                for number in valid
            ]

            overlap = grounding_overlap(text, cited)

            return {
                "status": "answered",
                "mode": "llm",
                "question": question,
                "answer": text,
                "citations": [
                    _citation(
                        number,
                        hits[number - 1][0],
                        hits[number - 1][1],
                    )
                    for number in valid
                ],
                "support": {
                    "coverage": round(coverage, 3),
                    "top_score": round(hits[0][1], 3),
                    "grounding_overlap": round(overlap, 3),
                },
                "removed_invalid_citations": removed,
                "warnings": (
                    [
                        "Parts of this answer use wording not found "
                        "in the cited passages; check the excerpts."
                    ]
                    if overlap < 0.4
                    else []
                ),
                "elapsed_ms": int(
                    (time.time() - started) * 1000
                ),
            }

        except LLMError as error:
            llm_error = str(error)

    # Offline mode, or fallback when the LLM call fails.
    text, used = extractive_answer(
        index,
        q_terms,
        hits,
    )

    if not text:
        return _unsupported(
            question,
            hits,
            "No sentence in the retrieved passages addresses "
            "the question.",
            cfg,
            "extractive",
            started,
            coverage,
        )

    result = {
        "status": "answered",
        "mode": (
            "extractive"
            if llm_error is None
            else "extractive_fallback"
        ),
        "question": question,
        "answer": text,
        "citations": [
            _citation(
                number,
                hits[number - 1][0],
                hits[number - 1][1],
            )
            for number in used
        ],
        "support": {
            "coverage": round(coverage, 3),
            "top_score": round(hits[0][1], 3),
        },
        "removed_invalid_citations": [],
        "warnings": [],
        "elapsed_ms": int(
            (time.time() - started) * 1000
        ),
    }

    if llm_error:
        result["llm_error"] = llm_error

    return result