
"""
Benchmark / evaluation pipeline.

    python -m eval.run_benchmark                 # offline (extractive) mode
    python -m eval.run_benchmark --llm           # also evaluate configured LLM
    python -m eval.run_benchmark --skip-simulation
    python -m eval.run_benchmark --out /path/to/report

Measures:
1. Retrieval, answerability gate, and expected-location citation metrics.
2. Question-generation structural and source-location audit.
3. Adaptive-policy simulation using a SYNTHETIC learner, not student data.

Writes report.json, report.md, and review_sample.csv to eval/results/<UTC timestamp>/.

Important:
- Expected-location matching does not prove semantic support for every claim.
- Dev results may be optimistic if used during tuning.
- Once heldout influences tuning, it must be treated as historical, not untouched.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from learnpath import assessment, create_app
from learnpath.db import connect
from learnpath.questions import validate_question
from learnpath.tutor import answer_question

from .metrics import any_match, mrr, prf, rate


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [
    "sample_course.pdf",
    "sample_deck.pptx",
    "sample_lecture.vtt",
    "sample_figure.pdf",
]


def norm(s):
    """Normalize text for source-location comparison."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def load_jsonl(path):
    """Load JSON Lines data."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha(path):
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_app(tmp, llm=None):
    """Create an isolated app and ingest the sample corpus."""
    overrides = {"DATA_DIR": Path(tmp)}

    if llm is None:
        overrides.update(
            {
                "LLM_API_KEY": "",
                "LLM_BASE_URL": "",
                "LLM_MODEL": "",
            }
        )

    app = create_app(overrides)
    client = app.test_client()

    response = client.post(
        "/api/documents/sample",
        json={"names": SAMPLES},
    )
    assert response.status_code in (200, 201), response.get_json()

    return app, client


# ---------------------------------------------------------------------------
# 1. Retrieval, answerability gate, and citation evaluation
# ---------------------------------------------------------------------------

def eval_questions(app, items, use_llm=False):
    """Evaluate each question and record retrieval and citation outcomes."""
    config = app.config
    conn = connect(config["DB_PATH"])
    index = app.extensions["retriever"].index(conn)
    llm = app.extensions["llm"] if use_llm else None

    records = []

    try:
        for item in items:
            result = answer_question(
                item["question"],
                index,
                llm,
                config,
            )

            hits = index.search(
                item["question"],
                k=config["TOP_K"],
            )

            retrieval_rank = next(
                (
                    rank
                    for rank, (chunk, _score) in enumerate(hits, start=1)
                    if any_match(
                        {
                            "doc_name": chunk["filename"],
                            "loc_kind": chunk["loc_kind"],
                            "loc_start": chunk["loc_start"],
                            "loc_end": chunk["loc_end"],
                        },
                        item["expected"],
                    )
                ),
                None,
            )

            citations = result.get("citations", [])
            citation_matches = [
                any_match(citation, item["expected"])
                for citation in citations
            ]

            records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answerable": item["answerable"],
                    "note": item.get("note", ""),
                    "status": result["status"],
                    "mode": result.get("mode", ""),
                    "reason": result.get("reason", ""),
                    "cited": [
                        f"{citation.get('doc_name', 'unknown')} "
                        f"{citation.get('label', '')}"
                        for citation in citations
                    ],
                    "cited_match": citation_matches,
                    "retrieval_rank": retrieval_rank,
                    "grounding_overlap": result.get("support", {}).get(
                        "grounding_overlap"
                    ),
                }
            )
    finally:
        conn.close()

    return records


def summarize_questions(records):
    """
    Summarize gate, citation, and retrieval metrics.

    Citation metrics use answerable questions that were actually answered.
    Citation precision uses individual citations as its denominator.
    """
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    tp = sum(1 for r in answerable if r["status"] == "answered")
    fn = len(answerable) - tp
    fp = sum(1 for r in unanswerable if r["status"] == "answered")
    tn = len(unanswerable) - fp

    answered = [
        r for r in answerable
        if r["status"] == "answered"
    ]

    all_citation_flags = [
        flag
        for record in answered
        for flag in record["cited_match"]
    ]

    
    first_correct = sum(
        1
        for record in answerable
        if record["status"] == "answered"
        and record["cited_match"]
        and record["cited_match"][0]
    )

    any_correct = sum(
        1
        for record in answerable
        if record["status"] == "answered"
        and any(record["cited_match"])
    )


    good_citations = sum(all_citation_flags)

    failures = [
        {
            "id": record["id"],
            "question": record["question"],
            "answerable": record["answerable"],
            "status": record["status"],
            "cited": record["cited"],
            "retrieval_rank": record["retrieval_rank"],
            "reason": record["reason"],
            "note": record["note"],
        }
        for record in records
        if (
            record["answerable"]
            and (
                record["status"] != "answered"
                or not any(record["cited_match"])
            )
        )
        or (
            not record["answerable"]
            and record["status"] == "answered"
        )
    ]

    return {
        "counts": {
            "total": len(records),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },
        "gate": {
            "confusion": {
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            },
            **prf(tp, fp, fn),
            "false_refusal_rate": rate(fn, len(answerable)),
            "correct_refusal_rate": rate(tn, len(unanswerable)),
            "false_answer_rate": rate(fp, len(unanswerable)),
        },
        "citations": {
            "answered_answerable_n": len(answered),
            "citation_count": len(all_citation_flags),
            "first_citation_correct": rate(
                first_correct,
                len(answerable),
            ),
            "any_citation_correct": rate(
                any_correct,
                len(answered),
            ),
            "citation_precision": rate(
                good_citations,
                len(all_citation_flags),
            ),
            "metric_definition": (
                "Expected-location match, not semantic verification "
                "that every answer claim is supported."
            ),
        },
        "retrieval_independent_of_gate": {
            "hit_at_1": rate(
                sum(
                    1 for record in answerable
                    if record["retrieval_rank"] == 1
                ),
                len(answerable),
            ),
            "hit_at_k": rate(
                sum(
                    1 for record in answerable
                    if record["retrieval_rank"] is not None
                ),
                len(answerable),
            ),
            "mrr": mrr(
                [record["retrieval_rank"] for record in answerable]
            ),
        },
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# 2. Question-generation audit
# ---------------------------------------------------------------------------

def _vtt_cues(path):
    """Extract cue start times and their following text lines."""
    output = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines):
        match = re.match(
            r"(\d\d):(\d\d):(\d\d)\.\d+\s+-->\s+"
            r"(\d\d):(\d\d):(\d\d)\.\d+",
            line,
        )

        if match and i + 1 < len(lines):
            h1, m1, s1, _h2, _m2, _s2 = map(int, match.groups())
            start = h1 * 3600 + m1 * 60 + s1
            output.append((start, lines[i + 1]))

    return output


def independent_location_text(
    upload_dir,
    doc,
    loc_kind,
    loc_start,
    loc_end,
):
    """
    Re-derive location text using an extractor separate from ingestion.

    Returns None when independent verification is unavailable.
    """
    path = Path(upload_dir) / doc["stored_name"]

    if loc_kind == "page":
        if not shutil.which("pdftotext"):
            return None

        return subprocess.check_output(
            [
                "pdftotext",
                "-f",
                str(int(loc_start)),
                "-l",
                str(int(loc_start)),
                str(path),
                "-",
            ],
            text=True,
        )

    if loc_kind == "slide":
        from pptx import Presentation

        presentation = Presentation(str(path))
        slide = presentation.slides[int(loc_start) - 1]
        parts = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)

            if getattr(shape, "has_table", False) and shape.has_table:
                parts.extend(
                    cell.text
                    for row in shape.table.rows
                    for cell in row.cells
                )

        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)

        return " ".join(parts)

    if loc_kind == "time":
        return " ".join(
            text
            for start, text in _vtt_cues(path)
            if loc_start <= start <= loc_end
        )

    return None


def audit_generation(app):
    """Audit generated question structure, duplication, and source fidelity."""
    client = app.test_client()
    conn = connect(app.config["DB_PATH"])

    n = 0

    try:
        while n < 500:
            response = client.post(
                "/api/quiz/next",
                json={},
            ).get_json()

            if response["status"] != "ok":
                break

            question_id = response["question"]["question_id"]
            row = conn.execute(
                "SELECT answer_index FROM questions WHERE id = ?",
                (question_id,),
            ).fetchone()

            if row is None:
                break

            client.post(
                "/api/quiz/answer",
                json={
                    "question_id": question_id,
                    "choice": row["answer_index"],
                },
            )
            n += 1

        questions = [
            dict(row)
            for row in conn.execute("SELECT * FROM questions")
        ]
        chunks = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM chunks")
        }
        documents = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM documents")
        }

        valid = 0
        verified = 0
        checked = 0

        for question in questions:
            chunk = chunks.get(question["chunk_id"])
            if chunk is None:
                continue

            dto = dict(
                question,
                options=json.loads(question["options_json"]),
            )
            is_valid, _reason = validate_question(dto, chunk["text"])
            valid += bool(is_valid)

            document = documents.get(question["doc_id"])
            if document is None:
                continue

            text = independent_location_text(
                app.config["UPLOAD_DIR"],
                document,
                question["loc_kind"],
                question["loc_start"],
                question["loc_end"],
            )

            if text is not None:
                checked += 1
                verified += (
                    norm(question["source_quote"]) in norm(text)
                )

        audit = {
            "n_questions": len(questions),
            "stopped_because": (
                "engine reported no further distinct questions"
                if n < 500
                else "safety cap"
            ),
            "revalidation_pass": rate(valid, len(questions)),
            "independent_location_verified": rate(verified, checked),
            "duplicate_fingerprints": (
                len(questions)
                - len({q["fingerprint"] for q in questions})
            ),
            "duplicate_source_sentences": (
                len(questions)
                - len({q["quote_hash"] for q in questions})
            ),
            "by_location_kind": dict(
                Counter(q["loc_kind"] for q in questions)
            ),
            "by_difficulty": {
                str(key): value
                for key, value in sorted(
                    Counter(q["difficulty"] for q in questions).items()
                )
            },
            "by_generator": dict(
                Counter(q["generator"] for q in questions)
            ),
            "distinct_topics": len(
                {(q["doc_id"], q["topic"]) for q in questions}
            ),
            "caveat": (
                "These checks measure structural validity and source "
                "fidelity, not pedagogical quality. Review review_sample.csv "
                "manually before claiming question quality."
            ),
        }

        rows = []
        for question in questions:
            options = json.loads(question["options_json"])
            document = documents.get(question["doc_id"], {})
            rows.append(
                {
                    "id": question["id"],
                    "location": (
                        f"{document.get('filename', 'unknown')} "
                        f"{assessment.label_of(question)}"
                    ),
                    "difficulty": question["difficulty"],
                    "stem": question["stem"],
                    "options": " | ".join(
                        option["text"] for option in options
                    ),
                    "key": options[question["answer_index"]]["text"],
                }
            )

        random.Random(7).shuffle(rows)
        return audit, rows[:20]

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. Synthetic adaptive-policy simulation
# ---------------------------------------------------------------------------

def simulate_adaptive(
    seeds=10,
    steps=30,
    p_known=0.9,
    p_weak=0.25,
):
    """
    Simulate a synthetic learner.

    v1 is the original post-coverage weak-topic share. It is confounded by
    the finite question-bank capacity.

    v2 is a post-hoc exploratory metric measuring how often the weak topic
    appears during a window where weak-topic questions remain.
    It is not independent evidence of real learning outcomes.
    """
    results = []

    for seed in range(seeds):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                {
                    "DATA_DIR": Path(tmp),
                    "LLM_API_KEY": "",
                    "LLM_BASE_URL": "",
                    "LLM_MODEL": "",
                }
            )
            client = app.test_client()
            client.post(
                "/api/documents/sample",
                json={"names": ["sample_course.pdf"]},
            )

            conn = connect(app.config["DB_PATH"])

            try:
                topics = [
                    row[0]
                    for row in conn.execute(
                        "SELECT topic FROM chunks "
                        "GROUP BY topic ORDER BY MIN(id)"
                    )
                ]

                if not topics:
                    continue

                rng = random.Random(seed)
                weak_topic = rng.choice(topics)
                asked = []

                def step(correct_fn):
                    response = client.post(
                        "/api/quiz/next",
                        json={},
                    ).get_json()

                    if response["status"] != "ok":
                        return None

                    question_id = response["question"]["question_id"]
                    row = conn.execute(
                        "SELECT topic, answer_index FROM questions WHERE id = ?",
                        (question_id,),
                    ).fetchone()

                    if row is None:
                        return None

                    is_correct = correct_fn(row["topic"])
                    choice = (
                        row["answer_index"]
                        if is_correct
                        else (row["answer_index"] + 1) % 4
                    )

                    client.post(
                        "/api/quiz/answer",
                        json={
                            "question_id": question_id,
                            "choice": choice,
                        },
                    )
                    return row["topic"]

                for _ in range(steps):
                    topic = step(
                        lambda current_topic: (
                            rng.random()
                            < (
                                p_weak
                                if current_topic == weak_topic
                                else p_known
                            )
                        )
                    )

                    if topic is None:
                        break

                    asked.append(topic)

                # Drain the bank; these answers are excluded from measurement.
                while step(lambda _topic: True) is not None:
                    pass

                capacity = Counter(
                    row[0]
                    for row in conn.execute(
                        "SELECT topic FROM questions"
                    )
                )

                topic_count = len(topics)
                post_coverage = asked[topic_count:]
                weak_in_coverage = sum(
                    topic == weak_topic
                    for topic in asked[:topic_count]
                )

                weak_remaining = max(
                    0,
                    capacity[weak_topic] - weak_in_coverage,
                )
                total_remaining = max(
                    0,
                    sum(capacity.values()) - topic_count,
                )

                window = min(
                    weak_remaining,
                    len(post_coverage),
                )

                observed = (
                    sum(
                        topic == weak_topic
                        for topic in post_coverage[:window]
                    ) / window
                    if window
                    else None
                )

                baseline = (
                    weak_remaining / total_remaining
                    if total_remaining
                    else None
                )

                results.append(
                    {
                        "seed": seed,
                        "weak_topic": weak_topic,
                        "bank_capacity_weak": capacity[weak_topic],
                        "bank_total": sum(capacity.values()),
                        "n_post_coverage": len(post_coverage),
                        "v1_weak_share_post_coverage": (
                            sum(
                                topic == weak_topic
                                for topic in post_coverage
                            ) / len(post_coverage)
                            if post_coverage
                            else None
                        ),
                        "v2_m": window,
                        "v2_observed": observed,
                        "v2_random_baseline": baseline,
                    }
                )

            finally:
                conn.close()

    def mean(key):
        values = [
            result[key]
            for result in results
            if result[key] is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "SYNTHETIC": (
            "Simulated learner (P(correct)=%.2f on known topics, "
            "%.2f on one weak topic). Not student data."
        ) % (p_known, p_weak),
        "seeds": seeds,
        "steps": steps,
        "chance_share_if_uniform": round(1 / 6, 3),
        "v1_mean_weak_share": mean("v1_weak_share_post_coverage"),
        "v2_posthoc_note": (
            "v2 was defined after v1 appeared close to chance. "
            "The original metric is constrained by question-bank capacity."
        ),
        "v2_mean_observed": mean("v2_observed"),
        "v2_mean_random_baseline": mean("v2_random_baseline"),
        "per_seed": results,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def pct(value):
    return "n/a" if value is None else f"{100 * value:.0f}%"


def rate_str(result):
    """Format a rate() result, including empty denominators."""
    if result is None or result.get("n", 0) == 0:
        return "n/a (n=0)"

    value = pct(result.get("value"))
    k = result.get("k", 0)
    n = result.get("n", 0)
    ci = result.get("ci95")

    if not ci or len(ci) != 2:
        return f"{value} ({k}/{n}; CI unavailable)"

    return (
        f"{value} ({k}/{n}, 95% CI "
        f"{pct(ci[0])}–{pct(ci[1])})"
    )


def render_md(report):
    """Render a Markdown report from measured results."""
    lines = [
        "# LearnPath benchmark report",
        "",
        (
            f"Run: {report['meta']['timestamp_utc']}  |  "
            f"mode: **{report['meta']['mode']}**  |  "
            f"Python {report['meta']['python']}"
        ),
        "",
        (
            "> Read the caveats at the bottom before quoting any number. "
            "Datasets are small and confidence intervals may be wide."
        ),
        "",
    ]

    for name in ("dev", "heldout", "independent"):
        summary = report["datasets"].get(name)
        if not summary:
            continue

        if name == "dev":
            tag = "Development set; potentially optimistic after tuning"
        elif name == "heldout":
            tag = "Historical held-out set; do not call untouched after tuning"
        else:
            tag = "Independent evaluation set; check its provenance in the report"

        gate = summary["gate"]
        citations = summary["citations"]
        retrieval = summary["retrieval_independent_of_gate"]

        lines += [
            f"## {name} ({tag})",
            (
                f"{summary['counts']['answerable']} answerable + "
                f"{summary['counts']['unanswerable']} unanswerable questions."
            ),
            "",
            "| Metric | Result |",
            "|---|---|",
            (
                "| Answered when answerable (1 - false refusal) | "
                f"{rate_str(rate(gate['confusion']['tp'], summary['counts']['answerable']))} |"
            ),
            (
                "| Refused when unanswerable | "
                f"{rate_str(gate['correct_refusal_rate'])} |"
            ),
            (
                "| Gate precision / recall / F1 | "
                f"{pct(gate.get('precision'))} / "
                f"{pct(gate.get('recall'))} / "
                f"{pct(gate.get('f1'))} |"
            ),
            (
                "| First citation at expected location | "
                f"{rate_str(citations['first_citation_correct'])} |"
            ),
            (
                "| Any citation at expected location | "
                f"{rate_str(citations['any_citation_correct'])} |"
            ),
            (
                "| Citation precision (expected-location match) | "
                f"{rate_str(citations['citation_precision'])} |"
            ),
            (
                "| Raw retrieval hit@1 / hit@k (ignores gate) | "
                f"{pct(retrieval['hit_at_1'].get('value'))} / "
                f"{pct(retrieval['hit_at_k'].get('value'))} |"
            ),
            (
                "| Raw retrieval MRR | "
                f"{'n/a' if retrieval['mrr'] is None else f'{retrieval['mrr']:.3f}'} |"
            ),
            "",
            (
                "Citation note: these metrics check expected source locations; "
                "they do not establish semantic support for every answer claim."
            ),
            "",
        ]

        if summary["failures"]:
            lines += ["Failures:", ""]

            for failure in summary["failures"]:
                if not failure["answerable"]:
                    kind = "should have refused"
                elif failure["status"] != "answered":
                    kind = "refused an answerable question"
                else:
                    kind = "no citation at expected location"

                note = (
                    f" ({failure['note']})"
                    if failure.get("note")
                    else ""
                )

                lines.append(
                    f"- `{failure['id']}` {kind}: "
                    f"“{failure['question']}” "
                    f"cited={failure['cited']} "
                    f"rank={failure['retrieval_rank']}{note}"
                )

            lines.append("")

    audit = report.get("generation_audit")
    if audit:
        lines += [
            "## Question-generation audit",
            "",
            f"- Questions generated: **{audit['n_questions']}** "
            f"({audit['stopped_because']})",
            f"- Re-validation pass: {rate_str(audit['revalidation_pass'])}",
            (
                "- Source location independently verified: "
                f"{rate_str(audit['independent_location_verified'])}"
            ),
            (
                f"- Duplicate fingerprints: {audit['duplicate_fingerprints']}; "
                "duplicate source sentences: "
                f"{audit['duplicate_source_sentences']}"
            ),
            (
                f"- By location kind: {audit['by_location_kind']}; "
                f"by difficulty: {audit['by_difficulty']}; "
                f"distinct topics: {audit['distinct_topics']}"
            ),
            f"- Caveat: {audit['caveat']}",
            "",
        ]

    simulation = report.get("adaptive_simulation")
    if simulation:
        lines += [
            "## Adaptive policy under a synthetic learner",
            "",
            f"**{simulation['SYNTHETIC']}**",
            "",
            (
                f"- v1 weak-topic share after coverage: "
                f"**{pct(simulation['v1_mean_weak_share'])}** "
                f"over {simulation['seeds']} seeds; uniform chance is "
                f"{pct(simulation['chance_share_if_uniform'])}. "
                "This metric is constrained by finite question-bank capacity."
            ),
            (
                f"- v2 post-hoc observed share: "
                f"**{pct(simulation['v2_mean_observed'])}** vs "
                f"**{pct(simulation['v2_mean_random_baseline'])}** "
                "random-order baseline."
            ),
            f"- Note: {simulation['v2_posthoc_note']}",
            "",
        ]

    lines += [
        "## LLM mode",
        "",
        report["llm_mode"]["note"],
        "",
        "## Caveats",
        "",
    ]
    lines.extend(
        f"- {caveat}"
        for caveat in report["caveats"]
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also evaluate the configured LLM; requires LLM environment variables.",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip the synthetic adaptive-policy simulation.",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.out)
        if args.out
        else ROOT / "eval" / "results" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = ROOT / "eval" / "datasets"

    dataset_hashes = {
        path.name: sha(path)
        for path in sorted(dataset_dir.glob("*.jsonl"))
    }

    corpus_hashes = {}
    for name in SAMPLES:
        path = ROOT / "sample_data" / name
        if path.is_file():
            corpus_hashes[name] = sha(path)
        else:
            corpus_hashes[name] = None

    report = {
        "meta": {
            "timestamp_utc": timestamp,
            "python": platform.python_version(),
            "mode": "offline extractive",
            "dataset_sha256": dataset_hashes,
            "corpus_sha256": corpus_hashes,
        },
        "datasets": {},
        "records": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        app, _client = fresh_app(tmp)
        report["meta"]["thresholds"] = {
            key: app.config[key]
            for key in (
                "TOP_K",
                "MIN_COVERAGE",
                "MIN_BM25_SCORE",
            )
            if key in app.config
        }

        for name in ("dev", "heldout"):
            path = dataset_dir / f"{name}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required evaluation dataset not found: {path}"
                )

            items = load_jsonl(path)
            records = eval_questions(app, items)

            report["records"][name] = records
            report["datasets"][name] = summarize_questions(records)

    with tempfile.TemporaryDirectory() as tmp:
        app, _client = fresh_app(tmp)
        report["generation_audit"], sample_rows = audit_generation(app)

    if not args.skip_simulation:
        report["adaptive_simulation"] = simulate_adaptive()

    llm_note = (
        "Not run: --llm was not requested. No LLM metrics are reported."
    )

    if args.llm:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app({"DATA_DIR": Path(tmp)})

            if not app.extensions["llm"].configured:
                llm_note = (
                    "Not run: --llm was passed, but the required "
                    "LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL settings "
                    "are not all configured."
                )
            else:
                client = app.test_client()
                response = client.post(
                    "/api/documents/sample",
                    json={"names": SAMPLES},
                )
                assert response.status_code in (200, 201), response.get_json()

                heldout_path = dataset_dir / "heldout.jsonl"
                records = eval_questions(
                    app,
                    load_jsonl(heldout_path),
                    use_llm=True,
                )

                report["datasets"]["heldout_llm"] = summarize_questions(records)
                report["records"]["heldout_llm"] = records

                overlaps = [
                    record["grounding_overlap"]
                    for record in records
                    if (
                        record["status"] == "answered"
                        and record["grounding_overlap"] is not None
                    )
                ]

                model = app.extensions["llm"].model
                report["meta"]["mode"] = (
                    f"offline extractive + LLM ({model})"
                )

                if overlaps:
                    llm_note = (
                        f"Ran heldout with model {model}. Mean grounding "
                        f"overlap (a lexical heuristic, not a faithfulness "
                        f"judgment) over {len(overlaps)} answered items: "
                        f"{sum(overlaps) / len(overlaps):.2f}."
                    )
                else:
                    llm_note = "LLM run produced no answered items."

    report["llm_mode"] = {"note": llm_note}

    report["caveats"] = [
        (
            "Datasets are small and project-authored. Confidence intervals "
            "are uncertain with small samples."
        ),
        (
            "The corpus contains four short sample files and may be easier "
            "than real course material."
        ),
        (
            "Development results can be optimistic because they may guide "
            "design decisions."
        ),
        (
            "Treat heldout as historical if its questions or results inform "
            "tuning. Do not continue describing it as untouched."
        ),
        (
            "Dataset hashes are recorded for reproducibility; a hash alone "
            "does not prove a dataset was created independently of tuning."
        ),
        (
            "Citation metrics measure expected source-location matching, "
            "not semantic support for each answer claim."
        ),
        (
            "Retrieval uses the project's configured retrieval implementation. "
            "Lexical retrieval can miss paraphrases without shared vocabulary."
        ),
        (
            "Question-generation checks measure structural validity and source "
            "fidelity, not pedagogical quality or learning outcomes."
        ),
        (
            "Adaptive simulation uses a synthetic learner and says nothing "
            "about real students' learning outcomes."
        ),
        (
            "The v2 adaptive metric is post-hoc and exploratory, not an "
            "independent confirmatory evaluation."
        ),
    ]

    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    review_csv = output_dir / "review_sample.csv"

    report_json.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    report_md.write_text(
        render_md(report),
        encoding="utf-8",
    )

    fieldnames = [
        "id",
        "location",
        "difficulty",
        "stem",
        "options",
        "key",
        "rating_1to5_clear_and_fair",
        "reviewer_notes",
    ]

    with review_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in sample_rows:
            writer.writerow(
                {
                    **row,
                    "rating_1to5_clear_and_fair": "",
                    "reviewer_notes": "",
                }
            )

    print(f"Wrote {report_md}")
    print(f"Wrote {report_json}")
    print(f"Wrote {review_csv}")

    return output_dir


if __name__ == "__main__":
    main()
