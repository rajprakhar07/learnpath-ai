"""SQLite access. Single-file DB, no ORM."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- pdf | pptx | transcript
    sha256      TEXT NOT NULL UNIQUE,
    stored_name TEXT NOT NULL,
    n_units     INTEGER NOT NULL,           -- pages / slides / segments
    n_empty     INTEGER NOT NULL DEFAULT 0, -- units with no extractable text
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,
    loc_kind   TEXT NOT NULL,               -- page | slide | time
    loc_start  REAL NOT NULL,               -- page/slide number, or seconds
    loc_end    REAL NOT NULL,
    text       TEXT NOT NULL,
    topic      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS figures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    loc_kind    TEXT NOT NULL,
    loc_start   REAL NOT NULL,
    stored_name TEXT NOT NULL,
    mime        TEXT NOT NULL,
    width       INTEGER, height INTEGER,
    alt_text    TEXT NOT NULL DEFAULT '',
    caption     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fig_doc ON figures(doc_id, loc_kind, loc_start);

CREATE TABLE IF NOT EXISTS questions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id           TEXT NOT NULL,
    doc_id               INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id             INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    loc_kind             TEXT NOT NULL,
    loc_start            REAL NOT NULL,
    loc_end              REAL NOT NULL,
    topic                TEXT NOT NULL,
    concept              TEXT NOT NULL,
    concept_key          TEXT NOT NULL,
    difficulty           INTEGER NOT NULL CHECK (difficulty IN (1,2,3)),  -- realized, measured
    requested_difficulty INTEGER NOT NULL,
    qtype                TEXT NOT NULL,         -- cloze_mcq | llm_mcq
    generator            TEXT NOT NULL,         -- template | llm
    stem                 TEXT NOT NULL,
    options_json         TEXT NOT NULL,         -- [{text, chunk_id, quote}]
    answer_index         INTEGER NOT NULL,
    source_quote         TEXT NOT NULL,         -- verbatim sentence from the chunk
    fingerprint          TEXT NOT NULL,
    quote_hash           TEXT NOT NULL,
    validation_json      TEXT NOT NULL,
    selection_reason     TEXT NOT NULL DEFAULT '',
    purpose              TEXT NOT NULL DEFAULT 'practice',   -- practice | retest
    status               TEXT NOT NULL DEFAULT 'pending',    -- pending | answered | skipped
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (learner_id, fingerprint),
    UNIQUE (learner_id, quote_hash)
);
CREATE TABLE IF NOT EXISTS attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id   TEXT NOT NULL,
    question_id  INTEGER NOT NULL UNIQUE REFERENCES questions(id) ON DELETE CASCADE,
    chosen_index INTEGER NOT NULL,
    correct      INTEGER NOT NULL,
    difficulty   INTEGER NOT NULL,
    topic        TEXT NOT NULL,
    concept_key  TEXT NOT NULL,
    before_json  TEXT NOT NULL,
    after_json   TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS topic_mastery (
    learner_id TEXT NOT NULL,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    topic      TEXT NOT NULL,
    a REAL NOT NULL, b REAL NOT NULL, n INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (learner_id, doc_id, topic)
);
CREATE TABLE IF NOT EXISTS mastery_loops (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id         TEXT NOT NULL,
    doc_id             INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    concept_key        TEXT NOT NULL,
    concept            TEXT NOT NULL,
    topic              TEXT NOT NULL,
    status             TEXT NOT NULL,           -- revising | retest_pending | complete
    diagnosis_json     TEXT NOT NULL,
    revision_json      TEXT NOT NULL,
    retest_question_id INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    result_json        TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS concept_mastery (
    learner_id  TEXT NOT NULL,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    topic       TEXT NOT NULL,
    concept_key TEXT NOT NULL,
    concept     TEXT NOT NULL,
    a REAL NOT NULL, b REAL NOT NULL, n INTEGER NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (learner_id, doc_id, concept_key)
);
"""


def connect(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_columns(conn):
    """Tiny forward-only migration so DBs created by earlier stages keep working."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)")}
    if "topic" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN topic TEXT NOT NULL DEFAULT ''")


def init_db(path):
    conn = connect(path)
    conn.executescript(SCHEMA)
    _ensure_columns(conn)
    conn.commit()
    conn.close()
