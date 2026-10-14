"""Document / chunk persistence and a cached retrieval index."""
import hashlib
from pathlib import Path

from .bm25 import BM25Index


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_document(conn, *, filename, kind, sha256, stored_name, n_units, n_empty, chunks) -> int:
    cur = conn.execute(
        "INSERT INTO documents (filename, kind, sha256, stored_name, n_units, n_empty) VALUES (?,?,?,?,?,?)",
        (filename, kind, sha256, stored_name, n_units, n_empty),
    )
    doc_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO chunks (doc_id, ord, loc_kind, loc_start, loc_end, text, topic) VALUES (?,?,?,?,?,?,?)",
        [(doc_id, i, c["loc_kind"], c["loc_start"], c["loc_end"], c["text"], c.get("topic", "")) for i, c in enumerate(chunks)],
    )
    conn.commit()
    return doc_id


def add_figures(conn, doc_id, figures, fig_dir: Path):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(figures):
        name = f"{doc_id}_{i}.{f['ext']}"
        (fig_dir / name).write_bytes(f["blob"])
        conn.execute(
            "INSERT INTO figures (doc_id, loc_kind, loc_start, stored_name, mime, width, height, alt_text, caption) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, f["loc_kind"], f["loc_start"], name, f["mime"], f.get("width"), f.get("height"), f.get("alt_text", ""), f.get("caption", "")))
    conn.commit()


def figures_at(conn, doc_id, loc_kind, loc_start):
    rows = conn.execute("SELECT id, alt_text, caption, width, height FROM figures WHERE doc_id=? AND loc_kind=? AND loc_start=? ORDER BY id",
                        (doc_id, loc_kind, loc_start)).fetchall()
    return [{"id": r["id"], "url": f"/figures/{r['id']}", "alt_text": r["alt_text"], "caption": r["caption"],
             "width": r["width"], "height": r["height"]} for r in rows]


def find_by_sha(conn, sha):
    return conn.execute("SELECT * FROM documents WHERE sha256=?", (sha,)).fetchone()


def list_documents(conn):
    rows = conn.execute(
        "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id) AS n_chunks, "
        "(SELECT COUNT(*) FROM figures f WHERE f.doc_id=d.id) AS n_figures "
        "FROM documents d ORDER BY d.id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_document(conn, doc_id):
    r = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    return dict(r) if r else None


def delete_document(conn, doc_id, upload_dir: Path, fig_dir: Path | None = None):
    doc = get_document(conn, doc_id)
    if not doc:
        return False
    fig_names = [r["stored_name"] for r in conn.execute("SELECT stored_name FROM figures WHERE doc_id=?", (doc_id,))]
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    p = Path(upload_dir) / doc["stored_name"]
    if p.exists():
        p.unlink()
    for n in fig_names:
        if fig_dir and (Path(fig_dir) / n).exists():
            (Path(fig_dir) / n).unlink()
    return True


class Retriever:
    """Rebuilds the BM25 index only when the chunk table changes."""

    def __init__(self):
        self._index = None
        self._version = None

    def index(self, conn) -> BM25Index:
        row = conn.execute("SELECT COUNT(*) n, COALESCE(MAX(id),0) m, COALESCE(SUM(id),0) s FROM chunks").fetchone()
        version = (row["n"], row["m"], row["s"])
        if self._index is None or version != self._version:
            rows = conn.execute(
                "SELECT c.id, c.doc_id, c.ord, c.loc_kind, c.loc_start, c.loc_end, c.text, c.topic, d.filename "
                "FROM chunks c JOIN documents d ON d.id=c.doc_id ORDER BY c.id"
            ).fetchall()
            self._index = BM25Index([dict(r) for r in rows])
            self._version = version
        return self._index
