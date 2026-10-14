"""PDF text extraction that preserves page numbers, plus page-bounded chunking.

Chunks never cross page boundaries, so a chunk's page number is exact.
"""
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .textutil import split_sentences


class IngestError(Exception):
    """Raised for user-facing ingestion problems (bad file, no text, ...)."""


def clean_text(text: str) -> str:
    """Collapse whitespace but keep a "\n" after heading-like lines.

    Heading heuristic: a line with no terminal punctuation that is clearly shorter (<60%) than the
    longest line on the page. Body lines wrap near full width, and paragraph-final lines end in
    punctuation, so this rarely fires mid-sentence.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)  # de-hyphenate line-wrapped words
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return ""
    longest = max(len(l) for l in lines)
    parts = []
    for i, l in enumerate(lines):
        parts.append(l)
        numbered = bool(re.match(r"^\d+(\.\d+)*[.)]?\s+[A-Z]", l)) and len(l) < 0.8 * longest
        short_plain = (not l.endswith((".", "!", ":", ";", ","))) and len(l) < 0.6 * longest
        is_heading = i < len(lines) - 1 and (short_plain or (numbered and not l.endswith(".")))
        parts.append("\n" if is_heading else " ")
    return "".join(parts).strip()


def _join(sentences):
    out = ""
    for s in sentences:
        if out:
            out += " " if out.rstrip().endswith((".", "!", "?")) else "\n"
        out += s
    return out


def extract_pages(path):
    """Return [{'page': 1-based int, 'text': str}] for every page (text may be '')."""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise IngestError("This PDF is password-protected. Remove the password and try again.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            pages.append({"page": i, "text": clean_text(raw)})
        return pages
    except IngestError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError) as e:
        raise IngestError(f"Could not read this PDF ({type(e).__name__}). Is it a valid, uncorrupted PDF?") from e


_NUM_PREFIX = re.compile(r"^\d+(?:\.\d+)*[.)]?\s*")


def clean_heading(h: str) -> str:
    h = _NUM_PREFIX.sub("", h.strip()).strip(" .:;?")
    return re.sub(r"\s+", " ", h)


def page_sentences(page_text: str):
    """-> list of (sentence, heading_or_None). A heading is the last sentence before a "\n" break
    (see clean_text); consecutive heading-only lines (wrapped headings) are merged into one heading."""
    out, pending = [], None  # pending = heading text being accumulated
    segments = page_text.split("\n")
    for si, seg in enumerate(segments):
        sents = split_sentences(seg)
        is_break = si < len(segments) - 1
        if is_break and sents:
            head, body = sents[-1], sents[:-1]
            if body:  # heading follows body text in the same segment
                if pending:
                    out.append((pending, pending))
                    pending = None
                out.extend((x, None) for x in body)
                pending = head
            else:  # heading-only line: merge with a directly preceding heading-only line
                pending = f"{pending} {head}" if pending else head
        else:
            if pending:
                out.append((pending, pending))
                pending = None
            out.extend((x, None) for x in sents)
    if pending:
        out.append((pending, pending))
    return out


def chunk_page(page_text: str, current_topic: str, target_words: int = 110, overlap_sentences: int = 1):
    """Pack sentences into ~target_words chunks. Returns ([(text, topic)], topic_at_end_of_page)."""
    items = []  # (sentence, topic_in_effect, is_heading)
    topic = current_topic
    for sent, head in page_sentences(page_text):
        if head is not None:
            topic = clean_heading(head) or topic
        words = sent.split()
        if len(words) > target_words * 2:  # very long "sentence" (tables, lists): window it
            for i in range(0, len(words), target_words):
                items.append((" ".join(words[i : i + target_words]), topic, False))
        else:
            items.append((sent, topic, head is not None))
    chunks, cur, cur_words, fresh = [], [], 0, 0

    def flush():
        # topic = topic in effect at the first prose sentence (heading sentences don't count)
        body = [c for c in cur if not c[2]] or cur
        chunks.append((_join([c[0] for c in cur]), body[0][1]))

    for item in items:
        cur.append(item)
        cur_words += len(item[0].split())
        fresh += 1
        if cur_words >= target_words:
            flush()
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_words = sum(len(x[0].split()) for x in cur)
            fresh = 0
    if cur and fresh > 0:
        flush()
    return chunks, topic


def chunk_text(text: str, target_words: int = 110, overlap_sentences: int = 1):
    """Plain-text convenience wrapper (used by tests and non-PDF sources)."""
    chunks, _ = chunk_page(text, "", target_words, overlap_sentences)
    return [c[0] for c in chunks]


def build_chunks(pages):
    """pages -> chunk dicts ready for the DB (loc_kind='page'), each with a topic taken from real headings."""
    out, topic = [], ""
    for p in pages:
        fallback = topic or f"Page {p['page']}"
        chunks, topic_end = chunk_page(p["text"], fallback)
        topic = topic_end
        for text, t in chunks:
            out.append({"loc_kind": "page", "loc_start": p["page"], "loc_end": p["page"], "text": text, "topic": t or f"Page {p['page']}"})
    return out


def ingest_pdf(path):
    """Return (pages, chunks, n_empty). Raises IngestError if no text can be extracted."""
    pages = extract_pages(Path(path))
    if not pages:
        raise IngestError("This PDF has no pages.")
    n_empty = sum(1 for p in pages if not p["text"])
    chunks = build_chunks(pages)
    if not chunks:
        raise IngestError(
            "No extractable text found. This looks like a scanned/image-only PDF; "
            "OCR is not supported yet. Try a text-based PDF."
        )
    return pages, chunks, n_empty
