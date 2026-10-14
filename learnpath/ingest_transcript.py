"""Timestamped transcript ingestion (WebVTT, SRT, and plain '[hh:mm:ss] text' lines).

Chunks are packed from whole cues, split at long pauses, so a citation is a tight time range.
Topics are fixed 5-minute windows ("Lecture 05:00–10:00"): transcripts carry no heading information, so no
semantic topic is invented.  Unpunctuated auto-captions work for search/tutoring but yield few assessment questions
(question generation needs sentence boundaries).
"""
import html
import re

from .ingest_pdf import IngestError
from .locfmt import fmt_time

TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
CUE = re.compile(rf"{TS}\s*-->\s*{TS}")
LINE_TS = re.compile(rf"^\s*\[?{TS}\]?\s*[-–:]?\s+(.+)$")
WINDOW = 300          # seconds per topic window
TARGET_WORDS = 45
MAX_WORDS = 90
MAX_GAP = 30.0        # seconds of silence that forces a new chunk


def _secs(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + (int((ms or "0").ljust(3, "0")) / 1000.0)


# Only genuine caption markup is removed (speaker <v Bob>, <i>, <b>, <u>, <c.cls>, <lang>, <ruby>, inline <00:01:02.000>
# timestamps). Anything else, e.g. "if x<y and y>z", is lecture content and must be kept.
_MARKUP = re.compile(r"</?(?:v|c|i|b|u|ruby|rt|lang)(?:[ .][^>]*)?>|<\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?>", re.I)


def _clean(t: str) -> str:
    t = _MARKUP.sub("", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("latin-1")


def parse_cues(text: str):
    """-> [{'start','end','text'}] sorted by start."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    if CUE.search(text):
        for block in re.split(r"\n\s*\n", text):
            lines = [l for l in block.split("\n") if l.strip()]
            for i, l in enumerate(lines):
                m = CUE.search(l)
                if m:
                    g = m.groups()
                    body = _clean(" ".join(lines[i + 1:]))
                    if body:
                        cues.append({"start": _secs(*g[:4]), "end": _secs(*g[4:8]), "text": body})
                    break
    else:
        raw = []
        for l in text.split("\n"):
            m = LINE_TS.match(l)
            if m:
                g = m.groups()
                raw.append([_secs(*g[:4]), _clean(g[4])])
            elif raw and l.strip():           # continuation line of the previous timestamped entry
                raw[-1][1] += " " + _clean(l)
        for i, (st, tx) in enumerate(raw):
            if tx:
                en = raw[i + 1][0] if i + 1 < len(raw) else st + 5
                cues.append({"start": st, "end": max(en, st), "text": tx})
    cues.sort(key=lambda c: c["start"])
    return cues


def build_chunks(cues):
    chunks, cur, words = [], [], 0

    def flush():
        nonlocal cur, words
        if cur:
            start = cur[0]["start"]
            w = int(start // WINDOW)
            chunks.append({"loc_kind": "time", "loc_start": start, "loc_end": max(c["end"] for c in cur),
                           "text": " ".join(c["text"] for c in cur),
                           "topic": f"Lecture {fmt_time(w * WINDOW)}–{fmt_time((w + 1) * WINDOW)}"})
        cur, words = [], 0

    for c in cues:
        if cur and c["start"] - cur[-1]["end"] > MAX_GAP:
            flush()
        cur.append(c)
        words += len(c["text"].split())
        ends_sentence = c["text"].rstrip().endswith((".", "!", "?"))
        if (words >= TARGET_WORDS and ends_sentence) or words >= MAX_WORDS:
            flush()
    flush()
    return chunks


def ingest_transcript(path):
    with open(path, "rb") as f:
        data = f.read()
    cues = parse_cues(decode(data))
    if not cues:
        raise IngestError("No timestamped lines found. Expected WebVTT/SRT cues or lines like “[00:01:23] text”.")
    return cues, build_chunks(cues), []
