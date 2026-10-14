"""Tokenising, light stemming and sentence splitting (no external NLP deps)."""
import re

# Only true grammatical function words and question-framing verbs are stopped.
# Content-bearing query signals like "purpose", "explain", "describe", "memorize",
# "difference", "comparison", etc. are NOT stopped so they contribute to query coverage.
# Exception: "work"/"works" and "differ"/"differs" are stopped because they appear
# in questions ("How does X work?", "How does A differ from B?") but the corpus uses
# different vocabulary for those concepts, causing false-positive coverage boosts.
STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being am do does did
    doing have has had having it its this that these those i you he she we they them his her our your their my me
    what which who whom whose when where why how can could should would will shall may might must about into over
    under again further then once here there all any both each few more most other some such no nor not only own
    same so than too very just also s t don now
    work works worked working differ differs differently difference differences""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
# Split after . ! ? before a capital/digit, but not after a bare section number such as "1." or "2.3."
_SENT_SPLIT = re.compile(r"(?<=[.!?])(?<!^\d\.)(?<!\s\d\.)(?<!^\d\d\.)(?<!\s\d\d\.)\s+(?=[A-Z0-9(\[\"'])")

_SUFFIXES = (
    ("ization", "ize"), ("ations", "ate"), ("ation", "ate"), ("ingly", ""), ("ies", "y"),
    ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", ""),
)


def stem(w):
    """Tiny suffix stripper. Only needs to be *consistent* between query and corpus."""
    if len(w) <= 3 or w.isdigit():
        return w
    for suf, rep in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf == "s" and w.endswith("ss"):
                continue
            w = w[: -len(suf)] + rep
            if suf in ("ing", "ed", "edly", "ingly") and len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]  # fitting -> fitt -> fit
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokens(text):
    return _TOKEN.findall(text.lower())


def terms(text):
    """Content-bearing stemmed terms (stopwords and 1-char tokens removed)."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text):
    """Split on sentence punctuation and on newlines (newlines only mark heading breaks)."""
    out = []
    for seg in text.split("\n"):
        seg = re.sub(r"\s+", " ", seg).strip()
        if seg:
            out.extend(s.strip() for s in _SENT_SPLIT.split(seg) if s.strip())
    return out


def is_full_sentence(s: str) -> bool:
    """True for prose sentences (ends with . ! ? and has a few words); False for headings/fragments."""
    return s.rstrip().endswith((".", "!", "?")) and len(s.split()) >= 5
