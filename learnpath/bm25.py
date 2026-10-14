"""In-memory BM25 over chunks, plus idf-weighted query coverage used for answerability."""
import math
from collections import Counter

from .textutil import terms


class BM25Index:
    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [Counter(terms(c["text"])) for c in self.chunks]
        self.termsets = [set(t) for t in self.tf]
        self.dl = [sum(t.values()) for t in self.tf]
        self.N = len(self.chunks)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.df = Counter()
        for ts in self.termsets:
            self.df.update(ts)

    def idf(self, term: str) -> float:
        """Terms absent from the corpus get the maximum idf (they are maximally informative)."""
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5, doc_ids=None):
        q = list(dict.fromkeys(terms(query)))
        scored = []
        for i, c in enumerate(self.chunks):
            if doc_ids is not None and c["doc_id"] not in doc_ids:
                continue
            score, tf, dl = 0.0, self.tf[i], self.dl[i]
            for t in q:
                f = tf.get(t, 0)
                if f:
                    score += self.idf(t) * (f * (self.k1 + 1)) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    )
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(self.chunks[i], s) for s, i in scored[:k]]

    def union_terms(self, chunks) -> set:
        by_id = {c["id"]: i for i, c in enumerate(self.chunks)}
        out = set()
        for c in chunks:
            out |= self.termsets[by_id[c["id"]]]
        return out

    def weighted_coverage(self, q_terms, hit_chunks) -> float:
        """Share of query-term idf mass that appears in the union of the given chunks."""
        if not q_terms:
            return 0.0
        present = self.union_terms(hit_chunks)
        total = sum(self.idf(t) for t in q_terms)
        got = sum(self.idf(t) for t in q_terms if t in present)
        return got / total if total else 0.0
