"""Stage 2 - RETRIEVE: pure-Python BM25 over an inverted index (no LLM)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import config
from kb_loader import Chunk
from text import content_terms


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk with its BM25 score and the stems of its TEXT (for coverage)."""

    chunk: Chunk
    score: float
    text_terms: frozenset[str]

    @property
    def parent_id(self) -> str:
        """Id of the KB document this passage came from (always what gets cited)."""
        return self.chunk.parent_id

    @property
    def title(self) -> str:
        """Title of the parent KB document."""
        return self.chunk.title


class BM25Index:
    """Inverted index (term -> postings) built once, scored with BM25."""

    def __init__(self, chunks: list[Chunk], k1: float = config.BM25_K1, b: float = config.BM25_B):
        """Index title + text stems of every chunk; keep text-only stems for coverage."""
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.n_chunks = len(chunks)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.text_terms: list[frozenset[str]] = []
        for idx, chunk in enumerate(chunks):
            terms = content_terms(f"{chunk.title}\n{chunk.text}")
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, []).append((idx, tf))
            self.doc_len.append(len(terms))
            self.text_terms.append(frozenset(content_terms(chunk.text)))
        total = sum(self.doc_len)
        # Guard against an index with no tokens at all (avgdl of 0).
        self.avgdl = total / self.n_chunks if total > 0 else 1.0

    def idf(self, term: str) -> float:
        """Non-negative BM25 idf: ln((N - df + 0.5) / (df + 0.5) + 1)."""
        df = len(self.postings.get(term, ()))
        return math.log((self.n_chunks - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, terms: list[str]) -> dict[int, float]:
        """BM25 score per chunk index, touching only the postings of the query terms."""
        scores: dict[int, float] = {}
        for term in dict.fromkeys(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for idx, tf in postings:
                norm = 1.0 - self.b + self.b * self.doc_len[idx] / self.avgdl
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores

    def retrieve(
        self, terms: list[str], top_k: int = config.TOP_K, min_score: float = config.MIN_BM25_SCORE
    ) -> list[Passage]:
        """Top-k passages, one per parent doc, score > min_score, ties by KB order."""
        best: dict[str, tuple[float, int]] = {}
        for idx, raw in self.score(terms).items():
            score = round(raw, 9)  # stabilise float noise so ties are real ties
            if score <= min_score:
                continue
            parent = self.chunks[idx].parent_id
            if parent not in best or (-score, idx) < (-best[parent][0], best[parent][1]):
                best[parent] = (score, idx)
        ranked = sorted(best.values(), key=lambda s: (-s[0], self.chunks[s[1]].doc_order, s[1]))
        return [Passage(self.chunks[idx], score, self.text_terms[idx]) for score, idx in ranked[:top_k]]
