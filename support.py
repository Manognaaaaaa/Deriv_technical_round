"""Stage 3 - CHECK_SUPPORT: decide ANSWER vs ABSTAIN with one documented rule (no LLM).

THE SUPPORT RULE
----------------
Question content terms = question tokens minus stopwords and filler words,
stemmed, deduplicated. A term is *covered* by a passage if it equals one of the
passage's text stems, or both are >= 4 characters and share their first 4
characters (closing / closure / closed).

    coverage(passage) = covered terms in that ONE passage / total question terms
    support_score     = max coverage over retrieved passages (3 decimals)

ANSWER only if all hold (checked in the order 4, 1, 2, 3; the first failure is
recorded as ``abstain_reason``):

  4. out_of_scope_intent - the question contains none of ABSTAIN_INTENT_TERMS
     (whole stemmed tokens; multi-word terms as consecutive tokens).
  1. no_retrieval        - at least one passage scored > MIN_BM25_SCORE and the
     question has at least one content term.
  2. low_coverage        - support_score >= SUPPORT_THRESHOLD, i.e. the evidence
     sits in a single passage rather than being spread across several.
  3. number_mismatch     - every all-digit token in the question appears as a
     whole token in the best passage ("2" does not match "24").

``SupportDecision`` objects can only be created by ``check_support`` in this
module; answerers accept nothing else, so an answer cannot be produced without
a completed support check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from retriever import Passage
from text import numbers, query_terms, stem, stemmed_sequence, tokenize

_CREATION_KEY = object()


@dataclass(frozen=True)
class ScoredPassage:
    """A retrieved passage with its coverage and the question terms it covers."""

    passage: Passage
    coverage: float
    covered_terms: frozenset[str]


@dataclass(frozen=True)
class SupportDecision:
    """Result of CHECK_SUPPORT. Only ``check_support`` may construct one."""

    should_answer: bool
    abstain_reason: str | None
    support_score: float
    threshold: float
    query_terms: tuple[str, ...]
    passages: tuple[Passage, ...]
    supporting: tuple[ScoredPassage, ...]
    _key: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        """Refuse construction from anywhere except ``check_support``."""
        if self._key is not _CREATION_KEY:
            raise TypeError("SupportDecision can only be created by support.check_support()")

    @property
    def retrieved_ids(self) -> list[str]:
        """Parent doc ids of the retrieved passages, in ranked order."""
        return [p.parent_id for p in self.passages]


def is_covered(term: str, passage_terms: frozenset[str] | set[str]) -> bool:
    """True if ``term`` equals a passage stem or shares a 4-char prefix with one (both >= 4 chars)."""
    if term in passage_terms:
        return True
    n = config.PREFIX_MATCH_LEN
    if len(term) < n:
        return False
    return any(len(p) >= n and p[:n] == term[:n] for p in passage_terms)


def covered_terms(terms: list[str] | tuple[str, ...], passage_terms: frozenset[str] | set[str]) -> frozenset[str]:
    """The subset of ``terms`` covered by ``passage_terms``."""
    return frozenset(t for t in terms if is_covered(t, passage_terms))


def coverage(terms: list[str] | tuple[str, ...], passage_terms: frozenset[str] | set[str]) -> float:
    """Fraction of question terms covered by a single passage (0.0 if no terms)."""
    if not terms:
        return 0.0
    return len(covered_terms(terms, passage_terms)) / len(terms)


def find_intent_term(question: str, intent_terms: list[str] = config.ABSTAIN_INTENT_TERMS) -> str | None:
    """Return the first out-of-scope intent term found in the question, else None."""
    seq = stemmed_sequence(question)
    for term in intent_terms:
        pattern = [stem(t) for t in tokenize(term)]
        if not pattern:
            continue
        for i in range(len(seq) - len(pattern) + 1):
            if seq[i : i + len(pattern)] == pattern:
                return term
    return None


def check_support(
    question: str,
    passages: list[Passage],
    threshold: float = config.SUPPORT_THRESHOLD,
    min_score: float = config.MIN_BM25_SCORE,
) -> SupportDecision:
    """Apply the support rule (see module docstring) and return a SupportDecision."""
    terms = tuple(query_terms(question))
    scored = [
        ScoredPassage(p, coverage(terms, p.text_terms), covered_terms(terms, p.text_terms)) for p in passages
    ]
    # Best passage: highest coverage, then highest BM25 score, then retrieval rank.
    ranked = sorted(enumerate(scored), key=lambda e: (-e[1].coverage, -e[1].passage.score, e[0]))
    ranked_scored = [s for _, s in ranked]
    support_score = round(ranked_scored[0].coverage, 3) if ranked_scored else 0.0
    best = ranked_scored[0] if ranked_scored else None

    reason: str | None = None
    if find_intent_term(question) is not None:
        reason = "out_of_scope_intent"
    elif not terms or not any(p.score > min_score for p in passages):
        reason = "no_retrieval"
    elif support_score < threshold:
        reason = "low_coverage"
    elif not numbers(question) <= set(tokenize(best.passage.chunk.text)):
        reason = "number_mismatch"

    supporting = tuple(s for s in ranked_scored if s.coverage >= threshold) if reason is None else ()
    return SupportDecision(
        should_answer=reason is None,
        abstain_reason=reason,
        support_score=support_score,
        threshold=threshold,
        query_terms=terms,
        passages=tuple(passages),
        supporting=supporting,
        _key=_CREATION_KEY,
    )
