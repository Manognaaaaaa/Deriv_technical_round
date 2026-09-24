"""Deterministic text preprocessing shared by every stage.

Defined once here so indexing, retrieval, support checking and answering all
see exactly the same tokens.
"""

from __future__ import annotations

import re
import unicodedata

from config import QUESTION_FILLER_TERMS, STOPWORDS

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?](?=\s)")
_SUFFIXES = ("ings", "ing", "edly", "ed", "ies", "es", "s")
_ES_ENDINGS = ("sses", "xes", "zes", "ches", "shes")


def normalize(text: str) -> str:
    """Apply Unicode NFKC normalisation and lowercase."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    """Split normalised text on non-alphanumerics. Digits are kept ("2FA" -> "2fa")."""
    return [t for t in _TOKEN_SPLIT.split(normalize(text)) if t]


def stem(token: str) -> str:
    """Strip the first applicable suffix, keeping at least 3 characters.

    All-digit tokens are never stemmed. "es" is only stripped after
    sses/xes/zes/ches/shes, and "s" is never stripped after "ss", so
    "address" stays "address" while "reviews" becomes "review".
    """
    if token.isdigit():
        return token
    for suffix in _SUFFIXES:
        if not token.endswith(suffix):
            continue
        if suffix == "es" and not token.endswith(_ES_ENDINGS):
            continue
        if suffix == "s" and token.endswith("ss"):
            continue
        base = token[: -len(suffix)]
        if len(base) < 3:
            continue
        return base + "y" if suffix == "ies" else base
    return token


def content_terms(text: str, is_question: bool = False) -> list[str]:
    """Tokens with stopwords (and, for questions, filler words) removed, then stemmed."""
    drop = STOPWORDS | QUESTION_FILLER_TERMS if is_question else STOPWORDS
    return [stem(t) for t in tokenize(text) if t not in drop]


def query_terms(question: str) -> list[str]:
    """Deduplicated question content terms, in first-seen order."""
    return list(dict.fromkeys(content_terms(question, is_question=True)))


def stemmed_sequence(text: str) -> list[str]:
    """Every token stemmed, stopwords kept, order preserved (for phrase matching)."""
    return [stem(t) for t in tokenize(text)]


def numbers(text: str) -> set[str]:
    """Whole tokens made only of digits ("2fa" is not a number)."""
    return {t for t in tokenize(text) if t.isdigit()}


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of sentences in the ORIGINAL text.

    A sentence ends after '.', '!' or '?' followed by whitespace. Offsets are
    trimmed of surrounding whitespace so ``text[start:end]`` is the sentence.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        _append_span(spans, text, start, match.end())
        start = match.end()
    _append_span(spans, text, start, len(text))
    return spans


def _append_span(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    """Trim whitespace from [start, end) and append it if anything remains."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))
