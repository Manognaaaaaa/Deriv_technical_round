"""Stage 1 - LOAD_KB: read, validate and chunk the knowledge base (no LLM)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

import config
from models import KBDoc
from text import split_sentences

_DOCS_ADAPTER = TypeAdapter(list[KBDoc])


class KBError(Exception):
    """Raised when the knowledge base is missing or invalid (CLI exit code 3)."""


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. Short docs are one chunk; long docs are sentence windows.

    ``text`` is always ``parent_text[start:end]``, so snippets sliced from it are
    exact substrings of the parent document.
    """

    chunk_id: str
    parent_id: str
    title: str
    text: str
    start: int
    end: int
    parent_text: str
    doc_order: int


def load_kb(path: str | os.PathLike) -> list[KBDoc]:
    """Read and validate kb.json. Raises KBError with a clear message on any problem."""
    p = Path(path)
    if not p.exists():
        raise KBError(f"knowledge base file not found: {p}")
    if not p.is_file():
        raise KBError(f"knowledge base path is not a file: {p}")
    size = os.stat(p).st_size
    if size > config.MAX_KB_BYTES:
        raise KBError(f"knowledge base file is too large: {size} bytes (max {config.MAX_KB_BYTES})")
    try:
        data = json.loads(p.read_bytes().decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise KBError(f"knowledge base file is not valid UTF-8: {p}") from None
    except json.JSONDecodeError as exc:
        raise KBError(f"knowledge base is not valid JSON ({p}, line {exc.lineno}): {exc.msg}") from None
    if not isinstance(data, list):
        raise KBError("knowledge base must be a JSON list of {id, title, text} objects")
    if not data:
        raise KBError("knowledge base is empty")
    try:
        docs = _DOCS_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise KBError("knowledge base schema error: " + _format_errors(exc)) from None
    seen: set[str] = set()
    for doc in docs:
        if doc.id in seen:
            raise KBError(f"knowledge base has duplicate id: {doc.id!r}")
        seen.add(doc.id)
    return docs


def _format_errors(exc: ValidationError, limit: int = 3) -> str:
    """Summarise pydantic errors as 'item N field: message' (first few only)."""
    parts = []
    for err in exc.errors()[:limit]:
        loc = err["loc"]
        where = f"item {loc[0]}" + (f" field '{loc[1]}'" if len(loc) > 1 else "")
        msg = "missing required key" if err["type"] == "missing" else err["msg"]
        parts.append(f"{where}: {msg}")
    return "; ".join(parts)


def chunk_documents(docs: list[KBDoc], max_words: int = config.MAX_CHUNK_WORDS) -> list[Chunk]:
    """Keep short docs whole; split docs over ``max_words`` into overlapping sentence windows."""
    chunks: list[Chunk] = []
    for order, doc in enumerate(docs):
        if len(doc.text.split()) <= max_words:
            chunks.append(Chunk(doc.id, doc.id, doc.title, doc.text, 0, len(doc.text), doc.text, order))
            continue
        for n, (start, end) in enumerate(_sentence_windows(doc.text)):
            chunks.append(
                Chunk(f"{doc.id}#{n}", doc.id, doc.title, doc.text[start:end], start, end, doc.text, order)
            )
    return chunks


def _sentence_windows(text: str) -> list[tuple[int, int]]:
    """Group sentences into ~CHUNK_TARGET_WORDS windows sharing CHUNK_OVERLAP_SENTENCES."""
    spans = split_sentences(text) or [(0, len(text))]
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        j, words = i, 0
        while j < len(spans) and (words < config.CHUNK_TARGET_WORDS or j == i):
            words += len(text[spans[j][0] : spans[j][1]].split())
            j += 1
        windows.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        # Step back for overlap, but always make progress.
        i = max(j - config.CHUNK_OVERLAP_SENTENCES, i + 1)
    return windows
