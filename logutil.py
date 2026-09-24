"""Small JSONL logging helpers with PII redaction. Logging never raises."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import config

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(text: str) -> str:
    """Replace email addresses and phone-number-like sequences."""
    return _PHONE.sub("[REDACTED_PHONE]", _EMAIL.sub("[REDACTED_EMAIL]", text))


def sha256(text: str) -> str:
    """Hex SHA-256 of a string (used for question and prompt hashes)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """Current UTC time in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one JSON record to a .jsonl file, swallowing I/O errors."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_failure(question: str, stage: str, error: str, log_dir: str | Path = config.LOG_DIR) -> None:
    """Record a per-question failure in logs/failures.jsonl (question stored only as a hash)."""
    append_jsonl(
        Path(log_dir) / "failures.jsonl",
        {
            "timestamp": now_iso(),
            "question_hash": sha256(question),
            "stage": stage,
            "error": redact(error),
        },
    )
