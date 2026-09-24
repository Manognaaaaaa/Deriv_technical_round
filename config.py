"""Central configuration: thresholds, word lists, paths and LLM settings.

Every tunable value lives here so behaviour can be audited in one place.
Secrets are never stored here; they are read from environment variables
(optionally populated from a local ``.env`` file via python-dotenv).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def _load_env() -> None:
    """Load ``.env`` into the environment unless SKIP_DOTENV=1 (used by tests/validate)."""
    if os.getenv("SKIP_DOTENV") != "1":
        load_dotenv(override=False)


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default`` on absence or bad value."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


_load_env()

# --- Knowledge base -------------------------------------------------------
DEFAULT_KB_PATH = "kb.json"
MAX_KB_BYTES = 10 * 1024 * 1024  # 10 MB, checked with os.stat before reading
MAX_CHUNK_WORDS = 200  # documents longer than this are split into sentence windows
CHUNK_TARGET_WORDS = 150  # approximate size of each window
CHUNK_OVERLAP_SENTENCES = 1  # sentences shared between consecutive windows

# --- Input ----------------------------------------------------------------
MAX_QUESTION_CHARS = 1000

# --- Retrieval (BM25) -----------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K = 3
MIN_BM25_SCORE = 0.0  # passages must score strictly above this

# --- Support check --------------------------------------------------------
SUPPORT_THRESHOLD = 0.8  # min fraction of question terms covered by ONE passage
PREFIX_MATCH_LEN = 4  # closing / closure / closed share "clos"
MAX_ANSWER_PASSAGES = 2  # at most this many supporting passages feed an answer
MAX_SNIPPET_SENTENCES = 3

# Out-of-scope intents: matched on the stemmed question token sequence as whole
# tokens (multi-word terms as consecutive tokens). Applied generically.
ABSTAIN_INTENT_TERMS = [
    "guarantee",
    "guaranteed",
    "promise",
    "legal advice",
    "tax advice",
    "financial advice",
]

ABSTAIN_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that. "
    "Please contact human support for help."
)

# --- Text preprocessing ---------------------------------------------------
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were be been being
    do does did i me my we our you your he she it its they them their this that these those
    what which who whom how when where why can could will would should may might must shall
    have has had not no so than then there here into about up down out over under again all
    any each few more most other some such only own same too very just also
    """.split()
)

# Removed from QUESTION content terms only (never from passages).
QUESTION_FILLER_TERMS = frozenset(
    "need needs want get tell know please help like way able".split()
)

# --- Answer modes ---------------------------------------------------------
ANSWER_MODES = ("rule", "llm")
DEFAULT_ANSWER_MODE = "rule"

# --- LLM (optional, --mode llm only) --------------------------------------
LLM_PROVIDER = "groq"
LLM_API_KEY_ENV = "GROQ_API_KEY"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1200  # cap on output tokens per call (reasoning models use part of it)
LLM_TIMEOUT_S = 20.0
LLM_MAX_CALLS_PER_RUN = 50  # hard cap on total LLM calls per process
LLM_NETWORK_RETRIES = 2  # retries on 429 / 5xx only, separate from the validation retry
LLM_BACKOFF_BASE_S = 1.0
LLM_BACKOFF_JITTER_S = 0.5
PROMPT_VERSION = "v1"
# USD per 1M tokens (estimates; override in .env)
LLM_PRICE_INPUT_PER_1M = _env_float("LLM_PRICE_INPUT_PER_1M", 0.15)
LLM_PRICE_OUTPUT_PER_1M = _env_float("LLM_PRICE_OUTPUT_PER_1M", 0.60)

# --- Output files ---------------------------------------------------------
LOG_DIR = os.getenv("LOG_DIR", "logs")
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
