"""CLI entry point for the citation-grounded support-answer service.

Usage:
    python app.py "<question>"
    python app.py --json '{"question": "<question>"}'
    echo '{"question": "..."}' | python app.py
Options: --kb PATH, --mode rule|llm, --verbose

stdout carries ONLY the final JSON; stage names and errors go to stderr.
Exit codes: 0 success (answer or abstain), 1 internal error, 2 invalid input, 3 invalid KB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pydantic import ValidationError

import config
from kb_loader import KBError
from logutil import log_failure
from models import QuestionRequest

EXIT_OK, EXIT_INTERNAL, EXIT_INPUT, EXIT_KB = 0, 1, 2, 3


class InputError(Exception):
    """Invalid CLI input (exit code 2)."""


def _eprint(message: str) -> None:
    """Print a message to stderr."""
    print(message, file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Define and parse CLI arguments (argparse exits with 2 on bad usage)."""
    parser = argparse.ArgumentParser(description="Answer a support question from a local FAQ knowledge base.")
    parser.add_argument("question", nargs="?", help="the question to answer")
    parser.add_argument("--json", dest="json_input", help='request JSON, e.g. \'{"question": "..."}\'')
    parser.add_argument("--kb", help="path to kb.json (default: KB_PATH env var or ./kb.json)")
    parser.add_argument("--mode", choices=config.ANSWER_MODES, help="answer mode (default: ANSWER_MODE env or rule)")
    parser.add_argument("--verbose", action="store_true", help="add mode, thresholds, abstain_reason, latencies")
    return parser.parse_args(argv)


def resolve_request(args: argparse.Namespace) -> QuestionRequest:
    """Build a validated QuestionRequest from the positional arg, --json or stdin JSON."""
    if args.question is not None and args.json_input is not None:
        raise InputError("give the question either as an argument or via --json, not both")
    try:
        if args.question is not None:
            return QuestionRequest(question=args.question)
        raw = args.json_input
        if raw is None and sys.stdin is not None and not sys.stdin.isatty():
            raw = sys.stdin.read()
        if raw is None or not raw.strip():
            raise InputError("no question provided (pass it as an argument, via --json, or as JSON on stdin)")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InputError(f"request is not valid JSON: {exc.msg}") from None
        if not isinstance(payload, dict):
            raise InputError('request JSON must be an object like {"question": "..."}')
        return QuestionRequest.model_validate(payload)
    except ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "request"
        raise InputError(f"invalid {field}: {err['msg'].removeprefix('Value error, ')}") from None


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    try:
        request = resolve_request(args)
    except InputError as exc:
        _eprint(f"error: {exc}")
        return EXIT_INPUT

    mode = args.mode or os.getenv("ANSWER_MODE") or config.DEFAULT_ANSWER_MODE
    if mode not in config.ANSWER_MODES:
        _eprint(f"error: ANSWER_MODE must be one of {', '.join(config.ANSWER_MODES)}")
        return EXIT_INPUT
    kb_path = args.kb or os.getenv("KB_PATH") or config.DEFAULT_KB_PATH

    # Imported here so argument errors stay fast and dependency-light.
    from pipeline import Pipeline

    try:
        pipeline = Pipeline(kb_path=kb_path, mode=mode)
    except KBError as exc:
        _eprint(f"error: invalid knowledge base: {exc}")
        return EXIT_KB
    except Exception as exc:  # never show a stack trace to the user
        _eprint(f"error: internal error while loading the knowledge base ({type(exc).__name__})")
        return EXIT_INTERNAL

    try:
        output = pipeline.run(request.question).to_output(verbose=args.verbose)
    except Exception as exc:
        log_failure(request.question, "RUN", f"{type(exc).__name__}: {exc}")
        _eprint(f"error: internal error while answering ({type(exc).__name__})")
        return EXIT_INTERNAL

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
