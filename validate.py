"""End-to-end validation of app.py in rule mode with NO API key.

Runs app.py as a subprocess (ANSWER_MODE=rule, API key removed, --mode rule),
checks the output contract, and prints PASS/FAIL per check. Exits non-zero on
any failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["SKIP_DOTENV"] = "1"  # never load a real key, even in-process
os.environ["ANSWER_MODE"] = "rule"

import config  # noqa: E402

os.environ.pop(config.LLM_API_KEY_ENV, None)

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "kb.json"
# Probe questions come from the eval set (never hardcoded in source).
_EVAL = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))
ANSWERABLE = next(q["question"] for q in _EVAL if q["expected_decision"] == "answer")
UNSUPPORTED = next(q["question"] for q in _EVAL if q["expected_decision"] == "abstain")
REQUIRED_KEYS = {"question", "decision", "answer", "citations", "debug"}

REPLACEMENT_KB = [
    {
        "id": "ship_1",
        "title": "Parcel tracking",
        "text": "Parcels can be tracked from the Orders page using the tracking number. "
        "Tracking updates appear within 6 hours of dispatch.",
    },
    {
        "id": "ship_2",
        "title": "Gift wrapping",
        "text": "Gift wrapping is available at checkout for a small fee.",
    },
]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one PASS/FAIL line."""
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def safe_env() -> dict[str, str]:
    """Environment with rule mode forced, the API key removed and .env loading disabled."""
    env = dict(os.environ)
    env.pop(config.LLM_API_KEY_ENV, None)
    env["ANSWER_MODE"] = "rule"
    env["SKIP_DOTENV"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_app(question: str, *extra: str) -> subprocess.CompletedProcess:
    """Run app.py in a subprocess in rule mode."""
    return subprocess.run(
        [sys.executable, str(ROOT / "app.py"), "--mode", "rule", *extra, question],
        cwd=ROOT,
        env=safe_env(),
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def parse_stdout(proc: subprocess.CompletedProcess) -> dict | None:
    """Parse stdout as a single JSON document, or None if anything else is present."""
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def check_output_shape(label: str, out: dict) -> None:
    """Contract checks shared by every response."""
    check(f"{label}: exact top-level keys", set(out) == REQUIRED_KEYS, str(sorted(out)))
    check(f"{label}: decision is answer|abstain", out.get("decision") in ("answer", "abstain"))
    debug = out.get("debug", {})
    check(f"{label}: debug has exactly retrieved_ids + support_score", set(debug) == {"retrieved_ids", "support_score"})
    check(
        f"{label}: retrieved_ids is a list of strings",
        isinstance(debug.get("retrieved_ids"), list) and all(isinstance(i, str) for i in debug["retrieved_ids"]),
    )
    score = debug.get("support_score")
    check(f"{label}: support_score is a float in [0, 1]", isinstance(score, float) and 0.0 <= score <= 1.0)


def check_citations(label: str, out: dict, kb: list[dict]) -> None:
    """Every citation id exists in the KB and every snippet is an exact substring."""
    docs = {d["id"]: d for d in kb}
    for c in out.get("citations", []):
        check(f"{label}: citation id {c['id']} exists in KB", c["id"] in docs)
        check(
            f"{label}: snippet of {c['id']} is an exact substring",
            c["id"] in docs and c["snippet"] in docs[c["id"]]["text"],
        )


def main() -> int:
    """Run all checks; return 0 if all pass."""
    # 1. KB loads from disk.
    from kb_loader import KBError, load_kb

    try:
        load_kb(KB_PATH)
        kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
        check("kb.json loads and validates", True)
    except KBError as exc:
        check("kb.json loads and validates", False, str(exc))
        return 1

    # 2. Answerable query.
    proc = run_app(ANSWERABLE)
    out = parse_stdout(proc)
    check("answerable: exit code 0", proc.returncode == 0, proc.stderr.strip())
    check("answerable: stdout is only JSON", out is not None)
    if out:
        check_output_shape("answerable", out)
        check("answerable: decision is 'answer'", out["decision"] == "answer")
        check("answerable: at least 1 citation", len(out["citations"]) >= 1)
        check_citations("answerable", out, kb)
    stderr_stages = [line.split()[-1] for line in proc.stderr.splitlines() if line.startswith("[stage]")]
    check(
        "answerable: stderr shows RETRIEVE before ANSWER",
        "RETRIEVE" in stderr_stages and "ANSWER" in stderr_stages
        and stderr_stages.index("RETRIEVE") < stderr_stages.index("ANSWER"),
        str(stderr_stages),
    )

    # 3. Unsupported query.
    proc = run_app(UNSUPPORTED)
    out = parse_stdout(proc)
    check("unsupported: exit code 0", proc.returncode == 0, proc.stderr.strip())
    check("unsupported: stdout is only JSON", out is not None)
    if out:
        check_output_shape("unsupported", out)
        check("unsupported: decision is 'abstain'", out["decision"] == "abstain")
        check("unsupported: citations empty", out["citations"] == [])

    # 4. Stage order recorded by the pipeline itself.
    from pipeline import Pipeline

    pipe = Pipeline(str(KB_PATH), mode="rule", echo_stages=False)
    for q in (ANSWERABLE, UNSUPPORTED):
        pipe.run(q)
        stages = pipe.last_stages
        final = next((s for s in ("ANSWER", "ABSTAIN") if s in stages), None)
        check(
            f"stage order for {q!r}: LOAD_KB -> RETRIEVE -> CHECK_SUPPORT before {final}",
            final is not None
            and stages[:3] == ["LOAD_KB", "RETRIEVE", "CHECK_SUPPORT"]
            and stages.index("RETRIEVE") < stages.index(final),
            str(stages),
        )

    # 5. Reads kb.json rather than embedded constants.
    with tempfile.TemporaryDirectory() as tmp:
        alt = Path(tmp) / "kb_alt.json"
        alt.write_text(json.dumps(REPLACEMENT_KB), encoding="utf-8")
        proc = run_app("How can I track my parcel?", "--kb", str(alt))
        out = parse_stdout(proc)
        ok = bool(out) and out["decision"] == "answer" and [c["id"] for c in out["citations"]] == ["ship_1"]
        check("replacement KB: answer and citations come from the replacement", ok, proc.stdout[:200])
        if out:
            check_citations("replacement KB", out, REPLACEMENT_KB)
            check(
                "replacement KB: only replacement ids retrieved",
                set(out["debug"]["retrieved_ids"]) <= {d["id"] for d in REPLACEMENT_KB},
            )

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
