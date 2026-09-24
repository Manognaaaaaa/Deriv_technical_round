"""CLI contract: stdout JSON only, exact schema, input validation, exit codes."""

import json

import pytest

import config
from conftest import run_cli
from models import Response

REQUIRED_KEYS = {"question", "decision", "answer", "citations", "debug"}


def test_answer_output_schema():
    proc = run_cli("How do I reset my password?")
    assert proc.returncode == 0
    out = json.loads(proc.stdout)  # stdout is ONLY the JSON
    assert set(out) == REQUIRED_KEYS
    assert set(out["debug"]) == {"retrieved_ids", "support_score"}
    assert out["decision"] == "answer"
    assert set(out["citations"][0]) == {"id", "title", "snippet"}
    assert isinstance(out["debug"]["support_score"], float)
    stages = [line.split()[-1] for line in proc.stderr.splitlines() if line.startswith("[stage]")]
    assert stages == ["LOAD_KB", "RETRIEVE", "CHECK_SUPPORT", "ANSWER"]


def test_abstain_output():
    proc = run_cli("Can you guarantee my withdrawal will finish in 2 hours?")
    out = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert out["decision"] == "abstain" and out["citations"] == []
    assert out["answer"] == config.ABSTAIN_MESSAGE
    assert set(out["debug"]) == {"retrieved_ids", "support_score"}


def test_verbose_debug():
    out = json.loads(run_cli("--verbose", "Can support change my email address for me?").stdout)
    assert {"mode", "thresholds", "abstain_reason", "latency_ms"} <= set(out["debug"])
    assert out["debug"]["abstain_reason"] == "low_coverage"
    assert out["debug"]["thresholds"]["support_threshold"] == 0.8


def test_json_flag_and_stdin():
    a = run_cli("--json", '{"question": "How do I reset my password?"}')
    b = run_cli(stdin='{"question": "How do I reset my password?"}')
    assert a.returncode == b.returncode == 0
    assert json.loads(a.stdout) == json.loads(b.stdout)


@pytest.mark.parametrize(
    "args,stdin",
    [
        (("",), ""),
        (("   \t ",), ""),
        (("x" * 1001,), ""),
        (("--json", '{"question": "   "}'), ""),
        (("--json", "not json"), ""),
        (("--json", '{"q": "hi"}'), ""),
        ((), ""),
        ((), '{"question": ""}'),
    ],
)
def test_invalid_questions_exit_2(args, stdin):
    proc = run_cli(*args, stdin=stdin)
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "error:" in proc.stderr and "Traceback" not in proc.stderr


def test_1000_chars_is_allowed():
    assert run_cli("reset " * 166 + "pass").returncode == 0  # 1000 chars


def test_response_model_rejects_bad_decision():
    base = {"question": "q", "answer": "a", "citations": [], "debug": {"retrieved_ids": [], "support_score": 0.0}}
    with pytest.raises(ValueError):
        Response.model_validate({**base, "decision": "maybe"})
    with pytest.raises(ValueError):
        Response.model_validate({**base, "decision": "answer"})  # answer needs a citation
    with pytest.raises(ValueError):
        Response.model_validate({**base, "decision": "abstain", "debug": {**base["debug"], "extra": 1}})
