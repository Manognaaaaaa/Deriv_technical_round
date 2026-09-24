"""Test setup: rule mode, no API key, no .env loading, no network for the LLM client."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must happen before config is imported anywhere.
os.environ["SKIP_DOTENV"] = "1"
os.environ["ANSWER_MODE"] = "rule"
os.environ.pop("GROQ_API_KEY", None)

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *a, **k: False  # a real .env key can never leak into tests

import pytest  # noqa: E402

import answerers  # noqa: E402
import config  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from support import check_support  # noqa: E402
from text import query_terms  # noqa: E402

KB_PATH = ROOT / "kb.json"

MUST_ANSWER = {
    "How do I reset my password?": "doc_1",
    "What 2FA methods are supported?": "doc_2",
    "How long do withdrawal reviews take?": "doc_3",
    "What do I need before closing my account?": "doc_4",
}
MUST_ABSTAIN = {
    "Can you guarantee my withdrawal will finish in 2 hours?": "out_of_scope_intent",
    "What tax advice do you provide for closed accounts?": "out_of_scope_intent",
    "Can support change my email address for me?": "low_coverage",
}


@pytest.fixture(autouse=True)
def _block_llm(monkeypatch):
    """No test may reach the network: the SDK import fails and client creation raises."""
    monkeypatch.delenv(config.LLM_API_KEY_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "groq", None)

    def _blocked(api_key):
        raise answerers.LLMCallError("network blocked in tests")

    monkeypatch.setattr(answerers, "_create_client", _blocked)


@pytest.fixture(scope="session")
def pipe() -> Pipeline:
    """One rule-mode pipeline over the sample KB, reused across tests."""
    return Pipeline(str(KB_PATH), mode="rule", echo_stages=False)


@pytest.fixture
def write_kb(tmp_path):
    """Write a KB (list or raw string) to a temp file and return its path."""

    def _write(content, name: str = "kb.json") -> Path:
        path = tmp_path / name
        text = content if isinstance(content, str) else json.dumps(content)
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def sample_kb() -> list[dict]:
    """The sample KB as Python data."""
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def decide(pipeline: Pipeline, question: str, threshold: float = config.SUPPORT_THRESHOLD):
    """Retrieve + support-check a question without touching the stage machine."""
    return check_support(question, pipeline.index.retrieve(query_terms(question)), threshold)


def run_cli(*args: str, stdin: str = "") -> subprocess.CompletedProcess:
    """Run app.py in a subprocess with rule mode forced and no API key."""
    env = dict(os.environ)
    env.pop("GROQ_API_KEY", None)
    env.update(ANSWER_MODE="rule", SKIP_DOTENV="1", PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(ROOT / "app.py"), *args],
        cwd=ROOT,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


class FakeLLMClient:
    """Mimics client.chat.completions.create; replays scripted outputs (str or Exception)."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        """Return the next scripted output (the last one repeats)."""
        self.calls += 1
        self.requests.append(kwargs)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        content = out if isinstance(out, str) else json.dumps(out)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
        )
