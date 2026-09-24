"""KB loading: invalid files fail clearly (KBError in-process, exit code 3 from the CLI)."""

import pytest

import config
from conftest import KB_PATH, run_cli
from kb_loader import KBError, load_kb

GOOD = [{"id": "a", "title": "A", "text": "Alpha text."}]

BAD_KBS = {
    "invalid_json": ("[{not json", "not valid JSON"),
    "not_a_list": ('{"id": "a"}', "must be a JSON list"),
    "empty_list": ("[]", "empty"),
    "missing_key": ('[{"id": "a", "title": "A"}]', "missing required key"),
    "empty_text": ('[{"id": "a", "title": "A", "text": "  "}]', "non-empty"),
    "duplicate_ids": (
        '[{"id": "a", "title": "A", "text": "x"}, {"id": "a", "title": "B", "text": "y"}]',
        "duplicate id",
    ),
}


def test_loads_sample_kb_and_ignores_extra_fields(write_kb):
    assert len(load_kb(KB_PATH)) == 6
    docs = load_kb(write_kb([{**GOOD[0], "extra": 1}]))
    assert docs[0].id == "a"


def test_missing_file(tmp_path):
    with pytest.raises(KBError, match="not found"):
        load_kb(tmp_path / "nope.json")


@pytest.mark.parametrize("name", BAD_KBS)
def test_invalid_kb_raises(write_kb, name):
    content, message = BAD_KBS[name]
    with pytest.raises(KBError, match=message):
        load_kb(write_kb(content))


def test_oversized_file_checked_before_reading(write_kb, monkeypatch):
    path = write_kb(GOOD)
    monkeypatch.setattr(config, "MAX_KB_BYTES", 10)
    with pytest.raises(KBError, match="too large"):
        load_kb(path)


@pytest.mark.parametrize("name", ["invalid_json", "missing_key", "duplicate_ids", "empty_list"])
def test_cli_exit_code_3(write_kb, name):
    proc = run_cli("--kb", str(write_kb(BAD_KBS[name][0])), "How do I reset my password?")
    assert proc.returncode == 3
    assert proc.stdout == ""
    assert "invalid knowledge base" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_missing_kb_exit_3(tmp_path):
    proc = run_cli("--kb", str(tmp_path / "missing.json"), "How do I reset my password?")
    assert proc.returncode == 3 and "not found" in proc.stderr


def test_cli_oversized_kb_exit_3(tmp_path):
    big = tmp_path / "big.json"
    with big.open("wb") as fh:
        fh.write(b"[" + b" " * (config.MAX_KB_BYTES + 1) + b"]")
    proc = run_cli("--kb", str(big), "How do I reset my password?")
    assert proc.returncode == 3 and "too large" in proc.stderr
