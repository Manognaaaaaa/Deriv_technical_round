# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Citation-grounded support-answer CLI (`app.py`) over a local FAQ KB (`kb.json`). Pipeline stages LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER/ABSTAIN are enforced by a state machine in `pipeline.py`. All thresholds and word lists live in `config.py`. See README.md for the support rule and design.

## Commands

```
pip install -r requirements.txt
python app.py "How do I reset my password?"
python validate.py
pytest -q
pytest tests/test_support.py::test_number_must_match_whole_token -q   # single test
python eval.py --sweep
```

## Environment

Development happens on Windows (PowerShell primary, Git Bash available). A local venv lives in `venv/`.

## Secrets

- Load all secrets from environment variables via `python-dotenv` (`load_dotenv()` + `os.getenv(...)`). Keys live in `.env` (e.g. `GROQ_API_KEY`), which is gitignored.
- Never hardcode keys.
- Never print secrets or stack traces to the client.
- Set `SKIP_DOTENV=1` to stop `.env` from loading (tests and validate.py do this).
