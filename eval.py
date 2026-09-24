"""Offline evaluation over eval_questions.json using ONE Pipeline instance.

    python eval.py            # metrics + per-question table, writes eval_report.json
    python eval.py --sweep    # also sweep SUPPORT_THRESHOLD 0.50 .. 0.90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import config
from kb_loader import KBError
from logutil import log_failure
from pipeline import Pipeline


def load_questions(path: str) -> list[dict]:
    """Read the eval set: [{question, expected_decision, expected_citation_ids?}]."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_all(pipe: Pipeline, items: list[dict], threshold: float | None = None) -> list[dict]:
    """Run every question; a failure on one question is logged and does not stop the run."""
    rows = []
    for item in items:
        q = item["question"]
        try:
            r = pipe.run(q, support_threshold=threshold)
            decision, cited, reason, score = r.decision, [c.id for c in r.citations], r.abstain_reason, r.support_score
        except Exception as exc:
            log_failure(q, "EVAL", f"{type(exc).__name__}: {exc}")
            decision, cited, reason, score = "error", [], type(exc).__name__, 0.0
        rows.append(
            {
                "question": q,
                "expected_decision": item["expected_decision"],
                "decision": decision,
                "correct": decision == item["expected_decision"],
                "expected_citation_ids": item.get("expected_citation_ids", []),
                "citation_ids": cited,
                "abstain_reason": reason,
                "support_score": score,
            }
        )
    return rows


def metrics(rows: list[dict]) -> dict:
    """Accuracy, false-answer rate, abstain rate, citation presence and correctness."""
    n = len(rows) or 1
    should_abstain = [r for r in rows if r["expected_decision"] == "abstain"]
    answered = [r for r in rows if r["decision"] == "answer"]
    graded = [r for r in answered if r["expected_citation_ids"]]
    return {
        "n_questions": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "false_answer_rate": round(
            sum(r["decision"] == "answer" for r in should_abstain) / (len(should_abstain) or 1), 3
        ),
        "abstain_rate": round(sum(r["decision"] == "abstain" for r in rows) / n, 3),
        "citation_presence_rate": round(sum(bool(r["citation_ids"]) for r in answered) / (len(answered) or 1), 3),
        "citation_correctness_rate": round(
            sum(bool(set(r["expected_citation_ids"]) & set(r["citation_ids"])) for r in graded) / (len(graded) or 1), 3
        ),
    }


def print_table(rows: list[dict]) -> None:
    """Per-question table."""
    print(f"{'ok':<4}{'expected':<10}{'got':<9}{'score':<7}{'reason':<21}{'cited':<12}question")
    for r in rows:
        print(
            f"{'Y' if r['correct'] else 'N':<4}{r['expected_decision']:<10}{r['decision']:<9}"
            f"{r['support_score']:<7}{str(r['abstain_reason'] or '-'):<21}{','.join(r['citation_ids']) or '-':<12}"
            f"{r['question']}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write eval_report.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", default=os.getenv("KB_PATH", config.DEFAULT_KB_PATH))
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--mode", choices=config.ANSWER_MODES, default="rule")
    parser.add_argument("--sweep", action="store_true", help="sweep SUPPORT_THRESHOLD 0.50..0.90")
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args(argv)

    try:
        pipe = Pipeline(args.kb, mode=args.mode, echo_stages=False)  # built ONCE
    except KBError as exc:
        print(f"error: invalid knowledge base: {exc}", file=sys.stderr)
        return 3
    items = load_questions(args.questions)

    rows = run_all(pipe, items)
    summary = metrics(rows)
    print_table(rows)
    print()
    for k, v in summary.items():
        print(f"{k:<27}{v}")
    report = {"mode": args.mode, "support_threshold": pipe.support_threshold, "metrics": summary, "questions": rows}

    if args.sweep:
        print(f"\n{'threshold':<11}{'accuracy':<10}{'false_answer':<14}abstain_rate")
        sweep = []
        for i in range(9):
            t = round(0.5 + 0.05 * i, 2)
            m = metrics(run_all(pipe, items, threshold=t))
            sweep.append({"threshold": t, **m})
            print(f"{t:<11}{m['accuracy']:<10}{m['false_answer_rate']:<14}{m['abstain_rate']}")
        report["sweep"] = sweep

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
