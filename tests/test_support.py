"""The support rule: coverage, prefix matching, threshold, numbers, intent."""

import pytest

from conftest import decide
from pipeline import Pipeline
from support import SupportDecision, coverage, find_intent_term, is_covered


def test_coverage_fraction():
    assert coverage(["reset", "password"], {"reset", "password", "link"}) == 1.0
    assert coverage(["support", "change", "email", "address"], {"support", "change", "address"}) == 0.75
    assert coverage([], {"x"}) == 0.0


def test_four_char_prefix_matching():
    assert is_covered("clos", {"closure"})  # closing -> clos, closure
    assert is_covered("closure", {"clos"})
    assert is_covered("long", {"longer"})
    assert not is_covered("tax", {"taxes"})  # < 4 chars: exact only
    assert not is_covered("reset", {"resort"})


def test_email_question_scores_075_and_abstains(pipe):
    d = decide(pipe, "Can support change my email address for me?")
    assert d.support_score == 0.75
    assert not d.should_answer and d.abstain_reason == "low_coverage"
    # At the old 0.6 / 0.75 threshold it would (wrongly) answer.
    assert decide(pipe, "Can support change my email address for me?", threshold=0.75).should_answer


def test_threshold_boundary_075_abstains_080_answers(write_kb):
    kb = write_kb([{"id": "p1", "title": "Parcels", "text": "Parcels are tracked by courier with a number."}])
    p = Pipeline(str(kb), echo_stages=False)
    low = decide(p, "parcel courier number refund")  # 3/4
    high = decide(p, "parcel courier number tracking refund")  # 4/5
    assert (low.support_score, low.should_answer, low.abstain_reason) == (0.75, False, "low_coverage")
    assert (high.support_score, high.should_answer) == (0.8, True)


def test_number_must_match_whole_token(pipe):
    wrong = decide(pipe, "Do withdrawal reviews take 2 hours?")
    assert wrong.support_score >= 0.8
    assert wrong.abstain_reason == "number_mismatch"  # "2" must not match "24"
    assert decide(pipe, "Do withdrawal reviews take 24 hours?").should_answer


def test_2fa_is_not_a_number(pipe):
    assert decide(pipe, "What 2FA methods are supported?").should_answer


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Can you guarantee it?", "guarantee"),
        ("Is this guaranteed?", "guaranteed"),
        ("Do you promise a refund?", "promise"),
        ("I want legal advice", "legal advice"),
        ("my account was compromised", None),
        ("advice on legal matters", None),  # not consecutive
    ],
)
def test_intent_terms(question, expected):
    found = find_intent_term(question)
    if expected is None:
        assert found is None
    else:
        assert found is not None


def test_intent_checked_before_other_rules(pipe):
    d = decide(pipe, "Can you guarantee my password reset?")
    assert d.abstain_reason == "out_of_scope_intent"


def test_no_retrieval_reason(pipe):
    assert decide(pipe, "What is the minimum deposit amount?").abstain_reason == "no_retrieval"
    assert decide(pipe, "what is the").abstain_reason == "no_retrieval"  # no content terms


def test_support_decision_cannot_be_forged():
    with pytest.raises(TypeError):
        SupportDecision(True, None, 1.0, 0.8, (), (), ())
