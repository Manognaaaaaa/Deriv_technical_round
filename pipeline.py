"""Pipeline: LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER | ABSTAIN, enforced by a state machine.

The KB is loaded and indexed ONCE per Pipeline; ``run`` then answers any number
of questions. Stage names are printed to stderr as each stage starts.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum

import config
from answerers import AnswerResult, Answerer, LLMAnswerer, RuleBasedAnswerer
from kb_loader import chunk_documents, load_kb
from models import Citation, Debug, Response, VerboseDebug
from retriever import BM25Index, Passage
from support import SupportDecision, check_support
from text import query_terms


class Stage(str, Enum):
    """Pipeline stages, in order."""

    LOAD_KB = "LOAD_KB"
    RETRIEVE = "RETRIEVE"
    CHECK_SUPPORT = "CHECK_SUPPORT"
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"


class StageOrderError(RuntimeError):
    """Raised when a stage is entered out of order."""


# Allowed transitions. ANSWER -> ABSTAIN covers an LLM downgrade or empty citations.
_NEXT: dict[Stage | None, set[Stage]] = {
    None: set(),
    Stage.LOAD_KB: {Stage.RETRIEVE},
    Stage.RETRIEVE: {Stage.CHECK_SUPPORT},
    Stage.CHECK_SUPPORT: {Stage.ANSWER, Stage.ABSTAIN},
    Stage.ANSWER: {Stage.ABSTAIN},
    Stage.ABSTAIN: set(),
}


class StageMachine:
    """Tracks and enforces stage order for the current question."""

    def __init__(self, echo: bool = True):
        """``echo`` prints each stage name to stderr as it starts."""
        self.echo = echo
        self.kb_loaded = False
        self.stages: list[Stage] = []

    def _announce(self, stage: Stage) -> None:
        """Print the stage name to stderr (stdout is reserved for the final JSON)."""
        if self.echo:
            print(f"[stage] {stage.value}", file=sys.stderr, flush=True)

    def load_kb(self) -> None:
        """Enter LOAD_KB (once per Pipeline)."""
        if self.kb_loaded:
            raise StageOrderError("LOAD_KB already completed for this pipeline")
        self._announce(Stage.LOAD_KB)

    def begin_question(self) -> None:
        """Start a new question: its trace begins with the (already done) LOAD_KB."""
        if not self.kb_loaded:
            raise StageOrderError("LOAD_KB must complete before any question is processed")
        self.stages = [Stage.LOAD_KB]

    def enter(self, stage: Stage) -> None:
        """Move to ``stage`` if the transition is allowed, else raise StageOrderError."""
        if stage in (Stage.ANSWER, Stage.ABSTAIN):
            missing = [s.value for s in (Stage.RETRIEVE, Stage.CHECK_SUPPORT) if s not in self.stages]
            if missing:
                raise StageOrderError(f"{stage.value} called before {', '.join(missing)} completed")
        prev = self.stages[-1] if self.stages else None
        if stage not in _NEXT[prev]:
            raise StageOrderError(f"cannot enter {stage.value} after {prev.value if prev else 'nothing'}")
        self.stages.append(stage)
        self._announce(stage)

    def finish(self) -> list[str]:
        """End the question and return its recorded stage order."""
        done = [s.value for s in self.stages]
        self.stages = []
        return done


@dataclass
class RunResult:
    """Everything produced for one question."""

    question: str
    decision: str
    answer: str
    citations: list[Citation]
    retrieved_ids: list[str]
    support_score: float
    abstain_reason: str | None
    stages: list[str]
    mode: str
    thresholds: dict[str, float]
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_response(self, verbose: bool = False) -> Response:
        """Build and validate the output model (Debug has only 2 keys unless verbose)."""
        if verbose:
            debug = VerboseDebug(
                retrieved_ids=self.retrieved_ids,
                support_score=self.support_score,
                mode=self.mode,
                thresholds=self.thresholds,
                abstain_reason=self.abstain_reason,
                latency_ms=self.latency_ms,
            )
        else:
            debug = Debug(retrieved_ids=self.retrieved_ids, support_score=self.support_score)
        return Response(
            question=self.question,
            decision=self.decision,
            answer=self.answer,
            citations=self.citations,
            debug=debug,
        )

    def to_output(self, verbose: bool = False) -> dict:
        """Validated JSON-ready dict for stdout."""
        return self.to_response(verbose).model_dump(mode="json")


class Pipeline:
    """Loads the KB once and answers many questions through the enforced stages."""

    def __init__(
        self,
        kb_path: str = config.DEFAULT_KB_PATH,
        mode: str = config.DEFAULT_ANSWER_MODE,
        answerer: Answerer | None = None,
        support_threshold: float = config.SUPPORT_THRESHOLD,
        top_k: int = config.TOP_K,
        echo_stages: bool = True,
    ):
        """Run LOAD_KB: read + validate kb.json, chunk it, build the BM25 index."""
        if mode not in config.ANSWER_MODES:
            raise ValueError(f"mode must be one of {config.ANSWER_MODES}")
        self.mode = mode
        self.support_threshold = support_threshold
        self.top_k = top_k
        self.answerer = answerer or (LLMAnswerer() if mode == "llm" else RuleBasedAnswerer())
        self.machine = StageMachine(echo=echo_stages)
        self.last_stages: list[str] = []

        start = time.perf_counter()
        self.machine.load_kb()
        self.docs = load_kb(kb_path)
        self.chunks = chunk_documents(self.docs)
        self.index = BM25Index(self.chunks)
        self.machine.kb_loaded = True
        self.load_ms = round((time.perf_counter() - start) * 1000, 2)

    # --- individual stages (public so tests can prove the order is enforced) ---

    def retrieve(self, question: str) -> list[Passage]:
        """RETRIEVE: start a new question and return top-k BM25 passages."""
        self.machine.begin_question()
        self.machine.enter(Stage.RETRIEVE)
        return self.index.retrieve(query_terms(question), top_k=self.top_k)

    def check_support(self, question: str, passages: list[Passage], threshold: float | None = None) -> SupportDecision:
        """CHECK_SUPPORT: apply the support rule to the retrieved passages."""
        self.machine.enter(Stage.CHECK_SUPPORT)
        return check_support(question, passages, self.support_threshold if threshold is None else threshold)

    def answer(self, question: str, decision: SupportDecision) -> AnswerResult:
        """ANSWER: only after RETRIEVE and CHECK_SUPPORT, and only with a SupportDecision."""
        self.machine.enter(Stage.ANSWER)
        if not isinstance(decision, SupportDecision):
            raise TypeError("answer() requires a SupportDecision")
        return self.answerer.answer(question, decision)

    def abstain(self, decision: SupportDecision, reason: str | None) -> AnswerResult:
        """ABSTAIN: fixed message, no citations."""
        self.machine.enter(Stage.ABSTAIN)
        if not isinstance(decision, SupportDecision):
            raise TypeError("abstain() requires a SupportDecision")
        return AnswerResult("abstain", config.ABSTAIN_MESSAGE, [], reason, self.answerer.name)

    # --- orchestration ---

    def run(self, question: str, support_threshold: float | None = None) -> RunResult:
        """Run one question through RETRIEVE -> CHECK_SUPPORT -> ANSWER | ABSTAIN."""
        threshold = self.support_threshold if support_threshold is None else support_threshold
        latency: dict[str, float] = {Stage.LOAD_KB.value: self.load_ms}
        try:
            t = time.perf_counter()
            passages = self.retrieve(question)
            latency[Stage.RETRIEVE.value] = _ms(t)

            t = time.perf_counter()
            decision = self.check_support(question, passages, threshold)
            latency[Stage.CHECK_SUPPORT.value] = _ms(t)

            t = time.perf_counter()
            if decision.should_answer:
                result = self.answer(question, decision)
                latency[Stage.ANSWER.value] = _ms(t)
                if result.decision == "answer" and not result.citations:
                    result = self.abstain(decision, "no_citations")
                elif result.decision == "abstain":  # LLM downgrade
                    result = self.abstain(decision, result.abstain_reason)
            else:
                result = self.abstain(decision, decision.abstain_reason)
                latency[Stage.ABSTAIN.value] = _ms(t)
        finally:
            self.last_stages = self.machine.finish()

        return RunResult(
            question=question,
            decision=result.decision,
            answer=result.answer,
            citations=result.citations,
            retrieved_ids=decision.retrieved_ids,
            support_score=decision.support_score,
            abstain_reason=result.abstain_reason,
            stages=self.last_stages,
            mode=result.mode_used if result.decision == "answer" else self.mode,
            thresholds={
                "support_threshold": threshold,
                "min_bm25_score": config.MIN_BM25_SCORE,
                "top_k": float(self.top_k),
                "bm25_k1": self.index.k1,
                "bm25_b": self.index.b,
            },
            latency_ms=latency,
        )


def _ms(start: float) -> float:
    """Milliseconds elapsed since ``start`` (perf_counter)."""
    return round((time.perf_counter() - start) * 1000, 3)
