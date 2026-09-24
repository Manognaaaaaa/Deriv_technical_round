"""Pydantic schemas for KB documents, CLI request/response and LLM output."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

import config


class KBDoc(BaseModel):
    """One knowledge-base article. Extra fields in kb.json are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: StrictStr
    title: StrictStr
    text: StrictStr

    @field_validator("id", "title", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only strings."""
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class QuestionRequest(BaseModel):
    """CLI request: a single user question."""

    model_config = ConfigDict(extra="forbid")

    question: StrictStr

    @field_validator("question")
    @classmethod
    def _valid_question(cls, value: str) -> str:
        """Reject empty/whitespace-only and over-long questions; return it stripped."""
        if not value.strip():
            raise ValueError("question must not be empty")
        if len(value) > config.MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {config.MAX_QUESTION_CHARS} characters")
        return value.strip()


class Citation(BaseModel):
    """A cited KB document with the exact supporting snippet."""

    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    title: StrictStr
    snippet: str = Field(min_length=1)


class Debug(BaseModel):
    """Default debug block: exactly retrieved_ids and support_score."""

    model_config = ConfigDict(extra="forbid")

    retrieved_ids: list[StrictStr]
    support_score: float = Field(ge=0.0, le=1.0)


class VerboseDebug(Debug):
    """Debug block for --verbose: adds mode, thresholds, abstain_reason, latencies."""

    mode: str
    thresholds: dict[str, float]
    abstain_reason: Optional[str]
    latency_ms: dict[str, float]


class Response(BaseModel):
    """The only thing printed to stdout."""

    model_config = ConfigDict(extra="forbid")

    question: StrictStr
    decision: Literal["answer", "abstain"]
    answer: StrictStr
    citations: list[Citation]
    debug: Union[VerboseDebug, Debug]

    @model_validator(mode="after")
    def _citations_match_decision(self) -> "Response":
        """An answer needs at least one citation; an abstention has none."""
        if self.decision == "answer" and not self.citations:
            raise ValueError("decision 'answer' requires at least one citation")
        if self.decision == "abstain" and self.citations:
            raise ValueError("decision 'abstain' must have no citations")
        return self


class LLMCitation(BaseModel):
    """A citation proposed by the LLM; checked against KB text in code."""

    id: StrictStr
    quote: str = Field(min_length=1)


class LLMOutput(BaseModel):
    """Structured JSON the LLM must return. decision is whitelisted."""

    decision: Literal["answer", "abstain"]
    answer: StrictStr = ""
    citations: list[LLMCitation] = Field(default_factory=list)
