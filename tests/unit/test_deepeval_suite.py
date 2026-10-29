"""DeepEval integration: test-case construction is pure and tested here (no judge needed).
The judged run lives in tests/evals/ and needs ANTHROPIC_API_KEY."""

from __future__ import annotations

import pytest

pytest.importorskip("deepeval")

from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402

from store_support.evals.deepeval_suite import (  # noqa: E402
    build_knowledge_cases,
    build_tool_cases,
    knowledge_case,
    tool_case,
)
from store_support.evals.runner import Question, Scenario  # noqa: E402
from store_support.knowledge.pipeline import Citation, KnowledgeResult, Outcome  # noqa: E402
from store_support.knowledge.store import Hit  # noqa: E402


def test_knowledge_case_maps_answer_context_and_expectation() -> None:
    q = Question("q01", "How long to return?", "return-policy.md", ["30 days"])
    res = KnowledgeResult(
        outcome=Outcome.ANSWERED,
        answer="Within 30 days. [return-policy.md#Eligibility window]",
        citations=[Citation("return-policy.md", "Eligibility window")],
        hits=[Hit("return-policy.md", "Eligibility window", "Most items may be returned within 30 days", 0.9)],
    )
    tc = knowledge_case(q, res)
    assert isinstance(tc, LLMTestCase)
    assert tc.name == "q01"
    assert tc.input == "How long to return?"
    assert tc.actual_output.startswith("Within 30 days")
    assert tc.retrieval_context == ["Most items may be returned within 30 days"]
    assert "30 days" in tc.expected_output
    assert tc.metadata["expected_source"] == "return-policy.md"


def test_escalated_result_becomes_explicit_handoff_output() -> None:
    q = Question("q02", "quantum?", "x.md", ["y"])
    res = KnowledgeResult(outcome=Outcome.ESCALATED, reason="low_relevance", hits=[])
    tc = knowledge_case(q, res)
    assert "escalated" in tc.actual_output.lower()
    assert tc.retrieval_context == []


def test_tool_case_uses_scenario_expectations_and_observed_calls() -> None:
    sc = Scenario("s01", "demo", "propose_refund", ["lookup_order", "check_warranty", "create_ticket"], False)
    tc = tool_case(sc, observed=["lookup_order", "check_warranty", "create_ticket"], observed_action="propose_refund")
    assert tc.expected_tools == [ToolCall(name=n) for n in sc.expected_tools]
    assert [t.name for t in tc.tools_called] == sc.expected_tools
    assert "propose_refund" in tc.actual_output


def test_offline_suites_cover_all_questions_and_scenarios() -> None:
    assert len(build_knowledge_cases(mode="offline")) == 30
    assert len(build_tool_cases()) == 20
