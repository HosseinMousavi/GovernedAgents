"""Judged evals (DeepEval + Claude as judge). Opt-in: needs ANTHROPIC_API_KEY.

make eval-deepeval              # offline pipeline (hash embedder + extractive answers), judged
DEEPEVAL_MODE=online make eval-deepeval   # real pgvector + Claude answers, judged (stack must be up)
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("deepeval")
pytestmark = pytest.mark.llm

if not os.environ.get("ANTHROPIC_API_KEY"):
    pytest.skip("ANTHROPIC_API_KEY not set - DeepEval tier is opt-in", allow_module_level=True)

from deepeval import assert_test  # noqa: E402

from store_support.evals.deepeval_suite import (  # noqa: E402
    build_knowledge_cases,
    build_tool_cases,
    judge,
    knowledge_metrics,
    tool_metrics,
)

MODE = os.environ.get("DEEPEVAL_MODE", "offline")
JUDGE = judge()
KNOWLEDGE_CASES = build_knowledge_cases(mode=MODE)  # type: ignore[arg-type]
TOOL_CASES = build_tool_cases()


@pytest.mark.parametrize("case", KNOWLEDGE_CASES, ids=[c.name for c in KNOWLEDGE_CASES])
def test_knowledge_answer_is_faithful_and_relevant(case) -> None:
    assert_test(case, knowledge_metrics(JUDGE))


@pytest.mark.parametrize("case", TOOL_CASES, ids=[c.name for c in TOOL_CASES])
def test_resolver_calls_exactly_the_expected_tools(case) -> None:
    assert_test(case, tool_metrics(JUDGE))
