"""DeepEval tier: LLM-judged metrics on top of the deterministic scorecard.

    scorecard (runner.py)   deterministic: retrieval hit, citation validity, tool choice, scope.   CI, no keys.
    DeepEval (this file)    judged by Claude: faithfulness, answer relevancy, contextual relevancy,
                            tool correctness.                                                      `make eval-deepeval`

Same 30 questions and 20 scenarios; this module only converts results into DeepEval test cases.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

from deepeval.test_case import LLMTestCase, ToolCall

from store_support.access import AgentIdentity, RequestContext
from store_support.evals.runner import (
    DATASETS,
    ROOT,
    TODAY,
    ExtractiveLLM,
    Question,
    Scenario,
    _Knowledge,
    _offline_store,
    _principal,
    _seed_repo,
    load_questions,
    load_scenarios,
)
from store_support.governance.manifest import load_manifest
from store_support.intake.schema import IssueSpec
from store_support.knowledge.langgraph_agent import build_graph, run_graph
from store_support.knowledge.pipeline import KnowledgeResult, Outcome
from store_support.mcp_server.audit import AuditLog
from store_support.mcp_server.tools import ToolError, execute_tool
from store_support.resolver.agent import LocalToolClient, ResolverAgent

Mode = Literal["offline", "online"]


# ---------- pure conversions ----------


def knowledge_case(q: Question, res: KnowledgeResult) -> LLMTestCase:
    if res.outcome == Outcome.ANSWERED:
        actual = res.answer
    else:
        actual = f"Escalated to a human agent (reason: {res.reason}). No answer was given."
    return LLMTestCase(
        name=q.id,
        input=q.question,
        actual_output=actual,
        expected_output=f"An answer grounded in {q.expected_source} that mentions: {', '.join(q.keywords)}.",
        retrieval_context=[h.content for h in res.hits],
        metadata={"expected_source": q.expected_source, "outcome": res.outcome.value, "retries": res.retries},
    )


def tool_case(sc: Scenario, *, observed: Sequence[str], observed_action: str) -> LLMTestCase:
    return LLMTestCase(
        name=sc.id,
        input=sc.name,
        actual_output=f"action={observed_action}",
        expected_output=f"action={sc.expected_action}",
        tools_called=[ToolCall(name=n) for n in observed],
        expected_tools=[ToolCall(name=n) for n in sc.expected_tools],
    )


# ---------- builders ----------


def build_knowledge_cases(mode: Mode = "offline") -> list[LLMTestCase]:
    questions = load_questions(DATASETS / "questions.jsonl")
    if mode == "offline":
        graph = build_graph(store=_offline_store(), llm=ExtractiveLLM(), tenant_id="store-a", k=5, threshold=0.3)
        return [knowledge_case(q, run_graph(graph, q.question)) for q in questions]
    import asyncio

    from store_support.config import Settings
    from store_support.knowledge.factory import KnowledgeService

    with KnowledgeService(Settings.from_env(), "store-a") as svc:
        return [knowledge_case(q, asyncio.run(svc.ask(q.question))) for q in questions]


def build_tool_cases() -> list[LLMTestCase]:
    """Scenarios always run on the in-memory tool path: tool choice is deterministic code."""
    manifest = load_manifest(ROOT / "agents" / "resolver-agent" / "manifest.yaml")
    graph = build_graph(store=_offline_store(), llm=ExtractiveLLM(), tenant_id="store-a", k=5, threshold=0.3)
    cases = []
    for sc in load_scenarios(DATASETS / "scenarios.jsonl"):
        repo, audit = _seed_repo(), AuditLog()
        if sc.tool_call:
            tc = sc.tool_call
            ctx = RequestContext(_principal(tc["principal"]), AgentIdentity("eval-direct", "1"), sc.id)
            try:
                execute_tool(
                    ctx, tc["tool"], tc["args"], repo=repo, audit=audit, kill_switch=bool(tc.get("kill_switch"))
                )
                action = "ok"
            except ToolError as e:
                action = "denied" if e.code == "forbidden" else e.code
        else:
            assert sc.principal and sc.spec
            ctx = RequestContext(_principal(sc.principal), AgentIdentity(manifest.name, manifest.version), sc.id)
            agent = ResolverAgent(
                manifest=manifest, knowledge=_Knowledge(graph), tools=LocalToolClient(repo, audit), today=TODAY
            )
            action = agent.resolve(ctx, IssueSpec(**sc.spec)).action.value
        cases.append(tool_case(sc, observed=[e["tool"] for e in audit.entries], observed_action=action))
    return cases


# ---------- judge + metrics ----------


def judge() -> Any:
    """Claude as the DeepEval judge. Raises if no key: the judged tier is opt-in."""
    from deepeval.models import AnthropicModel

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for the DeepEval (judged) tier")
    return AnthropicModel(model=os.environ.get("DEEPEVAL_JUDGE_MODEL", "claude-sonnet-4-5"), api_key=key, temperature=0)


def knowledge_metrics(model: Any) -> list[Any]:
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric

    return [
        FaithfulnessMetric(threshold=0.8, model=model),  # claims in the answer are supported by retrieval_context
        AnswerRelevancyMetric(threshold=0.7, model=model),  # answer addresses the question
        ContextualRelevancyMetric(threshold=0.5, model=model),  # retrieved chunks are on-topic
    ]


def tool_metrics(model: Any) -> list[Any]:
    from deepeval.metrics import ToolCorrectnessMetric

    return [ToolCorrectnessMetric(threshold=1.0, model=model, should_exact_match=True, should_consider_ordering=True)]
