"""Knowledge agent, Claude Agent SDK edition.

Here the *model* drives the loop. We give it three in-process MCP tools and a system
prompt that describes the policy; the SDK's agentic loop decides when to call what.
Governance stays in code, not in the prompt:

  * `RetrievalBudget` caps searches at 1 + max_retries (the model cannot loop forever)
  * tenant id is bound at construction - the model never sees or chooses it
  * `validate_answer` runs on the final text exactly as in the LangGraph build
  * a PreToolUse hook denies every tool when the kill switch is engaged

Requires the Claude Code CLI runtime (`npm i -g @anthropic-ai/claude-code`) at run time;
the tool handlers and budget are unit-tested without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from store_support.governance.kill_switch import KillSwitch
from store_support.knowledge import pipeline as p
from store_support.knowledge.llm import LLM
from store_support.knowledge.store import Hit, VectorStore

SYSTEM_PROMPT = """You are a store support knowledge agent.
Procedure:
1. Call search_policies with the customer's question.
2. Call grade_context. If score >= {threshold}, write the final answer ONLY from the returned
   context, citing each claim with the exact tag shown, e.g. [return-policy.md#Refund timing].
3. If score < {threshold}, rewrite the question using policy vocabulary and search again.
   You may search at most {max_searches} times in total.
4. If you cannot answer from context, call escalate with a short reason and stop.
Never invent policy. Never ask for or repeat personal data (names, emails, phone numbers).
"""


@dataclass(slots=True)
class RetrievalBudget:
    max_searches: int
    searches: int = 0
    queries: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    last_hits: list[Hit] = field(default_factory=list)
    escalated_reason: str | None = None

    @property
    def exhausted(self) -> bool:
        return self.searches >= self.max_searches


@dataclass(slots=True)
class SdkKnowledgeAgent:
    store: VectorStore
    llm: LLM  # used only for grading (same grader as LangGraph build -> comparable evals)
    tenant_id: str
    model: str = "claude-sonnet-4-5"
    k: int = 5
    max_retries: int = 2
    threshold: float = 0.6
    kill_switch: KillSwitch = field(default_factory=KillSwitch)

    # ----- tool handlers (plain async functions; testable without the CLI) -----

    def handlers(self, budget: RetrievalBudget) -> dict[str, Any]:
        async def search_policies(args: dict[str, Any]) -> dict[str, Any]:
            if budget.exhausted:
                return _err(f"search budget exhausted ({budget.max_searches}); call escalate")
            q = str(args["query"])
            budget.searches += 1
            budget.queries.append(q)
            budget.last_hits = p.retrieve(self.store, self.tenant_id, q, k=self.k)
            if not budget.last_hits:
                return _err("no context found; call escalate with reason no_context")
            text = "\n\n".join(f"{h.citation}\n{h.content}" for h in budget.last_hits)
            return {"content": [{"type": "text", "text": text}]}

        async def grade_context(args: dict[str, Any]) -> dict[str, Any]:
            if not budget.last_hits:
                return _err("nothing retrieved yet")
            score = p.grade(self.llm, str(args["question"]), budget.last_hits)
            budget.scores.append(score)
            return {"content": [{"type": "text", "text": f'{{"score": {score:.2f}}}'}]}

        async def escalate(args: dict[str, Any]) -> dict[str, Any]:
            budget.escalated_reason = str(args.get("reason") or "model_escalated")[:60]
            return {"content": [{"type": "text", "text": "escalated to a human; stop."}]}

        return {"search_policies": search_policies, "grade_context": grade_context, "escalate": escalate}

    def finalize(self, final_text: str, budget: RetrievalBudget) -> p.KnowledgeResult:
        res = p.KnowledgeResult(outcome=p.Outcome.ESCALATED, retries=max(0, budget.searches - 1),
                                queries=budget.queries, scores=budget.scores, hits=budget.last_hits)
        if budget.escalated_reason:
            res.reason = budget.escalated_reason
            return res
        cites, reason = p.validate_answer(final_text, budget.last_hits)
        if reason:
            res.reason = reason
            return res
        res.outcome, res.answer, res.citations = p.Outcome.ANSWERED, final_text, cites
        return res

    # ----- runtime (needs the SDK + CLI) -----

    async def run(self, question: str) -> p.KnowledgeResult:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            TextBlock,
            create_sdk_mcp_server,
            query,
            tool,
        )

        budget = RetrievalBudget(max_searches=1 + self.max_retries)
        h = self.handlers(budget)
        tools = [
            tool("search_policies", "Search store policy and manual chunks.", {"query": str})(h["search_policies"]),
            tool("grade_context", "Grade how well the last search answers the question (0..1).", {"question": str})(h["grade_context"]),
            tool("escalate", "Hand off to a human with a short reason.", {"reason": str})(h["escalate"]),
        ]
        server = create_sdk_mcp_server("knowledge", tools=tools)

        async def deny_when_killed(_input: dict[str, Any], _tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
            if self.kill_switch.engaged:
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                               "permissionDecisionReason": "kill switch engaged"}}
            return {}

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=SYSTEM_PROMPT.format(threshold=self.threshold, max_searches=budget.max_searches),
            mcp_servers={"knowledge": server},
            allowed_tools=["mcp__knowledge__search_policies", "mcp__knowledge__grade_context", "mcp__knowledge__escalate"],
            permission_mode="bypassPermissions",
            max_turns=4 + 2 * budget.max_searches,
            hooks={"PreToolUse": [HookMatcher(hooks=[deny_when_killed])]},
        )
        final = ""
        async for msg in query(prompt=question, options=options):
            if isinstance(msg, AssistantMessage):
                final = "".join(b.text for b in msg.content if isinstance(b, TextBlock)) or final
        return self.finalize(final, budget)


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}
