# Knowledge agent: LangGraph vs Claude Agent SDK

Same job, same steps (`pipeline.py`), same grader, same citation validator, same evals.
Only the orchestration differs.

| | LangGraph (`langgraph_agent.py`) | Claude Agent SDK (`sdk_agent.py`) |
|---|---|---|
| Who decides the next step | **Code.** A `StateGraph` with explicit conditional edges. | **The model.** It reads a procedure in the system prompt and calls tools. |
| Retry limit | An edge condition (`retries < max_retries`). | `RetrievalBudget` inside the tool handler returns an error after N searches. |
| Tenant isolation | Bound in the `retrieve` node closure. | Bound in the tool closure; a `tenant_id` arg from the model is ignored. |
| Kill switch | Checked by the MCP server for business tools; the knowledge graph itself has no tools to kill. | `PreToolUse` hook denies every tool call when engaged. |
| Citation check | `generate` node -> `validate_answer` -> `escalate` edge. | `finalize()` runs the same `validate_answer` on the final text. |
| LLM calls per happy path | 2 (grade, answer). | 2-3 (model turn to search, grade via tool, answer turn). |
| Determinism / testability | High: every node is a function; the graph is unit-tested with a fake LLM. | Handlers and budget are unit-tested; the loop itself needs the CLI runtime and is covered by evals. |
| Lines of orchestration | ~70 | ~60 (+ prompt) |
| Best when | Compliance needs a fixed, auditable control flow. | Steps vary per case and the model's judgement about *when* to retry adds value. |

## What we learned building both

1. **Keep governance out of the prompt.** In the SDK build the prompt says "search at most 3 times",
   but the thing that actually stops it is `RetrievalBudget`. Prompts are advice; code is policy.
2. **Share the grader.** Using one grader implementation makes the eval numbers comparable across
   builds. Otherwise you compare prompts, not architectures.
3. **Validate the output in code in both builds.** A model-driven loop can produce a fluent,
   uncited answer. `validate_answer` rejects it in both paths and routes to a human.
4. **Tracing shape differs.** LangGraph gives you one span per node for free (`step.retrieve`,
   `step.grade` ...). In the SDK build, spans come from the tool handlers, so model "thinking" turns
   are not spans; you see tool calls and the final answer.

## Picking one for production

For this project the LangGraph build is the default (`KnowledgeService(flavor="langgraph")`).
Customer-facing policy answers benefit more from a fixed control flow than from model autonomy.
The SDK build is kept as the comparison and for cases where the agent later gets more tools
(order lookup, warranty check) and the ordering genuinely depends on the conversation.
