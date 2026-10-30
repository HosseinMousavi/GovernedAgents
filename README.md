# Store Support Agent

A governed multi-agent customer-issue resolver. A customer describes a problem, an intake agent
turns it into a fixed-schema spec, a knowledge agent answers from store policy with citations, a
resolver proposes an action through scoped MCP tools, and a human manager approves anything that
moves money. Every step is traced, every tool call is access-checked and audited, and a kill
switch stops all tool execution at once.

![demo](docs/demo.gif)

**Governance, in one line each**

| Concern | Where it lives | Enforced by |
|---|---|---|
| Entitlement (may this scope call this tool?) | `access/rules.py` `authorize_tool` | MCP server, every call |
| Data authorization (may this user see this row?) | `access/rules.py` `row_filter` -> SQL `WHERE` | Repository, every query |
| Tenant isolation | `tenant_id` is mandatory on every read/write; checked first | Repository + vector store |
| Agent identity vs user identity | `RequestContext(principal, agent)` logged in separate fields | Audit log, spans |
| No PII in logs/traces | Presidio + regex redaction as the *first* structlog processor | `governance/redaction.py` |
| Agent allow-list | `agents/<name>/manifest.yaml` `tools:` | `ResolverAgent.call_tool` |
| Kill switch | `KILL_SWITCH=true` or `touch /tmp/ssa.KILL` | MCP server + SDK `PreToolUse` hook |
| Money needs a human | Resolver *proposes*; only a `manager:refund` token can `issue_refund` | Scope check + n8n approval |
| Answers only from context | Grader + citation validator; otherwise escalate | `knowledge/pipeline.py` |
| Cost: which model runs what | `governance/routing.py` `route()` (pure) from task + signals + budget | `RoutedAnthropicLLM`, every call |
| Cost: how much was spent | Token usage -> USD per call, per request, per tenant per day | spans, `Resolution.cost_usd`, `CostLedger` |
| Cost: a ceiling | Per-tenant daily budget: soft -> cheap tier only; hard -> escalate to a human | `Budget` + `BudgetExhaustedError` |

## Architecture

```mermaid
flowchart LR
    subgraph n8n
        CT[Chat trigger] --> IA[Intake agent<br/>fills IssueSpec JSON]
        IA --> V{validate}
        V -- missing --> IA
        MA[Manager approval<br/>form]
        NT[Ticket notification<br/>PII guard]
    end
    V -- ok --> AG[/agents service<br/>POST /resolve/]
    subgraph agents["agents (Python)"]
        AG --> RS[Resolver agent<br/>manifest allow-list]
        RS --> KA[Knowledge agent<br/>LangGraph or Claude Agent SDK]
        KA --> PG[(pgvector)]
        KA --> EMB[Ollama BGE<br/>-> OpenAI fallback]
        RS --> RULES[decide: pure rules]
    end
    RS -- customer token --> MCP
    subgraph mcp["MCP server (Python MCP SDK)"]
        MCP[tools: lookup_order, check_inventory,<br/>check_warranty, create_ticket,<br/>schedule_service, issue_refund]
        MCP --> AUTH[scope check -> row filter -> audit]
        AUTH --> DB[(Postgres)]
    end
    AG -- requires_manager --> MA
    MA -- manager token: issue_refund --> MCP
    MA --> NT
    MCP -. OTLP .-> J[Jaeger / CloudWatch]
    agents -. OTLP .-> J
    KS[KILL_SWITCH] -. disables .-> MCP
```

Two identities travel with every request: the **user** (scoped token: `customer:read_own`,
`staff:read_all`, `manager:refund`) and the **agent** (`X-Agent-Name/Version`, from its
manifest). The server re-derives the principal from the token, so an agent can never upgrade its
own scopes; the audit row and the trace carry both.

## Quick start

```bash
conda create -n ssa python=3.12 -y && conda activate ssa   # or: python3.12 -m venv .venv && source .venv/bin/activate
make install                # pip install -e ".[dev]"; add ".[pii]" + `python -m spacy download en_core_web_lg` for Presidio
make test                   # 140 unit + integration tests, no services needed (Postgres tests skip)
make eval                   # offline eval -> scorecard (zero network)
python scripts/demo_offline.py   # the whole demo flow in-process

cp .env.example .env        # add ANTHROPIC_API_KEY (and OPENAI_API_KEY for embedding fallback)
make up                     # postgres+pgvector (host port 5433), ollama (pulls bge-m3), n8n, jaeger, mcp, agents
make tokens >> .env         # demo tokens for n8n; then `docker compose up -d n8n` again
make demo                   # scripts/demo.sh against the live stack; `make reset-demo` to run it again
```

Docker Desktop needs **at least 6 GB** of memory for the full stack (Ollama's BGE-M3 alone is ~1.1 GB;
Settings -> Resources -> Memory). Containers use the small spaCy model for Presidio; set
`SPACY_MODEL=en_core_web_lg` for higher-recall NER where memory allows.

Then open n8n at `http://localhost:5678` and import `n8n/workflows/*.json` (see `n8n/README.md`),
Jaeger at `http://localhost:16686`.

## Demo flow

1. Customer: *"my steering wheel cover arrived damaged, order 1042"* -> n8n intake agent fills the
   `IssueSpec`; `/intake/validate` lists missing fields and n8n re-asks.
2. `/resolve` under the **customer** token: knowledge agent retrieves, grades, answers with
   `[manual-steering-wheel-cover.md#Common issues]`; resolver calls `lookup_order` and
   `check_warranty`; pure rules propose a full refund (`requires_manager=true`); `create_ticket`.
3. Manager approves in n8n; n8n calls MCP `issue_refund` with the **manager's** token.
4. Show the trace (one trace, spans per step and tool), the access log (agent and user separate,
   subject pseudonymized), `make eval` scorecard, then flip the kill switch and watch every call
   return `forbidden: kill_switch`.

## Repository map

```
src/store_support/
  access/        rules.py (pure entitlement/data-auth), identity.py, tokens.py (HMAC scoped tokens)
  mcp_server/    server.py (MCP SDK, streamable HTTP), tools.py (registry + single execution path),
                 repo_postgres.py / repo_memory.py, audit.py
  knowledge/     pipeline.py (steps), langgraph_agent.py, sdk_agent.py, COMPARISON.md,
                 embeddings.py (Ollama -> OpenAI fallback), store.py (pgvector), chunking.py, ingest.py
  resolver/      rules.py (pure decisions), agent.py, service.py (HTTP for n8n), mcp_client.py
  intake/        schema.py (IssueSpec JSON schema + validation)
  governance/    manifest.py, kill_switch.py, redaction.py (Presidio), logging_setup.py,
                 routing.py (model tiers, pure), cost.py (prices, usage, ledger, budget)
  telemetry/     tracing.py (OTel: request > step.* > tool.*)
  evals/         datasets/ (30 questions, 20 scenarios), metrics.py, runner.py, drift.py, deepeval_suite.py
  lora/          distillation dataset, PEFT training, Ollama serving, student-vs-teacher eval
  data/          seed.py (Faker: 2 tenants, 200 customers, 50 SKUs, 500 orders)
agents/*/manifest.yaml     name, version, model, owner, tools, memory, limits
data/policies/*.md         13 policy and product-manual docs for the vector store
n8n/workflows/*.json       intake chat, manager approval, ticket notification
infra/terraform/           ECS Fargate x2 + RDS Postgres + ADOT sidecar (free-tier sized)
.github/workflows/         ci (lint, tests, evals, build), deploy (tag -> ECR + terraform), weekly drift
```

## Evals

`make eval` runs 30 labelled questions and 20 scripted scenarios and prints:

```
metric                     score   target
retrieval_hit_rate         0.967   >= 0.80  ok
faithfulness               0.867   >= 0.80  ok
tool_choice_accuracy       1.000   >= 1.00  ok
scope_enforcement          1.000   >= 1.00  ok
```

Offline mode (default, used in CI) swaps in a hashing embedder and an extractive answerer so the
numbers measure the pipeline deterministically. `--online` uses pgvector + Anthropic for the two
retrieval/generation metrics. The two governance metrics must be 1.0 or the job fails.
`weekly-eval.yml` reruns and `evals/drift.py` fails on a > 0.05 drop against the stored baseline.

### Judged tier: DeepEval

The scorecard is deterministic by design. A second, opt-in tier uses [DeepEval](https://github.com/confident-ai/deepeval)
with **Claude as the judge** on the same 30 questions and 20 scenarios:

| Metric | What it judges | Threshold |
|---|---|---|
| `FaithfulnessMetric` | every claim in the answer is supported by the retrieved chunks | 0.8 |
| `AnswerRelevancyMetric` | the answer addresses the question | 0.7 |
| `ContextualRelevancyMetric` | the retrieved chunks are on-topic | 0.5 |
| `ToolCorrectnessMetric` | resolver called exactly the expected tools, in order | 1.0 |

```bash
pip install -e ".[deepeval]"
make eval-deepeval                      # offline pipeline, judged (needs ANTHROPIC_API_KEY)
DEEPEVAL_MODE=online make eval-deepeval # live pgvector + Claude answers, judged (stack must be up)
```

`src/store_support/evals/deepeval_suite.py` converts results into `LLMTestCase`s; `tests/evals/`
holds the parametrized `assert_test` calls. The weekly workflow runs it when a key is configured.

## Reuse patterns

### Add an MCP tool in ~20 lines

```python
# 1. access/rules.py - declare the scope (the ONLY place scopes are mapped)
TOOL_SCOPES["cancel_order"] = Scope.STAFF_READ_ALL

# 2. mcp_server/tools.py - input model + handler + registry line
class CancelOrderIn(BaseModel):
    order_id: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=200)

def cancel_order(ctx: RequestContext, repo: Repository, a: CancelOrderIn) -> dict[str, Any]:
    o = _order_or_404(repo, ctx, a.order_id)          # row filter applied here
    if o.status != "placed":
        raise ToolError("business_rule", "only placed orders can be cancelled")
    return {"order_id": o.order_id, "status": repo.cancel(order=o, reason=a.reason)}

TOOLS["cancel_order"] = ToolSpec("cancel_order", "Cancel a placed order.", CancelOrderIn, cancel_order)

# 3. tests/unit/test_mcp_tools.py - write the denied/allowed/not_found cases first
```

The JSON schema, the scope check, the audit row, the span, and the kill switch come for free.

### Add a scope

```python
class Scope(StrEnum):
    ...
    SUPPORT_TRIAGE = "support:triage"

_SATISFIES[Scope.SUPPORT_TRIAGE] = frozenset({Scope.SUPPORT_TRIAGE, Scope.STAFF_READ_ALL})
```

Add a row to `test_access_rules.py` for what the scope must and must not satisfy. Tokens that
carry an unknown scope are rejected by `issue_token`/`parse_token`, so nothing else changes.

### Add an agent manifest

```yaml
# agents/triage-agent/manifest.yaml
name: triage-agent
version: 1.0.0
owner: support-platform@store.example
model: claude-sonnet-4-5
tools: [lookup_order, create_ticket]     # must exist in TOOL_SCOPES or the manifest fails to load
memory: { kind: session, ttl_minutes: 30 }
```

`load_all()` validates it, `/manifests` serves it, and `ResolverAgent.call_tool` refuses any tool
not in the list.

## Cost optimization & model routing

Every LLM call goes through `RoutedAnthropicLLM`, which asks a pure function which model tier should run it:

| Task | Tier (policy `routed`) | Why |
|---|---|---|
| grade the retrieved context | **cheap** (Haiku) | structured JSON output, high volume |
| rewrite the query | **cheap** | short, structured |
| filter chunks (context compaction) | **cheap** | returns a list of indices |
| answer, well-graded context | **default** (Sonnet) | customer-facing |
| answer after a retry, or on a borderline grade | **strong** (Opus) | spend more only where the first attempt struggled |
| anything, once the tenant's soft daily budget is spent | **cheap** (degraded) | keep serving, cheaper |
| anything, once the hard daily budget is spent | refused -> `escalate: budget_exhausted` | a person follows up |

Tiers come from the agent manifest (`agents/knowledge-agent/manifest.yaml`: `model` + `models: {cheap, strong}`),
the policy from `ROUTING_POLICY` (`routed | all_default | all_cheap`), budgets from `BUDGET_SOFT_USD_PER_DAY` /
`BUDGET_HARD_USD_PER_DAY`.

Four things make a request cheaper by construction:

1. **Context compaction** - a cheap filter call keeps only the relevant chunks before the answer model sees them
   (`step.compact`; the scorecard reports the compaction rate). This also lifted DeepEval's *contextual
   relevancy*, which was the weakest judged metric at the baseline (0.29 mean with k=5, no compaction).
2. **Prompt caching** - the system prompt and the retrieved-context block carry `cache_control`, so
   grade -> filter -> answer on the same context pays for that context once.
3. **Routing** - helper calls (3 of 4 per request) run on the cheap tier.
4. **Accounting you can act on** - every span carries `llm.model`, `llm.tier`, `llm.route_reason`,
   `llm.input_tokens`, `llm.cache_read_tokens`, `llm.cost_usd`; every `/resolve` response carries `cost_usd`,
   `tokens` and `models`; the `llm_call` log line sums by tenant in CloudWatch Insights.

```bash
make compare-routing     # live stack: all_default vs routed vs all_cheap on the 30 questions
```

prints a table of avg cost/question, tokens, faithfulness and retrieval hit rate per policy (saved to
`eval_results/routing_comparison.json`). Measured on the live stack (pgvector + Ollama BGE-M3 + Claude), 30 questions:

| policy | avg cost / question | avg tokens | faithfulness | retrieval hit | context compaction |
|---|---|---|---|---|---|
| all_default (Sonnet everywhere) | $0.0050 | 1212 | 0.95 | 1.00 | 76% |
| **routed** (Haiku for grade/rewrite/filter, Sonnet answer, Opus on retry) | **$0.0026 (-46%)** | 1220 | 0.97 | 1.00 | 77% |
| all_cheap (Haiku everywhere) | $0.0018 (-64%) | 1229 | 0.97 | 1.00 | 77% |

Tokens are flat across policies: the saving is routing, not shorter prompts. `routed` is the default
because the deterministic faithfulness metric is coarse (citation validity + a labelled fact); the
strong-tier upgrade on retries buys headroom on questions the first attempt struggles with, and the
DeepEval tier is where answer-quality differences between tiers would show.

## LoRA: a local, fine-tuned cheap tier

The cheap tier does two structured jobs - grade the context (`{"score": 0.8}`) and pick the
relevant chunks (`[0, 2]`). Those are small-model tasks, so the project includes a distillation
pipeline that fine-tunes a local model to replace Haiku for them:

```
make lora-dataset    # Haiku (teacher) labels grade/filter on ~200 synthetic questions -> data/lora/*.jsonl  (~$0.50)
make lora-train      # PEFT LoRA (r=16) on Qwen2.5-1.5B-Instruct; micro-batch 1 + grad checkpointing so it fits a 24 GB Mac
make lora-train-small  # 0.5B student, 1 epoch: ~9 min on an M-series Mac (what the numbers below were measured with)
make lora-serve      # merged weights -> GGUF (llama.cpp) -> Ollama container as `ssa-grader`
make lora-eval       # student vs teacher agreement on 128 held-out examples (tuned vs untuned base)
make compare-lora    # the routing table with the local tier as `cheap`
```

Measured (teacher = Haiku 4.5; 392 training examples; 64 held-out grade + 64 held-out filter prompts;
Qwen2.5-0.5B-Instruct, LoRA r=16, 1 epoch, 8m50s on an Apple-silicon Mac, train loss 1.17 -> 0.18):

| student                         | grade decision agreement (>= 0.7) | grade MAE | grade within +-0.15 | filter Jaccard |
|---------------------------------|-----------------------------------|-----------|---------------------|----------------|
| Qwen2.5-0.5B-Instruct, untuned  | 0.36                              | 0.66      | 0.31                | 0.57           |
| **+ LoRA (`ssa-grader`)**       | **0.84**                          | **0.18**  | **0.77**            | **0.69**       |

The untuned base is near chance on the pass/fail grading decision; nine minutes of LoRA brings it
to 84% agreement with the teacher at $0 per call (`eval_results/lora_*.json`).

Design points:

- **Same prompts.** The student is trained on the exact `GRADER_SYSTEM` / `FILTER_SYSTEM` prompt
  strings the router sends, so switching tiers is one env var (`CHEAP_MODEL=ollama/ssa-grader`),
  not a prompt change. Only assistant tokens are trained on.
- **Held-out means held-out.** The 30 labelled eval questions go into the validation split only;
  `lora-eval` reports grade MAE, agreement on the >= threshold decision, and filter Jaccard against
  the teacher on questions the student never saw.
- **Governance unchanged.** A local model is just another `ollama/...` model name in the manifest;
  routing, budgets, redaction, spans and the citation validator all apply. Cost is recorded as $0.
- **When it is worth it.** Not at demo volume - Haiku is already cheap. It pays when the grader must
  not send text off-box (PII residency), at high volume, or when helper-call latency dominates.

`src/store_support/lora/`: `dataset.py` (pure, tested), `build_dataset.py`, `train.py`, `evaluate.py`
(pure, tested), `eval_adapter.py`.

## LangGraph vs Claude Agent SDK

The knowledge agent is built twice over the same step functions. Short version: LangGraph puts
control flow in code (auditable, deterministic, easy to unit-test); the Agent SDK lets the model
drive while code still enforces the budget, the tenant, the citation check and the kill switch.
Details and the trade-offs in [`src/store_support/knowledge/COMPARISON.md`](src/store_support/knowledge/COMPARISON.md).

## AWS

`infra/terraform` deploys the same image as two Fargate services plus RDS Postgres (db.t3.micro)
with an ADOT sidecar exporting traces to X-Ray and logs to CloudWatch. Ollama is not deployed;
`FallbackEmbedder` moves to OpenAI automatically. `git tag v0.1.0 && git push --tags` runs
`deploy.yml`. See `infra/terraform/README.md`.

## Independent review

A separate reviewer audited tenant isolation, scope bypass, PII leakage and the kill switch after
the build. Two real bugs were found and fixed with tests (`tests/unit/test_hardening.py`,
`tests/integration/test_repo_postgres.py`): the Postgres refund was never committed on pooled
connections, and two concurrent approvals could both refund (now an atomic conditional `UPDATE`).
The one accepted limitation is documented in `n8n/README.md` (demo customer token).

## Development rules used here

- TDD: every module above has its tests written first (`tests/unit`, `tests/integration`).
- Pure functions for anything that decides (access rules, resolution rules, metrics).
- `tenant_id` is a required argument, never a default.
- PII columns (`full_name`, `email`, `phone`) are never selected by tool handlers.
- Python 3.11+, `ruff`, `mypy --strict` config in `pyproject.toml`.
