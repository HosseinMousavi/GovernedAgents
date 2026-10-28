"""Online eval: same datasets, real embeddings + LLM + Postgres + MCP server.

Questions go through KnowledgeService (pgvector + Anthropic). Scenarios still run against the
in-memory tool path - tool choice and scope enforcement are deterministic code, so the online
run measures the parts that can drift: retrieval and generation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from store_support.config import Settings
from store_support.evals.metrics import faithfulness, retrieval_hit
from store_support.evals.runner import DATASETS, _mean, load_questions, run_offline
from store_support.knowledge.factory import KnowledgeService


def run_online() -> dict[str, Any]:
    settings = Settings.from_env()
    svc = KnowledgeService(settings, "store-a")
    rows = []
    for q in load_questions(DATASETS / "questions.jsonl"):
        res = asyncio.run(svc.ask(q.question))
        rows.append({"id": q.id, "hit": retrieval_hit(res.hits, q.expected_source),
                     "faithfulness": faithfulness(res.answer, res.citations, res.hits, q.keywords),
                     "outcome": res.outcome.value, "retries": res.retries})
    offline = run_offline()  # governance metrics come from the deterministic path
    return {
        "mode": "online", "ts": time.time(), "n_questions": len(rows), "n_scenarios": offline["n_scenarios"],
        "metrics": {
            "retrieval_hit_rate": _mean([r["hit"] for r in rows]),
            "faithfulness": _mean([r["faithfulness"] for r in rows]),
            "tool_choice_accuracy": offline["metrics"]["tool_choice_accuracy"],
            "scope_enforcement": offline["metrics"]["scope_enforcement"],
        },
        "questions": rows, "scenarios": offline["scenarios"],
    }
