"""Wire real components from Settings. Everything else takes them as arguments."""

from __future__ import annotations

from typing import Literal

import psycopg

from store_support.config import Settings
from store_support.knowledge.embeddings import default_embedder
from store_support.knowledge.langgraph_agent import build_graph, run_graph
from store_support.knowledge.llm import AnthropicLLM
from store_support.knowledge.pipeline import KnowledgeResult
from store_support.knowledge.sdk_agent import SdkKnowledgeAgent
from store_support.knowledge.store import PgVectorStore

Flavor = Literal["langgraph", "sdk"]


class KnowledgeService:
    def __init__(self, settings: Settings, tenant_id: str, flavor: Flavor = "langgraph") -> None:
        self._s = settings
        self._tenant = tenant_id
        self._flavor = flavor
        self._conn = psycopg.connect(settings.database_url)
        emb = default_embedder(
            ollama_base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._store = PgVectorStore(self._conn, emb)
        self._llm = AnthropicLLM(settings.anthropic_api_key, settings.chat_model)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> KnowledgeService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def ask(self, question: str) -> KnowledgeResult:
        if self._flavor == "sdk":
            return await SdkKnowledgeAgent(
                store=self._store, llm=self._llm, tenant_id=self._tenant, model=self._s.chat_model
            ).run(question)
        graph = build_graph(store=self._store, llm=self._llm, tenant_id=self._tenant)
        return run_graph(graph, question)
