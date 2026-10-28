"""Vector store. Tenant id is a mandatory argument on every read and write."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from psycopg import Connection

    from store_support.knowledge.embeddings import Embedder


@dataclass(frozen=True, slots=True)
class Chunk:
    source: str
    section: str
    content: str


@dataclass(frozen=True, slots=True)
class Hit:
    source: str
    section: str
    content: str
    score: float  # cosine similarity, higher is better

    @property
    def citation(self) -> str:
        return f"[{self.source}#{self.section}]"


class VectorStore(Protocol):
    def add(self, tenant_id: str, chunks: Sequence[Chunk]) -> int: ...
    def search(self, tenant_id: str, query: str, *, k: int = 5) -> list[Hit]: ...


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are unit-norm


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._e = embedder
        self._rows: list[tuple[str, Chunk, list[float]]] = []

    def add(self, tenant_id: str, chunks: Sequence[Chunk]) -> int:
        vecs = self._e.embed([c.content for c in chunks]) if chunks else []
        self._rows.extend((tenant_id, c, v) for c, v in zip(chunks, vecs, strict=True))
        return len(chunks)

    def search(self, tenant_id: str, query: str, *, k: int = 5) -> list[Hit]:
        q = self._e.embed([query])[0]
        scored = [(_cos(q, v), c) for t, c, v in self._rows if t == tenant_id]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [Hit(c.source, c.section, c.content, s) for s, c in scored[:k]]


class PgVectorStore:
    def __init__(self, conn: Connection, embedder: Embedder) -> None:
        from pgvector.psycopg import register_vector

        register_vector(conn)
        self._c = conn
        self._e = embedder

    def add(self, tenant_id: str, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        vecs = self._e.embed([c.content for c in chunks])
        with self._c.cursor() as cur:
            cur.executemany(
                "INSERT INTO doc_chunks (tenant_id, source, section, content, embedding) VALUES (%s,%s,%s,%s,%s)",
                [(tenant_id, c.source, c.section, c.content, v) for c, v in zip(chunks, vecs, strict=True)],
            )
        self._c.commit()
        return len(chunks)

    def clear(self, tenant_id: str, source: str | None = None) -> None:
        if source:
            self._c.execute("DELETE FROM doc_chunks WHERE tenant_id=%s AND source=%s", [tenant_id, source])
        else:
            self._c.execute("DELETE FROM doc_chunks WHERE tenant_id=%s", [tenant_id])
        self._c.commit()

    def search(self, tenant_id: str, query: str, *, k: int = 5) -> list[Hit]:
        q = self._e.embed([query])[0]
        rows = self._c.execute(
            """SELECT source, section, content, 1 - (embedding <=> %s::vector) AS score
               FROM doc_chunks WHERE tenant_id = %s
               ORDER BY embedding <=> %s::vector LIMIT %s""",
            [q, tenant_id, q, k],
        ).fetchall()
        return [Hit(r[0], r[1], r[2], float(r[3])) for r in rows]
