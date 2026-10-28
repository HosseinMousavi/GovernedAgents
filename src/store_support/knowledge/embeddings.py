"""Embedding providers. Ollama (local BGE) first, OpenAI fallback, hash for tests.

All providers return 1024-d vectors (the pgvector column width). OpenAI's
text-embedding-3-small supports `dimensions=1024` natively.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

DIM = 1024


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_STOP = frozenset("a an the my i me it is are was be to of in on for and or what can do with this that you your".split())
_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing. Not semantic; good enough for tests and offline evals."""

    name = "hash"

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in _TOKEN.findall(t.lower()):
                if tok in _STOP:
                    continue
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class OllamaEmbedder:
    name = "ollama"
    dim = DIM

    def __init__(self, base_url: str, model: str = "bge-m3", timeout: float = 60.0) -> None:
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        r = self._client.post(self._url, json={"model": self._model, "input": list(texts)})
        r.raise_for_status()
        return r.json()["embeddings"]  # type: ignore[no-any-return]


class OpenAIEmbedder:
    name = "openai"
    dim = DIM

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        res = self._client.embeddings.create(model=self._model, input=list(texts), dimensions=DIM)
        return [d.embedding for d in res.data]


class FallbackEmbedder:
    """Try primary; on any error use fallback. Records which one served the last call (for traces)."""

    def __init__(self, primary: Embedder, fallback: Embedder) -> None:
        assert primary.dim == fallback.dim
        self._p, self._f = primary, fallback
        self.dim = primary.dim
        self.name = f"{primary.name}|{fallback.name}"
        self.last_used = primary.name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            out = self._p.embed(texts)
            self.last_used = self._p.name
            return out
        except Exception:  # noqa: BLE001 - any provider failure triggers fallback
            out = self._f.embed(texts)
            self.last_used = self._f.name
            return out


def default_embedder(*, ollama_base_url: str, model: str, openai_api_key: str) -> Embedder:
    primary = OllamaEmbedder(ollama_base_url, model)
    if openai_api_key:
        return FallbackEmbedder(primary, OpenAIEmbedder(openai_api_key))
    return primary
