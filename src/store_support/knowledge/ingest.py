"""Ingest markdown policies into pgvector for one tenant.

python -m store_support.knowledge.ingest data/policies --tenant store-a
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
import typer

from store_support.config import Settings
from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import default_embedder
from store_support.knowledge.store import PgVectorStore

app = typer.Typer(add_completion=False)


@app.command()
def main(folder: Path, tenant: str = "store-a", replace: bool = True) -> None:
    s = Settings.from_env()
    emb = default_embedder(ollama_base_url=s.ollama_base_url, model=s.embedding_model, openai_api_key=s.openai_api_key)
    with psycopg.connect(s.database_url) as conn:
        store = PgVectorStore(conn, emb)
        if replace:
            store.clear(tenant)
        total = 0
        for f in sorted(folder.glob("*.md")):
            n = store.add(tenant, chunk_markdown(f.read_text(), source=f.name))
            total += n
            print(f"{f.name}: {n} chunks", file=sys.stderr)
    print(f"ingested {total} chunks for {tenant} via {emb.name}", file=sys.stderr)


if __name__ == "__main__":
    app()
