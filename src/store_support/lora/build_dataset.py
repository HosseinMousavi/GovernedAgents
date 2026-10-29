"""Build the distillation dataset for the local cheap tier.

    python -m store_support.lora.build_dataset                 # teacher = Haiku (needs ANTHROPIC_API_KEY)
    python -m store_support.lora.build_dataset --offline       # teacher = extractive stand-in (no keys; smoke test)

Writes data/lora/train.jsonl and data/lora/val.jsonl. The 30 labelled eval questions are added to
the validation split only, so the held-out agreement numbers are on questions the student never saw.
"""

from __future__ import annotations

import sys

import typer

from store_support.evals.runner import DATASETS, ROOT, ExtractiveLLM, load_questions
from store_support.knowledge.chunking import chunk_markdown
from store_support.knowledge.embeddings import HashEmbedder
from store_support.knowledge.store import Chunk, InMemoryVectorStore
from store_support.lora.dataset import (
    build_examples,
    generate_questions_with_teacher,
    split_by_question,
    synthetic_questions,
    write_jsonl,
)

app = typer.Typer(add_completion=False)
OUT = ROOT / "data" / "lora"


def _chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for f in sorted((ROOT / "data" / "policies").glob("*.md")):
        out.extend(chunk_markdown(f.read_text(), source=f.name))
    return out


@app.command()
def main(offline: bool = False, per_chunk: int = 4, seed: int = 0, k: int = 5) -> None:
    chunks = _chunks()
    store = InMemoryVectorStore(HashEmbedder(dim=512))
    store.add("store-a", chunks)

    if offline:
        teacher = ExtractiveLLM()
        questions = synthetic_questions(chunks, per_chunk=per_chunk, seed=seed)
    else:
        from anthropic import Anthropic

        from store_support.config import Settings
        from store_support.governance.cost import CostLedger
        from store_support.governance.routing import ModelTiers
        from store_support.knowledge.llm import RoutedAnthropicLLM

        s = Settings.from_env()
        client = Anthropic(api_key=s.anthropic_api_key)
        teacher = RoutedAnthropicLLM(
            client=client,
            tiers=ModelTiers("claude-haiku-4-5", "claude-haiku-4-5", "claude-haiku-4-5"),
            tenant_id="lora-teacher",
            policy="all_cheap",
            ledger=CostLedger(),
        )
        print(f"generating ~{per_chunk} questions x {len(chunks)} chunks with Haiku...", file=sys.stderr)
        questions = generate_questions_with_teacher(chunks, client, per_chunk=per_chunk)
        questions += synthetic_questions(chunks, per_chunk=1, seed=seed)  # a few template ones for coverage

    print(f"labelling {len(questions)} questions ({'offline' if offline else 'Haiku'} teacher)...", file=sys.stderr)
    examples = build_examples(questions, store, "store-a", teacher, k=k)
    train, val = split_by_question(examples, val_fraction=0.15, seed=seed)
    heldout = build_examples(
        [q.question for q in load_questions(DATASETS / "questions.jsonl")], store, "store-a", teacher, k=k
    )
    val += heldout

    write_jsonl(train, OUT / "train.jsonl")
    write_jsonl(val, OUT / "val.jsonl")
    if not offline:
        _, cost, _ = teacher.drain_usage()
        print(f"teacher cost: ${cost:.3f}", file=sys.stderr)
    print(f"train={len(train)} val={len(val)} (incl. {len(heldout)} held-out eval questions) -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    app()
