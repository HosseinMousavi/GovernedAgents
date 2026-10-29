"""LoRA distillation dataset: (question, chunks) -> labels from a teacher -> chat-format JSONL.
The prompts are the SAME strings the router sends to the cheap tier, so the student replaces it 1:1."""

from __future__ import annotations

import json
from pathlib import Path

from store_support.knowledge.embeddings import HashEmbedder
from store_support.knowledge.llm import FILTER_SYSTEM, GRADER_SYSTEM, FakeLLM
from store_support.knowledge.store import Chunk, InMemoryVectorStore
from store_support.lora.dataset import (
    Example,
    build_examples,
    label_with_teacher,
    split_by_question,
    synthetic_questions,
    write_jsonl,
)

CHUNKS = [
    Chunk("return-policy.md", "Refund timing", "Approved refunds post within 5-7 business days."),
    Chunk("return-policy.md", "Eligibility window", "Most items may be returned within 30 days of delivery."),
    Chunk("shipping-policy.md", "Tracking", "A tracking number is emailed when shipped."),
]


def test_synthetic_questions_cover_every_section_deterministically() -> None:
    qs = synthetic_questions(CHUNKS, per_chunk=2, seed=1)
    assert len(qs) == 6
    assert qs == synthetic_questions(CHUNKS, per_chunk=2, seed=1)
    assert any("Refund timing" in q.text or "refund" in q.text.lower() for q in qs)
    assert all(q.source_section for q in qs)


def test_label_with_teacher_produces_grade_and_filter_examples() -> None:
    store = InMemoryVectorStore(HashEmbedder(dim=64))
    store.add("store-a", CHUNKS)
    teacher = FakeLLM(grades=[0.8], keep=[0])
    ex = label_with_teacher("When will my refund arrive?", store, "store-a", teacher, k=3)
    assert {e.task for e in ex} == {"grade", "filter"}
    g = next(e for e in ex if e.task == "grade")
    assert g.messages[0] == {"role": "system", "content": GRADER_SYSTEM}
    assert "Question: When will my refund arrive?" in g.messages[1]["content"]
    assert json.loads(g.messages[2]["content"])["score"] == 0.8
    f = next(e for e in ex if e.task == "filter")
    assert f.messages[0]["content"] == FILTER_SYSTEM
    assert f.messages[2]["content"] == "[0]"


def test_build_examples_and_split_keep_questions_disjoint() -> None:
    store = InMemoryVectorStore(HashEmbedder(dim=64))
    store.add("store-a", CHUNKS)
    qs = synthetic_questions(CHUNKS, per_chunk=2, seed=3)
    teacher = FakeLLM(grades=[0.5] * 20, keep=[0, 1])
    examples = build_examples(qs, store, "store-a", teacher, k=3)
    assert len(examples) == 2 * len(qs)
    train, val = split_by_question(examples, val_fraction=0.34, seed=0)
    assert {e.question for e in train}.isdisjoint({e.question for e in val})
    assert len(train) + len(val) == len(examples)


def test_write_jsonl_round_trip(tmp_path: Path) -> None:
    ex = [Example(task="grade", question="q", messages=[{"role": "user", "content": "x"}])]
    p = write_jsonl(ex, tmp_path / "train.jsonl")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert rows[0]["messages"][0]["content"] == "x" and rows[0]["task"] == "grade"


def test_examples_never_contain_pii_from_questions() -> None:
    store = InMemoryVectorStore(HashEmbedder(dim=64))
    store.add("store-a", CHUNKS)
    ex = label_with_teacher("refund for jane@example.com please", store, "store-a", FakeLLM(grades=[0.5]), k=3)
    assert all("jane@example.com" not in m["content"] for e in ex for m in e.messages)
