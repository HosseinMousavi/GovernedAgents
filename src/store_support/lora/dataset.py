"""Distillation dataset for the local cheap tier.

teacher (Haiku, or FakeLLM in tests) labels the two structured tasks the cheap tier performs:
  grade  : (question, k chunks) -> {"score": s, "why": "..."}
  filter : (question, k chunks) -> [i, j]
The student is trained on the exact prompt strings `RoutedAnthropicLLM` sends, so swapping the
tier is a config change, not a prompt change. Questions are PII-redacted before anything is stored.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from store_support.governance.redaction import redact_text
from store_support.knowledge.llm import FILTER_SYSTEM, GRADER_SYSTEM, LLM, _context_block, _numbered_block
from store_support.knowledge.store import Chunk, VectorStore

Message = dict[str, str]

_TEMPLATES = [
    "What does the policy say about {section}?",
    "Can you explain {section_lower}?",
    "I have a question about {section_lower} for my order.",
    "How does {section_lower} work?",
    "{title}: {section_lower} - what are the rules?",
    "Is there anything I should know about {section_lower}?",
]


@dataclass(frozen=True, slots=True)
class SyntheticQuestion:
    text: str
    source: str
    source_section: str


@dataclass(slots=True)
class Example:
    task: str  # grade | filter
    question: str
    messages: list[Message] = field(default_factory=list)


def synthetic_questions(chunks: Sequence[Chunk], *, per_chunk: int = 3, seed: int = 0) -> list[SyntheticQuestion]:
    """Template questions per section. Deterministic. (Online generation with a teacher is in
    `generate_questions_with_teacher`; templates keep tests and the offline path key-free.)"""
    rng = random.Random(seed)  # noqa: S311 - not security
    out: list[SyntheticQuestion] = []
    for c in chunks:
        title = c.content.split("\n", 1)[0].split(" - ")[0]
        picks = rng.sample(_TEMPLATES, k=min(per_chunk, len(_TEMPLATES)))
        for t in picks:
            text = t.format(section=c.section, section_lower=c.section.lower(), title=title)
            out.append(SyntheticQuestion(text=text, source=c.source, source_section=c.section))
    return out


def label_with_teacher(question: str, store: VectorStore, tenant_id: str, teacher: LLM, *, k: int = 5) -> list[Example]:
    question = redact_text(question)
    hits = store.search(tenant_id, question, k=k)
    if not hits:
        return []
    score = teacher.grade(question, hits)
    kept = teacher.filter(question, hits)
    kept_idx = sorted({i for i, h in enumerate(hits) if h in kept})
    grade_ex = Example(
        task="grade",
        question=question,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user", "content": f"Context:\n{_context_block(hits)}\n\nQuestion: {question}"},
            {"role": "assistant", "content": json.dumps({"score": round(score, 2), "why": "teacher label"})},
        ],
    )
    filter_ex = Example(
        task="filter",
        question=question,
        messages=[
            {"role": "system", "content": FILTER_SYSTEM},
            {
                "role": "user",
                "content": f"Chunks:\n{_numbered_block(hits)}\n\nQuestion: {question}\nRelevant chunk numbers:",
            },
            {"role": "assistant", "content": json.dumps(kept_idx)},
        ],
    )
    return [grade_ex, filter_ex]


def build_examples(
    questions: Sequence[SyntheticQuestion | str], store: VectorStore, tenant_id: str, teacher: LLM, *, k: int = 5
) -> list[Example]:
    out: list[Example] = []
    for q in questions:
        text = q if isinstance(q, str) else q.text
        out.extend(label_with_teacher(text, store, tenant_id, teacher, k=k))
    return out


def split_by_question(
    examples: Sequence[Example], *, val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Split on question so grade/filter pairs for one question never straddle train/val."""
    questions = sorted({e.question for e in examples})
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(questions)
    n_val = max(1, int(len(questions) * val_fraction)) if questions else 0
    val_q = set(questions[:n_val])
    return [e for e in examples if e.question not in val_q], [e for e in examples if e.question in val_q]


def write_jsonl(examples: Sequence[Example], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in examples:
            f.write(json.dumps({"task": e.task, "question": e.question, "messages": e.messages}) + "\n")
    return path


def generate_questions_with_teacher(
    chunks: Sequence[Chunk], teacher_client: object, *, per_chunk: int = 4, model: str = "claude-haiku-4-5"
) -> list[SyntheticQuestion]:
    """Online: ask the teacher for natural customer questions per chunk (more varied than templates)."""
    out: list[SyntheticQuestion] = []
    for c in chunks:
        msg = teacher_client.messages.create(  # type: ignore[attr-defined]
            model=model,
            max_tokens=400,
            system="You write realistic, varied customer-support questions. Reply with a JSON list of strings only.",
            messages=[
                {
                    "role": "user",
                    "content": f"Write {per_chunk} questions a customer might ask that this text answers:\n\n"
                    f"{c.content}",
                }
            ],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            start, end = text.index("["), text.rindex("]") + 1
            qs = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            continue
        out.extend(SyntheticQuestion(text=str(q), source=c.source, source_section=c.section) for q in qs[:per_chunk])
    return out
