"""Eval metrics. Pure functions over observed vs expected."""

from __future__ import annotations

from collections.abc import Sequence

from store_support.knowledge.pipeline import Citation
from store_support.knowledge.store import Hit


def retrieval_hit(hits: Sequence[Hit], expected_source: str) -> bool:
    """Was the labelled source among the retrieved chunks?"""
    return any(h.source == expected_source for h in hits)


def faithfulness(answer: str, citations: Sequence[Citation], hits: Sequence[Hit], keywords: Sequence[str]) -> float:
    """1.0 when every citation points at a retrieved chunk AND the labelled fact appears in the answer.
    0.5 when only one of the two holds. Answers with no citation score 0 (unsupported)."""
    if not answer or not citations:
        return 0.0
    retrieved = {(h.source, h.section) for h in hits} | {(h.source, None) for h in hits}
    cited_ok = all((c.source, c.section) in retrieved or (c.source, None) in retrieved for c in citations)
    fact_ok = any(k.lower() in answer.lower() for k in keywords)
    return (0.5 if cited_ok else 0.0) + (0.5 if fact_ok else 0.0)


def tool_choice(observed: Sequence[str], expected: Sequence[str]) -> bool:
    return list(observed) == list(expected)


def scope_enforced(*, expected_denied: bool, observed_denied: bool) -> bool:
    return expected_denied == observed_denied
