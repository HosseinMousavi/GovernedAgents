"""Student-vs-teacher agreement on held-out examples. Pure metrics + a runner over Ollama."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from store_support.lora.dataset import Example


def grade_from_text(text: str) -> float | None:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return float(json.loads(m.group(0))["score"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def indices_from_text(text: str) -> list[int] | None:
    m = re.search(r"\[[\d,\s]*\]", text)
    if not m:
        return None
    try:
        return sorted({int(i) for i in json.loads(m.group(0))})
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def agreement(
    val: Sequence[Example], student_outputs: Sequence[str], *, grade_tolerance: float = 0.15, threshold: float = 0.6
) -> dict[str, float | int]:
    """grade: MAE and agreement on the >= threshold decision. filter: mean Jaccard of index sets.
    Unparseable student output counts as a miss (that is the failure mode that matters in production)."""
    maes: list[float] = []
    decisions: list[float] = []
    jaccards: list[float] = []
    for ex, out in zip(val, student_outputs, strict=True):
        truth = ex.messages[-1]["content"]
        if ex.task == "grade":
            t, s = grade_from_text(truth), grade_from_text(out)
            if t is None:
                continue
            if s is None:
                maes.append(1.0)
                decisions.append(0.0)
                continue
            maes.append(abs(t - s))
            decisions.append(1.0 if (t >= threshold) == (s >= threshold) else 0.0)
        else:
            t_idx, s_idx = indices_from_text(truth), indices_from_text(out)
            if t_idx is None:
                continue
            if s_idx is None:
                jaccards.append(0.0)
                continue
            a, b = set(t_idx), set(s_idx)
            jaccards.append(1.0 if not a and not b else len(a & b) / len(a | b))
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
    return {
        "n_grade": len(maes),
        "n_filter": len(jaccards),
        "grade_mae": mean(maes),
        "grade_decision_agreement": mean(decisions),
        "grade_within_tolerance": mean([1.0 if m <= grade_tolerance else 0.0 for m in maes]),
        "filter_jaccard": mean(jaccards),
    }


def run_student_over(val: Sequence[Example], chat: Any, model: str) -> list[str]:
    """chat: OllamaChat-like with .chat(model=, system=, user=) -> (text, usage)."""
    outs = []
    for ex in val:
        system, user = ex.messages[0]["content"], ex.messages[1]["content"]
        text, _ = chat.chat(model=model, system=system, user=user)
        outs.append(text)
    return outs


def load_examples(path: Path) -> list[Example]:
    return [
        Example(task=r["task"], question=r["question"], messages=r["messages"])
        for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ]
