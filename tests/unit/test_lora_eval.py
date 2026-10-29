"""Adapter evaluation: how often does the student agree with the teacher on held-out examples?"""

from __future__ import annotations

from store_support.lora.dataset import Example
from store_support.lora.evaluate import agreement, grade_from_text, indices_from_text


def _ex(task: str, assistant: str) -> Example:
    return Example(
        task=task,
        question="q",
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": assistant},
        ],
    )


def test_parsers_tolerate_noise() -> None:
    assert grade_from_text('sure: {"score": 0.75, "why": "x"}') == 0.75
    assert grade_from_text("garbage") is None
    assert indices_from_text("[0, 2]") == [0, 2]
    assert indices_from_text("none") is None


def test_agreement_metrics() -> None:
    val = [
        _ex("grade", '{"score": 0.8}'),
        _ex("grade", '{"score": 0.2}'),
        _ex("filter", "[0, 1]"),
        _ex("filter", "[2]"),
    ]
    student = ['{"score": 0.7}', '{"score": 0.9}', "[0, 1]", "[1, 2]"]
    m = agreement(val, student, grade_tolerance=0.15, threshold=0.6)
    assert m["grade_mae"] == 0.4  # (0.1 + 0.7) / 2
    assert m["grade_decision_agreement"] == 0.5  # 0.8/0.7 both >= 0.6; 0.2/0.9 disagree
    assert m["filter_jaccard"] == 0.75  # 1.0 and 0.5
    assert m["n_grade"] == 2 and m["n_filter"] == 2
