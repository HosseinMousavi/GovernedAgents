from __future__ import annotations

import json
from pathlib import Path

from store_support.evals.metrics import faithfulness, retrieval_hit, scope_enforced, tool_choice
from store_support.evals.runner import load_questions, load_scenarios, run_offline
from store_support.knowledge.pipeline import Citation
from store_support.knowledge.store import Hit

DATA = Path(__file__).resolve().parents[2] / "src" / "store_support" / "evals" / "datasets"


def test_datasets_have_required_sizes_and_fields() -> None:
    qs = load_questions(DATA / "questions.jsonl")
    sc = load_scenarios(DATA / "scenarios.jsonl")
    assert len(qs) == 30 and len(sc) == 20
    assert all(q.expected_source and q.keywords for q in qs)
    assert all(s.expected_action for s in sc)


def test_retrieval_hit() -> None:
    hits = [Hit("return-policy.md", "x", "c", 0.9), Hit("other.md", "y", "c", 0.5)]
    assert retrieval_hit(hits, "return-policy.md") is True
    assert retrieval_hit(hits, "nope.md") is False


def test_faithfulness_requires_keywords_and_known_citations() -> None:
    hits = [Hit("return-policy.md", "Refund timing", "Approved refunds post within 5-7 business days.", 0.9)]
    cites = [Citation("return-policy.md", "Refund timing")]
    assert (
        faithfulness(
            "Refunds post in 5-7 business days [return-policy.md#Refund timing]", cites, hits, ["5-7 business days"]
        )
        == 1.0
    )
    assert faithfulness("Refunds are instant", [], hits, ["5-7 business days"]) == 0.0


def test_tool_choice_exact_sequence() -> None:
    assert tool_choice(["lookup_order", "create_ticket"], ["lookup_order", "create_ticket"]) is True
    assert tool_choice(["lookup_order"], ["lookup_order", "create_ticket"]) is False


def test_scope_enforced() -> None:
    assert scope_enforced(expected_denied=True, observed_denied=True) is True
    assert scope_enforced(expected_denied=False, observed_denied=True) is False


def test_offline_run_produces_scorecard(tmp_path: Path) -> None:
    card = run_offline(out_dir=tmp_path)
    assert set(card["metrics"]) == {"retrieval_hit_rate", "faithfulness", "tool_choice_accuracy", "scope_enforcement"}
    assert card["metrics"]["scope_enforcement"] == 1.0  # access rules are deterministic
    assert card["metrics"]["tool_choice_accuracy"] == 1.0
    assert card["metrics"]["retrieval_hit_rate"] >= 0.8
    assert json.loads((tmp_path / "latest.json").read_text())["n_questions"] == 30


def test_drift_compare_flags_only_real_regressions() -> None:
    from store_support.evals.drift import compare

    base = {"retrieval_hit_rate": 0.9, "faithfulness": 0.8}
    assert compare(base, {"retrieval_hit_rate": 0.88, "faithfulness": 0.8}, 0.05) == []
    assert compare(base, {"retrieval_hit_rate": 0.7, "faithfulness": 0.8}, 0.05) == [
        "retrieval_hit_rate: 0.900 -> 0.700"
    ]
