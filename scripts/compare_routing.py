"""Routing policy comparison on the live stack: all_default vs routed vs all_cheap.

    python scripts/compare_routing.py            # needs the compose stack + ANTHROPIC_API_KEY
    python scripts/compare_routing.py --policies routed,all_default

Runs the 30 labelled questions under each policy through the real pipeline (pgvector + Claude),
then prints the table that answers "what did routing cost us in quality, and save us in dollars?".
Writes eval_results/routing_comparison.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from store_support.evals.online import run_online
from store_support.evals.runner import ROOT
from store_support.governance.logging_setup import configure_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(policies: str = "all_default,routed,all_cheap") -> None:
    import logging

    configure_logging(level=logging.WARNING)
    results: dict[str, dict] = {}
    for policy in [p.strip() for p in policies.split(",") if p.strip()]:
        os.environ["ROUTING_POLICY"] = policy
        print(f"\n--- policy={policy} ---", file=sys.stderr)
        results[policy] = run_online()
    out = ROOT / "eval_results" / "routing_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n| policy | avg cost / question | avg tokens | faithfulness | retrieval hit | context compaction |")
    print("|---|---|---|---|---|---|")
    base = results.get("all_default", {}).get("cost", {}).get("avg_cost_usd")
    for policy, card in results.items():
        c, m = card["cost"], card["metrics"]
        delta = f" ({(c['avg_cost_usd'] / base - 1):+.0%})" if base and policy != "all_default" else ""
        print(
            f"| {policy} | ${c['avg_cost_usd']:.4f}{delta} | {c['avg_tokens']:.0f} | {m['faithfulness']:.2f} | "
            f"{m['retrieval_hit_rate']:.2f} | {c['avg_context_compaction']:.0%} |"
        )
    print(f"\nsaved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    app()
