"""Evaluate the served student (Ollama) against teacher labels on the held-out split.

    python -m store_support.lora.eval_adapter --model ssa-grader
    python -m store_support.lora.eval_adapter --model qwen2.5:1.5b-instruct     # the untuned base, for the before/after

Prints agreement metrics and writes eval_results/lora_<model>.json.
"""

from __future__ import annotations

import json
import sys

import typer

from store_support.config import Settings
from store_support.evals.runner import ROOT
from store_support.knowledge.llm import OllamaChat
from store_support.lora.evaluate import agreement, load_examples, run_student_over

app = typer.Typer(add_completion=False)


@app.command()
def main(model: str = "ssa-grader", limit: int = 0) -> None:
    val = load_examples(ROOT / "data" / "lora" / "val.jsonl")
    if limit:
        val = val[:limit]
    chat = OllamaChat(Settings.from_env().ollama_base_url, timeout=300)
    print(f"running {len(val)} held-out examples through ollama/{model}...", file=sys.stderr)
    outs = run_student_over(val, chat, model)
    m = agreement(val, outs)
    out = ROOT / "eval_results" / f"lora_{model.replace('/', '_').replace(':', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": model, **m}, indent=2))
    print(f"\n=== student vs teacher agreement: ollama/{model} ===")
    for k, v in m.items():
        print(f"{k:<28}{v}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    app()
