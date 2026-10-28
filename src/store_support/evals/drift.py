"""Weekly drift check: compare eval_results/latest.json to baseline.json.

Exit 1 if any metric dropped more than `tolerance` (default 0.05). Run by the scheduled
GitHub Actions job (.github/workflows/weekly-eval.yml) and by `make drift`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from store_support.evals.runner import ROOT

app = typer.Typer(add_completion=False)


def compare(baseline: dict[str, float], latest: dict[str, float], tolerance: float) -> list[str]:
    return [
        f"{k}: {baseline[k]:.3f} -> {latest.get(k, 0.0):.3f}"
        for k in baseline
        if latest.get(k, 0.0) < baseline[k] - tolerance
    ]


@app.command()
def main(results_dir: Path = ROOT / "eval_results", tolerance: float = 0.05) -> None:
    base = json.loads((results_dir / "baseline.json").read_text())["metrics"]
    latest = json.loads((results_dir / "latest.json").read_text())["metrics"]
    regressions = compare(base, latest, tolerance)
    for k in base:
        print(f"{k:<24}{base[k]:>8.3f} -> {latest.get(k, 0.0):.3f}")
    if regressions:
        print("DRIFT: " + "; ".join(regressions), file=sys.stderr)
        raise typer.Exit(code=1)
    print("no drift beyond tolerance")


if __name__ == "__main__":
    app()
