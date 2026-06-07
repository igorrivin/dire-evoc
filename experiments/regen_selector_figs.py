#!/usr/bin/env python3
"""Regenerate the guarded-auto selector docs figures + CSV from a selector-auto-full tree."""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "examples")
from plot_benchmark_metrics import write_selector_comparison  # noqa: E402

rows = write_selector_comparison(
    Path("artifacts/selector-auto-full"), [], Path("docs/_static"), "warm_mean"
)
print(f"generated {len(rows)} selector rows + benchmark_selector_auto_*.svg")
