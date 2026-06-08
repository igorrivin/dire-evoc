#!/usr/bin/env python3
"""Render the restart-regime comparison figure from consensus_experiment_results.json.

Grouped bars per dataset: single-run (mean, with min-max whisker), best-of-silhouette,
and consensus vote (best threshold). Writes an SVG into docs/_static for the docs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"single": "#5f6368", "best": "#c56b37", "consensus": "#2f6f9f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/consensus_experiment_results.json")
    ap.add_argument("--out", default="docs/_static/benchmark_consensus_voting.svg")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    names = [d["dataset"] for d in data]
    x = np.arange(len(names))
    w = 0.27

    single = np.array([d["single_run"]["ari_mean"] for d in data])
    lo = np.array([d["single_run"]["ari_min"] for d in data])
    hi = np.array([d["single_run"]["ari_max"] for d in data])
    best = np.array([d["best_of_silhouette"]["ari"] for d in data])
    cons = np.array([d["consensus"][d["consensus_best_threshold"]]["ari"] for d in data])
    yerr = np.stack([single - lo, hi - single])

    fig, ax = plt.subplots(figsize=(max(7.5, len(names) * 2.0), 4.8))
    b1 = ax.bar(x - w, single, w, yerr=yerr, capsize=4, color=COLORS["single"],
                label="single-run (mean; whisker=min-max)")
    b2 = ax.bar(x, best, w, color=COLORS["best"], label="best-of-silhouette")
    b3 = ax.bar(x + w, cons, w, color=COLORS["consensus"], label="consensus vote")

    n_runs = data[0]["restarts"]
    ax.set_ylabel("Adjusted Rand Index (vs ground truth)")
    ax.set_title(f"Restart regimes on small volatile datasets ({n_runs} restarts)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['dataset']}\n(n={d['n']})" for d in data])
    ax.set_ylim(0, max(hi.max(), best.max(), cons.max()) * 1.18)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for bars in (b1, b2, b3):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}", (rect.get_x() + rect.get_width() / 2,
                        rect.get_height()), ha="center", va="bottom", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="svg")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
