#!/usr/bin/env python3
"""Rebuild the scaling/stability figures from the committed experiment artifacts.

The scaling/stability study (``experiments/scale_compare.py`` and
``experiments/stability.py``) asks two questions about clustering a million
high-dimensional Wikipedia embeddings: does vanilla EVoC return at that scale, and is its
clustering *stable* as data is added? Timing and cluster counts come from
``experiments/scale_results.json``; the shared-anchor stability ARI comes from
``experiments/stability_results.json``. This script turns both into the documentation SVGs
under ``docs/_static`` so the figures stay in lockstep with the recorded numbers.

    python scripts/plot_scaling_metrics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCALE_RESULTS = ROOT / "experiments" / "scale_results.json"
STABILITY_RESULTS = ROOT / "experiments" / "stability_results.json"
DOCS_STATIC = ROOT / "docs" / "_static"

VANILLA_COLOR = "#c56b37"
DIRE_COLOR = "#2f6f9f"


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def normalize_svg(path: Path) -> None:
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    normalize_svg(path)


def load_scale_rows(path: Path) -> list[dict]:
    """Flatten the ``method@n`` checkpoint JSON into per-N vanilla/dire rows."""
    raw = json.loads(path.read_text())
    by_n: dict[int, dict] = {}
    for key, rec in raw.items():
        if rec.get("status") != "ok":
            continue
        method, n = key.split("@")
        n = int(n)
        row = by_n.setdefault(n, {"n": n})
        row[f"{method}_fit_s"] = rec.get("fit_s")
        row[f"{method}_clusters"] = rec.get("n_clusters")
        row[f"{method}_noise"] = rec.get("noise_frac")
        if "intrinsic_dim" in rec:
            row["intrinsic_dim"] = rec["intrinsic_dim"]
    return [by_n[n] for n in sorted(by_n)]


def annotate_points(ax, xs, ys, color, fmt="{:.0f}", dy=8) -> None:
    for x, y in zip(xs, ys):
        if y is None or not np.isfinite(y):
            continue
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )


def plot_stability_ari(stability: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    measurements = stability["measurements"]
    labels = [m["label"] for m in measurements]
    vanilla = np.array([m["vanilla_value"] for m in measurements], dtype=float)
    dire = np.array([m["dire_value"] for m in measurements], dtype=float)
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 2.3), 5.4))
    b1 = ax.bar(x - width / 2, vanilla, width, label="vanilla EVoC", color=VANILLA_COLOR)
    b2 = ax.bar(x + width / 2, dire, width, label="DiReEVoC", color=DIRE_COLOR)
    ax.set_title(
        "Stability ARI on a shared 100k anchor as N grows to 1M\n"
        "(self-consistency, not ground truth — higher = labels survive added data)"
    )
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(min(-0.15, float(min(vanilla.min(), dire.min())) - 0.05), 1.08)
    ax.axhline(0.0, color="#2b2b2b", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
            )
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def plot_fit_seconds(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    ns = np.array([r["n"] for r in rows], dtype=float)
    vanilla = np.array([r.get("vanilla_fit_s", np.nan) for r in rows], dtype=float)
    dire = np.array([r.get("dire_fit_s", np.nan) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(ns, vanilla, "o-", color=VANILLA_COLOR, label="vanilla EVoC", linewidth=2)
    ax.plot(ns, dire, "s-", color=DIRE_COLOR, label="DiReEVoC", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fit time vs N (Cohere Wikipedia, 1024-d) — both near-linear")
    ax.set_xlabel("points (N)")
    ax.set_ylabel("fit time (s), log scale")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n):,}" for n in ns], rotation=20, ha="right")
    annotate_points(ax, ns, vanilla, VANILLA_COLOR, "{:.0f}s", dy=-14)
    annotate_points(ax, ns, dire, DIRE_COLOR, "{:.0f}s", dy=8)
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)
    ax.grid(axis="x", alpha=0.12)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def plot_cluster_counts(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    ns = np.array([r["n"] for r in rows], dtype=float)
    vanilla = np.array([r.get("vanilla_clusters", np.nan) for r in rows], dtype=float)
    dire = np.array([r.get("dire_clusters", np.nan) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(ns, vanilla, "o-", color=VANILLA_COLOR, label="vanilla EVoC", linewidth=2)
    ax.plot(ns, dire, "s-", color=DIRE_COLOR, label="DiReEVoC", linewidth=2)
    ax.set_xscale("log")
    ax.set_title("Cluster count vs N — vanilla swings wildly, DiReEVoC holds a band")
    ax.set_xlabel("points (N)")
    ax.set_ylabel("number of clusters")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n):,}" for n in ns], rotation=20, ha="right")
    annotate_points(ax, ns, vanilla, VANILLA_COLOR, "{:.0f}", dy=8)
    annotate_points(ax, ns, dire, DIRE_COLOR, "{:.0f}", dy=-14)
    ax.legend(frameon=False, loc="center right")
    style_axis(ax)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="a scaling artifact tree (downloaded from the GPU host) containing "
        "scale_results.json and stability_results.json; overrides the committed defaults",
    )
    parser.add_argument("--scale-results", type=Path, default=SCALE_RESULTS)
    parser.add_argument("--stability-results", type=Path, default=STABILITY_RESULTS)
    parser.add_argument("--docs-static", type=Path, default=DOCS_STATIC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.artifacts is not None:
        args.scale_results = args.artifacts / "scale_results.json"
        args.stability_results = args.artifacts / "stability_results.json"
    rows = load_scale_rows(args.scale_results)
    stability = json.loads(args.stability_results.read_text())

    outputs = {
        "stability_ari": args.docs_static / "benchmark_scaling_stability_ari.svg",
        "fit_seconds": args.docs_static / "benchmark_scaling_fit_seconds.svg",
        "clusters": args.docs_static / "benchmark_scaling_clusters.svg",
    }
    plot_stability_ari(stability, outputs["stability_ari"])
    plot_fit_seconds(rows, outputs["fit_seconds"])
    plot_cluster_counts(rows, outputs["clusters"])

    print(json.dumps({
        "scale_points": [r["n"] for r in rows],
        "outputs": {k: str(v) for k, v in outputs.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
