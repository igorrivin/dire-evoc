#!/usr/bin/env python3
"""Generate benchmark comparison figures for the README and Sphinx docs.

By default this reads ``artifacts/cuda/*/metrics.json`` if present. The fallback data
below is the latest committed A10 Docker run so documentation can be rebuilt without
shipping the large artifact directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_static"


FALLBACK = [
    {
        "dataset": "DBpedia-14 BGE-small",
        "n": 49000,
        "d": 384,
        "classes": 14,
        "dire_auto": 0.6119548845413553,
        "evoc_auto": 0.5715246720755588,
        "dire_fixed": 0.6123469610019592,
        "evoc_fixed": 0.5715246720755588,
    },
    {
        "dataset": "Fashion-MNIST",
        "n": 60000,
        "d": 784,
        "classes": 10,
        "dire_auto": 0.36014040353610344,
        "evoc_auto": 0.32284983623757796,
        "dire_fixed": 0.39796536606907473,
        "evoc_fixed": 0.27259050699260645,
    },
    {
        "dataset": "Olivetti faces",
        "n": 400,
        "d": 4096,
        "classes": 40,
        "dire_auto": 0.07295727476998595,
        "evoc_auto": 0.05666574354700534,
        "dire_fixed": 0.02098216636204316,
        "evoc_fixed": 0.04109347396703659,
    },
    {
        "dataset": "Synthetic blobs 12k",
        "n": 12000,
        "d": 32,
        "classes": 8,
        "dire_auto": 0.2226520138450453,
        "evoc_auto": 1.0,
        "dire_fixed": 1.0,
        "evoc_fixed": 1.0,
    },
]

ORDER = {
    "DBpedia-14 BGE-small": 0,
    "Fashion-MNIST": 1,
    "Olivetti faces": 2,
    "Synthetic blobs 12k": 3,
}


def load_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        data = metrics["data"]
        dire = metrics["dire_evoc"]
        evoc = metrics["vanilla_evoc"]
        rows.append(
            {
                "dataset": data["label"],
                "n": data["n_samples"],
                "d": data["n_features"],
                "classes": data["n_classes"],
                "dire_auto": dire["auto_ari"],
                "evoc_auto": evoc["auto_ari"],
                "dire_fixed": dire.get("fixed_k_ari", float("nan")),
                "evoc_fixed": evoc.get("fixed_k_ari", float("nan")),
            }
        )
    return sorted(rows, key=lambda row: ORDER.get(row["dataset"], 100))


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def add_labels(ax, bars, fmt="{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def grouped_bar(
    rows: list[dict],
    path: Path,
    title: str,
    field_a: str,
    field_b: str,
    ylabel: str,
) -> None:
    labels = [row["dataset"] for row in rows]
    dire = np.array([row[field_a] for row in rows])
    evoc = np.array([row[field_b] for row in rows])

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    b1 = ax.bar(x - width / 2, dire, width, label="DiReEVoC", color="#2f6f9f")
    b2 = ax.bar(x + width / 2, evoc, width, label="Vanilla EVoC", color="#c56b37")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    add_labels(ax, b1)
    add_labels(ax, b2)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def delta_bar(rows: list[dict], path: Path) -> None:
    labels = [row["dataset"] for row in rows]
    delta = np.array([row["dire_auto"] - row["evoc_auto"] for row in rows])
    colors = ["#2f6f9f" if value >= 0 else "#c56b37" for value in delta]

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    bars = ax.bar(np.arange(len(labels)), delta, color=colors)
    ax.axhline(0.0, color="#2b2b2b", linewidth=1)
    ax.set_title("Auto ARI delta: DiReEVoC minus vanilla EVoC")
    ax.set_ylabel("Adjusted Rand Index delta")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    style_axis(ax)
    for bar, value in zip(bars, delta):
        offset = 3 if value >= 0 else -12
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            f"{value:+.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "artifacts" / "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = load_rows(input_dir) if input_dir.exists() else []
    if not rows:
        rows = FALLBACK

    OUT.mkdir(parents=True, exist_ok=True)
    grouped_bar(
        rows,
        OUT / "benchmark_auto_ari.svg",
        "CUDA benchmark auto ARI",
        "dire_auto",
        "evoc_auto",
        "Adjusted Rand Index",
    )
    grouped_bar(
        rows,
        OUT / "benchmark_fixed_k_ari.svg",
        "CUDA benchmark fixed-k ARI",
        "dire_fixed",
        "evoc_fixed",
        "Adjusted Rand Index",
    )
    delta_bar(rows, OUT / "benchmark_auto_delta.svg")
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
