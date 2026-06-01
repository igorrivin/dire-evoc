#!/usr/bin/env python3
"""Generate benchmark comparison figures for the README and Sphinx docs.

By default this reads the tuned CUDA artifact tree if present, then the conventional
``artifacts/cuda/*/metrics.json`` path. The embedded fallback data mirrors that tuned
A10 run so documentation can be rebuilt without shipping the large artifact directory.
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
    {"dataset": "Synthetic blobs 12k", "n": 9600, "d": 32, "classes": 8, "dire_auto": 1.0, "evoc_auto": 1.0, "dire_fixed": 1.0, "evoc_fixed": 1.0, "dire_fit": 10.3323, "evoc_fit": 11.7014},
    {"dataset": "Synthetic moons 12k", "n": 9600, "d": 8, "classes": 2, "dire_auto": 1.0, "evoc_auto": 1.0, "dire_fixed": 1.0, "evoc_fixed": 1.0, "dire_fit": 5.6835, "evoc_fit": 0.2690},
    {"dataset": "Synthetic circles 12k", "n": 9600, "d": 8, "classes": 2, "dire_auto": 1.0, "evoc_auto": 1.0, "dire_fixed": 1.0, "evoc_fixed": 1.0, "dire_fit": 5.6593, "evoc_fit": 0.2594},
    {"dataset": "MNIST", "n": 24000, "d": 784, "classes": 10, "dire_auto": 0.7225, "evoc_auto": 0.8128, "dire_fixed": 0.7730, "evoc_fixed": 0.8476, "dire_fit": 27.9191, "evoc_fit": 0.7386},
    {"dataset": "sklearn digits", "n": 1438, "d": 64, "classes": 10, "dire_auto": 0.2938, "evoc_auto": 0.3054, "dire_fixed": 0.8030, "evoc_fixed": 0.8725, "dire_fit": 9.9067, "evoc_fit": 0.1929},
    {"dataset": "sklearn iris", "n": 120, "d": 4, "classes": 3, "dire_auto": 0.5616, "evoc_auto": 0.2358, "dire_fixed": 0.5616, "evoc_fixed": 0.5817, "dire_fit": 6.6348, "evoc_fit": 0.2283},
    {"dataset": "sklearn wine", "n": 142, "d": 13, "classes": 3, "dire_auto": 0.2765, "evoc_auto": 0.1215, "dire_fixed": 0.4124, "evoc_fixed": 0.2570, "dire_fit": 7.3069, "evoc_fit": 0.1579},
    {"dataset": "sklearn breast cancer", "n": 455, "d": 30, "classes": 2, "dire_auto": 0.4038, "evoc_auto": 0.1160, "dire_fixed": 0.6691, "evoc_fixed": 0.5108, "dire_fit": 7.5310, "evoc_fit": 0.1720},
    {"dataset": "Fashion-MNIST", "n": 24000, "d": 784, "classes": 10, "dire_auto": 0.4397, "evoc_auto": 0.3903, "dire_fixed": 0.3983, "evoc_fixed": 0.2729, "dire_fit": 27.5410, "evoc_fit": 0.7639},
    {"dataset": "DBpedia-14 BGE-small", "n": 39200, "d": 384, "classes": 14, "dire_auto": 0.6067, "evoc_auto": 0.6228, "dire_fixed": 0.5592, "evoc_fixed": 0.6252, "dire_fit": 23.7640, "evoc_fit": 0.7596},
    {"dataset": "Olivetti faces", "n": 320, "d": 4096, "classes": 40, "dire_auto": 0.0275, "evoc_auto": 0.0634, "dire_fixed": 0.0056, "evoc_fixed": 0.0564, "dire_fit": 7.3803, "evoc_fit": 0.2827},
    {"dataset": "Forest CoverType", "n": 40000, "d": 54, "classes": 7, "dire_auto": 0.0158, "evoc_auto": -0.0256, "dire_fixed": 0.0064, "evoc_fixed": -0.0327, "dire_fit": 23.4873, "evoc_fit": 0.5902},
    {"dataset": "CIFAR-10", "n": 1200, "d": 3072, "classes": 10, "dire_auto": 0.0299, "evoc_auto": 0.0232, "dire_fixed": 0.0039, "evoc_fixed": 0.0080, "dire_fit": 10.3288, "evoc_fit": 0.4058},
]

DEFAULT_INPUT_DIRS = (
    ROOT / "artifacts" / "cuda-bmc-tuned",
    ROOT / "artifacts" / "cuda",
    ROOT / "artifacts" / "cuda-broad-163-192-33-223-fragment-merge2",
)

ORDER = {
    row["dataset"]: i for i, row in enumerate(FALLBACK)
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
                "dire_fit": dire.get("fit_seconds", float("nan")),
                "evoc_fit": evoc.get("fit_seconds", float("nan")),
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

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 5.2))
    b1 = ax.bar(x - width / 2, dire, width, label="DiReEVoC", color="#2f6f9f")
    b2 = ax.bar(x + width / 2, evoc, width, label="Pure EVoC", color="#c56b37")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    min_value = min(float(np.nanmin(dire)), float(np.nanmin(evoc)))
    ax.set_ylim(min(0.0, min_value * 1.2), 1.05)
    ax.axhline(0.0, color="#2b2b2b", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    add_labels(ax, b1)
    add_labels(ax, b2)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def runtime_bar(rows: list[dict], path: Path) -> None:
    labels = [row["dataset"] for row in rows]
    dire = np.array([row["dire_fit"] for row in rows])
    evoc = np.array([row["evoc_fit"] for row in rows])
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 5.2))
    b1 = ax.bar(x - width / 2, dire, width, label="DiReEVoC", color="#2f6f9f")
    b2 = ax.bar(x + width / 2, evoc, width, label="Pure EVoC", color="#c56b37")
    ax.set_title("CUDA benchmark fit runtime")
    ax.set_ylabel("seconds, log scale")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    add_labels(ax, b1, "{:.1f}s")
    add_labels(ax, b2, "{:.1f}s")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def runtime_ratio_bar(rows: list[dict], path: Path) -> None:
    labels = [row["dataset"] for row in rows]
    ratios = np.array([row["dire_fit"] / row["evoc_fit"] for row in rows])
    colors = ["#2f6f9f" if value <= 1.0 else "#c56b37" for value in ratios]

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 4.9))
    bars = ax.bar(np.arange(len(labels)), ratios, color=colors)
    ax.axhline(1.0, color="#2b2b2b", linewidth=1)
    ax.set_title("Runtime ratio: DiReEVoC / Pure EVoC")
    ax.set_ylabel("fit time ratio, log scale")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    style_axis(ax)
    for bar, value in zip(bars, ratios):
        ax.annotate(
            f"{value:.1f}x",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def delta_bar(rows: list[dict], path: Path) -> None:
    labels = [row["dataset"] for row in rows]
    delta = np.array([row["dire_auto"] - row["evoc_auto"] for row in rows])
    colors = ["#2f6f9f" if value >= 0 else "#c56b37" for value in delta]

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 4.9))
    bars = ax.bar(np.arange(len(labels)), delta, color=colors)
    ax.axhline(0.0, color="#2b2b2b", linewidth=1)
    ax.set_title("Auto ARI delta: DiReEVoC minus Pure EVoC")
    ax.set_ylabel("Adjusted Rand Index delta")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
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
    parser.add_argument("--input-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(args.input_dir)] if args.input_dir else list(DEFAULT_INPUT_DIRS)
    rows = []
    input_dir = None
    for candidate in input_dirs:
        if candidate.exists():
            rows = load_rows(candidate)
            if rows:
                input_dir = candidate
                break
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
    runtime_bar(rows, OUT / "benchmark_fit_seconds.svg")
    runtime_ratio_bar(rows, OUT / "benchmark_fit_ratio.svg")
    source = input_dir if input_dir is not None else "embedded fallback"
    print(f"Wrote figures to {OUT} from {source}")


if __name__ == "__main__":
    main()
