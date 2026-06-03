#!/usr/bin/env python3
"""Rebuild README and Sphinx benchmark figures from benchmark artifacts.

The benchmark artifacts remain the source of truth. By default this script reads the
local CUDA repeated-timing artifact trees, regenerates suite-level summaries using warm
mean fit time as the primary runtime metric, writes the combined smoke+default figures,
optionally rebuilds a scale-suite artifact tree, and copies the documentation SVG names
into ``docs/_static``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from benchmark_dataset_suite import metrics_row, plot_suite_summary, write_summary_csv
from benchmark_dataset_suite import timing_title


DEFAULT_ARTIFACTS = ROOT / "artifacts" / "cuda-timing-repeats-default"
SMOKE_ARTIFACTS = ROOT / "artifacts" / "cuda-timing-repeats-smoke"
COMBINED_ARTIFACTS = ROOT / "artifacts" / "cuda-timing-repeats-combined"
SCALE_ARTIFACTS = ROOT / "artifacts" / "cuda-scale"
DOCS_STATIC = ROOT / "docs" / "_static"


def read_suite_slugs(artifact_dir: Path) -> list[str]:
    csv_path = artifact_dir / "suite_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            return [row["slug"] for row in csv.DictReader(f)]
    return [
        path.parent.name
        for path in sorted(artifact_dir.glob("*/metrics.json"))
        if path.parent.name != "datasets"
    ]


def load_suite_rows(artifact_dir: Path, timing_primary: str) -> list[dict]:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact directory does not exist: {artifact_dir}")
    rows = []
    for slug in read_suite_slugs(artifact_dir):
        metrics_path = artifact_dir / slug / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics file: {metrics_path}")
        rows.append(metrics_row(slug, json.loads(metrics_path.read_text()), timing_primary))
    if not rows:
        raise RuntimeError(f"no per-dataset metrics found in {artifact_dir}")
    return rows


def write_artifact_summary(rows: list[dict], artifact_dir: Path, timing_primary: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(rows, artifact_dir / "suite_metrics.csv")
    plot_suite_summary(rows, artifact_dir, timing_primary)
    normalize_svg_outputs(artifact_dir.glob("suite_*.svg"))


def prefixed_rows(source: str, rows: Iterable[dict]) -> list[dict]:
    prefixed = []
    for row in rows:
        combined = dict(row)
        combined["source"] = source
        combined["combined_slug"] = f"{source}_{row['slug']}"
        combined["dataset"] = f"{source}:{row['dataset']}"
        combined["output_dir"] = f"{source}/{row['output_dir']}"
        prefixed.append(combined)
    return prefixed


def write_combined_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    leading = ["source", "slug", "combined_slug"]
    fieldnames = leading + [key for key in rows[0] if key not in leading]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg(path: Path) -> None:
    if path.suffix != ".svg":
        return
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def normalize_svg_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            normalize_svg(path)


def copy_svg(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"expected SVG was not generated: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    normalize_svg(dst)


def copy_suite_svgs_to_docs(
    default_dir: Path,
    combined_dir: Path,
    docs_static: Path,
    include_combined: bool,
) -> None:
    copies = {
        default_dir / "suite_auto_ari.svg": docs_static / "benchmark_auto_ari.svg",
        default_dir / "suite_auto_delta.svg": docs_static / "benchmark_auto_delta.svg",
        default_dir / "suite_fit_seconds.svg": docs_static / "benchmark_fit_seconds.svg",
        default_dir / "suite_fit_seconds_warm.svg": docs_static / "benchmark_fit_seconds_warm.svg",
        default_dir
        / "suite_fit_seconds_mean_all.svg": docs_static
        / "benchmark_fit_seconds_mean_all.svg",
    }
    if include_combined:
        copies.update(
            {
                combined_dir
                / "combined_auto_ari.svg": docs_static
                / "benchmark_combined_auto_ari.svg",
                combined_dir
                / "combined_auto_delta.svg": docs_static
                / "benchmark_combined_auto_delta.svg",
                combined_dir
                / "combined_fit_seconds.svg": docs_static
                / "benchmark_combined_fit_seconds.svg",
                combined_dir
                / "combined_fit_seconds_warm.svg": docs_static
                / "benchmark_combined_fit_seconds_warm.svg",
                combined_dir
                / "combined_fit_seconds_mean_all.svg": docs_static
                / "benchmark_combined_fit_seconds_mean_all.svg",
            }
        )
    for src, dst in copies.items():
        copy_svg(src, dst)


def copy_scale_svgs_to_docs(scale_dir: Path, docs_static: Path) -> None:
    copies = {
        scale_dir / "suite_auto_ari.svg": docs_static / "benchmark_scale_auto_ari.svg",
        scale_dir / "suite_auto_delta.svg": docs_static / "benchmark_scale_auto_delta.svg",
        scale_dir / "suite_fit_seconds.svg": docs_static / "benchmark_scale_fit_seconds.svg",
        scale_dir
        / "suite_fit_seconds_warm.svg": docs_static
        / "benchmark_scale_fit_seconds_warm.svg",
        scale_dir
        / "suite_fit_seconds_mean_all.svg": docs_static
        / "benchmark_scale_fit_seconds_mean_all.svg",
        scale_dir / "suite_fixed_k_ari.svg": docs_static / "benchmark_scale_fixed_k_ari.svg",
        scale_dir / "suite_fit_ratio.svg": docs_static / "benchmark_scale_fit_ratio.svg",
    }
    for src, dst in copies.items():
        copy_svg(src, dst)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def clean_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def add_labels(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def grouped_ari_bar(
    rows: list[dict],
    path: Path,
    title: str,
    field_a: str,
    field_b: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["dataset"] for row in rows]
    dire = np.array([clean_float(row[field_a]) for row in rows])
    pure = np.array([clean_float(row[field_b]) for row in rows])
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 5.2))
    b1 = ax.bar(x - width / 2, dire, width, label="DiReEVoC", color="#2f6f9f")
    b2 = ax.bar(x + width / 2, pure, width, label="Pure EVoC", color="#c56b37")
    ax.set_title(title)
    ax.set_ylabel("Adjusted Rand Index")
    finite = np.concatenate([dire[np.isfinite(dire)], pure[np.isfinite(pure)]])
    min_value = min(0.0, float(np.min(finite)) * 1.2) if finite.size else 0.0
    ax.set_ylim(min_value, 1.05)
    ax.axhline(0.0, color="#2b2b2b", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    add_labels(ax, b1, "{:.3f}")
    add_labels(ax, b2, "{:.3f}")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    normalize_svg(path)


def runtime_ratio_bar(rows: list[dict], path: Path, timing_primary: str) -> None:
    import matplotlib.pyplot as plt

    labels = [row["dataset"] for row in rows]
    ratios = np.array(
        [
            clean_float(row["dire_fit_seconds"]) / clean_float(row["pure_fit_seconds"])
            for row in rows
        ]
    )
    colors = ["#2f6f9f" if value <= 1.0 else "#c56b37" for value in ratios]

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 4.9))
    bars = ax.bar(np.arange(len(labels)), ratios, color=colors)
    ax.axhline(1.0, color="#2b2b2b", linewidth=1)
    ax.set_title(f"Runtime ratio: DiReEVoC / Pure EVoC ({timing_title(rows, timing_primary)})")
    ax.set_ylabel("fit time ratio, log scale")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    style_axis(ax)
    for bar, value in zip(bars, ratios):
        if not np.isfinite(value):
            continue
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
    normalize_svg(path)


def rename_combined_suite_svgs(combined_dir: Path) -> None:
    renames = {
        "suite_auto_ari.svg": "combined_auto_ari.svg",
        "suite_auto_delta.svg": "combined_auto_delta.svg",
        "suite_fit_seconds.svg": "combined_fit_seconds.svg",
        "suite_fit_seconds_warm.svg": "combined_fit_seconds_warm.svg",
        "suite_fit_seconds_mean_all.svg": "combined_fit_seconds_mean_all.svg",
    }
    for src_name, dst_name in renames.items():
        src = combined_dir / src_name
        dst = combined_dir / dst_name
        copy_svg(src, dst)
        src.unlink()


def build_combined_artifacts(
    smoke_rows: list[dict],
    default_rows: list[dict],
    combined_dir: Path,
    timing_primary: str,
    smoke_artifacts: Path,
    default_artifacts: Path,
) -> list[dict]:
    combined_rows = [
        *prefixed_rows("smoke", smoke_rows),
        *prefixed_rows("default", default_rows),
    ]
    combined_dir.mkdir(parents=True, exist_ok=True)
    write_combined_csv(combined_rows, combined_dir / "combined_suite_metrics.csv")
    plot_suite_summary(combined_rows, combined_dir, timing_primary)
    rename_combined_suite_svgs(combined_dir)
    summary = {
        "timing_primary": timing_primary,
        "timing_policy": (
            "mean_all_first_included"
            if timing_primary == "mean_all"
            else "warm_mean_first_discarded"
        ),
        "sources": {
            "smoke": str(smoke_artifacts),
            "default": str(default_artifacts),
        },
        "rows": combined_rows,
    }
    (combined_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return combined_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--smoke-artifacts", type=Path, default=SMOKE_ARTIFACTS)
    parser.add_argument("--combined-artifacts", type=Path, default=COMBINED_ARTIFACTS)
    parser.add_argument("--scale-artifacts", type=Path, default=SCALE_ARTIFACTS)
    parser.add_argument("--docs-static", type=Path, default=DOCS_STATIC)
    parser.add_argument(
        "--timing-primary",
        default="warm_mean",
        choices=["warm_mean", "mean_all"],
        help="runtime summary used for primary fit_seconds figures",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="only rebuild the default-suite docs figures",
    )
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="do not rebuild scale-suite docs figures, even if scale artifacts exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_rows = load_suite_rows(args.default_artifacts, args.timing_primary)
    write_artifact_summary(default_rows, args.default_artifacts, args.timing_primary)
    grouped_ari_bar(
        default_rows,
        args.default_artifacts / "suite_fixed_k_ari.svg",
        "CUDA benchmark fixed-k ARI",
        "dire_fixed_k_ari",
        "pure_fixed_k_ari",
    )
    runtime_ratio_bar(
        default_rows,
        args.default_artifacts / "suite_fit_ratio.svg",
        args.timing_primary,
    )

    combined_rows = []
    if not args.skip_combined:
        smoke_rows = load_suite_rows(args.smoke_artifacts, args.timing_primary)
        write_artifact_summary(smoke_rows, args.smoke_artifacts, args.timing_primary)
        combined_rows = build_combined_artifacts(
            smoke_rows,
            default_rows,
            args.combined_artifacts,
            args.timing_primary,
            args.smoke_artifacts,
            args.default_artifacts,
        )

    copy_suite_svgs_to_docs(
        args.default_artifacts,
        args.combined_artifacts,
        args.docs_static,
        include_combined=not args.skip_combined,
    )
    copy_svg(
        args.default_artifacts / "suite_fixed_k_ari.svg",
        args.docs_static / "benchmark_fixed_k_ari.svg",
    )
    copy_svg(
        args.default_artifacts / "suite_fit_ratio.svg",
        args.docs_static / "benchmark_fit_ratio.svg",
    )

    scale_rows = []
    if not args.skip_scale and args.scale_artifacts.exists():
        scale_rows = load_suite_rows(args.scale_artifacts, args.timing_primary)
        write_artifact_summary(scale_rows, args.scale_artifacts, args.timing_primary)
        grouped_ari_bar(
            scale_rows,
            args.scale_artifacts / "suite_fixed_k_ari.svg",
            "CUDA scale benchmark fixed-k ARI",
            "dire_fixed_k_ari",
            "pure_fixed_k_ari",
        )
        runtime_ratio_bar(
            scale_rows,
            args.scale_artifacts / "suite_fit_ratio.svg",
            args.timing_primary,
        )
        copy_scale_svgs_to_docs(args.scale_artifacts, args.docs_static)

    print(
        json.dumps(
            {
                "timing_primary": args.timing_primary,
                "default_rows": len(default_rows),
                "combined_rows": len(combined_rows),
                "scale_rows": len(scale_rows),
                "docs_static": str(args.docs_static),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
