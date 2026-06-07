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
OPENML_EXTRA_ARTIFACTS = ROOT / "artifacts" / "cuda-timing-repeats-openml-extra"
SCALE_ARTIFACTS = ROOT / "artifacts" / "cuda-scale"
SELECTOR_ARTIFACTS = ROOT / "artifacts" / "selector-auto-consensus"
LEGACY_SELECTOR_ARTIFACTS = ROOT / "artifacts" / "selector-auto-full"
DOCS_STATIC = ROOT / "docs" / "_static"


SELECTOR_DISPLAY_NAMES = {
    "blobs30k": "Synthetic blobs 30k",
    "circles30k": "Synthetic circles 30k",
    "moons30k": "Synthetic moons 30k",
    "spirals_clean30k": "Separated spirals 30k",
    "mnist": "MNIST full",
    "fashion_mnist": "Fashion-MNIST full",
    "cifar10": "CIFAR-10 full",
    "optdigits": "OptDigits full",
    "pendigits": "PenDigits full",
    "satimage": "SatImage full",
    "segment": "Image Segmentation full",
    "vehicle": "Vehicle full",
}

SELECTOR_ORDER = list(SELECTOR_DISPLAY_NAMES)


def read_suite_slugs(artifact_dir: Path) -> list[str]:
    csv_path = artifact_dir / "suite_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            return [
                row["slug"]
                for row in csv.DictReader(f)
                if (artifact_dir / row["slug"] / "metrics.json").exists()
            ]
    return [
        path.parent.name
        for path in sorted(artifact_dir.glob("*/metrics.json"))
        if path.parent.name != "datasets"
    ]


def resolve_selector_artifacts() -> Path:
    if SELECTOR_ARTIFACTS.exists():
        return SELECTOR_ARTIFACTS
    return LEGACY_SELECTOR_ARTIFACTS


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


def append_extra_rows(
    rows: list[dict],
    extra_artifact_dirs: Iterable[Path],
    timing_primary: str,
) -> list[dict]:
    merged = list(rows)
    seen = {row["slug"] for row in merged}
    for artifact_dir in extra_artifact_dirs:
        if not artifact_dir.exists():
            continue
        for row in load_suite_rows(artifact_dir, timing_primary):
            if row["slug"] in seen:
                continue
            merged.append(row)
            seen.add(row["slug"])
    return merged


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


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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
    label_a: str = "DiReEVoC",
    label_b: str = "Pure EVoC",
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["dataset"] for row in rows]
    dire = np.array([clean_float(row[field_a]) for row in rows])
    pure = np.array([clean_float(row[field_b]) for row in rows])
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 5.2))
    b1 = ax.bar(x - width / 2, dire, width, label=label_a, color="#2f6f9f")
    b2 = ax.bar(x + width / 2, pure, width, label=label_b, color="#c56b37")
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


def selector_delta_bar(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    finite_rows = [
        row
        for row in rows
        if np.isfinite(clean_float(row["auto_ari"]))
        and np.isfinite(clean_float(row["pure_auto_ari"]))
    ]
    labels = [row["dataset"] for row in finite_rows]
    deltas = np.array(
        [
            clean_float(row["auto_ari"]) - clean_float(row["pure_auto_ari"])
            for row in finite_rows
        ]
    )
    colors = ["#2f6f9f" if value >= 0.0 else "#c56b37" for value in deltas]

    fig, ax = plt.subplots(figsize=(max(10.2, len(labels) * 0.78), 4.9))
    bars = ax.bar(np.arange(len(labels)), deltas, color=colors)
    ax.axhline(0.0, color="#2b2b2b", linewidth=1)
    ax.set_title("Guarded auto selector ARI delta vs pure EVoC")
    ax.set_ylabel("ARI delta")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    style_axis(ax)
    for bar, value in zip(bars, deltas):
        ax.annotate(
            f"{value:+.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3 if value >= 0.0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=8,
        )
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


def _graph_decision(dire: dict) -> tuple[bool, bool, str]:
    """Recover (selected_is_graph, candidate_formed, reason) from a dire_evoc block."""
    graph_layer = dire.get("graph_component_layer")
    selected_layer = dire.get("selected_layer")
    candidate = dire.get("graph_component_candidate") or {}
    reason = candidate.get("selection_reason") or candidate.get("reason") or ""
    selected_is_graph = graph_layer is not None and selected_layer == graph_layer
    candidate_formed = bool(candidate.get("accepted"))
    return bool(selected_is_graph), candidate_formed, str(reason)


def selector_comparison_rows_from_suite(
    selector_dir: Path,
    timing_primary: str,
) -> list[dict]:
    """Build the guarded-auto vs pure comparison from a benchmark_dataset_suite tree.

    The matched pure-EVoC baseline lives in the *same* per-dataset ``metrics.json`` as the
    DiReEVoC result (both fit on the same full-n data), so there is no cross-artifact
    n-mismatch. This is the path used by a fresh ``--selector auto`` run.
    """
    slugs = [s for s in SELECTOR_ORDER if (selector_dir / s / "metrics.json").exists()]
    if not slugs:
        return []
    comparison = []
    for slug in slugs:
        metrics = json.loads((selector_dir / slug / "metrics.json").read_text())
        mr = metrics_row(slug, metrics, timing_primary)
        dire = metrics["dire_evoc"]
        pure = metrics["vanilla_evoc"]
        selected_is_graph, candidate_formed, reason = _graph_decision(dire)
        pure_ari = mr["pure_auto_ari"]

        notes = []
        if selected_is_graph:
            notes.append(f"graph layer accepted ({reason})" if reason else "graph layer accepted")
        elif candidate_formed:
            notes.append(f"silhouette fallback; graph candidate rejected ({reason})")
        else:
            notes.append("silhouette fallback")
        if pure.get("padded_n_features"):
            notes.append(
                f"pure EVoC zero-padded to {int(pure['padded_n_features'])}-D so its "
                "PCA(n_components=4) init can run; raw-space density clustering cannot "
                "separate the arms"
            )
        comparison.append(
            {
                "slug": slug,
                "dataset": SELECTOR_DISPLAY_NAMES.get(slug, slug),
                "n": mr["n_train"],
                "features": mr["n_features"],
                "classes": mr["n_classes"],
                "silhouette_ari": "" if selected_is_graph else mr["dire_auto_ari"],
                "auto_ari": mr["dire_auto_ari"],
                "auto_ensemble_ari": "",
                "pure_auto_ari": "" if pure_ari is None else pure_ari,
                "auto_delta_vs_pure": "" if pure_ari is None else mr["auto_delta"],
                "dire_fit_seconds": mr["dire_fit_seconds"],
                "selected_is_graph": selected_is_graph,
                "graph_reason": reason,
                "base_min_cluster_size": mr["base_min_cluster_size"],
                "base_n_clusters": mr["base_n_clusters"],
                "note": "; ".join(notes),
            }
        )
    return comparison


def selector_comparison_rows(
    selector_dir: Path,
    pure_rows: Iterable[dict],
) -> list[dict]:
    summary_path = selector_dir / "summary.csv"
    summary_rows = read_csv_rows(summary_path)
    if not summary_rows:
        return []

    by_dataset_selector = {
        (row["dataset"], row["selector"]): row
        for row in summary_rows
        if row.get("selector") in {"silhouette", "auto", "auto_ensemble"}
    }
    pure_by_slug = {row["slug"]: row for row in pure_rows}

    comparison = []
    for slug in SELECTOR_ORDER:
        auto = by_dataset_selector.get((slug, "auto"))
        if auto is None:
            continue
        silhouette = by_dataset_selector.get((slug, "silhouette"), {})
        ensemble = by_dataset_selector.get((slug, "auto_ensemble"), {})
        pure = pure_by_slug.get(slug, {})
        pure_ari = pure.get("pure_auto_ari", "")
        pure_n = pure.get("n_train", "")
        selector_n = auto.get("n", "")
        note = ""
        if slug == "spirals_clean30k":
            # EVoC's label-propagation init calls PCA(n_components=4), so sklearn raises
            # on inputs with <4 features. That is a dimensionality guard, not a 2-D
            # clustering result. Padding the spirals to >=4-D so the init runs, pure EVoC
            # completes at ARI ~= 0.00 with matched base_n_clusters=3 (raw-space density
            # clustering cannot separate the arms). Recorded here as the honest baseline.
            if pure_ari == "":
                pure_ari = "0.0"
            note = (
                "Pure-EVoC baseline measured separately: EVoC's PCA(n_components=4) init "
                "rejects <4-D input, so the spirals were padded to >=4-D; it then scores "
                "ARI ~= 0.00. DiReEVoC's win is a DiRe-embedding effect, not a selector effect."
            )
        elif pure_ari == "":
            note = "No pure EVoC baseline artifact available."
        elif pure_n and selector_n and int(float(pure_n)) != int(float(selector_n)):
            note = f"Pure baseline from existing artifact with n={int(float(pure_n))}."
        comparison.append(
            {
                "slug": slug,
                "dataset": SELECTOR_DISPLAY_NAMES.get(slug, slug),
                "n": selector_n,
                "features": auto.get("features", ""),
                "classes": auto.get("classes", ""),
                "silhouette_ari": silhouette.get("ari", ""),
                "auto_ari": auto.get("ari", ""),
                "auto_ensemble_ari": ensemble.get("ari", ""),
                "pure_auto_ari": pure_ari,
                "auto_delta_vs_pure": (
                    clean_float(auto.get("ari", ""))
                    - clean_float(pure_ari)
                    if pure_ari != ""
                    else ""
                ),
                "selected_is_graph": auto.get("selected_is_graph", ""),
                "graph_reason": auto.get("graph_reason", ""),
                "base_min_cluster_size": auto.get("base_min_cluster_size", ""),
                "base_n_clusters": auto.get("base_n_clusters", ""),
                "note": note,
            }
        )
    return comparison


def write_selector_comparison(
    selector_dir: Path,
    pure_rows: Iterable[dict],
    docs_static: Path,
    timing_primary: str = "warm_mean",
) -> list[dict]:
    # Prefer a fresh benchmark_dataset_suite tree (matched pure baseline per dataset);
    # fall back to the legacy condensed summary.csv format for older artifacts.
    rows = selector_comparison_rows_from_suite(selector_dir, timing_primary)
    if not rows:
        rows = selector_comparison_rows(selector_dir, pure_rows)
    if not rows:
        return []

    selector_dir.mkdir(parents=True, exist_ok=True)
    path = selector_dir / "selector_vs_pure.csv"
    with path.open("w", newline="") as f:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped_ari_bar(
        rows,
        selector_dir / "selector_auto_vs_pure.svg",
        "Guarded auto selector ARI vs pure EVoC",
        "auto_ari",
        "pure_auto_ari",
        label_a="DiReEVoC guarded auto",
        label_b="Pure EVoC auto",
    )
    selector_delta_bar(rows, selector_dir / "selector_auto_delta_vs_pure.svg")
    copy_svg(
        selector_dir / "selector_auto_vs_pure.svg",
        docs_static / "benchmark_selector_auto_vs_pure.svg",
    )
    copy_svg(
        selector_dir / "selector_auto_delta_vs_pure.svg",
        docs_static / "benchmark_selector_auto_delta_vs_pure.svg",
    )
    return rows


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
    parser.add_argument(
        "--default-extra-artifacts",
        type=Path,
        action="append",
        default=[OPENML_EXTRA_ARTIFACTS],
        help="optional artifact tree(s) appended to the default benchmark figures",
    )
    parser.add_argument("--scale-artifacts", type=Path, default=SCALE_ARTIFACTS)
    parser.add_argument("--selector-artifacts", type=Path, default=resolve_selector_artifacts())
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
    default_rows = append_extra_rows(
        default_rows,
        args.default_extra_artifacts,
        args.timing_primary,
    )
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

    pure_rows = {row["slug"]: row for row in default_rows}
    pure_rows.update({row["slug"]: row for row in scale_rows})
    selector_rows = write_selector_comparison(
        args.selector_artifacts,
        pure_rows.values(),
        args.docs_static,
        args.timing_primary,
    )

    print(
        json.dumps(
            {
                "timing_primary": args.timing_primary,
                "default_rows": len(default_rows),
                "combined_rows": len(combined_rows),
                "scale_rows": len(scale_rows),
                "selector_rows": len(selector_rows),
                "default_extra_artifacts": [
                    str(path) for path in args.default_extra_artifacts if path.exists()
                ],
                "docs_static": str(args.docs_static),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
