#!/usr/bin/env python3
"""Profile DiReEVoC against pure EVoC and write reproducible visual outputs.

This extends ``benchmark_quality.py`` with the diagnostics that are useful when deciding
whether the bottleneck is DiRe, EVoC, layer selection, or prediction:

* auto/oracle/fixed-k ARI and NMI comparisons;
* DiReEVoC phase timings;
* DiRe intrinsic dimension from the fitted reducer's existing kNN graph;
* noise fraction, selected-layer cluster counts, and kNN graph shapes;
* SVG plots plus JSON/CSV metrics.

Examples:
    python examples/benchmark_visuals.py --synthetic-blobs --n-samples 12000 \
        --test-fraction 0.2 --output-dir artifacts/benchmark-visuals

    python examples/benchmark_visuals.py --npz /tmp/ng20_emb.npz --label "20NG BGE" \
        --output-dir artifacts/benchmark-visuals/ng20
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


COLORS = {
    "dire": "#2f6f9f",
    "evoc": "#c56b37",
    "neutral": "#5f6368",
    "accent": "#4b8f4f",
}


def gaussian_blobs(n=12_000, k=8, d=32, spread=0.10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    per = n // k
    parts, labels = [], []
    for c in range(k):
        pts = centers[c] + rng.normal(scale=spread, size=(per, d)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        parts.append(pts)
        labels += [c] * per
    return np.ascontiguousarray(np.vstack(parts)), np.array(labels)


def load_npz(path):
    z = np.load(path)
    return z["X"].astype(np.float32), z["y"]


def load_openml(name):
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(name, version=1, as_frame=False)
    X = np.ascontiguousarray(ds.data, dtype=np.float32)
    _, y = np.unique(ds.target, return_inverse=True)
    return X, y


def subsample(X, y, n, seed):
    if n is None or n >= len(X):
        return np.ascontiguousarray(X), y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return np.ascontiguousarray(X[idx]), y[idx]


def train_test_split(X, y, test_fraction, seed):
    if test_fraction <= 0.0:
        return X, y, None, None
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(round(len(X) * test_fraction)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (
        np.ascontiguousarray(X[train_idx]),
        y[train_idx],
        np.ascontiguousarray(X[test_idx]),
        y[test_idx],
    )


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - t0


def cluster_count(labels):
    labels = np.asarray(labels)
    return int(np.unique(labels[labels >= 0]).size)


def noise_fraction(labels):
    labels = np.asarray(labels)
    return float((labels < 0).mean())


def layer_scores(layers, y):
    scores = []
    for i, labels in enumerate(layers):
        scores.append(
            {
                "layer": i,
                "ari": float(adjusted_rand_score(y, labels)),
                "nmi": float(normalized_mutual_info_score(y, labels)),
                "clusters": cluster_count(labels),
                "noise_fraction": noise_fraction(labels),
            }
        )
    return scores


def evaluate_auto_oracle(labels, layers, y):
    scores = layer_scores(layers, y)
    best = max(scores, key=lambda row: row["ari"])
    return {
        "auto_ari": float(adjusted_rand_score(y, labels)),
        "auto_nmi": float(normalized_mutual_info_score(y, labels)),
        "auto_clusters": cluster_count(labels),
        "auto_noise_fraction": noise_fraction(labels),
        "oracle_ari": best["ari"],
        "oracle_nmi": best["nmi"],
        "oracle_layer": best["layer"],
        "oracle_clusters": best["clusters"],
        "layers": scores,
    }


def fit_vanilla_evoc(X, y, args):
    """Fit pure EVoC directly on the original feature vectors.

    The returned metrics key remains ``vanilla_evoc`` for compatibility with existing
    artifacts, but plots and logs call this baseline "Pure EVoC".
    """
    from evoc import EVoC

    ev, fit_seconds = timed(
        lambda: EVoC(
            min_samples=args.min_samples,
            base_min_cluster_size=args.base_min_cluster_size,
            base_n_clusters=args.base_n_clusters,
            random_state=args.seed,
        ).fit(X)
    )
    result = evaluate_auto_oracle(ev.labels_, ev.cluster_layers_, y)
    result["fit_seconds"] = fit_seconds
    result["total_seconds"] = fit_seconds
    return ev, result


def fit_dire_evoc_profile(X_train, y_train, X_test, y_test, args):
    from dire_evoc import DiReEVoC
    from dire_evoc.dimension import intrinsic_dimension_from_dire
    from dire_evoc.predict import knn_vote_predict, prepare_reference

    model = DiReEVoC(
        n_components=args.n_components,
        min_samples=args.min_samples,
        base_min_cluster_size=args.base_min_cluster_size,
        base_n_clusters=args.base_n_clusters,
        use_gpu=args.use_gpu,
        selector=args.selector,
        random_state=args.seed,
        n_neighbors=args.n_neighbors,
        max_iter_layout=args.max_iter_layout,
    )

    phases = {}
    on_gpu, phases["resolve_backend_seconds"] = timed(model._resolve_gpu)
    (layout, reducer), phases["dire_layout_seconds"] = timed(
        lambda: model._build_layout(X_train)
    )
    (layers, membership_layers, persistence), phases["evoc_cluster_seconds"] = timed(
        lambda: model._cluster(layout, on_gpu)
    )
    (selected_layer, selected_score), phases["layer_selection_seconds"] = timed(
        lambda: model._best_layer(layout, layers)
    )
    (
        (
            selected_score,
            selected_layer,
            layers,
            membership_layers,
            persistence,
            selected_min_samples,
            selected_base_min_cluster_size,
        ),
        phases["adaptive_smoothing_seconds"],
    ) = timed(
        lambda: model._maybe_adapt_smoothing(
            layout,
            on_gpu,
            layers,
            membership_layers,
            persistence,
            selected_layer,
            selected_score,
        )
    )
    (
        layers,
        membership_layers,
        persistence,
        selected_layer,
        fragment_merge,
    ), phases["fragment_merge_seconds"] = timed(
        lambda: model._maybe_merge_fragments(
            layout,
            layers,
            membership_layers,
            persistence,
            selected_layer,
            reducer,
        )
    )
    labels = layers[selected_layer]

    try:
        intrinsic_dimension, phases["intrinsic_dimension_seconds"] = timed(
            lambda: intrinsic_dimension_from_dire(reducer, method="twonn")
        )
    except Exception as exc:
        intrinsic_dimension = None
        phases["intrinsic_dimension_seconds"] = 0.0
        phases["intrinsic_dimension_error"] = repr(exc)

    reference, phases["prepare_reference_seconds"] = timed(
        lambda: prepare_reference(X_train)
    )

    holdout = None
    if X_test is not None:
        query = prepare_reference(X_test)
        (pred_labels, pred_membership), phases["predict_seconds"] = timed(
            lambda: knn_vote_predict(
                query,
                reference,
                labels,
                args.predict_neighbors,
            )
        )
        holdout = {
            "ari": float(adjusted_rand_score(y_test, pred_labels)),
            "nmi": float(normalized_mutual_info_score(y_test, pred_labels)),
            "clusters": cluster_count(pred_labels),
            "noise_fraction": noise_fraction(pred_labels),
            "mean_membership": float(np.mean(pred_membership)),
        }
    else:
        phases["predict_seconds"] = 0.0

    result = evaluate_auto_oracle(labels, layers, y_train)
    result.update(
        {
            "fit_seconds": float(
                phases["dire_layout_seconds"]
                + phases["evoc_cluster_seconds"]
                + phases["layer_selection_seconds"]
                + phases["adaptive_smoothing_seconds"]
                + phases["fragment_merge_seconds"]
                + phases["intrinsic_dimension_seconds"]
                + phases["prepare_reference_seconds"]
            ),
            "total_seconds": float(sum(v for v in phases.values() if isinstance(v, float))),
            "timings": phases,
            "selected_layer": int(selected_layer),
            "selected_min_samples": int(selected_min_samples),
            "selected_base_min_cluster_size": int(selected_base_min_cluster_size),
            "fragment_merge": fragment_merge,
            "on_gpu": bool(on_gpu),
            "layout_shape": list(layout.shape),
            "intrinsic_dimension": None
            if intrinsic_dimension is None
            else float(intrinsic_dimension),
            "knn_indices_shape": shape_or_none(getattr(reducer, "_knn_indices", None)),
            "knn_distances_shape": shape_or_none(getattr(reducer, "_knn_distances", None)),
            "holdout": holdout,
            "membership_mean": float(np.mean(membership_layers[selected_layer])),
            "persistence_scores": to_jsonable(persistence),
        }
    )
    return layout, result


def compute_fixed_k(X_train, y_train, layout, n_classes, args):
    from evoc import EVoC
    from evoc.clustering import build_cluster_layers

    ev_fixed, ev_seconds = timed(
        lambda: EVoC(
            min_samples=args.min_samples,
            base_n_clusters=n_classes,
            random_state=args.seed,
        )
        .fit(X_train)
        .cluster_layers_[0]
    )
    de_fixed, de_seconds = timed(
        lambda: build_cluster_layers(
            layout,
            min_samples=args.min_samples,
            base_min_cluster_size=args.base_min_cluster_size,
            base_n_clusters=n_classes,
        )[0][0]
    )
    return {
        "vanilla_evoc": {
            "fixed_k_ari": float(adjusted_rand_score(y_train, ev_fixed)),
            "fixed_k_nmi": float(normalized_mutual_info_score(y_train, ev_fixed)),
            "fixed_k_clusters": cluster_count(ev_fixed),
            "fixed_k_noise_fraction": noise_fraction(ev_fixed),
            "fixed_k_seconds": ev_seconds,
        },
        "dire_evoc": {
            "fixed_k_ari": float(adjusted_rand_score(y_train, de_fixed)),
            "fixed_k_nmi": float(normalized_mutual_info_score(y_train, de_fixed)),
            "fixed_k_clusters": cluster_count(de_fixed),
            "fixed_k_noise_fraction": noise_fraction(de_fixed),
            "fixed_k_seconds": de_seconds,
        },
    }


def shape_or_none(value):
    if value is None:
        return None
    return list(np.asarray(value).shape)


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def add_value_labels(ax, bars, fmt="{:.3f}"):
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


def plot_quality(metrics, output_dir):
    import matplotlib.pyplot as plt

    categories = ["auto", "oracle"]
    dire_values = [
        metrics["dire_evoc"]["auto_ari"],
        metrics["dire_evoc"]["oracle_ari"],
    ]
    evoc_values = [
        metrics["vanilla_evoc"]["auto_ari"],
        metrics["vanilla_evoc"]["oracle_ari"],
    ]
    if "fixed_k_ari" in metrics["dire_evoc"]:
        categories.append("fixed-k")
        dire_values.append(metrics["dire_evoc"]["fixed_k_ari"])
        evoc_values.append(metrics["vanilla_evoc"]["fixed_k_ari"])

    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - width / 2, dire_values, width, label="DiReEVoC", color=COLORS["dire"])
    b2 = ax.bar(x + width / 2, evoc_values, width, label="Pure EVoC", color=COLORS["evoc"])
    ax.set_title("Quality comparison by selection regime")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    add_value_labels(ax, b1)
    add_value_labels(ax, b2)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_ari.svg", format="svg")
    plt.close(fig)


def plot_timings(metrics, output_dir):
    import matplotlib.pyplot as plt

    timings = metrics["dire_evoc"]["timings"]
    labels = [
        "Pure EVoC fit",
        "DiRe layout",
        "EVoC on layout",
        "Layer selector",
        "Adaptive smoothing",
        "Fragment merge",
        "ID",
        "Predict",
    ]
    values = [
        metrics["vanilla_evoc"]["fit_seconds"],
        timings.get("dire_layout_seconds", 0.0),
        timings.get("evoc_cluster_seconds", 0.0),
        timings.get("layer_selection_seconds", 0.0),
        timings.get("adaptive_smoothing_seconds", 0.0),
        timings.get("fragment_merge_seconds", 0.0),
        timings.get("intrinsic_dimension_seconds", 0.0),
        timings.get("predict_seconds", 0.0),
    ]
    colors = [
        COLORS["evoc"],
        COLORS["dire"],
        COLORS["dire"],
        COLORS["neutral"],
        COLORS["neutral"],
        COLORS["neutral"],
        COLORS["accent"],
        COLORS["accent"],
    ]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.barh(y, values, color=colors)
    ax.set_title("Runtime breakdown")
    ax.set_xlabel("seconds")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    max_value = max(values) if values else 0.0
    ax.set_xlim(0.0, max_value * 1.18 if max_value else 1.0)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}s",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "timing_breakdown.svg", format="svg")
    plt.close(fig)


def plot_diagnostics(metrics, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))

    methods = ["DiReEVoC", "Pure EVoC"]
    noise = [
        100.0 * metrics["dire_evoc"]["auto_noise_fraction"],
        100.0 * metrics["vanilla_evoc"]["auto_noise_fraction"],
    ]
    clusters = [
        metrics["dire_evoc"]["auto_clusters"],
        metrics["vanilla_evoc"]["auto_clusters"],
    ]
    method_colors = [COLORS["dire"], COLORS["evoc"]]

    axes[0].bar(methods, noise, color=method_colors)
    axes[0].set_title("Selected-layer noise")
    axes[0].set_ylabel("percent")
    axes[0].set_ylim(0.0, max(100.0, max(noise) * 1.15))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(methods, clusters, color=method_colors)
    axes[1].set_title("Selected clusters")
    axes[1].set_ylabel("count")
    axes[1].grid(axis="y", alpha=0.25)

    id_value = metrics["dire_evoc"].get("intrinsic_dimension")
    ambient = metrics["data"]["n_features"]
    id_labels = ["Estimated ID", "Ambient d"]
    id_values = [0.0 if id_value is None else id_value, ambient]
    axes[2].bar(id_labels, id_values, color=[COLORS["accent"], COLORS["neutral"]])
    axes[2].set_title("Dimension diagnostic")
    axes[2].set_ylabel("dimension")
    axes[2].grid(axis="y", alpha=0.25)
    if id_value is None:
        axes[2].text(0, 0, "unavailable", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle(metrics["data"]["label"], y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostics.svg", format="svg")
    plt.close(fig)


def write_csv(metrics, path):
    rows = []

    def emit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                emit(f"{prefix}.{key}" if prefix else key, child)
        elif isinstance(value, list):
            rows.append((prefix, json.dumps(value)))
        else:
            rows.append((prefix, value))

    emit("", metrics)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic-blobs", action="store_true")
    source.add_argument("--npz")
    source.add_argument("--openml")
    p.add_argument("--label")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=12_000)
    p.add_argument("--n-classes", type=int, default=8)
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--spread", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--base-min-cluster-size", type=int, default=5)
    p.add_argument("--base-n-clusters", type=int, default=None)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--predict-neighbors", type=int, default=15)
    p.add_argument("--max-iter-layout", type=int, default=100)
    p.add_argument("--use-gpu", default="auto", choices=["auto", "true", "false"])
    p.add_argument(
        "--selector",
        default="silhouette",
        choices=["silhouette", "silhouette_raw", "base", "none"],
    )
    p.add_argument("--skip-fixed-k", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.use_gpu == "true":
        args.use_gpu = True
    elif args.use_gpu == "false":
        args.use_gpu = False

    if args.synthetic_blobs:
        X, y = gaussian_blobs(
            n=args.n_samples,
            k=args.n_classes,
            d=args.n_features,
            spread=args.spread,
            seed=args.seed,
        )
        label = args.label or "synthetic blobs"
    elif args.npz:
        X, y = load_npz(args.npz)
        label = args.label or Path(args.npz).stem
    else:
        X, y = load_openml(args.openml)
        label = args.label or f"OpenML:{args.openml}"

    X, y = subsample(X, y, args.subsample, args.seed)
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, args.test_fraction, args.seed
    )
    n_classes = int(np.unique(y_train).size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{label}: train={X_train.shape}, classes={n_classes}")
    if X_test is not None:
        print(f"holdout={X_test.shape}")

    print("fitting pure EVoC...")
    _, vanilla = fit_vanilla_evoc(X_train, y_train, args)

    print("profiling DiReEVoC...")
    layout, dire = fit_dire_evoc_profile(X_train, y_train, X_test, y_test, args)

    if not args.skip_fixed_k:
        print("computing fixed-k comparison...")
        fixed = compute_fixed_k(X_train, y_train, layout, n_classes, args)
        vanilla.update(fixed["vanilla_evoc"])
        dire.update(fixed["dire_evoc"])

    metrics = {
        "data": {
            "label": label,
            "n_samples": int(X.shape[0]),
            "n_train": int(X_train.shape[0]),
            "n_test": 0 if X_test is None else int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": n_classes,
            "min_samples": int(args.min_samples),
            "base_min_cluster_size": int(args.base_min_cluster_size),
            "base_n_clusters": None
            if args.base_n_clusters is None
            else int(args.base_n_clusters),
        },
        "vanilla_evoc": vanilla,
        "dire_evoc": dire,
    }
    metrics = to_jsonable(metrics)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_csv(metrics, output_dir / "metrics.csv")
    plot_quality(metrics, output_dir)
    plot_timings(metrics, output_dir)
    plot_diagnostics(metrics, output_dir)

    print(f"wrote metrics and SVG figures to {output_dir}")
    print(
        "auto ARI: "
        f"DiReEVoC={dire['auto_ari']:.4f}, pure EVoC={vanilla['auto_ari']:.4f}"
    )
    print(
        "intrinsic dimension: "
        f"{dire.get('intrinsic_dimension', None)}; "
        f"DiRe layout seconds={dire['timings']['dire_layout_seconds']:.3f}; "
        f"EVoC-on-layout seconds={dire['timings']['evoc_cluster_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
