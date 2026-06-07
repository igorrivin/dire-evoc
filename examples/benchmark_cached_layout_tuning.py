#!/usr/bin/env python3
"""Tune EVoC controls on cached DiRe layouts.

The regular visual benchmark measures full DiReEVoC fits. That is the right end-to-end
number, but it rebuilds the DiRe layout for every fit. This script separates the two
questions:

* how much does each DiRe layout dimension cost?
* once that layout exists, which EVoC controls work best on the cached layout?

Examples:
    python examples/benchmark_cached_layout_tuning.py \
        --npz artifacts/cuda/datasets/fashion_mnist.npz \
        --label Fashion-MNIST \
        --output-dir artifacts/cached-layout/fashion_mnist

    python examples/benchmark_cached_layout_tuning.py --synthetic-blobs \
        --output-dir artifacts/cached-layout/blobs \
        --n-components-grid 4,8 \
        --base-min-cluster-size-grid 25,50,100 \
        --base-n-clusters-grid none,classes
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from benchmark_visuals import (
    cluster_count,
    evaluate_auto_oracle,
    gaussian_blobs,
    load_npz,
    load_openml,
    noise_fraction,
    shape_or_none,
    subsample,
    timed,
    to_jsonable,
    train_test_split,
)


VALID_SELECTORS = {"silhouette", "silhouette_raw", "base", "none"}


def parse_int_grid(value: str) -> list[int]:
    items = []
    seen = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed <= 0:
            raise ValueError(f"grid values must be positive integers: {value!r}")
        if parsed not in seen:
            seen.add(parsed)
            items.append(parsed)
    if not items:
        raise ValueError("grid must contain at least one integer")
    return items


def parse_base_n_clusters_grid(value: str, n_classes: int) -> list[int | None]:
    items = []
    seen = set()
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "auto", "null"}:
            parsed = None
        elif token in {"classes", "n_classes", "true_k"}:
            parsed = int(n_classes)
        else:
            parsed = int(token)
            if parsed <= 0:
                raise ValueError(f"base_n_clusters must be positive or none: {value!r}")
        key = "none" if parsed is None else parsed
        if key not in seen:
            seen.add(key)
            items.append(parsed)
    if not items:
        raise ValueError("base_n_clusters grid must contain at least one value")
    return items


def parse_bool_grid(value: str) -> list[bool]:
    mapping = {
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "off": False,
    }
    items = []
    seen = set()
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise ValueError(f"boolean grid value must be true/false: {token!r}")
        parsed = mapping[token]
        if parsed not in seen:
            seen.add(parsed)
            items.append(parsed)
    if not items:
        raise ValueError("boolean grid must contain at least one value")
    return items


def parse_selectors(value: str) -> list[str]:
    selectors = []
    seen = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in VALID_SELECTORS:
            raise ValueError(
                f"unknown selector {token!r}; expected one of {sorted(VALID_SELECTORS)}"
            )
        if token not in seen:
            seen.add(token)
            selectors.append(token)
    if not selectors:
        raise ValueError("selector grid must contain at least one selector")
    return selectors


def normalize_use_gpu(value):
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def load_data(args):
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
    return X, y, label


def fit_pure_evoc(X_train, y_train, args, base_min_cluster_size):
    from evoc import EVoC

    ev, seconds = timed(
        lambda: EVoC(
            min_samples=args.min_samples,
            base_min_cluster_size=base_min_cluster_size,
            random_state=args.seed,
        ).fit(X_train)
    )
    result = evaluate_auto_oracle(ev.labels_, ev.cluster_layers_, y_train)
    result.update(
        {
            "fit_seconds": float(seconds),
            "base_min_cluster_size": int(base_min_cluster_size),
            "auto_clusters": cluster_count(ev.labels_),
            "auto_noise_fraction": noise_fraction(ev.labels_),
        }
    )
    return result


def build_cached_layout(X_train, n_components, args):
    from dire_evoc import DiReEVoC
    from dire_evoc.dimension import intrinsic_dimension_from_dire

    model = DiReEVoC(
        n_components=n_components,
        metric=args.metric,
        min_samples=args.min_samples,
        use_gpu=args.use_gpu,
        random_state=args.seed,
        n_neighbors=args.n_neighbors,
        max_iter_layout=args.max_iter_layout,
    )
    on_gpu, resolve_seconds = timed(model._resolve_gpu)
    (layout, reducer), layout_seconds = timed(lambda: model._build_layout(X_train))
    try:
        intrinsic_dimension, id_seconds = timed(
            lambda: intrinsic_dimension_from_dire(reducer, method="twonn")
        )
        id_error = None
    except Exception as exc:
        intrinsic_dimension = None
        id_seconds = 0.0
        id_error = repr(exc)

    info = {
        "n_components": int(n_components),
        "on_gpu": bool(on_gpu),
        "resolve_backend_seconds": float(resolve_seconds),
        "dire_layout_seconds": float(layout_seconds),
        "intrinsic_dimension_seconds": float(id_seconds),
        "intrinsic_dimension": None
        if intrinsic_dimension is None
        else float(intrinsic_dimension),
        "intrinsic_dimension_error": id_error,
        "layout_shape": list(layout.shape),
        "knn_indices_shape": shape_or_none(getattr(reducer, "_knn_indices", None)),
        "knn_distances_shape": shape_or_none(getattr(reducer, "_knn_distances", None)),
    }
    return layout, reducer, on_gpu, info


def evaluate_cached_trial(
    layout,
    reducer,
    on_gpu,
    y_train,
    n_components,
    base_min_cluster_size,
    base_n_clusters,
    selector,
    adaptive_min_samples,
    layout_seconds,
    args,
):
    from dire_evoc import DiReEVoC

    model = DiReEVoC(
        n_components=n_components,
        metric=args.metric,
        min_samples=args.min_samples,
        base_min_cluster_size=base_min_cluster_size,
        base_n_clusters=base_n_clusters,
        use_gpu=args.use_gpu,
        selector=selector,
        adaptive_min_samples=adaptive_min_samples,
        random_state=args.seed,
        n_neighbors=args.n_neighbors,
        max_iter_layout=args.max_iter_layout,
    )
    phases = {}
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
    labels = layers[selected_layer]
    cached_trial_seconds = float(
        phases["evoc_cluster_seconds"]
        + phases["layer_selection_seconds"]
        + phases["adaptive_smoothing_seconds"]
    )
    result = evaluate_auto_oracle(labels, layers, y_train)
    result.update(
        {
            "n_components": int(n_components),
            "min_samples": int(args.min_samples),
            "base_min_cluster_size": int(base_min_cluster_size),
            "base_n_clusters": None
            if base_n_clusters is None
            else int(base_n_clusters),
            "selector": selector,
            "adaptive_min_samples": bool(adaptive_min_samples),
            "selected_layer": int(selected_layer),
            "selected_score": float(selected_score),
            "selected_min_samples": int(selected_min_samples),
            "selected_base_min_cluster_size": int(selected_base_min_cluster_size),
            "cached_trial_seconds": cached_trial_seconds,
            "full_once_seconds": float(layout_seconds + cached_trial_seconds),
            "timings": phases,
            "auto_clusters": cluster_count(labels),
            "auto_noise_fraction": noise_fraction(labels),
            "membership_mean": float(np.mean(membership_layers[selected_layer])),
            "persistence_scores": to_jsonable(persistence),
        }
    )
    return result


def write_trials_csv(trials, path: Path) -> None:
    columns = [
        "n_components",
        "base_min_cluster_size",
        "base_n_clusters",
        "selector",
        "adaptive_min_samples",
        "auto_ari",
        "oracle_ari",
        "auto_nmi",
        "auto_clusters",
        "auto_noise_fraction",
        "selected_layer",
        "selected_score",
        "selected_min_samples",
        "selected_base_min_cluster_size",
        "cached_trial_seconds",
        "full_once_seconds",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for trial in trials:
            writer.writerow({column: trial.get(column) for column in columns})


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic-blobs", action="store_true")
    source.add_argument("--npz")
    source.add_argument("--openml")
    parser.add_argument("--label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=12_000)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--spread", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--n-components-grid", default="4,8")
    parser.add_argument("--base-min-cluster-size-grid", default="5,25,100")
    parser.add_argument("--base-n-clusters-grid", default="none,classes")
    parser.add_argument("--selectors", default="silhouette")
    parser.add_argument("--adaptive-min-samples-grid", default="true,false")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--max-iter-layout", type=int, default=100)
    parser.add_argument("--use-gpu", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--skip-pure", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.use_gpu = normalize_use_gpu(args.use_gpu)

    X, y, label = load_data(args)
    X_train, y_train, X_test, _y_test = train_test_split(
        X,
        y,
        args.test_fraction,
        args.seed,
    )
    n_classes = int(np.unique(y_train).size)
    n_features = int(X_train.shape[1])
    n_components_grid = [
        min(value, n_features) for value in parse_int_grid(args.n_components_grid)
    ]
    n_components_grid = list(dict.fromkeys(n_components_grid))
    base_min_cluster_sizes = parse_int_grid(args.base_min_cluster_size_grid)
    base_n_clusters_grid = parse_base_n_clusters_grid(
        args.base_n_clusters_grid,
        n_classes,
    )
    selectors = parse_selectors(args.selectors)
    adaptive_grid = parse_bool_grid(args.adaptive_min_samples_grid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{label}: train={X_train.shape}, test="
        f"{None if X_test is None else X_test.shape}, classes={n_classes}"
    )
    print(
        "cached layout sweep: "
        f"n_components={n_components_grid}, "
        f"base_min_cluster_size={base_min_cluster_sizes}, "
        f"base_n_clusters={base_n_clusters_grid}, "
        f"selectors={selectors}, adaptive={adaptive_grid}"
    )

    pure = None
    if not args.skip_pure:
        print("fitting pure EVoC baseline...")
        pure = fit_pure_evoc(
            X_train,
            y_train,
            args,
            base_min_cluster_size=base_min_cluster_sizes[0],
        )

    layouts = []
    trials = []
    for n_components in n_components_grid:
        print(f"building DiRe layout once for n_components={n_components}...")
        layout, reducer, on_gpu, layout_info = build_cached_layout(
            X_train,
            n_components,
            args,
        )
        layouts.append(layout_info)
        layout_seconds = layout_info["dire_layout_seconds"]

        for bmc, bnc, selector, adaptive in product(
            base_min_cluster_sizes,
            base_n_clusters_grid,
            selectors,
            adaptive_grid,
        ):
            print(
                "  trial "
                f"bmc={bmc} base_n_clusters={bnc} "
                f"selector={selector} adaptive={adaptive}"
            )
            try:
                trial = evaluate_cached_trial(
                    layout,
                    reducer,
                    on_gpu,
                    y_train,
                    n_components,
                    bmc,
                    bnc,
                    selector,
                    adaptive,
                    layout_seconds,
                    args,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                trial = {
                    "n_components": int(n_components),
                    "base_min_cluster_size": int(bmc),
                    "base_n_clusters": None if bnc is None else int(bnc),
                    "selector": selector,
                    "adaptive_min_samples": bool(adaptive),
                    "error": repr(exc),
                }
            trials.append(trial)

    successful_trials = [trial for trial in trials if "error" not in trial]
    best = None
    if successful_trials:
        best = max(
            successful_trials,
            key=lambda trial: (
                trial["auto_ari"],
                trial["oracle_ari"],
                -trial["cached_trial_seconds"],
            ),
        )
        print(
            "best cached-layout trial: "
            f"n_components={best['n_components']} "
            f"bmc={best['base_min_cluster_size']} "
            f"base_n_clusters={best['base_n_clusters']} "
            f"selector={best['selector']} "
            f"adaptive={best['adaptive_min_samples']} "
            f"auto_ari={best['auto_ari']:.4f} "
            f"oracle_ari={best['oracle_ari']:.4f} "
            f"cached_seconds={best['cached_trial_seconds']:.3f} "
            f"full_once_seconds={best['full_once_seconds']:.3f}"
        )

    metrics = {
        "data": {
            "label": label,
            "n_samples": int(X.shape[0]),
            "n_train": int(X_train.shape[0]),
            "n_test": 0 if X_test is None else int(X_test.shape[0]),
            "n_features": n_features,
            "n_classes": n_classes,
            "metric": args.metric,
            "min_samples": int(args.min_samples),
            "n_neighbors": int(args.n_neighbors),
            "max_iter_layout": int(args.max_iter_layout),
        },
        "grids": {
            "n_components": n_components_grid,
            "base_min_cluster_size": base_min_cluster_sizes,
            "base_n_clusters": base_n_clusters_grid,
            "selectors": selectors,
            "adaptive_min_samples": adaptive_grid,
        },
        "pure_evoc": pure,
        "layouts": layouts,
        "trials": trials,
        "best_by_auto_ari": best,
    }
    metrics = to_jsonable(metrics)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_trials_csv(metrics["trials"], output_dir / "trials.csv")
    print(f"wrote cached-layout tuning metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
