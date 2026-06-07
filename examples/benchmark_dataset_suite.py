#!/usr/bin/env python3
"""Run visual DiReEVoC-vs-pure-EVoC comparisons across many datasets.

This is a thin orchestration layer around ``examples/benchmark_visuals.py``. It fetches
or builds labeled datasets, caches them as ``.npz`` files, runs the same visual benchmark
for each dataset, and writes suite-level CSV/SVG summaries.

Examples:
    python examples/benchmark_dataset_suite.py --list-datasets

    python examples/benchmark_dataset_suite.py --datasets smoke \
        --output-dir artifacts/smoke-suite --test-fraction 0.0

    python examples/benchmark_dataset_suite.py --datasets default \
        --output-dir artifacts/openml-suite --test-fraction 0.2

    python examples/benchmark_dataset_suite.py --datasets scale \
        --output-dir artifacts/scale-suite --prefer-hf --timing-repeats 1

    python examples/benchmark_dataset_suite.py --datasets blobs,mnist,fashion_mnist,covertype \
        --output-dir artifacts/openml-smoke --subsample 12000 --skip-existing

The per-dataset outputs are exactly the outputs from ``benchmark_visuals.py``:
``metrics.json``, ``metrics.csv``, ``quality_ari.svg``, ``timing_breakdown.svg``, and
``diagnostics.svg``. The baseline labeled "Pure EVoC" is EVoC fitted directly to the
original feature vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    label: str
    source: str
    openml_name: str | None = None
    openml_id: int | None = None
    hf_name: str | None = None
    hf_kind: str = "image"
    hf_split: str = "train"
    hf_full_split: str | None = None
    hf_image_column: str = "image"
    hf_label_column: str = "label"
    hf_max_rows: int | None = None
    hf_image_pool: int = 1
    sklearn_fallback: str | None = None
    default_subsample: int | None = None
    default_base_min_cluster_size: int | None = None
    default_base_n_clusters: int | None = None
    metric: str = "cosine"
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "blobs": DatasetSpec(
        "blobs",
        "Synthetic blobs 12k",
        "synthetic_blobs",
        default_base_min_cluster_size=100,
        note="Generated locally; useful sanity check for density-separated clusters.",
    ),
    "moons": DatasetSpec(
        "moons",
        "Synthetic moons 12k",
        "synthetic_moons",
        default_base_min_cluster_size=100,
        default_base_n_clusters=2,
        note="Generated locally; nonlinear two-cluster geometry.",
    ),
    "circles": DatasetSpec(
        "circles",
        "Synthetic circles 12k",
        "synthetic_circles",
        default_base_min_cluster_size=100,
        default_base_n_clusters=2,
        note="Generated locally; nested nonlinear cluster geometry.",
    ),
    "digits": DatasetSpec(
        "digits",
        "sklearn digits",
        "sklearn_digits",
        default_base_min_cluster_size=10,
        note="Small 8x8 handwritten digits; CPU-friendly smoke dataset.",
    ),
    "iris": DatasetSpec(
        "iris",
        "sklearn iris",
        "sklearn_iris",
        note="Small tabular flower benchmark.",
    ),
    "wine": DatasetSpec(
        "wine",
        "sklearn wine",
        "sklearn_wine",
        note="Small tabular chemistry benchmark.",
    ),
    "breast_cancer": DatasetSpec(
        "breast_cancer",
        "sklearn breast cancer",
        "sklearn_breast_cancer",
        note="Small tabular diagnostic benchmark.",
    ),
    "olivetti_faces": DatasetSpec(
        "olivetti_faces",
        "Olivetti faces",
        "sklearn_olivetti",
        default_base_min_cluster_size=2,
        note="400 face images, 40 identities; intentionally tiny for this pipeline.",
    ),
    "mnist": DatasetSpec(
        "mnist",
        "MNIST",
        "openml",
        openml_id=554,
        hf_name="ylecun/mnist",
        hf_kind="image",
        hf_full_split="train+test",
        default_subsample=30000,
        default_base_min_cluster_size=100,
        note="OpenML mnist_784; falls back to Hugging Face ylecun/mnist.",
    ),
    "fashion_mnist": DatasetSpec(
        "fashion_mnist",
        "Fashion-MNIST",
        "openml",
        openml_id=40996,
        hf_name="zalando-datasets/fashion_mnist",
        hf_kind="image",
        hf_full_split="train+test",
        default_subsample=30000,
        default_base_min_cluster_size=100,
        note="OpenML Fashion-MNIST; falls back to Hugging Face zalando-datasets/fashion_mnist.",
    ),
    "dbpedia14": DatasetSpec(
        "dbpedia14",
        "DBpedia-14 BGE-small",
        "hf_dbpedia14",
        default_subsample=49000,
        default_base_min_cluster_size=100,
        note="DBpedia ontology text embedded with BGE-small; requires datasets and sentence-transformers.",
    ),
    "cifar10": DatasetSpec(
        "cifar10",
        "CIFAR-10",
        "openml",
        openml_id=40927,
        hf_name="uoft-cs/cifar10",
        hf_kind="image",
        hf_full_split="train+test",
        hf_image_column="img",
        hf_image_pool=2,
        default_subsample=30000,
        default_base_min_cluster_size=100,
        note=(
            "32x32 RGB pixels average-pooled to 16x16 RGB for the benchmark; "
            "falls back to Hugging Face uoft-cs/cifar10."
        ),
    ),
    "usps": DatasetSpec(
        "usps",
        "USPS digits",
        "openml",
        openml_id=41082,
        note="OpenML USPS handwritten digits.",
    ),
    "svhn": DatasetSpec(
        "svhn",
        "SVHN",
        "openml",
        openml_id=41081,
        default_subsample=6000,
        note="Street-view digits from OpenML.",
    ),
    "umist_faces": DatasetSpec(
        "umist_faces",
        "UMIST faces cropped",
        "openml",
        openml_id=41084,
        note="Face images with 20 identities.",
    ),
    "devnagari": DatasetSpec(
        "devnagari",
        "Devnagari Script",
        "openml",
        openml_id=40923,
        default_subsample=30000,
        note="46-class handwritten character dataset.",
    ),
    "emnist_balanced": DatasetSpec(
        "emnist_balanced",
        "EMNIST Balanced",
        "openml",
        openml_id=41039,
        default_subsample=30000,
        note="47-class handwritten character dataset.",
    ),
    "covertype": DatasetSpec(
        "covertype",
        "Forest CoverType",
        "openml",
        openml_name="covertype",
        hf_name="mstz/covertype",
        hf_kind="table",
        hf_label_column="cover_type",
        hf_max_rows=50000,
        sklearn_fallback="covtype",
        default_subsample=50000,
        default_base_min_cluster_size=100,
        note="Forest cover classification; sklearn fallback is preferred when OpenML is unavailable; HF mirror is declared.",
    ),
    "letter": DatasetSpec(
        "letter",
        "Letter Recognition",
        "openml",
        openml_name="letter",
        note="Classic UCI letter recognition.",
    ),
    "optdigits": DatasetSpec(
        "optdigits",
        "OptDigits",
        "openml",
        openml_name="optdigits",
        note="Optical handwritten digits.",
    ),
    "pendigits": DatasetSpec(
        "pendigits",
        "PenDigits",
        "openml",
        openml_name="pendigits",
        note="Pen-based handwritten digits.",
    ),
    "satimage": DatasetSpec(
        "satimage",
        "SatImage",
        "openml",
        openml_name="satimage",
        note="Landsat image classification.",
    ),
    "segment": DatasetSpec(
        "segment",
        "Image Segmentation",
        "openml",
        openml_name="segment",
        note="Image segmentation region features.",
    ),
    "vehicle": DatasetSpec(
        "vehicle",
        "Vehicle",
        "openml",
        openml_name="vehicle",
        note="Vehicle silhouette classification.",
    ),
    "wiki_cs": DatasetSpec(
        "wiki_cs",
        "Wiki-CS",
        "pyg_wiki_cs",
        default_subsample=30000,
        default_base_min_cluster_size=100,
        note="Optional graph-feature dataset; requires torch-geometric.",
    ),
    "ogbn_arxiv": DatasetSpec(
        "ogbn_arxiv",
        "OGBN-Arxiv",
        "ogb_arxiv",
        default_subsample=50000,
        default_base_min_cluster_size=100,
        note="Optional graph-feature dataset; requires ogb.",
    ),
}

SCALE_DATASETS: dict[str, DatasetSpec] = {
    "blobs30k": DatasetSpec(
        "blobs30k",
        "Synthetic blobs 30k",
        "synthetic_blobs_30k",
        default_base_min_cluster_size=100,
        note="Generated locally; 30k density-separated clusters.",
    ),
    "moons30k": DatasetSpec(
        "moons30k",
        "Synthetic moons 30k",
        "synthetic_moons_30k",
        default_base_min_cluster_size=100,
        default_base_n_clusters=2,
        note="Generated locally; 30k nonlinear two-cluster geometry.",
    ),
    "circles30k": DatasetSpec(
        "circles30k",
        "Synthetic circles 30k",
        "synthetic_circles_30k",
        default_base_min_cluster_size=100,
        default_base_n_clusters=2,
        note="Generated locally; 30k nested nonlinear cluster geometry.",
    ),
    "spirals_clean30k": DatasetSpec(
        "spirals_clean30k",
        "Separated spirals 30k",
        "synthetic_spirals_clean_30k",
        default_base_min_cluster_size=100,
        default_base_n_clusters=3,
        metric="euclidean",
        note="Generated locally; three separated 2-D spiral arms. Raw 2-D euclidean so "
        "DiRe must untangle the arms; pure EVoC cannot run on <4-D input.",
    ),
}

DATASETS.update(SCALE_DATASETS)


SMOKE_DATASETS = (
    "blobs",
    "moons",
    "circles",
)
DEFAULT_DATASETS = (
    *SMOKE_DATASETS,
    "mnist",
    "digits",
    "iris",
    "wine",
    "breast_cancer",
    "fashion_mnist",
    "dbpedia14",
    "olivetti_faces",
    "covertype",
    "cifar10",
    "optdigits",
    "pendigits",
    "satimage",
    "segment",
    "vehicle",
)
GRAPH_DATASETS = ("wiki_cs", "ogbn_arxiv")
SCALE_BENCHMARK_DATASETS = (
    "blobs30k",
    "moons30k",
    "circles30k",
    "mnist",
    "fashion_mnist",
    "cifar10",
    "dbpedia14",
    "covertype",
    "olivetti_faces",
    "wiki_cs",
    "ogbn_arxiv",
)


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")


def encode_labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.reshape(-1)
    mask = np.array([value is not None and str(value) != "nan" for value in y])
    if not mask.all():
        y = y[mask]
    _, encoded = np.unique(y.astype(str), return_inverse=True)
    return encoded.astype(np.int64), mask


def coerce_features(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.dtype.kind in "OUS":
        from sklearn.preprocessing import OrdinalEncoder

        X = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        ).fit_transform(X)
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        finite = np.isfinite(X)
        medians = np.zeros(X.shape[1], dtype=np.float32)
        for j in range(X.shape[1]):
            column = X[:, j]
            medians[j] = np.median(column[finite[:, j]]) if finite[:, j].any() else 0.0
        X = np.where(finite, X, medians)
    return np.ascontiguousarray(X, dtype=np.float32)


def save_npz(path: Path, X, y) -> None:
    encoded_y, label_mask = encode_labels(y)
    X = coerce_features(X)
    if len(label_mask) == len(X) and not label_mask.all():
        X = X[label_mask]
    if len(X) != len(encoded_y):
        raise ValueError(f"feature/label length mismatch: X={len(X)} y={len(encoded_y)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=encoded_y)


def fetch_openml_npz(
    spec: DatasetSpec,
    path: Path,
    allow_hf_fallback: bool = True,
    full: bool = False,
) -> None:
    from sklearn.datasets import fetch_openml

    kwargs = {"as_frame": False}
    if spec.openml_id is not None:
        kwargs["data_id"] = spec.openml_id
    else:
        kwargs["name"] = spec.openml_name
        kwargs["version"] = 1
    try:
        try:
            ds = fetch_openml(**kwargs, parser="auto")
        except TypeError:
            ds = fetch_openml(**kwargs)
    except Exception:
        fetch_openml_fallback_npz(spec, path, allow_hf=allow_hf_fallback, full=full)
        return
    save_npz(path, ds.data, ds.target)


def fetch_openml_fallback_npz(
    spec: DatasetSpec,
    path: Path,
    allow_hf: bool = True,
    full: bool = False,
) -> None:
    """Fallback policy for OpenML-backed specs.

    If the dataset is fully available through sklearn, use sklearn as the fallback.
    Otherwise use a declared Hugging Face mirror. This keeps sklearn-native datasets off
    the network mirror path and avoids retrying the same failing source twice.
    """
    if spec.sklearn_fallback is not None:
        print(
            f"OpenML fetch failed for {spec.slug}; "
            f"falling back to sklearn {spec.sklearn_fallback}"
        )
        fetch_sklearn_fallback_npz(spec, path)
        return
    if allow_hf and spec.hf_name is not None:
        print(f"OpenML fetch failed for {spec.slug}; falling back to Hugging Face {spec.hf_name}")
        fetch_hf_npz(spec, path, full=full)
        return
    raise RuntimeError(
        f"OpenML fetch failed for {spec.slug} and no sklearn/Hugging Face fallback is declared"
    )


def fetch_hf_npz(spec: DatasetSpec, path: Path, full: bool = False) -> None:
    if spec.hf_kind == "image":
        fetch_hf_image_npz(spec, path, full=full)
    elif spec.hf_kind == "table":
        fetch_hf_table_npz(spec, path, full=full)
    else:
        raise ValueError(f"unknown Hugging Face fallback kind: {spec.hf_kind}")


def maybe_limit_hf_rows(dataset, max_rows: int | None):
    if max_rows is None or len(dataset) <= max_rows:
        return dataset
    return dataset.select(range(max_rows))


def image_to_features(image, pool: int = 1) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if pool > 1:
        if array.ndim == 2:
            h, w = array.shape
            h -= h % pool
            w -= w % pool
            array = array[:h, :w].reshape(h // pool, pool, w // pool, pool).mean(axis=(1, 3))
        elif array.ndim == 3:
            h, w, c = array.shape
            h -= h % pool
            w -= w % pool
            array = (
                array[:h, :w, :]
                .reshape(h // pool, pool, w // pool, pool, c)
                .mean(axis=(1, 3))
            )
    return array.reshape(-1)


def fetch_hf_image_npz(spec: DatasetSpec, path: Path, full: bool = False) -> None:
    from datasets import load_dataset

    split = spec.hf_full_split if full and spec.hf_full_split is not None else spec.hf_split
    dataset = load_dataset(spec.hf_name, split=split)
    dataset = maybe_limit_hf_rows(dataset, spec.hf_max_rows)
    images = dataset[spec.hf_image_column]
    labels = dataset[spec.hf_label_column]
    X = np.stack([image_to_features(image, spec.hf_image_pool) for image in images])
    save_npz(path, X, np.asarray(labels))


def fetch_hf_table_npz(spec: DatasetSpec, path: Path, full: bool = False) -> None:
    from datasets import load_dataset

    dataset = load_dataset(spec.hf_name, split=spec.hf_split)
    dataset = maybe_limit_hf_rows(dataset, None if full else spec.hf_max_rows)
    if spec.hf_label_column not in dataset.column_names:
        raise ValueError(
            f"label column {spec.hf_label_column!r} not found in {spec.hf_name}; "
            f"available columns: {dataset.column_names}"
        )
    feature_columns = [name for name in dataset.column_names if name != spec.hf_label_column]
    X = np.column_stack([np.asarray(dataset[name]) for name in feature_columns])
    save_npz(path, X, np.asarray(dataset[spec.hf_label_column]))


def fetch_sklearn_fallback_npz(spec: DatasetSpec, path: Path) -> None:
    if spec.sklearn_fallback != "covtype":
        raise ValueError(f"unknown sklearn fallback: {spec.sklearn_fallback}")
    from sklearn.datasets import fetch_covtype

    ds = fetch_covtype()
    save_npz(path, ds.data, ds.target)


def synthetic_blobs_npz(path: Path, n: int = 12000) -> None:
    from benchmark_visuals import gaussian_blobs

    X, y = gaussian_blobs(n=n, k=8, d=32, spread=0.10, seed=0)
    save_npz(path, X, y)


def synthetic_moons_npz(path: Path, n: int = 12000) -> None:
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n, noise=0.06, random_state=0)
    x0 = X[:, 0]
    x1 = X[:, 1]
    X = np.column_stack(
        [
            x0,
            x1,
            x0 * x0,
            x1 * x1,
            x0 * x1,
            np.sin(np.pi * x0),
            np.cos(np.pi * x1),
            x0 - x1,
        ]
    )
    save_npz(path, X, y)


def synthetic_circles_npz(path: Path, n: int = 12000) -> None:
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=n, noise=0.04, factor=0.42, random_state=0)
    x0 = X[:, 0]
    x1 = X[:, 1]
    radius = np.sqrt(x0 * x0 + x1 * x1)
    angle = np.arctan2(x1, x0)
    X = np.column_stack(
        [
            x0,
            x1,
            radius,
            angle,
            radius * radius,
            np.sin(angle),
            np.cos(angle),
            x0 * x1,
        ]
    )
    save_npz(path, X, y)


def sklearn_digits_npz(path: Path) -> None:
    from sklearn.datasets import load_digits

    ds = load_digits()
    save_npz(path, ds.data, ds.target)


def sklearn_iris_npz(path: Path) -> None:
    from sklearn.datasets import load_iris

    ds = load_iris()
    save_npz(path, ds.data, ds.target)


def sklearn_wine_npz(path: Path) -> None:
    from sklearn.datasets import load_wine

    ds = load_wine()
    save_npz(path, ds.data, ds.target)


def sklearn_breast_cancer_npz(path: Path) -> None:
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    save_npz(path, ds.data, ds.target)


def sklearn_olivetti_npz(path: Path) -> None:
    from sklearn.datasets import fetch_olivetti_faces

    ds = fetch_olivetti_faces()
    save_npz(path, ds.data, ds.target)


def pyg_wiki_cs_npz(path: Path, cache_root: Path) -> None:
    from torch_geometric.datasets import WikiCS

    data = WikiCS(root=str(cache_root / "pyg" / "wiki_cs"))[0]
    save_npz(path, data.x.cpu().numpy(), data.y.cpu().numpy())


def ogb_arxiv_npz(path: Path, cache_root: Path) -> None:
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(cache_root / "ogb"))
    graph, labels = dataset[0]
    save_npz(path, graph["node_feat"], labels.reshape(-1))


def synthetic_blobs30k_npz(path: Path) -> None:
    synthetic_blobs_npz(path, n=30000)


def synthetic_moons30k_npz(path: Path) -> None:
    synthetic_moons_npz(path, n=30000)


def synthetic_circles30k_npz(path: Path) -> None:
    synthetic_circles_npz(path, n=30000)


def make_spirals_clean(n: int = 30000, arms: int = 3, seed: int = 0):
    """Three well-separated, low-noise 2-D spiral arms.

    Kept as raw 2-D (euclidean) so the DiRe layout has to untangle the arms. Pure EVoC
    cannot run on this directly: its label-propagation init calls ``PCA(n_components=4)``,
    which scikit-learn rejects for <4 features. That is a dimensionality guard, not a
    clustering verdict; the honest pure-EVoC baseline is measured separately by padding
    the same points to >=4-D (it scores ARI ~= 0.00).
    """
    rng = np.random.default_rng(seed)
    per_arm = n // arms
    xs, ys = [], []
    for arm in range(arms):
        t = np.linspace(0.6, 3.2, per_arm)
        angle = t * 2.0 + arm * (2.0 * np.pi / arms)
        pts = np.column_stack([t * np.cos(angle), t * np.sin(angle)])
        pts = pts + rng.normal(scale=0.03, size=pts.shape)
        xs.append(pts)
        ys.append(np.full(per_arm, arm))
    X = np.vstack(xs).astype(np.float64)
    y = np.concatenate(ys)
    return X, y


def synthetic_spirals_clean30k_npz(path: Path) -> None:
    X, y = make_spirals_clean(n=30000, arms=3, seed=0)
    save_npz(path, X, y)


def hf_dbpedia14_npz(path: Path, full: bool = False) -> None:
    from prepare_datasets import dbpedia14

    dbpedia14(str(path), per_class=None if full else 3500)


LOCAL_LOADERS: dict[str, Callable[[Path], None]] = {
    "synthetic_blobs": synthetic_blobs_npz,
    "synthetic_moons": synthetic_moons_npz,
    "synthetic_circles": synthetic_circles_npz,
    "synthetic_blobs_30k": synthetic_blobs30k_npz,
    "synthetic_moons_30k": synthetic_moons30k_npz,
    "synthetic_circles_30k": synthetic_circles30k_npz,
    "synthetic_spirals_clean_30k": synthetic_spirals_clean30k_npz,
    "sklearn_digits": sklearn_digits_npz,
    "sklearn_iris": sklearn_iris_npz,
    "sklearn_wine": sklearn_wine_npz,
    "sklearn_breast_cancer": sklearn_breast_cancer_npz,
    "sklearn_olivetti": sklearn_olivetti_npz,
    "hf_dbpedia14": hf_dbpedia14_npz,
}


def materialize_dataset(
    spec: DatasetSpec,
    dataset_dir: Path,
    force: bool,
    prefer_hf: bool,
    full: bool = False,
) -> Path:
    path = dataset_cache_path(spec, dataset_dir, full)
    if path.exists() and not force:
        return path

    print(f"preparing {spec.slug}: {spec.label}")
    if spec.source == "openml":
        if prefer_hf and spec.sklearn_fallback is not None:
            print(f"using sklearn fallback for {spec.slug}: {spec.sklearn_fallback}")
            fetch_sklearn_fallback_npz(spec, path)
        elif prefer_hf and spec.hf_name is not None:
            print(f"using Hugging Face mirror for {spec.slug}: {spec.hf_name}")
            try:
                fetch_hf_npz(spec, path, full=full)
            except Exception:
                print(f"Hugging Face mirror failed for {spec.slug}; falling back to OpenML")
                fetch_openml_npz(spec, path, allow_hf_fallback=False, full=full)
        else:
            fetch_openml_npz(spec, path, full=full)
    elif spec.source in LOCAL_LOADERS:
        if spec.source == "hf_dbpedia14":
            hf_dbpedia14_npz(path, full=full)
        else:
            LOCAL_LOADERS[spec.source](path)
    elif spec.source == "pyg_wiki_cs":
        pyg_wiki_cs_npz(path, dataset_dir)
    elif spec.source == "ogb_arxiv":
        ogb_arxiv_npz(path, dataset_dir)
    else:
        raise ValueError(f"unknown dataset source: {spec.source}")
    return path


def dataset_cache_path(spec: DatasetSpec, dataset_dir: Path, full: bool = False) -> Path:
    suffix = "_full" if full else ""
    return dataset_dir / f"{spec.slug}{suffix}.npz"


def parse_dataset_selection(value: str, include_graphs: bool) -> list[str]:
    if value == "smoke":
        slugs = list(SMOKE_DATASETS)
    elif value == "default":
        slugs = list(DEFAULT_DATASETS)
    elif value == "scale":
        slugs = list(SCALE_BENCHMARK_DATASETS)
    elif value == "all":
        slugs = [
            slug
            for slug in DATASETS
            if slug not in GRAPH_DATASETS and slug not in SCALE_DATASETS
        ]
    elif value == "graphs":
        slugs = list(GRAPH_DATASETS)
    else:
        slugs = [item.strip() for item in value.split(",") if item.strip()]

    if include_graphs and value != "graphs":
        slugs.extend(slug for slug in GRAPH_DATASETS if slug not in slugs)

    unknown = [slug for slug in slugs if slug not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset slug(s): {', '.join(unknown)}")
    return slugs


def benchmark_command(spec: DatasetSpec, npz_path: Path, output_dir: Path, args) -> list[str]:
    if args.dry_run and not npz_path.exists():
        n_components = args.n_components
    else:
        with np.load(npz_path) as data:
            n_features = int(data["X"].shape[1])
        n_components = min(args.n_components, n_features)
    base_min_cluster_size = (
        args.base_min_cluster_size
        if args.base_min_cluster_size is not None
        else spec.default_base_min_cluster_size or 5
    )
    base_n_clusters = (
        args.base_n_clusters
        if args.base_n_clusters is not None
        else spec.default_base_n_clusters
    )
    command = [
        sys.executable,
        "examples/benchmark_visuals.py",
        "--npz",
        str(npz_path),
        "--label",
        spec.label,
        "--output-dir",
        str(output_dir / spec.slug),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--min-samples",
        str(args.min_samples),
        "--base-min-cluster-size",
        str(base_min_cluster_size),
        "--n-components",
        str(n_components),
        "--n-neighbors",
        str(args.n_neighbors),
        "--predict-neighbors",
        str(args.predict_neighbors),
        "--max-iter-layout",
        str(args.max_iter_layout),
        "--timing-repeats",
        str(args.timing_repeats),
        "--timing-primary",
        args.timing_primary,
        "--use-gpu",
        args.use_gpu,
        "--metric",
        spec.metric,
        "--selector",
        args.selector,
        "--restarts",
        str(args.restarts),
        "--restart-combine",
        args.restart_combine,
        "--restart-threshold",
        str(args.restart_threshold),
        "--restart-min-cluster-size",
        str(args.restart_min_cluster_size),
    ]
    command.append(
        "--graph-component-candidates"
        if args.graph_component_candidates
        else "--no-graph-component-candidates"
    )
    command.extend(
        [
            "--graph-component-min-stability",
            str(args.graph_component_min_stability),
            "--graph-component-k-radius",
            str(args.graph_component_k_radius),
            "--graph-component-min-cluster-ratio",
            str(args.graph_component_min_cluster_ratio),
        ]
    )
    if base_n_clusters is not None:
        command.extend(["--base-n-clusters", str(base_n_clusters)])
    subsample = None if args.full else (args.subsample or spec.default_subsample)
    if subsample is not None:
        command.extend(["--subsample", str(subsample)])
    if args.skip_fixed_k:
        command.append("--skip-fixed-k")
    return command


def run_command(name: str, command: list[str], log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.txt"
    stderr_path = log_dir / f"{name}.stderr.txt"
    print("running:", " ".join(command))
    t0 = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    seconds = time.perf_counter() - t0
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "seconds": seconds,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def metric_seconds(section: dict, key: str, timing_primary: str):
    warm_key = f"{key}_warm_mean"
    mean_all_key = f"{key}_mean_all"
    if timing_primary == "warm_mean" and section.get(warm_key) is not None:
        return section[warm_key]
    if timing_primary == "mean_all":
        return section.get(mean_all_key, section[key])
    return section[key]


def timing_policy(timing_repeats: int, timing_primary: str) -> str:
    if max(1, int(timing_repeats)) == 1:
        return "single_run"
    if timing_primary == "mean_all":
        return "mean_all_first_included"
    return "warm_mean_first_discarded"


def timing_title(rows: list[dict], timing_primary: str) -> str:
    if rows and all(max(1, int(row.get("timing_repeats") or 1)) == 1 for row in rows):
        return "single run"
    if timing_primary == "mean_all":
        return "mean, first run included"
    return "warm mean, first run discarded"


def metrics_row(slug: str, metrics: dict, timing_primary: str) -> dict:
    dire = metrics["dire_evoc"]
    pure = metrics["vanilla_evoc"]
    timing_repeats = metrics["data"].get("timing_repeats", 1)
    pure_auto = pure.get("auto_ari")
    return {
        "slug": slug,
        "dataset": metrics["data"]["label"],
        "n_train": metrics["data"]["n_train"],
        "n_features": metrics["data"]["n_features"],
        "n_classes": metrics["data"]["n_classes"],
        "min_samples": metrics["data"].get("min_samples"),
        "base_min_cluster_size": metrics["data"].get("base_min_cluster_size"),
        "base_n_clusters": metrics["data"].get("base_n_clusters"),
        "metric": metrics["data"].get("metric", "cosine"),
        "dire_auto_ari": dire["auto_ari"],
        "pure_auto_ari": pure_auto,
        "auto_delta": (dire["auto_ari"] - pure_auto) if pure_auto is not None else None,
        "dire_oracle_ari": dire["oracle_ari"],
        "pure_oracle_ari": pure.get("oracle_ari"),
        "dire_fixed_k_ari": dire.get("fixed_k_ari"),
        "pure_fixed_k_ari": pure.get("fixed_k_ari"),
        "dire_fit_seconds": metric_seconds(dire, "fit_seconds", timing_primary),
        "pure_fit_seconds": metric_seconds(pure, "fit_seconds", timing_primary),
        "dire_fit_seconds_mean_all": dire.get("fit_seconds_mean_all", dire["fit_seconds"]),
        "pure_fit_seconds_mean_all": pure.get("fit_seconds_mean_all", pure["fit_seconds"]),
        "dire_fit_seconds_first": dire.get("fit_seconds_first"),
        "pure_fit_seconds_first": pure.get("fit_seconds_first"),
        "dire_fit_seconds_warm_mean": dire.get("fit_seconds_warm_mean"),
        "pure_fit_seconds_warm_mean": pure.get("fit_seconds_warm_mean"),
        "timing_repeats": timing_repeats,
        "timing_primary": timing_primary,
        "timing_policy": timing_policy(timing_repeats, timing_primary),
        "restart_regime": dire.get("restart_regime", "single_run"),
        "output_dir": slug,
    }


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_suite_summary(rows: list[dict], output_dir: Path, timing_primary: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping suite SVG plots; matplotlib unavailable: {exc}")
        return

    labels = [row["dataset"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig_width = max(9.0, len(rows) * 0.72)

    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar(x - width / 2, [row["dire_auto_ari"] for row in rows], width, label="DiReEVoC")
    ax.bar(x + width / 2, [row["pure_auto_ari"] for row in rows], width, label="Pure EVoC")
    ax.set_title("Auto ARI by dataset")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "suite_auto_ari.svg", format="svg")
    plt.close(fig)

    deltas = [row["auto_delta"] for row in rows]
    colors = ["#2f6f9f" if delta >= 0 else "#c56b37" for delta in deltas]
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(x, deltas, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Auto ARI delta: DiReEVoC minus Pure EVoC")
    ax.set_ylabel("ARI delta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "suite_auto_delta.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar(x - width / 2, [row["dire_fit_seconds"] for row in rows], width, label="DiReEVoC")
    ax.bar(x + width / 2, [row["pure_fit_seconds"] for row in rows], width, label="Pure EVoC")
    ax.set_title(f"Fit time by dataset ({timing_title(rows, timing_primary)})")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "suite_fit_seconds.svg", format="svg")
    plt.close(fig)

    if any(row.get("dire_fit_seconds_mean_all") is not None for row in rows):
        fig, ax = plt.subplots(figsize=(fig_width, 5.0))
        ax.bar(
            x - width / 2,
            [row["dire_fit_seconds_mean_all"] for row in rows],
            width,
            label="DiReEVoC",
        )
        ax.bar(
            x + width / 2,
            [row["pure_fit_seconds_mean_all"] for row in rows],
            width,
            label="Pure EVoC",
        )
        ax.set_title(f"Fit time by dataset ({timing_title(rows, 'mean_all')})")
        ax.set_ylabel("seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / "suite_fit_seconds_mean_all.svg", format="svg")
        plt.close(fig)

    if any(row.get("dire_fit_seconds_warm_mean") is not None for row in rows):
        fig, ax = plt.subplots(figsize=(fig_width, 5.0))
        ax.bar(
            x - width / 2,
            [row["dire_fit_seconds_warm_mean"] for row in rows],
            width,
            label="DiReEVoC",
        )
        ax.bar(
            x + width / 2,
            [row["pure_fit_seconds_warm_mean"] for row in rows],
            width,
            label="Pure EVoC",
        )
        ax.set_title(f"Fit time by dataset ({timing_title(rows, 'warm_mean')})")
        ax.set_ylabel("seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / "suite_fit_seconds_warm.svg", format="svg")
        plt.close(fig)


def list_datasets() -> None:
    for slug, spec in DATASETS.items():
        source = spec.source
        if spec.openml_id is not None:
            source += f":{spec.openml_id}"
        elif spec.openml_name:
            source += f":{spec.openml_name}"
        smoke = " smoke" if slug in SMOKE_DATASETS else ""
        default = " default" if slug in DEFAULT_DATASETS else ""
        scale = " scale" if slug in SCALE_BENCHMARK_DATASETS else ""
        graph = " graph-optional" if slug in GRAPH_DATASETS else ""
        print(f"{slug:<20} {source:<28} {spec.label}{smoke}{default}{scale}{graph}")
        if spec.note:
            print(f"{'':<50}{spec.note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--datasets",
        default="default",
        help="comma-separated slugs, or one of: smoke, default, scale, all, graphs",
    )
    parser.add_argument("--include-graphs", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/dataset-suite")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument(
        "--prefer-hf",
        action="store_true",
        help="use declared mirrors before OpenML; sklearn fallbacks win over HF for sklearn-native datasets",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="ignore per-dataset default caps")
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--base-min-cluster-size", type=int, default=None)
    parser.add_argument(
        "--base-n-clusters",
        type=int,
        default=None,
        help="force a base-layer target cluster count for every dataset; by default only dataset-specific controls use this",
    )
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--predict-neighbors", type=int, default=15)
    parser.add_argument("--max-iter-layout", type=int, default=100)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help=(
            "repeat each timed benchmark this many times; suite fit_seconds follows "
            "--timing-primary and metrics also record first-run, mean-all, and warm-only fields"
        ),
    )
    parser.add_argument(
        "--timing-primary",
        default="warm_mean",
        choices=["warm_mean", "mean_all"],
        help="which repeated timing summary is reported in suite fit_seconds",
    )
    parser.add_argument("--use-gpu", default="auto", choices=["auto", "true", "false"])
    parser.add_argument(
        "--selector",
        default="silhouette",
        choices=["silhouette", "auto", "auto_ensemble", "silhouette_raw", "base", "none"],
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="restart regime forwarded to each dataset: 1 single run; >1 consensus voting",
    )
    parser.add_argument("--restart-combine", default="consensus", choices=["consensus"])
    parser.add_argument("--restart-threshold", type=float, default=0.5)
    parser.add_argument("--restart-min-cluster-size", type=int, default=1)
    parser.add_argument(
        "--graph-component-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--graph-component-min-stability", type=float, default=0.95)
    parser.add_argument("--graph-component-k-radius", type=int, default=4)
    parser.add_argument("--graph-component-min-cluster-ratio", type=float, default=1.5)
    parser.add_argument("--skip-fixed-k", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.timing_repeats = max(1, int(args.timing_repeats))
    if args.datasets == "scale":
        args.full = True
    if args.list_datasets:
        list_datasets()
        return 0

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "datasets"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    slugs = parse_dataset_selection(args.datasets, args.include_graphs)
    manifest = {
        "datasets": slugs,
        "output_dir": str(output_dir),
        "cache_dir": str(dataset_dir),
        "timing_repeats": int(args.timing_repeats),
        "timing_primary": args.timing_primary,
        "timing_policy": timing_policy(args.timing_repeats, args.timing_primary),
        "runs": [],
    }
    rows = []

    for slug in slugs:
        spec = DATASETS[slug]
        try:
            npz_path = (
                dataset_cache_path(spec, dataset_dir, full=args.full)
                if args.dry_run
                else materialize_dataset(
                    spec,
                    dataset_dir,
                    force=args.force_fetch,
                    prefer_hf=args.prefer_hf,
                    full=args.full,
                )
            )
            command = benchmark_command(spec, npz_path, output_dir, args)
            run_name = f"benchmark_visuals_{safe_slug(slug)}"
            if args.dry_run:
                print("dry-run:", " ".join(command))
                continue
            metrics_path = output_dir / spec.slug / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"skipping existing metrics: {metrics_path}")
                run = {"name": run_name, "returncode": 0, "skipped": True}
            else:
                run = run_command(run_name, command, log_dir)
            manifest["runs"].append(run)
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

            if run["returncode"] != 0:
                if args.continue_on_error:
                    continue
                return run["returncode"]
            if metrics_path.exists():
                rows.append(
                    metrics_row(
                        spec.slug,
                        json.loads(metrics_path.read_text()),
                        args.timing_primary,
                    )
                )
        except Exception as exc:
            failure = {"name": slug, "returncode": 1, "error": repr(exc)}
            manifest["runs"].append(failure)
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if not args.continue_on_error:
                raise

    write_summary_csv(rows, output_dir / "suite_metrics.csv")
    plot_suite_summary(rows, output_dir, args.timing_primary)
    manifest["success"] = all(run.get("returncode") == 0 for run in manifest["runs"])
    manifest["summary_csv"] = str(output_dir / "suite_metrics.csv")
    manifest["summary_svgs"] = [
        str(output_dir / "suite_auto_ari.svg"),
        str(output_dir / "suite_auto_delta.svg"),
        str(output_dir / "suite_fit_seconds.svg"),
        str(output_dir / "suite_fit_seconds_warm.svg"),
        str(output_dir / "suite_fit_seconds_mean_all.svg"),
    ]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
