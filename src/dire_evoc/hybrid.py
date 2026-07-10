"""GPU-accelerated EVoC Phase 2: reuse cuML's single-linkage tree, keep EVoC's layers.

EVoC's ``build_cluster_layers`` spends ~94% of its time (at 1M points) building the
Boruvka MST / single-linkage tree (SLT) on the mutual-reachability graph -- which is
*exactly* HDBSCAN*'s SLT, and cuML/RAFT already builds that on the GPU. The remaining
~6% (condense + persistence + per-layer labels) is the cheap O(n)-tree post-process
that makes EVoC != HDBSCAN. So: build the SLT on the GPU, hand it to EVoC's post-process
unchanged.

Format note (validated): cuML returns ``single_linkage_tree_`` as an
``hdbscan.plots.SingleLinkageTree``; ``.to_numpy()`` is *exactly* EVoC's linkage matrix
-- (n-1, 4) rows ``[child_a, child_b, mutual_reachability_distance, size]``, point ids
0..n-1, internal nodes n..2n-2, root 2(n-1). cuML puts the smaller child first and EVoC
the larger, but EVoC's ``condense_tree`` is order-independent. The SLT depends only on
``min_samples`` (core distances), NOT ``min_cluster_size``, so only ``min_samples`` must
match between the two engines.

Two traps the benchmark surfaced (encoded as defaults here): cuML's ``build_algo``
default ``brute_force`` is O(n^2) -- pass ``nn_descent`` at scale; and feeding a *device*
array triggers a slower RAFT MST-reconnection path on well-separated data -- feed a host
numpy array (the H2D copy is trivial).
"""
import operator

import numpy as np

__all__ = ["gpu_single_linkage_tree", "hybrid_cluster_layers"]


def gpu_single_linkage_tree(layout, min_samples=10, build_algo="nn_descent",
                            metric="euclidean"):
    """Build the HDBSCAN* single-linkage tree on ``layout`` using cuML (GPU).

    Returns the SLT as a float64 ``(n-1, 4)`` linkage matrix in EVoC/scipy format,
    ready for :func:`evoc.cluster_trees.condense_tree`.

    Parameters
    ----------
    layout : array (n, d)
        The (low-dimensional) DiRe layout to cluster. Passed as a *host* numpy array
        on purpose (see module docstring).
    min_samples : int
        HDBSCAN* core-distance parameter; MUST match the value EVoC would use for an
        exact correspondence.
    build_algo : {'nn_descent', 'brute_force'}
        kNN construction. 'nn_descent' (approximate, sub-quadratic) is required at
        scale; 'brute_force' is exact but O(n^2).
    """
    X = np.ascontiguousarray(np.asarray(layout), dtype=np.float32)   # host array
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("layout must be a two-dimensional array with at least two rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("layout must contain only finite values")
    if isinstance(min_samples, (bool, np.bool_)):
        raise ValueError("min_samples must be a positive integer")
    try:
        min_samples = operator.index(min_samples)
    except TypeError as exc:
        raise ValueError("min_samples must be a positive integer") from exc
    if min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if build_algo not in {"nn_descent", "brute_force"}:
        raise ValueError("build_algo must be 'nn_descent' or 'brute_force'")
    from cuml.cluster import HDBSCAN

    model = HDBSCAN(min_samples=min_samples, min_cluster_size=max(2, min_samples),
                    metric=metric, build_algo=build_algo)
    model.fit(X)
    return model.single_linkage_tree_.to_numpy().astype(np.float64)


def _cluster_layers_from_linkage(
    uncondensed_tree,
    n_samples,
    *,
    base_min_cluster_size=10,
    base_n_clusters=None,
    min_similarity_threshold=0.2,
    max_layers=10,
):
    """Run EVoC's layer post-process on an already-built linkage tree."""
    from evoc.cluster_trees import (
        condense_tree,
        extract_leaves,
        get_cluster_label_vector,
        get_point_membership_strength_vector,
        mask_condensed_tree,
    )
    from evoc.clustering_utilities import (
        _binary_search_for_n_clusters,
        compute_total_persistence,
        extract_clusters_by_id,
        find_peaks,
        min_cluster_size_barcode,
        select_diverse_peaks,
    )

    min_cluster_size = base_min_cluster_size
    cluster_layers = []
    membership_strength_layers = []
    persistence_scores = []

    if base_n_clusters is not None:
        _leaves, clusters, strengths = _binary_search_for_n_clusters(
            uncondensed_tree,
            base_n_clusters,
            n_samples=n_samples,
        )
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        if len(cluster_sizes) > 0:
            min_cluster_size = max(1, np.min(cluster_sizes))
        condensed_tree = condense_tree(uncondensed_tree, min_cluster_size)
    else:
        condensed_tree = condense_tree(uncondensed_tree, base_min_cluster_size)
        leaves = extract_leaves(condensed_tree)
        clusters = get_cluster_label_vector(condensed_tree, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(
            condensed_tree,
            leaves,
            clusters,
        )

    mask = condensed_tree.child >= n_samples
    cluster_tree = mask_condensed_tree(condensed_tree, mask)

    if len(cluster_tree.child) > 0 and cluster_tree.child[-1] >= n_samples:
        births, deaths, _parents, lambda_deaths = min_cluster_size_barcode(
            cluster_tree,
            n_samples,
            min_cluster_size,
        )
        sizes, total_persistence = compute_total_persistence(
            births,
            deaths,
            lambda_deaths,
        )
        peaks = find_peaks(total_persistence)
    else:
        births = np.array([])
        deaths = np.array([])
        sizes = np.array([])
        total_persistence = np.array([])
        peaks = np.array([], dtype=np.int64)

    cluster_layers.append(clusters)
    membership_strength_layers.append(strengths)
    persistence_scores.append(0.0)

    selected_peaks = select_diverse_peaks(
        peaks,
        total_persistence,
        sizes,
        births,
        deaths,
        min_similarity_threshold=min_similarity_threshold,
        max_layers=max_layers - 1,
    )

    for peak in selected_peaks:
        best_birth = sizes[peak]
        persistence = total_persistence[peak]
        selected_clusters = (
            np.where((births <= best_birth) & (deaths > best_birth))[0] + n_samples
        )
        labels, strengths = extract_clusters_by_id(condensed_tree, selected_clusters)
        cluster_layers.append(labels)
        membership_strength_layers.append(strengths)
        persistence_scores.append(persistence)

    n_clusters_per_layer = [layer.max() + 1 for layer in cluster_layers]
    sorted_indices = np.argsort(n_clusters_per_layer)[::-1]

    return (
        [cluster_layers[i] for i in sorted_indices],
        [membership_strength_layers[i] for i in sorted_indices],
        [persistence_scores[i] for i in sorted_indices],
    )


def hybrid_cluster_layers(layout, min_samples=10, build_algo="nn_descent", **kwargs):
    """Full hybrid Phase 2: GPU SLT -> EVoC's condense/persistence/layer machinery.

    Builds the single-linkage tree on the GPU, then runs EVoC's exact post-process via
    the ``precomputed_linkage`` seam when available. Current PyPI EVoC does not expose
    that parameter, so the fallback calls the same EVoC tree utility functions directly.
    """
    metric = kwargs.pop("metric", "euclidean")
    try:
        from evoc.clustering import build_cluster_layers
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "hybrid_cluster_layers requires the `evoc` package (Igor's fork) to be "
            "installed for the layer post-process."
        ) from exc

    import inspect

    has_precomputed_seam = (
        "precomputed_linkage" in inspect.signature(build_cluster_layers).parameters
    )
    fallback_kwargs = None
    if not has_precomputed_seam:
        fallback_kwargs = {
            "base_min_cluster_size": kwargs.pop("base_min_cluster_size", 10),
            "base_n_clusters": kwargs.pop("base_n_clusters", None),
            "min_similarity_threshold": kwargs.pop("min_similarity_threshold", 0.2),
            "max_layers": kwargs.pop("max_layers", 10),
        }
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                f"installed EVoC lacks the precomputed_linkage API; unsupported hybrid "
                f"keyword(s): {unsupported}"
            )

    slt = gpu_single_linkage_tree(
        layout,
        min_samples=min_samples,
        build_algo=build_algo,
        metric=metric,
    )
    if has_precomputed_seam:
        return build_cluster_layers(layout, precomputed_linkage=slt,
                                    min_samples=min_samples, **kwargs)

    return _cluster_layers_from_linkage(
        slt,
        np.asarray(layout).shape[0],
        **fallback_kwargs,
    )
