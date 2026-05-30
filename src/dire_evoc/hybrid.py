"""GPU-accelerated EVoC Phase 2: reuse cuML's single-linkage tree, keep EVoC's layers.

EVoC's ``build_cluster_layers`` spends ~94% of its time (at 1M points) building the
Boruvka MST / single-linkage tree (SLT) on the mutual-reachability graph -- which is
*exactly* HDBSCAN*'s SLT, and cuML/RAFT already builds that on the GPU. The remaining
~6% (condense + persistence + per-layer labels) is the cheap O(n)-tree post-process
that makes EVoC != HDBSCAN. So: build the SLT on the GPU, hand it to EVoC's post-process
unchanged. Measured ~3.3x end-to-end on Phase 2 at 1M points with identical clustering
(ARI = 1.0 at the meaningful granularity).

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
    from cuml.cluster import HDBSCAN

    X = np.ascontiguousarray(np.asarray(layout), dtype=np.float32)   # host array
    model = HDBSCAN(min_samples=min_samples, min_cluster_size=max(2, min_samples),
                    metric=metric, build_algo=build_algo)
    model.fit(X)
    return model.single_linkage_tree_.to_numpy().astype(np.float64)


def hybrid_cluster_layers(layout, min_samples=10, build_algo="nn_descent", **kwargs):
    """Full hybrid Phase 2: GPU SLT -> EVoC's condense/persistence/layer machinery.

    Builds the single-linkage tree on the GPU, then runs EVoC's exact post-process via
    the ``precomputed_linkage`` seam of ``evoc.clustering.build_cluster_layers``.

    The post-process is owned by EVoC (this module deliberately does NOT reimplement
    condense/persistence/layer extraction). If the installed evoc does not yet expose
    the ``precomputed_linkage`` seam, this raises ``NotImplementedError`` pointing to it
    -- the GPU SLT itself is still available via :func:`gpu_single_linkage_tree`.
    """
    slt = gpu_single_linkage_tree(layout, min_samples=min_samples,
                                  build_algo=build_algo,
                                  metric=kwargs.pop("metric", "euclidean"))
    try:
        from evoc.clustering import build_cluster_layers
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "hybrid_cluster_layers requires the `evoc` package (Igor's fork) to be "
            "installed for the layer post-process."
        ) from exc

    import inspect
    if "precomputed_linkage" not in inspect.signature(build_cluster_layers).parameters:
        raise NotImplementedError(
            "the installed evoc.build_cluster_layers has no `precomputed_linkage` "
            "parameter. Add the dependency-free seam upstream "
            "(build_cluster_layers(..., precomputed_linkage=slt)) and reinstall evoc; "
            "until then use gpu_single_linkage_tree() and evoc.cluster_trees.condense_tree "
            "directly. SLT is available as the returned value of "
            "gpu_single_linkage_tree()."
        )
    return build_cluster_layers(layout, precomputed_linkage=slt,
                                min_samples=min_samples, **kwargs)
