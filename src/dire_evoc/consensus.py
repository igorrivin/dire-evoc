"""Consensus (voting) clustering over restarts via a SPARSE co-association graph.

Cluster ids are arbitrary across independently seeded runs (label switching) and the
cluster count can differ, so labels cannot be averaged directly. We deliberately avoid the
O(n^2) dense co-association matrix: candidate edges are restricted to each run's kNN graph
(which DiRe already computes), so the association graph is sparse -- O(N * n * k) edges.

An undirected pair ``(i, j)`` votes in run ``r`` when ``j`` is a kNN-neighbour of ``i``
(or vice versa) and the two points share a non-noise cluster in run ``r``. An edge is kept
when its vote fraction across the ``N`` runs is at least ``threshold``; the consensus
clusters are the connected components of that thresholded sparse graph, found with the
package union-find (no n x n matrix is ever built). Components smaller than
``min_cluster_size`` are demoted to noise (-1).
"""
from __future__ import annotations

import numpy as np

from ._unionfind import components_from_edges


def _canonical_pairs_for_run(labels, neighbors):
    """Undirected (i<j) pairs that co-cluster within this run's kNN graph, one per pair."""
    labels = np.asarray(labels)
    neighbors = np.asarray(neighbors)
    n = labels.shape[0]
    if neighbors.ndim != 2 or neighbors.shape[0] != n:
        raise ValueError("neighbor graph must be (n, k) aligned with labels")
    k = neighbors.shape[1]
    src = np.repeat(np.arange(n), k)
    dst = neighbors.reshape(-1)
    valid = (dst >= 0) & (dst < n) & (dst != src)
    src, dst = src[valid], dst[valid]
    same = (labels[src] >= 0) & (labels[src] == labels[dst])
    src, dst = src[same], dst[same]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    pairs = np.stack([lo, hi], axis=1)
    if pairs.shape[0] == 0:
        return pairs.reshape(0, 2)
    return np.unique(pairs, axis=0)  # one vote per undirected pair per run


def _finalize(labels, min_cluster_size):
    labels = np.asarray(labels).copy()
    if min_cluster_size > 1:
        vals, counts = np.unique(labels, return_counts=True)
        small = vals[counts < min_cluster_size]
        if small.size:
            labels[np.isin(labels, small)] = -1
    out = np.full(labels.shape, -1, dtype=np.int64)
    for new, old in enumerate(np.unique(labels[labels >= 0])):
        out[labels == old] = new
    return out


def consensus_labels(labelings, neighbor_lists, threshold=0.5, min_cluster_size=1):
    """Sparse co-association consensus over restarts.

    Parameters
    ----------
    labelings : list of (n,) int arrays
        Per-restart cluster labels on the SAME points in the SAME order (-1 = noise).
    neighbor_lists : list of (n, k) int arrays
        Per-restart kNN index graphs (e.g. ``reducer._knn_indices``). Candidate edges; this
        is what keeps the association graph sparse. ``k`` may differ between runs.
    threshold : float in [0, 1]
        Minimum fraction of runs that must vote for an edge to survive (inclusive).
    min_cluster_size : int
        Connected components smaller than this become noise (-1).

    Returns
    -------
    labels : (n,) int array, consensus clusters compacted to 0..k-1 (-1 = noise).
    """
    n_runs = len(labelings)
    if n_runs == 0:
        raise ValueError("need at least one run")
    if len(neighbor_lists) != n_runs:
        raise ValueError("need one neighbor graph per labeling")
    n = np.asarray(labelings[0]).shape[0]
    if any(np.asarray(lab).shape[0] != n for lab in labelings):
        raise ValueError("all labelings must cover the same points in the same order")

    per_run = [_canonical_pairs_for_run(labelings[r], neighbor_lists[r]) for r in range(n_runs)]
    per_run = [p for p in per_run if p.shape[0]]
    if per_run:
        stacked = np.concatenate(per_run, axis=0)
        pairs, votes = np.unique(stacked, axis=0, return_counts=True)
        edges = pairs[votes >= threshold * n_runs]
    else:
        edges = np.empty((0, 2), dtype=np.int64)  # no agreement -> all singletons

    labels = components_from_edges(n, edges)
    return _finalize(labels, min_cluster_size)


def consensus_over_restarts(run_fn, n_restarts, base_seed=0,
                            threshold=0.5, min_cluster_size=1):
    """Run ``run_fn(seed) -> (labels, knn_indices)`` ``n_restarts`` times and vote.

    The GPU/DiRe work lives entirely inside ``run_fn``; this orchestration is backend-free
    so it can be unit-tested with a stub. ``base_seed=None`` runs nondeterministically.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    labelings, neighbor_lists = [], []
    for i in range(n_restarts):
        seed = None if base_seed is None else int(base_seed) + i
        labels, neighbors = run_fn(seed)
        labelings.append(np.asarray(labels))
        neighbor_lists.append(np.asarray(neighbors))
    labels = consensus_labels(labelings, neighbor_lists, threshold, min_cluster_size)
    info = {
        "regime": f"consensus_of_{n_restarts}",
        "n_restarts": int(n_restarts),
        "threshold": float(threshold),
        "min_cluster_size": int(min_cluster_size),
        "n_consensus_clusters": int(np.unique(labels[labels >= 0]).size),
        "noise_fraction": float(np.mean(labels < 0)),
    }
    return labels, info
