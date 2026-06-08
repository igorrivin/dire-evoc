"""Sparse co-association consensus voting over restarts (CPU; numpy + scipy only)."""
import numpy as np
import pytest

from dire_evoc.consensus import (
    consensus_labels,
    consensus_over_restarts,
    _canonical_pairs_for_run,
)


def _ring_neighbors(n):
    """Each point's neighbors are its two ring neighbors (a sparse (n,2) kNN stand-in)."""
    idx = np.arange(n)
    return np.stack([(idx - 1) % n, (idx + 1) % n], axis=1)


def test_canonical_pairs_dedupe_and_noise():
    labels = np.array([0, 0, 0, -1])
    nbrs = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])  # point 3 is noise
    pairs = _canonical_pairs_for_run(labels, nbrs)
    # (0,1),(0,2),(1,2) once each; nothing involving the noise point 3
    assert set(map(tuple, pairs)) == {(0, 1), (0, 2), (1, 2)}


def test_two_blocks_become_two_components():
    # 6 points, two co-clustered triples in every run; neighbors only within each triple
    labs = [np.array([0, 0, 0, 1, 1, 1]), np.array([5, 5, 5, 2, 2, 2])]  # label-switched
    nbrs = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    out = consensus_labels(labs, [nbrs, nbrs], threshold=0.5)
    assert np.unique(out).size == 2
    assert len(set(out[:3])) == 1 and len(set(out[3:])) == 1
    assert out[0] != out[3]


def test_threshold_filters_minority_edges():
    # edge (0,1) co-clusters in only 1 of 2 runs
    labs = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0])]
    nbrs = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
    # pair (0,1): same-cluster only in run 0 -> 1/2 votes
    lenient = consensus_labels(labs, [nbrs, nbrs], threshold=0.5)   # 0.5 keeps it
    strict = consensus_labels(labs, [nbrs, nbrs], threshold=0.75)   # 0.75 drops it
    assert lenient[0] == lenient[1]
    assert strict[0] != strict[1]


def test_connected_components_merge_transitively():
    # chain 0-1-2-3 co-clusters pairwise across a ring graph -> one component
    n = 4
    labs = [np.zeros(n, dtype=int), np.zeros(n, dtype=int)]
    nbrs = _ring_neighbors(n)
    out = consensus_labels(labs, [nbrs, nbrs], threshold=0.5)
    assert np.unique(out).size == 1


def test_min_cluster_size_demotes_singletons_to_noise():
    # point 3 never co-clusters (its neighbors are in other clusters) -> singleton -> noise
    labs = [np.array([0, 0, 0, 1]), np.array([0, 0, 0, 2])]
    nbrs = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    out = consensus_labels(labs, [nbrs, nbrs], threshold=0.5, min_cluster_size=2)
    assert out[3] == -1
    assert len(set(out[:3])) == 1


def test_no_agreement_all_singletons():
    labs = [np.array([-1, -1, -1])]
    nbrs = np.array([[1, 2], [0, 2], [0, 1]])
    out = consensus_labels(labs, [nbrs], threshold=0.5, min_cluster_size=2)
    assert np.all(out == -1)


def test_consensus_over_restarts_with_stub_run_fn():
    nbrs = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    seeds = []

    def run_fn(seed):
        seeds.append(seed)
        # deterministic two-block labeling regardless of seed (label ids vary)
        return np.array([seed % 3] * 3 + [(seed % 3) + 9] * 3), nbrs

    labels, info = consensus_over_restarts(run_fn, n_restarts=4, base_seed=0, threshold=0.5)
    assert seeds == [0, 1, 2, 3]
    assert info["regime"] == "consensus_of_4"
    assert info["n_consensus_clusters"] == 2
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1


def test_scales_sparsely_without_dense_matrix():
    # 20k points, k=4 ring graph: must complete fast and never build an n^2 array
    n = 20_000
    nbrs = np.stack([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n,
                     (np.arange(n) - 2) % n, (np.arange(n) + 2) % n], axis=1)
    labs = [np.zeros(n, dtype=int), np.zeros(n, dtype=int)]
    out = consensus_labels(labs, [nbrs, nbrs], threshold=0.5)
    assert out.shape == (n,)
    assert np.unique(out).size == 1  # the ring is one connected component


def test_input_validation():
    with pytest.raises(ValueError, match="at least one run"):
        consensus_labels([], [])
    with pytest.raises(ValueError, match="one neighbor graph"):
        consensus_labels([np.zeros(3)], [])
    with pytest.raises(ValueError, match="same points"):
        consensus_labels([np.zeros(3), np.zeros(4)], [np.zeros((3, 2)), np.zeros((4, 2))])
