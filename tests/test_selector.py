"""Coverage-weighted layer selection (CPU; no GPU stack needed).

Locks in the fix for the over-segmentation failure: plain silhouette is computed only on
clustered points, so a layer that dumps half the data as noise but leaves tight residual
clusters wins. Coverage-weighted silhouette (mean over ALL points, noise scored 0) prefers
the layer that is both clean and covers the data. _select_layer is pure sklearn/numpy, so
this needs no cuml/dire/evoc.
"""
import numpy as np

from dire_evoc import DiReEVoC


def _scenario(seed=0):
    """4 well-separated blobs. Return (layout, clean_layer, noise_dumping_layer).

    clean_layer: the correct 4-cluster labelling, full coverage.
    noise-dumping layer: the SAME 4 clusters, but the lowest-silhouette half of the
    points (the diffuse ones near cluster edges) are relabelled noise. Removing the worst
    contributors necessarily *raises* the mean silhouette of what remains while halving
    coverage -- exactly the mechanism by which over-segmented/noisy layers fool a plain
    (clustered-points-only) silhouette. Coverage-weighting must prefer the clean layer.
    """
    from sklearn.metrics import silhouette_samples
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    per = 300
    pts, clean = [], []
    for c, ctr in enumerate(centers):
        pts.append(ctr + rng.normal(scale=1.2, size=(per, 2)))
        clean += [c] * per
    layout = np.vstack(pts)
    clean = np.array(clean)

    ss = silhouette_samples(layout, clean)
    keep = ss >= np.median(ss)                       # drop the worst-silhouette half
    noise_dumping = np.where(keep, clean, -1)
    return layout, clean, noise_dumping


def test_coverage_weighting_avoids_noise_dumping():
    layout, clean, noise_dumping = _scenario()
    layers = [noise_dumping, clean]

    cov = DiReEVoC(selector="silhouette")._select_layer(layout, layers)
    assert cov == 1, "coverage-weighted silhouette should pick the clean, full-coverage layer"


def test_raw_silhouette_is_fooled():
    """Sanity check the failure mode exists: raw silhouette favours the layer that
    discards low-silhouette points as noise (this is what coverage-weighting fixes)."""
    layout, clean, noise_dumping = _scenario()
    layers = [noise_dumping, clean]
    raw = DiReEVoC(selector="silhouette_raw")._select_layer(layout, layers)
    assert raw == 0, "raw silhouette is expected to be fooled by the noise-dumping layer"


def test_base_and_none_selectors():
    layout, clean, noise_dumping = _scenario()
    layers = [noise_dumping, clean]
    assert DiReEVoC(selector="base")._select_layer(layout, layers) == 0
    assert DiReEVoC(selector="none")._select_layer(layout, layers) == len(layers) - 1


def _knn_indices(X, k=10):
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        return_distance=False
    )[:, 1:]


def test_fragment_merge_unions_adjacent_chain_fragments():
    """Adjacent fragments of two continuous components should collapse by kNN links."""
    x = np.linspace(0.0, 10.0, 500)
    top = np.column_stack([x, np.zeros_like(x)])
    bottom = np.column_stack([x, np.ones_like(x) * 10.0])
    X = np.vstack([top, bottom])
    labels = np.concatenate(
        [
            np.repeat(np.arange(5), 100),
            np.repeat(np.arange(5, 10), 100),
        ]
    )

    merged, info = DiReEVoC()._fragment_merge_labels(labels, _knn_indices(X, k=12))

    assert info["merged"]
    assert np.unique(merged).size == 2
    assert np.unique(merged[:500]).size == 1
    assert np.unique(merged[500:]).size == 1
    assert merged[0] != merged[-1]


def test_fragment_merge_keeps_separated_blobs_apart():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    X = np.vstack([center + rng.normal(scale=0.15, size=(80, 2)) for center in centers])
    labels = np.repeat(np.arange(4), 80)

    merged, info = DiReEVoC()._fragment_merge_labels(labels, _knn_indices(X, k=12))

    assert not info["merged"]
    assert np.array_equal(merged, labels)
