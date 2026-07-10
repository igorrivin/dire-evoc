"""Coverage-weighted layer selection (CPU; no GPU stack needed).

Locks in the fix for the over-segmentation failure: plain silhouette is computed only on
clustered points, so a layer that dumps half the data as noise but leaves tight residual
clusters wins. Coverage-weighted silhouette (mean over ALL points, noise scored 0) prefers
the layer that is both clean and covers the data. _select_layer is pure sklearn/numpy, so
this needs no cuml/dire/evoc.
"""
import numpy as np

from dire_evoc import DiReEVoC
from dire_evoc.selectors import (
    coverage_silhouette_score,
    guarded_graph_component_decision,
    or_probability,
)


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


def test_silhouette_rejects_one_label_per_point_without_sklearn_error():
    X = np.arange(6, dtype=float)[:, None]
    assert coverage_silhouette_score(X, np.arange(6)) == -np.inf


def _knn_indices(X, k=10):
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        return_distance=False
    )[:, 1:]


def test_graph_component_candidate_accepts_stable_small_k_components():
    """Stable connected components in the stored kNN graph should become a candidate."""
    x = np.linspace(0.0, 10.0, 200)
    top = np.column_stack([x, np.zeros_like(x)])
    bottom = np.column_stack([x, np.ones_like(x) * 10.0])
    X = np.vstack([top, bottom])

    class Reducer:
        _knn_indices = _knn_indices(X, k=8)

    model = DiReEVoC(
        graph_component_min_stability=0.95,
        graph_component_k_radius=1,
        base_min_cluster_size=20,
    )
    labels, info = model._graph_component_candidate(Reducer(), min_component_size=20)

    assert info["accepted"]
    assert info["components"] == 2
    assert labels is not None
    assert np.unique(labels).size == 2
    assert np.unique(labels[:200]).size == 1
    assert np.unique(labels[200:]).size == 1
    assert labels[0] != labels[-1]


def test_graph_component_arbitration_rejects_binary_collapse_without_target_count():
    evoc_labels = np.repeat(np.arange(11), 20)
    graph_labels = np.repeat(np.arange(2), 110)
    model = DiReEVoC(graph_component_min_cluster_ratio=1.5)
    model.graph_component_candidate_ = {"accepted": True}

    use_graph, info = model._should_select_graph_component_layer(evoc_labels, graph_labels)

    assert not use_graph
    assert info["selection_reason"] == "too_few_graph_clusters_without_base_n_clusters"


def test_graph_component_arbitration_allows_binary_target_count_match():
    evoc_labels = np.repeat(np.arange(11), 20)
    graph_labels = np.repeat(np.arange(2), 110)
    model = DiReEVoC(base_n_clusters=2, graph_component_min_cluster_ratio=1.5)
    model.graph_component_candidate_ = {"accepted": True}

    use_graph, info = model._should_select_graph_component_layer(evoc_labels, graph_labels)

    assert use_graph
    assert info["selection_reason"] == "matches_base_n_clusters"


def test_graph_component_arbitration_allows_three_component_fragmentation_fix():
    evoc_labels = np.repeat(np.arange(9), 30)
    graph_labels = np.repeat(np.arange(3), 90)
    model = DiReEVoC(graph_component_min_cluster_ratio=1.5)
    model.graph_component_candidate_ = {"accepted": True}

    use_graph, info = model._should_select_graph_component_layer(evoc_labels, graph_labels)

    assert use_graph
    assert info["selection_reason"] == "evoc_fragmented_relative_to_graph"


def test_selector_noisy_or_combines_independent_confidence_scores():
    assert np.isclose(or_probability(0.2, 0.5), 0.6)
    assert np.isclose(or_probability(0.2, 0.5, 0.25), 0.7)


def test_auto_ensemble_keeps_hard_binary_graph_guard():
    evoc_labels = np.repeat(np.arange(11), 20)
    graph_labels = np.repeat(np.arange(2), 110)

    decision = guarded_graph_component_decision(
        evoc_labels=evoc_labels,
        graph_labels=graph_labels,
        graph_info={"accepted": True, "stability": 1.0},
        base_n_clusters=None,
        min_cluster_ratio=1.5,
        policy="auto_ensemble",
        silhouette_score=-0.5,
    )

    assert not decision.selected
    assert decision.reason == "too_few_graph_clusters_without_base_n_clusters"
    assert decision.diagnostics["graph_confidence"] == 0.0


def test_default_selector_skips_graph_component_sweep(monkeypatch):
    layout = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    model = DiReEVoC(selector="silhouette", adaptive_min_samples=False)

    class Reducer:
        n_neighbors = 2

    monkeypatch.setattr(model, "_build_layout", lambda X, seed=None: (layout, Reducer()))
    monkeypatch.setattr(
        model,
        "_cluster",
        lambda layout, on_gpu: ([labels], [np.ones(4)], [0.0]),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ordinary silhouette selection must not build graph candidates")

    monkeypatch.setattr(model, "_maybe_add_graph_component_layer", fail_if_called)
    model._fit_once(np.asarray(layout), on_gpu=False, seed=0)

    assert model.graph_component_candidate_["reason"] == "disabled_for_selector"
