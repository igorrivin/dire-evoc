"""Backend-boundary tests for the RAPIDS/EVoC hybrid without requiring a GPU."""

import sys
import types

import numpy as np
import pytest

from dire_evoc import hybrid


def _install_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_gpu_single_linkage_tree_adapts_cuml_result(monkeypatch):
    calls = {}
    expected = np.array([[0.0, 1.0, 0.25, 2.0]], dtype=np.float32)

    class FakeTree:
        def to_numpy(self):
            return expected

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            self.single_linkage_tree_ = FakeTree()

        def fit(self, X):
            calls["X"] = X

    _install_module(monkeypatch, "cuml")
    _install_module(monkeypatch, "cuml.cluster", HDBSCAN=FakeHDBSCAN)

    layout = np.array([[0, 1], [2, 3]], dtype=np.float64, order="F")
    result = hybrid.gpu_single_linkage_tree(layout, min_samples=2, build_algo="brute_force")

    assert calls["kwargs"] == {
        "min_samples": 2,
        "min_cluster_size": 2,
        "metric": "euclidean",
        "build_algo": "brute_force",
    }
    assert calls["X"].dtype == np.float32 and calls["X"].flags.c_contiguous
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_hybrid_uses_public_precomputed_linkage_seam(monkeypatch):
    slt = np.array([[0.0, 1.0, 0.5, 2.0]])
    calls = {}

    def build_cluster_layers(layout, precomputed_linkage=None, min_samples=None, **kwargs):
        calls.update(
            layout=layout,
            precomputed_linkage=precomputed_linkage,
            min_samples=min_samples,
            kwargs=kwargs,
        )
        return "layers"

    _install_module(monkeypatch, "evoc")
    _install_module(monkeypatch, "evoc.clustering", build_cluster_layers=build_cluster_layers)
    monkeypatch.setattr(hybrid, "gpu_single_linkage_tree", lambda *args, **kwargs: slt)
    layout = np.zeros((2, 3), dtype=np.float32)

    result = hybrid.hybrid_cluster_layers(
        layout,
        min_samples=7,
        base_min_cluster_size=4,
    )

    assert result == "layers"
    assert calls["layout"] is layout
    assert calls["precomputed_linkage"] is slt
    assert calls["min_samples"] == 7
    assert calls["kwargs"] == {"base_min_cluster_size": 4}


def test_hybrid_fallback_passes_only_supported_options(monkeypatch):
    slt = np.array([[0.0, 1.0, 0.5, 2.0]])
    calls = {}

    def build_cluster_layers(layout, min_samples=None):
        raise AssertionError("fallback should not call build_cluster_layers")

    def fallback(linkage, n_samples, **kwargs):
        calls.update(linkage=linkage, n_samples=n_samples, kwargs=kwargs)
        return "fallback"

    _install_module(monkeypatch, "evoc")
    _install_module(monkeypatch, "evoc.clustering", build_cluster_layers=build_cluster_layers)
    monkeypatch.setattr(hybrid, "gpu_single_linkage_tree", lambda *args, **kwargs: slt)
    monkeypatch.setattr(hybrid, "_cluster_layers_from_linkage", fallback)

    result = hybrid.hybrid_cluster_layers(
        np.zeros((5, 2)),
        base_min_cluster_size=3,
        base_n_clusters=2,
        max_layers=6,
    )

    assert result == "fallback"
    assert calls["linkage"] is slt
    assert calls["n_samples"] == 5
    assert calls["kwargs"] == {
        "base_min_cluster_size": 3,
        "base_n_clusters": 2,
        "min_similarity_threshold": 0.2,
        "max_layers": 6,
    }

    with pytest.raises(TypeError, match="unsupported hybrid keyword.*future_option"):
        hybrid.hybrid_cluster_layers(np.zeros((5, 2)), future_option=True)


@pytest.mark.parametrize(
    "layout, message",
    [
        (np.zeros(3), "two-dimensional"),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), "finite"),
    ],
)
def test_gpu_single_linkage_tree_validates_before_backend_fit(monkeypatch, layout, message):
    class UnexpectedHDBSCAN:
        def __init__(self, **kwargs):
            raise AssertionError("invalid input should fail before constructing cuML")

    _install_module(monkeypatch, "cuml")
    _install_module(monkeypatch, "cuml.cluster", HDBSCAN=UnexpectedHDBSCAN)
    with pytest.raises(ValueError, match=message):
        hybrid.gpu_single_linkage_tree(layout)
