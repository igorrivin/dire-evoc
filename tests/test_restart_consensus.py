"""DiReEVoC restart + consensus voting wiring (CPU; _fit_once stubbed, no GPU)."""
import numpy as np
import pytest

from dire_evoc import DiReEVoC

# 6 points, two triples; ring-ish kNN inside each triple
KNN = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
X = np.zeros((6, 4), dtype=np.float32)


class _FakeReducer:
    def __init__(self, knn):
        self._knn_indices = knn
        self.n_neighbors = knn.shape[1]


def _install_stub(model, per_run_labels):
    """Make _fit_once set carrier state from a scripted sequence of labelings."""
    calls = {"i": 0}
    seeds = []

    def fake_fit_once(X, on_gpu, seed):
        i = calls["i"]
        calls["i"] += 1
        seeds.append(seed)
        labels = np.asarray(per_run_labels[i])
        model.labels_ = labels
        model.reducer_ = _FakeReducer(KNN)
        model.layout_ = np.zeros((labels.shape[0], 2), dtype=np.float32)
        model.cluster_layers_ = [labels]
        model.selected_layer_ = 0
        model.membership_ = np.ones(labels.shape[0])
        model._reference_ = None

    model._fit_once = fake_fit_once
    return seeds


def test_single_run_default_unchanged(monkeypatch):
    model = DiReEVoC(random_state=3)  # restarts=1 default
    seeds = _install_stub(model, [np.array([0, 0, 0, 1, 1, 1])])
    monkeypatch.setattr(model, "_resolve_gpu", lambda: False)
    model.fit(X)
    assert model.restart_regime_ == "single_run"
    assert seeds == [3]
    assert model.labels_.tolist() == [0, 0, 0, 1, 1, 1]


def test_consensus_votes_over_restarts(monkeypatch):
    # 3 runs agree on the two triples (label ids switch); one run is garbage
    per_run = [
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([9, 9, 9, 4, 4, 4]),
        np.array([0, 1, 0, 1, 0, 1]),  # garbage, no within-triple kNN agreement
    ]
    model = DiReEVoC(restarts=3, restart_combine="consensus",
                     restart_threshold=0.5, random_state=0)
    seeds = _install_stub(model, per_run)
    monkeypatch.setattr(model, "_resolve_gpu", lambda: False)
    model.fit(X)
    assert seeds == [0, 1, 2]  # distinct per-restart seeds
    assert model.restart_regime_ == "consensus_of_3"
    # consensus recovers the two triples
    assert len(set(model.labels_[:3])) == 1 and len(set(model.labels_[3:])) == 1
    assert model.labels_[0] != model.labels_[3]
    # membership = per-point non-noise vote fraction (all non-noise here -> 1.0)
    assert np.allclose(model.membership_, 1.0)
    assert model.selected_layer_ is None


def test_unsupported_combine_raises(monkeypatch):
    model = DiReEVoC(restarts=4, restart_combine="best")
    _install_stub(model, [np.zeros(6, dtype=int)] * 4)
    monkeypatch.setattr(model, "_resolve_gpu", lambda: False)
    with pytest.raises(ValueError, match="restart_combine"):
        model.fit(X)
