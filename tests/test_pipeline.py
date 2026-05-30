"""Smoke test for the end-to-end DiReEVoC pipeline.

Requires the GPU stack (dire-rapids + cuml + evoc); skipped otherwise so the
pure-numpy CPU suite stays green on any machine. The substantive end-to-end
validation (correctness ARI, throughput) lives in scripts/validate_hybrid_e2e.py.
"""
import numpy as np
import pytest

pytest.importorskip("cuml")
pytest.importorskip("dire_rapids")
pytest.importorskip("evoc")

from dire_evoc import DiReEVoC


def _blobs(n=6000, k=6, d=24, spread=0.10, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(k, d)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    per = n // k
    parts, y = [], []
    for c in range(k):
        a = C[c] + rng.normal(scale=spread, size=(per, d)).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        parts.append(a)
        y += [c] * per
    return np.ascontiguousarray(np.vstack(parts)), np.array(y)


def test_pipeline_fit_predict_dimension():
    X, y = _blobs()
    model = DiReEVoC(min_samples=10, random_state=0)
    labels = model.fit_predict(X)

    assert labels.shape == (X.shape[0],)
    assert model.layout_.shape == (X.shape[0], model.n_components)
    assert 0 <= model.selected_layer_ < len(model.cluster_layers_)
    assert model.intrinsic_dimension_ is None or model.intrinsic_dimension_ > 0

    pred, mem = model.predict(X[:500], return_membership=True)
    assert pred.shape == (500,)
    assert mem.min() > 0.0 and mem.max() <= 1.0 + 1e-9

    with pytest.raises(ValueError):
        model.predict(X[:, :5])
