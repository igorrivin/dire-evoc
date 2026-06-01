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


def test_adaptive_smoothing_recovers_tight_blobs():
    """Ultra-tight blobs make the DiRe layout tighter than EVoC's default smoothing
    expects, so the base clustering over-fragments (~85% noise). The adaptive search
    should escalate smoothing and recover a clean clustering; disabling it should not."""
    X, y = _blobs(n=12000, k=8, d=32, spread=0.10)
    from sklearn.metrics import adjusted_rand_score

    adaptive = DiReEVoC(min_samples=10, adaptive_min_samples=True, random_state=0).fit(X)
    fixed = DiReEVoC(min_samples=10, adaptive_min_samples=False, random_state=0).fit(X)

    ari_adaptive = adjusted_rand_score(y, adaptive.labels_)
    ari_fixed = adjusted_rand_score(y, fixed.labels_)

    # Assert the MECHANISM, not an absolute ARI ceiling: the escalation must fire and must
    # substantially beat the over-fragmented fixed base. The absolute ARI reaches 1.0 in
    # isolation but wobbles with approximate cuVS-kNN nondeterminism / test-order GPU state,
    # so a fixed high threshold would be flaky; the gap over the ~0.0 base is robust.
    assert (
        adaptive.selected_min_samples_ > adaptive.min_samples
        or adaptive.selected_base_min_cluster_size_ > adaptive.base_min_cluster_size
    ), "should have escalated smoothing"
    assert ari_adaptive > ari_fixed + 0.3, (
        f"adaptive ({ari_adaptive:.3f}) should beat the fixed-min_samples base ({ari_fixed:.3f})")
    assert ari_adaptive > 0.5, f"adaptive should recover a real clustering, got {ari_adaptive:.3f}"
