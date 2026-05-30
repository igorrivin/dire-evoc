"""Out-of-sample kNN-vote predictor: self-consistency and held-out generalisation."""
import numpy as np

from dire_evoc.predict import prepare_reference, knn_vote_predict


def _blobs(n, k, d, spread, seed):
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


def test_vote_self_consistency():
    X, y = _blobs(4000, 8, 32, 0.08, 0)
    ref = prepare_reference(X)
    labels, mem = knn_vote_predict(ref, ref, y, n_neighbors=15)
    # predicting the reference on itself reproduces its labels almost exactly
    agree = (labels == y).mean()
    assert agree > 0.99
    assert mem.min() > 0.0 and mem.max() <= 1.0 + 1e-9


def test_vote_holdout():
    X, y = _blobs(5000, 6, 24, 0.10, 1)
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    ref = prepare_reference(X[tr])
    q = prepare_reference(X[te])
    labels, _ = knn_vote_predict(q, ref, y[tr], n_neighbors=15)
    assert (labels == y[te]).mean() > 0.9


def test_noise_competes_as_label():
    # a query nearest to noise-labelled training points is itself labelled noise
    X, y = _blobs(2000, 4, 16, 0.05, 3)
    y[:200] = -1                              # mark a chunk as noise
    ref = prepare_reference(X)
    labels, _ = knn_vote_predict(ref[:50], ref, y, n_neighbors=10)
    assert (labels[:50] == -1).mean() > 0.8
