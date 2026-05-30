#!/usr/bin/env python3
"""Minimal DiReEVoC quickstart: fit, intrinsic dimension, out-of-sample predict.

Requires the GPU stack (dire-rapids + cuml + evoc). Run, e.g.:
    conda run --no-capture-output -n rapids-26.04 python examples/quickstart.py
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from dire_evoc import DiReEVoC


def gaussian_blobs(n=12_000, k=8, d=32, spread=0.10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    per = n // k
    parts, labels = [], []
    for c in range(k):
        pts = centers[c] + rng.normal(scale=spread, size=(per, d)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        parts.append(pts)
        labels += [c] * per
    return np.ascontiguousarray(np.vstack(parts)), np.array(labels)


def main():
    X, y = gaussian_blobs()
    X_train, X_test = X[:10_000], X[10_000:]
    y_train, y_test = y[:10_000], y[10_000:]

    model = DiReEVoC(min_samples=10, random_state=0).fit(X_train)

    print(f"layout:               {model.layout_.shape}")
    print(f"clusters:             {model.labels_.max() + 1} "
          f"(noise {100 * (model.labels_ < 0).mean():.1f}%)")
    print(f"intrinsic dimension:  {model.intrinsic_dimension_:.2f}  (true ~{X.shape[1]})")
    print(f"train ARI vs labels:  {adjusted_rand_score(y_train, model.labels_):.4f}")

    pred = model.predict(X_test)
    print(f"held-out ARI:         {adjusted_rand_score(y_test, pred):.4f}")


if __name__ == "__main__":
    main()
