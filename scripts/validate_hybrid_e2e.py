#!/usr/bin/env python3
"""End-to-end validation of the packaged dire-evoc hybrid on a REAL DiRe layout.

Earlier work validated the GPU single-linkage-tree bridge at the linkage level
(scripts/validate_gpu_slt.py in the evoc repo). This validates the *packaged* API
all the way through:

  DiRe.fit_transform -> layout
    -> evoc.build_cluster_layers(layout)                       (native CPU path)
    -> dire_evoc.hybrid.hybrid_cluster_layers(layout)          (cuML GPU SLT -> EVoC post)

and checks three things on each dataset:

  CORRECTNESS  fixed-k base-layer ARI(native, hybrid) -- extract exactly k clusters
               from each engine's tree via EVoC's binary search (removes the
               auto-peak-selection confound). The mathematician's bar: ~1.0.
  QUALITY      ground-truth ARI of both vs true labels at that k -- the hybrid must
               not degrade clustering quality.
  THROUGHPUT   wall-clock native vs hybrid (GPU SLT + CPU post-process), at the
               reported numba thread count.

Also exercises dire_evoc.intrinsic_dimension_from_dire on the fitted reducer -- the
"free" intrinsic-dimension read off DiRe's existing kNN graph.

Run: conda run --no-capture-output -n rapids-26.04 python -u scripts/validate_hybrid_e2e.py
"""
import os
import sys
import time

# RAPIDS before torch (shared-library ordering), per the dire-evoc env notes.
import cupy   # noqa: F401
import cuml   # noqa: F401
import cuvs   # noqa: F401
import numpy as np
import numba
from sklearn.metrics import adjusted_rand_score as ari

from dire_rapids import create_dire                                  # noqa: E402
from evoc.clustering import build_cluster_layers                    # noqa: E402
from evoc.clustering_utilities import _binary_search_for_n_clusters  # noqa: E402
from dire_evoc.hybrid import (  # noqa: E402
    _cluster_layers_from_linkage,
    gpu_single_linkage_tree,
    hybrid_cluster_layers,
)
from dire_evoc.dimension import intrinsic_dimension_from_dire        # noqa: E402

MIN_SAMPLES = 10
N_COMPONENTS = 8       # cluster an 8-D DiRe layout (matches earlier profiling work)


def make_blobs(n=60_000, k=12, d=50, spread=0.11, seed=0):
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


def load_mnist():
    z = np.load("/tmp/mnist_emb.npz")
    return z["X"].astype(np.float32), z["y"]


def fixed_k_labels(tree, k, n_samples):
    """Extract exactly ~k clusters from a single-linkage tree (EVoC binary search)."""
    _, clusters, _ = _binary_search_for_n_clusters(tree, k, n_samples=n_samples)
    return clusters


def dire_layout(X, seed=0):
    dire = create_dire(
        backend="cuvs", n_components=N_COMPONENTS, metric="cosine",
        init="pca", normalize=False, random_state=seed, verbose=False,
    )
    t0 = time.perf_counter()
    layout = np.ascontiguousarray(dire.fit_transform(X), dtype=np.float32)
    return layout, time.perf_counter() - t0, dire


def run(name, X, y, k_true):
    print(f"\n{'='*80}\n{name}: {X.shape[0]:,} x {X.shape[1]}  (true k={k_true})\n{'='*80}")
    layout, t_layout, dire = dire_layout(X)
    n = layout.shape[0]
    print(f"  DiRe layout: {layout.shape} in {t_layout:.1f}s  | numba threads={numba.get_num_threads()}")

    # free intrinsic dimension off DiRe's kNN graph (sanity: sensible, not doubled/halved)
    try:
        id_twonn = intrinsic_dimension_from_dire(dire, method="twonn")
        id_mle = intrinsic_dimension_from_dire(dire, method="mle", k=15)
        print(f"  intrinsic dim (free, from DiRe kNN): TwoNN={id_twonn:.2f}  MLE={id_mle:.2f}")
    except Exception as e:
        print(f"  intrinsic dim: skipped ({type(e).__name__}: {e})")

    # ---- CORRECTNESS + QUALITY: fixed-k base layer from each engine ----
    nat = build_cluster_layers(layout, min_samples=MIN_SAMPLES, base_n_clusters=k_true)
    hyb = hybrid_cluster_layers(layout, min_samples=MIN_SAMPLES, base_n_clusters=k_true)
    nat_k, hyb_k = nat[0][0], hyb[0][0]
    print(f"\n  fixed-k base layer (k={k_true}):")
    print(f"    native  clusters={nat_k.max()+1:>3}  noise={100*(nat_k<0).mean():4.1f}%  ARI_gt={ari(y, nat_k):.4f}")
    print(f"    hybrid  clusters={hyb_k.max()+1:>3}  noise={100*(hyb_k<0).mean():4.1f}%  ARI_gt={ari(y, hyb_k):.4f}")
    print(f"    ARI(native, hybrid) = {ari(nat_k, hyb_k):.4f}   <- correctness bar")

    # ---- THROUGHPUT: native CPU vs hybrid (GPU SLT + CPU post) ----
    t0 = time.perf_counter(); build_cluster_layers(layout, min_samples=MIN_SAMPLES); t_nat = time.perf_counter() - t0
    t0 = time.perf_counter(); slt = gpu_single_linkage_tree(layout, min_samples=MIN_SAMPLES); t_slt = time.perf_counter() - t0
    t0 = time.perf_counter(); _cluster_layers_from_linkage(slt, n); t_post = time.perf_counter() - t0
    t_hyb = t_slt + t_post
    speed = t_nat / t_hyb if t_hyb else float("nan")
    print(f"\n  throughput (auto layers):")
    print(f"    native (CPU MST+post)     {t_nat:6.2f}s")
    print(f"    hybrid GPU SLT            {t_slt:6.2f}s")
    print(f"    hybrid EVoC post-process  {t_post:6.2f}s")
    print(f"    hybrid total              {t_hyb:6.2f}s   -> {speed:.2f}x vs native")


def main():
    print(f"numba threads = {numba.get_num_threads()} (NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS','unset')})")
    Xb, yb = make_blobs()
    run("synthetic blobs (50-D)", Xb, yb, k_true=12)
    Xm, ym = load_mnist()
    run("MNIST-784", Xm, ym, k_true=10)


if __name__ == "__main__":
    main()
