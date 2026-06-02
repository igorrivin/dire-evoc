#!/usr/bin/env python3
"""Quality-vs-epochs knee for the DiRe layout in the DiRe->EVoC pipeline.

Diagnosis (measured 2026-06-01, GH200 / rapids-26.04): at N <= 100k the pipeline is
*layout-bound*, not MST-bound -- DiRe's layout costs ~2-3s while EVoC's whole Phase 2
(KD-tree + Boruvka MST + condense) costs ~0.1s. So the high-EV speed lever is DiRe's
optimization loop length (`max_iter_layout`, default 128, linear LR anneal
alpha = 1 - i/max_iter), NOT the MST. evoc's own internal embedding uses only ~50
epochs to 4-D; DiRe runs 128 to 8-D, and the late, low-alpha epochs likely only nudge
positions that no longer change cluster assignments.

This sweeps `max_iter_layout` and, for each value, reports:
  - DiRe layout wall-clock (the thing we're trying to cut),
  - ARI vs ground truth at fixed k (does quality hold?),
  - ARI vs the FULL-epoch (reference) layout's clustering at the same seed (when do
    cluster assignments stop moving, independent of ground truth -- the real knee).

The smallest epoch count whose ref-ARI is ~1.0 and whose gt-ARI matches full is the
near-free speedup. Mirrors the DiRe path in scripts/validate_hybrid_e2e.py exactly
(backend=cuvs, n_components=8, metric=cosine, init=pca, normalize=False).

Run: conda run --no-capture-output -n rapids-26.04 python -u scripts/sweep_epochs.py
"""
import argparse
import sys
import time

import cupy   # noqa: F401  (RAPIDS before torch: shared-library ordering)
import cuml   # noqa: F401
import cuvs   # noqa: F401
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, "/home/igor/devel/dire-rapids")
from dire_rapids import create_dire                                  # noqa: E402
from evoc.clustering import build_cluster_layers                    # noqa: E402

N_COMPONENTS = 8
MIN_SAMPLES = 10
DEFAULT_ITERS = [128, 96, 64, 48, 32, 24, 16, 12, 8, 4]


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


def load_npz(path):
    z = np.load(path)
    return z["X"].astype(np.float32), z["y"]


def dire_layout(X, max_iter, seed):
    reducer = create_dire(
        backend="cuvs", n_components=N_COMPONENTS, metric="cosine",
        init="pca", normalize=False, random_state=seed, verbose=False,
        max_iter_layout=max_iter,
    )
    t0 = time.perf_counter()
    layout = np.ascontiguousarray(reducer.fit_transform(X), dtype=np.float32)
    return layout, time.perf_counter() - t0


def cluster_fixed_k(layout, k_true):
    """Base layer at fixed k via EVoC (removes the auto-peak-selection confound)."""
    layers = build_cluster_layers(layout, min_samples=MIN_SAMPLES, base_n_clusters=k_true)
    return layers[0][0]


def run(name, X, y, k_true, iters, seeds):
    print(f"\n{'='*84}\n{name}: {X.shape[0]:,} x {X.shape[1]}  (true k={k_true})  "
          f"seeds={list(seeds)}\n{'='*84}")
    ref_iter = max(iters)
    # reference clustering (full epochs) per seed
    ref_labels = {}
    for s in seeds:
        lay, _ = dire_layout(X, ref_iter, s)
        ref_labels[s] = cluster_fixed_k(lay, k_true)

    print(f"{'max_iter':>9} {'layout_s':>9} {'gt_ARI':>16} {'ref_ARI':>16} {'speedup':>8}")
    base_time = None
    for it in iters:
        times, gt, ref = [], [], []
        for s in seeds:
            lay, t = dire_layout(X, it, s)
            lab = cluster_fixed_k(lay, k_true)
            times.append(t)
            gt.append(ari(y, lab))
            ref.append(ari(ref_labels[s], lab))
        tmean = float(np.mean(times))
        if it == ref_iter:
            base_time = tmean
        spd = (base_time / tmean) if base_time else float("nan")
        print(f"{it:>9} {tmean:>8.2f}s "
              f"{np.mean(gt):>7.4f}+-{np.std(gt):.4f} "
              f"{np.mean(ref):>7.4f}+-{np.std(ref):.4f} "
              f"{spd:>7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, nargs="+", default=DEFAULT_ITERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mnist", default="/tmp/mnist_emb.npz")
    ap.add_argument("--extra-npz", default=None, help="extra labeled dataset")
    ap.add_argument("--extra-label", default="extra dataset")
    ap.add_argument("--extra-k", type=int, default=None)
    args = ap.parse_args()
    iters = sorted(set(args.iters), reverse=True)

    # warm cuVS + numba JIT so compilation isn't charged to the reference timing
    Xw, _ = make_blobs(n=4000, seed=99)
    lw, _ = dire_layout(Xw, 8, 0)
    cluster_fixed_k(lw, 12)

    Xb, yb = make_blobs()
    run("synthetic blobs (50-D)", Xb, yb, k_true=12, iters=iters, seeds=args.seeds)

    try:
        Xm, ym = load_npz(args.mnist)
        run("MNIST-784", Xm, ym, k_true=10, iters=iters, seeds=args.seeds)
    except FileNotFoundError:
        print(f"\n[skip] MNIST not found at {args.mnist}")

    if args.extra_npz:
        Xe, ye = load_npz(args.extra_npz)
        k = args.extra_k or int(len(np.unique(ye)))
        run(args.extra_label, Xe, ye, k_true=k, iters=iters, seeds=args.seeds)


if __name__ == "__main__":
    main()
