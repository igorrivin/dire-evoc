#!/usr/bin/env python3
"""Quality benchmark: DiReEVoC (cluster the DiRe layout) vs vanilla EVoC (cluster the raw
embeddings), scored against ground-truth labels.

The claim under test is that running EVoC on a DiRe layout gives *better* clustering
quality than running EVoC directly on the high-dimensional embeddings -- and how
universal that is across datasets. We report Adjusted Rand Index and NMI against the
labels, the number of clusters found, the noise fraction, and wall-clock time, for both
methods on the same data.

Data sources:
    --npz PATH        a .npz with arrays X (n, d) and y (n,)
    --openml NAME     an OpenML dataset (fetched via sklearn), e.g. Fashion-MNIST
    --subsample N     cap the number of rows (stratified-ish random subsample)

Examples:
    conda run --no-capture-output -n rapids-26.04 python examples/benchmark_quality.py \
        --npz /tmp/ng20_emb.npz --label "20 Newsgroups (BGE-384)"
    conda run --no-capture-output -n rapids-26.04 python examples/benchmark_quality.py \
        --openml Fashion-MNIST --subsample 30000
"""
import argparse
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_npz(path):
    z = np.load(path)
    return z["X"].astype(np.float32), z["y"]


def load_openml(name, seed=0):
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name, version=1, as_frame=False)
    X = np.ascontiguousarray(ds.data, dtype=np.float32)
    y = ds.target
    # encode string/categorical labels to integers
    _, y = np.unique(y, return_inverse=True)
    return X, y


def subsample(X, y, n, seed=0):
    if n is None or n >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return np.ascontiguousarray(X[idx]), y[idx]


def score(name, labels, layers, y, dt):
    """Report the AUTO-selected clustering and the ORACLE (best-layer) clustering.

    Separating these answers two different questions: AUTO is the operational quality
    you get out of the box (substrate + layer selector together); ORACLE is the quality
    of the best layer the substrate produced (substrate quality alone, selector removed).
    A large AUTO<ORACLE gap means the layer SELECTOR is the bottleneck, not the embedding.
    """
    k = int(labels.max() + 1)
    noise = 100.0 * (labels < 0).mean()
    auto = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    # oracle: best ARI over the method's own layers
    aris = [adjusted_rand_score(y, lab) for lab in layers]
    best_i = int(np.argmax(aris))
    oracle = aris[best_i]
    oracle_k = int(layers[best_i].max() + 1)
    print(f"  {name:<12} auto: k={k:>3} noise={noise:4.1f}% ARI={auto:.4f} NMI={nmi:.4f}"
          f"  |  oracle(best layer): k={oracle_k:>3} ARI={oracle:.4f}   ({dt:.1f}s)")
    return auto, oracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz")
    p.add_argument("--openml")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--label", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.npz:
        X, y = load_npz(args.npz)
        label = args.label or args.npz
    elif args.openml:
        X, y = load_openml(args.openml, args.seed)
        label = args.label or f"OpenML:{args.openml}"
    else:
        p.error("provide --npz or --openml")

    X, y = subsample(X, y, args.subsample, args.seed)
    n_classes = len(np.unique(y))
    print(f"\n{label}: {X.shape[0]:,} x {X.shape[1]}  ({n_classes} classes)")

    # vanilla EVoC: EVoC's own node embedding + clustering
    from evoc import EVoC
    t0 = time.perf_counter()
    ev = EVoC(min_samples=args.min_samples, random_state=args.seed).fit(X)
    ev_auto, ev_oracle = score("vanilla EVoC", ev.labels_, ev.cluster_layers_, y,
                               time.perf_counter() - t0)

    # DiReEVoC: EVoC's clustering on a DiRe layout
    from dire_evoc import DiReEVoC
    t0 = time.perf_counter()
    de = DiReEVoC(min_samples=args.min_samples, random_state=args.seed).fit(X)
    de_auto, de_oracle = score("DiReEVoC", de.labels_, de.cluster_layers_, y,
                               time.perf_counter() - t0)

    # fixed-k = n_classes: pure substrate quality at the true granularity, with the
    # layer SELECTOR and peak-granularity confounds removed from BOTH sides. EVoC's
    # base_n_clusters extracts ~k from its own embedding; build_cluster_layers does the
    # same on the (already-computed) DiRe layout.
    # Compare the BASE layer (cluster_layers_[0]) on each side -- base_n_clusters sets the
    # base-layer granularity, but .labels_/.fit_predict return the *selected* layer, which
    # is a different k. Take [0] on both for an apples-to-apples fixed-k comparison.
    from evoc.clustering import build_cluster_layers
    ev_fk = EVoC(min_samples=args.min_samples, base_n_clusters=n_classes,
                 random_state=args.seed).fit(X).cluster_layers_[0]
    de_fk = build_cluster_layers(de.layout_, min_samples=args.min_samples,
                                 base_n_clusters=n_classes)[0][0]
    ev_fk_ari = adjusted_rand_score(y, ev_fk)
    de_fk_ari = adjusted_rand_score(y, de_fk)

    def verdict(d, e):
        return "DiRe WINS" if d > e + 0.01 else ("tie" if abs(d - e) <= 0.01 else "vanilla wins")
    print(f"  -> auto    ARI (DiRe {de_auto:.4f} - vanilla {ev_auto:.4f}) = {de_auto - ev_auto:+.4f}  [{verdict(de_auto, ev_auto)}]   (operational)")
    print(f"  -> oracle  ARI (DiRe {de_oracle:.4f} - vanilla {ev_oracle:.4f}) = {de_oracle - ev_oracle:+.4f}  [{verdict(de_oracle, ev_oracle)}]   (best produced layer)")
    print(f"  -> fixed-k ARI (DiRe {de_fk_ari:.4f} - vanilla {ev_fk_ari:.4f}) = {de_fk_ari - ev_fk_ari:+.4f}  [{verdict(de_fk_ari, ev_fk_ari)}]   (pure substrate @ k={n_classes})")


if __name__ == "__main__":
    main()
