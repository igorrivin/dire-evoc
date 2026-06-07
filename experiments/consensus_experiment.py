#!/usr/bin/env python3
"""Sparse-consensus voting vs single-run vs best-of-silhouette on small volatile datasets.

For each dataset we run N independently seeded DiReEVoC fits, collecting each run's
operative labels, the DiRe kNN graph, the internal coverage-weighted silhouette, and the
ground-truth ARI. From those we compare three regimes on ground-truth ARI:

  - single-run        : mean/std/min/max over the N seeds (the variance problem)
  - best-of-silhouette: ARI of the run with the highest internal silhouette (#13)
  - consensus(vote)   : sparse co-association over each run's kNN graph -> connected
                        components (this module), swept over agreement thresholds

No pipeline changes: DiReEVoC is used as-is; voting lives entirely in dire_evoc.consensus.
"""
import argparse
import json
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

import sys
sys.path.insert(0, "examples")
from benchmark_visuals import load_openml  # noqa: E402
from dire_evoc import DiReEVoC  # noqa: E402
from dire_evoc.selectors import coverage_silhouette_score  # noqa: E402
from dire_evoc.consensus import consensus_labels  # noqa: E402

DATASETS = [("segment", 5), ("vehicle", 5), ("satimage", 5)]
COMMON = dict(selector="auto", metric="cosine", n_components=8, min_samples=10,
              n_neighbors=15, max_iter_layout=100)
THRESHOLDS = [0.4, 0.5, 0.6, 0.75]


def one_fit(X, y, seed, bmc):
    m = DiReEVoC(random_state=seed, base_min_cluster_size=bmc, **COMMON).fit(X)
    knn = np.asarray(getattr(m.reducer_, "_knn_indices"))
    return {
        "labels": np.asarray(m.labels_),
        "knn": knn,
        "silhouette": float(coverage_silhouette_score(m.layout_, m.labels_)),
        "ari": float(ari(y, m.labels_)),
    }


def run(slug, bmc, n):
    X, y = load_openml(slug)
    X = np.ascontiguousarray(X, dtype=np.float32)
    t0 = time.perf_counter()
    runs = [one_fit(X, y, s, bmc) for s in range(n)]
    fit_secs = time.perf_counter() - t0

    aris = np.array([r["ari"] for r in runs])
    sils = np.array([r["silhouette"] for r in runs])
    best = int(np.argmax(np.where(np.isfinite(sils), sils, -np.inf)))
    labelings = [r["labels"] for r in runs]
    knns = [r["knn"] for r in runs]

    consensus = {}
    for thr in THRESHOLDS:
        lab = consensus_labels(labelings, knns, threshold=thr, min_cluster_size=2)
        consensus[str(thr)] = {
            "ari": float(ari(y, lab)),
            "n_clusters": int(np.unique(lab[lab >= 0]).size),
            "noise_fraction": float(np.mean(lab < 0)),
        }
    best_thr = max(consensus, key=lambda t: consensus[t]["ari"])

    out = {
        "dataset": slug, "n": int(X.shape[0]), "restarts": n, "fit_seconds_total": fit_secs,
        "single_run": {"ari_mean": float(aris.mean()), "ari_std": float(aris.std()),
                       "ari_min": float(aris.min()), "ari_max": float(aris.max())},
        "best_of_silhouette": {"seed": best, "ari": float(aris[best])},
        "consensus": consensus, "consensus_best_threshold": best_thr,
    }
    sr, bo, co = out["single_run"], out["best_of_silhouette"], consensus[best_thr]
    print(f"\n=== {slug} (n={X.shape[0]}, restarts={n}) ===")
    print(f"  single-run ARI : mean={sr['ari_mean']:.4f} std={sr['ari_std']:.4f} "
          f"[{sr['ari_min']:.4f}, {sr['ari_max']:.4f}]")
    print(f"  best-silhouette: ARI={bo['ari']:.4f} (seed {bo['seed']})")
    print(f"  consensus vote : ARI={co['ari']:.4f} @thr={best_thr} "
          f"(k={co['n_clusters']}, noise={co['noise_fraction']:.2f})")
    print("    by threshold: " + ", ".join(
        f"{t}->{consensus[t]['ari']:.3f}(k{consensus[t]['n_clusters']})" for t in consensus))
    print(f"  consensus - single_mean = {co['ari'] - sr['ari_mean']:+.4f} | "
          f"consensus - best_sil = {co['ari'] - bo['ari']:+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--out", default="artifacts/consensus-experiment/results.json")
    args = ap.parse_args()
    results = [run(slug, bmc, args.restarts) for slug, bmc in DATASETS]
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
