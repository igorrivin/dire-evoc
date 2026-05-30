# dire-evoc

GPU-accelerated density clustering on [DiRe-RAPIDS](https://github.com/sashakolpakov/dire-rapids)
layouts, using [EVoC](https://github.com/TutteInstitute/evoc)'s multi-layer machinery.

`dire-evoc` is a thin integration layer over two upstream primitives — DiRe-RAPIDS
(GPU dimensionality reduction) and EVoC (density clustering) — adding the capabilities
that only make sense at their intersection. The DiRe layout is the common substrate; on
top of it this package provides:

- **Intrinsic dimension, essentially free** (`dire_evoc.dimension`). The DiRe pipeline
  already builds a kNN graph on the original high-dimensional embeddings; the intrinsic
  dimension is a pure function of that graph's neighbour distances. TwoNN (Facco et al.,
  *Sci. Rep.* 2017) and MLE (Levina–Bickel, NIPS 2004), computed directly on
  `reducer._knn_distances` — no second neighbour search, no `skdim` dependency.

  > **Metric matters.** These estimators are ratio-based, so the distance must be the
  > manifold geodesic. On the unit sphere (any cosine-normalised embedding) the geodesic
  > is the angle θ, while cosine *distance* `1 − cos θ ≈ θ²/2` is the **square** of that
  > radius — and since the estimator reads dimension off volume `~ r^d`, squaring the
  > radius **halves** the reported dimension. The estimators take a `metric` argument and
  > convert to angles first. Validated: on arXiv-BGE, exact-kNN and DiRe's approximate
  > cuVS kNN give the same ID to < 0.6 % (and identically on MNIST), so reusing DiRe's
  > approximate graph costs nothing.

- **GPU Phase-2 clustering** (`dire_evoc.hybrid`). EVoC spends ~94 % of its clustering
  time (at 1M points) building the single-linkage tree — which is exactly HDBSCAN*'s SLT,
  and cuML builds that on the GPU. `gpu_single_linkage_tree()` returns it in EVoC's
  linkage format; EVoC's cheap O(n)-tree post-process (condense / persistence / layers)
  runs unchanged. Measured ≈ 3.3× on Phase 2 at 1M points with identical clustering
  (ARI = 1.0 at the meaningful granularity).

- **Out-of-sample assignment** (`dire_evoc.predict`). Label new points by a
  distance-weighted kNN vote over the training points in the original embedding space —
  sidestepping re-embedding into the transductive layout. Noise competes as a label.

## Install

```bash
python -m pip install -e .                 # pure-numpy cores (dimension, predict)
python -m pip install -e '.[rapids,dire,cluster]'   # + GPU SLT, DiRe reducer, EVoC layers
```

The pure-numpy estimators import and test on any machine; cuml / cupy / dire-rapids /
evoc are imported lazily only by the functions that need them.

## Quick start

```python
import dire_evoc as de

# one call: DiRe layout -> EVoC clustering (GPU SLT when available) -> labels,
# intrinsic dimension, and out-of-sample predict.
model = de.DiReEVoC(min_samples=10).fit(X)
labels = model.labels_                 # operative-layer clustering (-1 = noise)
dim    = model.intrinsic_dimension_    # TwoNN, free off DiRe's kNN graph
new    = model.predict(X_new)          # kNN vote in the original space

# ... or use the pieces directly:
dim = de.intrinsic_dimension_from_dire(reducer, method="twonn")   # from a fitted reducer
dim = de.intrinsic_dimension(X, method="twonn")                   # standalone point cloud

from dire_evoc.hybrid import gpu_single_linkage_tree
slt = gpu_single_linkage_tree(layout, min_samples=10)             # GPU SLT for EVoC
```

## Status

Alpha, validated end to end. `dimension` and `predict` are implemented and tested on CPU
(13 tests). `hybrid` provides the GPU single-linkage-tree producer and feeds EVoC's
post-process through the `build_cluster_layers(..., precomputed_linkage=...)` seam (live
upstream). `DiReEVoC` (`pipeline.py`) ties it together. End-to-end validation
(`scripts/validate_hybrid_e2e.py`): the packaged hybrid reproduces native EVoC exactly
(ARI = 1.0 at fixed k on synthetic / MNIST / 1M points) and is **2.7× faster at 1M points
on 8 threads** (the win regime — at small n / many cores the CPU baseline is faster).

## Documentation

- [`docs/usage.md`](docs/usage.md) — full usage guide for all four capabilities.
- [`docs/benchmarks.md`](docs/benchmarks.md) — quality vs vanilla EVoC (honest: the win is
  throughput + intrinsic dimension; clustering quality is currently bottlenecked by layer
  selection, not the embedding).
- [`examples/`](examples/) — runnable `quickstart.py` and `benchmark_quality.py`.

## Lineage / credit

`dire-evoc` (the integration idea) is by Igor Rivin. It builds on EVoC by Leland McInnes
et al. (Tutte Institute) and DiRe-RAPIDS by Alexander Kolpakov and Igor Rivin. The
out-of-sample `predict` is also offered upstream to EVoC.
