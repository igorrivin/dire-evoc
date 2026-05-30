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

# intrinsic dimension, reusing a fitted DiRe reducer's kNN graph (free):
dim = de.intrinsic_dimension_from_dire(reducer, method="twonn")

# ... or standalone from a point cloud:
dim = de.intrinsic_dimension(X, method="twonn")

# GPU single-linkage tree for EVoC's post-process:
from dire_evoc.hybrid import gpu_single_linkage_tree
slt = gpu_single_linkage_tree(layout, min_samples=10)
```

## Status

Alpha. `dimension` and `predict` are implemented and tested on CPU; `hybrid` provides the
validated GPU single-linkage-tree producer and feeds EVoC's post-process through the
upstream `build_cluster_layers(..., precomputed_linkage=...)` seam.

## Lineage / credit

`dire-evoc` (the integration idea) is by Igor Rivin. It builds on EVoC by Leland McInnes
et al. (Tutte Institute) and DiRe-RAPIDS by Alexander Kolpakov and Igor Rivin. The
out-of-sample `predict` is also offered upstream to EVoC.
