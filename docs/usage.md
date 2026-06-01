# dire-evoc usage

`dire-evoc` layers three capabilities on top of a DiRe-RAPIDS layout and EVoC's density
clustering. Each is usable on its own; `DiReEVoC` runs the whole thing in one call.

## Install

```bash
pip install -e .                                   # CPU cores (dimension, predict)
pip install -e '.[rapids,dire,cluster]'            # + GPU SLT, DiRe reducer, EVoC layers
```

The pure-numpy cores import on any machine. `cuml`, `cupy`, `dire-rapids`, and `evoc` are
imported lazily inside the functions that need them, so a CPU-only install of the base
package still works for intrinsic-dimension and predict on precomputed neighbour data.

## 1. One-call pipeline — `DiReEVoC`

```python
from dire_evoc import DiReEVoC

model = DiReEVoC(min_samples=10, random_state=0).fit(X)   # X: (n, d) embeddings
model.labels_                 # operative-layer cluster labels (-1 = noise)
model.membership_             # per-point membership strength in (0, 1]
model.layout_                 # the DiRe layout EVoC clustered (n, n_components)
model.intrinsic_dimension_    # TwoNN ID, read off DiRe's kNN graph for free
model.cluster_layers_         # all EVoC layers, fine -> coarse
model.selected_layer_         # which layer silhouette picked for labels_

labels_new = model.predict(X_new)                          # out-of-sample assignment
```

Key parameters: `n_components` (layout dim, default 8), `min_samples` (HDBSCAN* core
distance), `selector` (`"silhouette"` default — the validated layer selector — or
`"base"` for the finest layer), `adaptive_min_samples` (retry density smoothing when a
tight layout produces mostly noise), `merge_fragments` (merge adjacent selected-layer
fragments using DiRe's original-space kNN graph), and `use_gpu` (`"auto"`). Extra
keywords are forwarded to `dire_rapids.create_dire` (e.g. `n_neighbors`,
`max_iter_layout`).

> **When is the partition the right object?** Density clustering assumes density-separated
> structure. On continuous / high-noise manifolds a hard partition is the wrong object —
> the DiRe layout itself (`model.layout_`) is the deliverable. Inspect
> `model.intrinsic_dimension_` and the noise fraction (`(labels_ == -1).mean()`) before
> trusting `labels_`.

## 2. Intrinsic dimension — free off the kNN graph

```python
from dire_evoc import intrinsic_dimension, intrinsic_dimension_from_dire

intrinsic_dimension(X, method="twonn")               # standalone (computes its own kNN)
intrinsic_dimension_from_dire(reducer, method="twonn")  # reuse a fitted DiRe reducer's graph
```

TwoNN (Facco et al. 2017) is the robust default (uses only the 1st/2nd neighbour, so it
is insensitive to approximate-kNN error); MLE (Levina–Bickel 2004) is a second opinion.

> **Metric.** On the unit sphere (any cosine-normalised embedding) the geodesic is the
> angle θ, but cosine *distance* `1 − cos θ ≈ θ²/2` is the squared radius — feeding it raw
> **halves** the reported dimension. The estimators take a `metric` argument
> (`"cosine"`/`"chordal"`/`"angular"`/`"euclidean"`) and convert to angles first;
> `intrinsic_dimension_from_dire` reads the reducer's metric automatically.

## 3. GPU Phase-2 clustering — the hybrid

```python
from dire_evoc.hybrid import gpu_single_linkage_tree, hybrid_cluster_layers

slt = gpu_single_linkage_tree(layout, min_samples=10)     # cuML SLT in EVoC linkage format
layers, memberships, persistence = hybrid_cluster_layers(layout, min_samples=10)
```

`hybrid_cluster_layers` builds the single-linkage tree on the GPU and feeds it to EVoC's
condense/persistence/layer post-process. If EVoC exposes
`build_cluster_layers(..., precomputed_linkage=slt)`, that seam is used; otherwise
`dire-evoc` calls EVoC's tree utilities directly. On the CUDA benchmark run, the
EVoC-on-layout phase is usually small relative to DiRe layout construction, though the
full 13-dataset repro shows that timings are strongly dataset- and backend-dependent.

## 4. Out-of-sample assignment — `predict`

New points are labelled by a distance-weighted kNN vote over the training points in the
original embedding space (noise competes as a label), sidestepping re-embedding into the
transductive layout. Available as `DiReEVoC.predict` or directly:

```python
from dire_evoc import prepare_reference, knn_vote_predict
ref = prepare_reference(X_train)
labels, membership = knn_vote_predict(prepare_reference(X_new), ref, train_labels, n_neighbors=15)
```

## 5. Benchmark suite

```bash
python examples/benchmark_dataset_suite.py --list-datasets
python examples/benchmark_dataset_suite.py --datasets default \
  --output-dir artifacts/dataset-suite --test-fraction 0.2
```

The default suite compares DiReEVoC against pure EVoC on synthetic blobs, moons, circles,
sklearn digits/iris/wine/breast-cancer/Olivetti, MNIST, Fashion-MNIST, DBpedia-14,
Forest CoverType, and CIFAR-10. OpenML is tried first by default; sklearn-native OpenML
datasets use sklearn as fallback, and other mirrored OpenML datasets use Hugging Face as
fallback.
