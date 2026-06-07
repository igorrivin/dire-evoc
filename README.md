# dire-evoc

`dire-evoc` connects three pieces that are useful together:

- [DiRe-RAPIDS](https://github.com/sashakolpakov/dire-rapids) builds a GPU DiRe layout
  from high-dimensional embeddings.
- [EVoC](https://github.com/TutteInstitute/evoc) extracts density-based cluster layers.
- `dire-evoc` adds the bridge code: a one-call pipeline, intrinsic-dimension estimation
  from DiRe's existing kNN graph, GPU single-linkage-tree handoff, and out-of-sample
  label assignment.

The package is deliberately thin. It does not reimplement DiRe or EVoC; it makes their
shared boundary explicit and keeps the heavy GPU dependencies behind lazy imports.

## What You Get

**One-call DiRe -> EVoC pipeline**

```python
from dire_evoc import DiReEVoC

model = DiReEVoC(min_samples=10, random_state=0).fit(X)

labels = model.labels_                  # selected EVoC layer, -1 means noise
layout = model.layout_                  # DiRe layout clustered by EVoC
dimension = model.intrinsic_dimension_  # TwoNN estimate from DiRe's kNN graph
new_labels = model.predict(X_new)       # kNN vote in the original embedding space
```

**Intrinsic dimension without a second neighbour search**

DiRe already builds a k-nearest-neighbour graph. `dire-evoc` can estimate intrinsic
dimension directly from that graph:

```python
from dire_evoc import intrinsic_dimension, intrinsic_dimension_from_dire

dim = intrinsic_dimension_from_dire(fitted_reducer)
dim_reference = intrinsic_dimension(X, method="twonn")
```

The estimators handle the common cosine-normalized case correctly by converting cosine
or chordal distances to angular/geodesic distances before applying ratio-based ID
estimators.

**GPU Phase-2 clustering bridge**

EVoC's expensive clustering phase is the HDBSCAN* single-linkage tree on mutual
reachability distances. cuML can build that tree on the GPU; EVoC can then run its own
condense, persistence, and layer extraction logic unchanged.

```python
from dire_evoc.hybrid import hybrid_cluster_layers

layers, memberships, persistence = hybrid_cluster_layers(layout, min_samples=10)
```

**Out-of-sample prediction**

New points are assigned by a distance-weighted kNN vote against the original training
embeddings. Noise is treated as a competing label.

```python
from dire_evoc import prepare_reference, knn_vote_predict

reference = prepare_reference(X_train)
query = prepare_reference(X_new)
labels, membership = knn_vote_predict(query, reference, train_labels, n_neighbors=15)
```

## Results: guarded graph-component selector

The selector branch adds an opt-in guarded graph-component selector
(`selector="auto"`). It keeps coverage-weighted silhouette as the fallback, then accepts
a stable small-k connected-component layer from the DiRe layout graph only when it matches
an explicit `base_n_clusters` control or passes fragmentation guards. The numbers below
are one matched, full-`n` sweep on an NVIDIA A10 (`artifacts/selector-auto-full/`,
`timing_policy=single_run`): for every dataset, DiReEVoC and pure EVoC are fit on the
*same data* with the same EVoC `base_min_cluster_size`/`base_n_clusters` controls, so the
deltas are strictly matched — no cross-`n` borrowing.

`ARI delta` is DiReEVoC guarded-auto minus pure EVoC auto. `Fit` is the full
`selector="auto"` DiReEVoC pipeline (DiRe layout + EVoC + layer post-processing); see the
timing discussion below for why that is not directly comparable to pure EVoC's seconds.

![Guarded auto selector ARI](docs/_static/benchmark_selector_auto_vs_pure.svg)

![Guarded auto selector ARI delta](docs/_static/benchmark_selector_auto_delta_vs_pure.svg)

| Dataset | n | DiReEVoC auto ARI | Pure EVoC auto ARI | ARI delta | DiReEVoC fit | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Synthetic blobs 30k | 30,000 | 1.000 | 1.000 | +0.000 | 21.1s | silhouette fallback |
| Synthetic circles 30k | 30,000 | 1.000 | 1.000 | +0.000 | 8.0s | graph layer accepted (matches `base_n_clusters=2`) |
| Synthetic moons 30k | 30,000 | 1.000 | 1.000 | +0.000 | 8.1s | graph layer accepted (matches `base_n_clusters=2`) |
| Separated spirals 30k | 30,000 | 1.000 | ≈0.00 | ≈+1.00 | 10.6s | graph layer accepted (matches `base_n_clusters=3`); pure EVoC zero-padded to 4-D to run, see below |
| MNIST full | 70,000 | 0.883 | 0.892 | −0.009 | 35.9s | silhouette fallback |
| Fashion-MNIST full | 70,000 | 0.384 | 0.327 | +0.057 | 35.0s | silhouette fallback |
| CIFAR-10 full | 60,000 | 0.029 | 0.009 | +0.019 | 25.2s | silhouette fallback |
| OptDigits full | 5,620 | 0.825 | 0.184 | +0.641 | 13.3s | silhouette fallback |
| PenDigits full | 10,992 | 0.717 | 0.155 | +0.562 | 21.4s | silhouette fallback; graph candidate rejected (`too_few_graph_clusters_without_base_n_clusters`) |
| SatImage full | 6,430 | 0.602 | 0.359 | +0.243 | 14.9s | silhouette fallback |
| Image Segmentation full | 2,310 | 0.327 | 0.094 | +0.234 | 8.1s | silhouette fallback |
| Vehicle full | 846 | 0.110 | 0.108 | +0.002 | 8.0s | silhouette fallback; graph candidate rejected (`too_few_graph_clusters_without_base_n_clusters`) |

What the guarded selector actually does in this sweep: it accepts the graph-component
layer only on circles/moons/spirals, where a stable small-`k` component structure matches
the supplied `base_n_clusters`. On PenDigits and Vehicle it forms a candidate but the guard
rejects it (`too_few_graph_clusters_without_base_n_clusters`); everywhere else no stable
large-component candidate forms and it falls back to coverage-weighted silhouette. So the
selector is conservative by design — most real-data wins here come from the DiRe layout
itself, not the graph override. Timings are single-run (`timing_repeats=1`), so expect
run-to-run variance on the GPU pipeline.

### The "Separated spirals" row, honestly

Earlier drafts reported the spirals pure-EVoC entry as `n/a — PCA initializer rejects
exact 2-D input`. That is misleading. EVoC's label-propagation initializer calls
`PCA(n_components=4)`, so scikit-learn raises `ValueError: n_components=4 must be between 0
and min(n_samples, n_features)` on *any* input with fewer than 4 features. It is a
dimensionality guard, not a verdict about clustering 2-D data.

The benchmark now zero-pads the same spirals to 4-D so the initializer runs, and pure EVoC
completes with the matched `base_n_clusters=3` scoring **ARI ≈ 0.00** (0.0003 in this run)
— raw-space density clustering cannot separate the interleaved spiral arms regardless of
dimension. DiReEVoC reaches ARI 1.000 because the DiRe layout untangles the arms before
EVoC sees them. So the real delta is ≈ **+1.00**, a genuine win — but it is a
*DiRe-embedding* effect on a 2-D toy, not an EVoC limitation. (The graph override is also
accepted here because the three arms form a clean small-`k` component structure matching
`base_n_clusters=3`, but silhouette on the same layout already recovers them.)

### Timing — DiRe also embeds, so the seconds are not apples-to-apples

Pure EVoC clusters the raw vectors directly; DiReEVoC first builds a DiRe layout and then
clusters that, so its fit time includes work pure EVoC never does. Per-phase timings from
this sweep make the split concrete (MNIST full, one fit):

| Phase | DiReEVoC | Pure EVoC |
| --- | ---: | ---: |
| DiRe layout (embedding) | 12.9s | — |
| EVoC clustering | 1.3s | 20.4s (whole fit) |
| Layer selection | 0.9s | — |
| Adaptive smoothing | 17.4s | — |
| Graph-component stage | 6.1s | — |
| **Total fit** | **38.8s** | **20.4s** |

Two things follow. First, the embedding is a real, irreducible cost that pure EVoC does
not pay (~13s here, ~33% of the fit) — but it is also what buys the quality on hard
manifolds and yields a reusable layout for visualization and out-of-sample prediction.
Second, on large data a large share of DiReEVoC's wall time is not the embedding at all
but the DiRe-specific layer post-processing (adaptive smoothing ≈ 17s here, plus a
~6s graph-component stage under `selector="auto"`). Pure EVoC's ~20s fits are the right
baseline to keep in mind: DiReEVoC is still the slower method,
and only earns its cost where the layout is a materially better substrate than the raw
vectors (e.g. OptDigits/PenDigits/Segment, where the matched delta is large).

### The honest boundary

- This is not a universal clustering-quality win. MNIST still favors pure EVoC, and
  blobs/circles/moons are ties at ≈1.0.
- DiRe is data-hungry; tiny datasets and weak feature representations are poor fits.
- The guarded graph override is conservative: in this sweep it fires only on the
  synthetic sets with an explicit `base_n_clusters`; on real data it almost always defers
  to silhouette. The real-data deltas are DiRe-layout effects, not graph-selector effects.
- A hard partition is often the wrong object on continuous manifolds. In that case,
  inspect the layout and intrinsic dimension before trusting cluster labels.
- DiReEVoC is the heavier method. Pure EVoC clusters raw vectors in ≈3s on these sets;
  DiReEVoC only repays its embedding + post-processing cost when the layout genuinely
  separates structure the raw space does not.

## Installation

Base install:

```bash
python -m pip install -e .
```

The base package supports the CPU-tested pieces: intrinsic-dimension utilities,
out-of-sample voting, and package imports.

Full GPU pipeline install:

```bash
python -m pip install -e '.[rapids,dire,cluster]'
```

Those extras pull in the packages needed by `DiReEVoC` and `dire_evoc.hybrid`. RAPIDS
packages are CUDA-version-specific, so many users will prefer to install them in a RAPIDS
conda environment first and then install this project editable.

Documentation build dependencies:

```bash
python -m pip install -e '.[docs]'
```

## Quick Start

Run the bundled smoke example in an environment with the GPU stack installed:

```bash
python examples/quickstart.py
```

Reproduce the guarded graph-component selector sweep (the results above):

```bash
python examples/benchmark_dataset_suite.py \
  --datasets blobs30k,circles30k,moons30k,spirals_clean30k,mnist,fashion_mnist,cifar10,optdigits,pendigits,satimage,segment,vehicle \
  --selector auto --output-dir artifacts/selector-auto-full \
  --prefer-hf --timing-repeats 1 --test-fraction 0.0 --full --skip-fixed-k --continue-on-error
```

The benchmark/tuning scripts under `examples/` (`benchmark_visuals.py`,
`benchmark_dataset_suite.py` with `--list-datasets`, `benchmark_cached_layout_tuning.py`)
cover the smaller suites, layout-dimension tuning, and dataset loaders; the full CUDA
Docker repro is `scripts/run_remote_cuda_repro.sh`. See `docs/benchmarks.md` for the
options and dataset notes. Rebuild the figures from local artifacts with
`python scripts/plot_benchmark_metrics.py`.

Minimal standalone intrinsic-dimension example:

```python
import numpy as np
from dire_evoc import intrinsic_dimension

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 12))

print(intrinsic_dimension(X, k=20, method="twonn"))
```

## Documentation

The Sphinx documentation lives in `docs/`.

```bash
python -m sphinx -b html docs docs/_build/html
```

Useful entry points:

- `docs/index.rst` - Sphinx documentation home.
- `docs/usage.md` - original long-form usage guide.
- `docs/benchmarks.md` - benchmark notes and results.
- `examples/benchmark_visuals.py` - reproducible benchmark metrics and SVG plots.
- `examples/` - quickstart and benchmark scripts.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The CPU suite covers intrinsic dimension, out-of-sample voting, and layer selection. The
end-to-end pipeline test is skipped unless `cuml`, `dire_rapids`, and `evoc` are
installed.

## Project Status

Alpha. The CPU-importable pieces are tested without the GPU stack. The CUDA path has a
Docker repro in this repository and was run end-to-end on an NVIDIA A10. The hybrid path
uses cuML for the single-linkage tree and EVoC's own tree utilities for condense,
persistence, and layer extraction; it also supports an upstream
`build_cluster_layers(..., precomputed_linkage=...)` seam if EVoC exposes one.

## Credit

`dire-evoc` is the integration layer by Igor Rivin. It builds on EVoC by Leland McInnes
et al. at the Tutte Institute and DiRe-RAPIDS by Alexander Kolpakov and Igor Rivin.
