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

## When It Helps

The strongest use case is medium-to-large embedding data where density-separated
structure is present and the DiRe layout is a better clustering substrate than the raw
vectors. The table below is from the checked CUDA Docker repro on an NVIDIA A10, using
the same dataset-specific EVoC `base_min_cluster_size` and `base_n_clusters` controls
for DiReEVoC and pure EVoC; see `docs/benchmarks.md` and the local
`artifacts/cuda-timing-repeats-default/` artifact tree. Timings are 20-run means with
the first run included; warm-only means are also recorded in the metrics.

![CUDA auto ARI comparison](docs/_static/benchmark_auto_ari.svg)

![CUDA auto ARI delta](docs/_static/benchmark_auto_delta.svg)

![CUDA fit runtime comparison](docs/_static/benchmark_fit_seconds.svg)

| Dataset | DiReEVoC auto ARI | Pure EVoC auto ARI | Notes |
| --- | ---: | ---: | --- |
| Synthetic blobs 12k | 0.968 | 1.000 | Matched `base_min_cluster_size=100` |
| Synthetic moons 12k | 1.000 | 1.000 | Matched `base_min_cluster_size=100`, `base_n_clusters=2` |
| Synthetic circles 12k | 1.000 | 1.000 | Matched `base_min_cluster_size=100`, `base_n_clusters=2` |
| MNIST | 0.720 | 0.813 | Pure EVoC wins in this 24k HF-backed run |
| sklearn digits | 0.261 | 0.305 | Small image data; fixed-k favors Pure EVoC |
| sklearn iris | 0.562 | 0.236 | DiReEVoC auto win; fixed-k is close |
| sklearn wine | 0.465 | 0.121 | DiReEVoC auto win in this tuned run |
| sklearn breast cancer | 0.404 | 0.116 | DiReEVoC win; fixed-k is stronger |
| Fashion-MNIST | 0.440 | 0.390 | DiReEVoC win; stronger at fixed-k |
| DBpedia-14 BGE-small | 0.607 | 0.623 | Pure EVoC slightly wins in this 39.2k run |
| Olivetti faces | 0.022 | 0.063 | Tiny data; both poor |
| Forest CoverType | 0.016 | -0.026 | DiReEVoC slight win; both poor |
| CIFAR-10 | 0.029 | 0.012 | 30k images, average-pooled to 16x16 RGB (`d=768`) |

Runtime from the same CUDA run:

| Dataset | DiReEVoC fit | Pure EVoC fit |
| --- | ---: | ---: |
| Synthetic blobs 12k | 3.43s | 0.73s |
| Synthetic moons 12k | 1.65s | 0.15s |
| Synthetic circles 12k | 1.67s | 0.15s |
| MNIST | 20.75s | 1.17s |
| sklearn digits | 3.08s | 0.08s |
| sklearn iris | 0.51s | 0.06s |
| sklearn wine | 0.57s | 0.06s |
| sklearn breast cancer | 0.96s | 0.06s |
| Fashion-MNIST | 23.33s | 0.63s |
| DBpedia-14 BGE-small | 15.83s | 0.65s |
| Olivetti faces | 0.89s | 0.20s |
| Forest CoverType | 16.21s | 0.48s |
| CIFAR-10 | 15.30s | 1.51s |

![CUDA fit runtime ratio](docs/_static/benchmark_fit_ratio.svg)

The honest boundary:

- This is not a universal clustering-quality win.
- DiRe is data-hungry; tiny datasets and weak feature representations are poor fits.
- Layer selection is still a real failure mode. The tuned run makes moons/circles
  feasible by setting the known two-cluster base layer for both methods, but MNIST,
  digits, Fashion-MNIST, DBpedia, Olivetti, and CoverType still expose selector and
  substrate limits.
- A hard partition is often the wrong object on continuous manifolds. In that case,
  inspect the layout and intrinsic dimension before trusting cluster labels.
- The GPU Phase-2 speedup is regime-dependent. It helps at large `n` and low CPU core
  count, but the CPU path can win on small data or many-core machines.

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

Generate a reproducible local benchmark/profile bundle with metrics and SVG figures:

```bash
python examples/benchmark_visuals.py --synthetic-blobs --n-samples 12000 \
  --test-fraction 0.2 --output-dir artifacts/benchmark-visuals
```

The visual benchmark example writes `metrics.json`, `metrics.csv`, `quality_ari.svg`,
`timing_breakdown.svg`, and `diagnostics.svg`. It reports DiReEVoC versus pure EVoC,
phase timings, intrinsic dimension from DiRe's existing kNN graph, selected-layer noise,
cluster counts, and optional hold-out prediction metrics.

Run the synthetic smoke suite, or run the broader default suite against the same pure
EVoC baseline:

```bash
python examples/benchmark_dataset_suite.py --list-datasets
python examples/benchmark_dataset_suite.py --datasets smoke \
  --output-dir artifacts/smoke-suite --timing-repeats 20
python examples/benchmark_dataset_suite.py --datasets default \
  --output-dir artifacts/dataset-suite --test-fraction 0.2 --timing-repeats 20
```

The `smoke` preset is only synthetic blobs, moons, and circles. The `default` suite keeps
those same three synthetic examples at the front of the larger reproducibility run, then
adds sklearn digits, iris, wine, breast cancer, MNIST, Fashion-MNIST, DBpedia-14
embeddings, Olivetti faces, Forest CoverType, and CIFAR-10. CIFAR-10 is average-pooled to
16x16 RGB features for this benchmark so it stays on the DiRe cuVS kNN path. Timing
summaries include the first run in `fit_seconds` and also write warm-only means such as
`fit_seconds_warm_mean`. OpenML is tried first by default. If OpenML fails, datasets that
are fully available in sklearn use the sklearn fallback first, as Forest CoverType does
through `fetch_covtype`; other mirrored OpenML datasets fall back to Hugging Face
(`ylecun/mnist`, `zalando-datasets/fashion_mnist`, or `uoft-cs/cifar10`).
OpenML-only datasets such as USPS, SVHN, UMIST faces, Letter,
OptDigits, PenDigits, SatImage, Segment, and Vehicle remain available by explicit slug.
Optional graph-feature datasets include Wiki-CS, Facebook Page-Page, and OGBN-Arxiv when
their extra loaders are installed.

Run the full CUDA Docker repro on a CUDA host and download the artifacts locally:

```bash
scripts/run_remote_cuda_repro.sh user@gpu-host
```

This syncs the working tree to the host, builds `dire-evoc:cuda128`, runs
`examples/run_cuda_repro.py` in Docker, and downloads `manifest.json`, logs, datasets,
`metrics.json`, `metrics.csv`, and SVG figures into `artifacts/cuda/`.

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
