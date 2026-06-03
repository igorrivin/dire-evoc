# Quality benchmarks: does DiRe->EVoC beat pure EVoC?

Reproduce with `examples/run_cuda_repro.py` or the SSH wrapper
`scripts/run_remote_cuda_repro.sh`. Each dataset is scored three ways:

- **auto** - operational under the declared run configuration: the layer the method
  actually selects.
- **oracle** - best ARI over layers the method produced; this removes selection error.
- **fixed-k** - the base layer forced to `k = n_classes`, isolating substrate quality.

"Pure EVoC" = `EVoC.fit_predict` on the original feature vectors. "DiReEVoC" = the same
EVoC clustering on a DiRe layout. ARI is against ground-truth labels.

## CUDA results

These numbers come from the CUDA Docker repro on an NVIDIA A10 using the Dockerfile in
this repository: CUDA 12.8 image, RAPIDS 26.2 Python packages. The run completed
successfully: quickstart, the 13-dataset default suite, corrected CIFAR-10 rerun,
summary regeneration, and artifact download all returned exit code 0. The local artifact
trees are `artifacts/cuda-timing-repeats-default/` and the merged
`artifacts/cuda-timing-repeats-combined/`.

This run uses matched EVoC controls for both methods. The dataset suite chooses
`base_min_cluster_size` case-by-case, and uses `base_n_clusters=2` for the synthetic
moons and circles controls where the feasible benchmark target is known. Each timed fit
is repeated 20 times. The primary timing fields are warm means with the first run
discarded; first-run and first-included means are also recorded.

Operational auto ARI vs ground truth:

| dataset | n train | d | cls | DiReEVoC auto | pure EVoC auto | delta | fixed-k substrate |
|---|---:|---:|---:|---:|---:|---:|---|
| Synthetic blobs 12k | 9,600 | 32 | 8 | 0.968 | 1.000 | -0.032 | DiRe 1.000 vs 1.000 |
| Synthetic moons 12k | 9,600 | 8 | 2 | 1.000 | 1.000 | +0.000 | DiRe 1.000 vs 1.000 |
| Synthetic circles 12k | 9,600 | 8 | 2 | 1.000 | 1.000 | +0.000 | DiRe 1.000 vs 1.000 |
| MNIST | 24,000 | 784 | 10 | 0.720 | 0.813 | -0.093 | DiRe 0.772 vs 0.848 |
| sklearn digits | 1,438 | 64 | 10 | 0.261 | 0.305 | -0.044 | DiRe 0.803 vs 0.873 |
| sklearn iris | 120 | 4 | 3 | 0.562 | 0.236 | +0.326 | DiRe 0.562 vs 0.582 |
| sklearn wine | 142 | 13 | 3 | 0.465 | 0.121 | +0.344 | DiRe 0.303 vs 0.257 |
| sklearn breast cancer | 455 | 30 | 2 | 0.404 | 0.116 | +0.288 | DiRe 0.669 vs 0.511 |
| Fashion-MNIST | 24,000 | 784 | 10 | 0.440 | 0.390 | +0.049 | DiRe 0.399 vs 0.273 |
| DBpedia-14 BGE-small | 39,200 | 384 | 14 | 0.607 | 0.623 | -0.016 | DiRe 0.560 vs 0.625 |
| Olivetti faces | 320 | 4,096 | 40 | 0.022 | 0.063 | -0.041 | DiRe 0.014 vs 0.056 |
| Forest CoverType | 40,000 | 54 | 7 | 0.016 | -0.026 | +0.042 | DiRe 0.006 vs -0.033 |
| CIFAR-10 | 24,000 | 768 | 10 | 0.029 | 0.012 | +0.017 | DiRe 0.004 vs 0.002 |

The code default is OpenML first, then fallback. The recorded remote artifact was rerun
with the declared Hugging Face mirror option for MNIST, Fashion-MNIST, and CIFAR-10 after
OpenML proved unreliable on that host. Forest CoverType used the sklearn `fetch_covtype`
fallback directly because it is fully available in sklearn. CIFAR-10 uses 30k images
average-pooled from 32x32 RGB to 16x16 RGB (`d=768`) so DiRe stays on the cuVS kNN path;
raw `d=3072` CIFAR was not used for the recorded benchmark.

Timing from the same CUDA run:

![CUDA fit runtime comparison](_static/benchmark_fit_seconds.svg)

| dataset | DiReEVoC warm fit | pure EVoC warm fit | DiRe mean all | pure mean all | intrinsic dim | matched base control |
|---|---:|---:|---:|---:|---:|---|
| Synthetic blobs 12k | 3.03s | 0.15s | 3.43s | 0.73s | 22.51 | min_samples=10, base_min_cluster_size=100 |
| Synthetic moons 12k | 1.43s | 0.15s | 1.65s | 0.15s | 2.03 | min_samples=10, base_min_cluster_size=100, base_n_clusters=2 |
| Synthetic circles 12k | 1.44s | 0.15s | 1.67s | 0.15s | 2.07 | min_samples=10, base_min_cluster_size=100, base_n_clusters=2 |
| MNIST | 20.37s | 0.59s | 20.75s | 1.17s | 13.86 | min_samples=10, base_min_cluster_size=100 |
| sklearn digits | 2.71s | 0.07s | 3.08s | 0.08s | 8.66 | min_samples=10, base_min_cluster_size=10 |
| sklearn iris | 0.20s | 0.06s | 0.51s | 0.06s | 2.92 | min_samples=10, base_min_cluster_size=5 |
| sklearn wine | 0.21s | 0.06s | 0.57s | 0.06s | 4.05 | min_samples=10, base_min_cluster_size=5 |
| sklearn breast cancer | 0.62s | 0.06s | 0.96s | 0.06s | 3.61 | min_samples=10, base_min_cluster_size=5 |
| Fashion-MNIST | 23.13s | 0.62s | 23.33s | 0.63s | 15.56 | min_samples=10, base_min_cluster_size=100 |
| DBpedia-14 BGE-small | 15.45s | 0.65s | 15.83s | 0.65s | 25.69 | min_samples=10, base_min_cluster_size=100 |
| Olivetti faces | 0.56s | 0.19s | 0.89s | 0.20s | 5.21 | min_samples=10, base_min_cluster_size=2 |
| Forest CoverType | 15.84s | 0.47s | 16.21s | 0.48s | 3.04 | min_samples=10, base_min_cluster_size=100 |
| CIFAR-10 | 14.87s | 0.60s | 15.30s | 1.51s | 26.39 | min_samples=10, base_min_cluster_size=100 |

![CUDA fit runtime ratio](_static/benchmark_fit_ratio.svg)

The smoke-only synthetic run and the default run are also merged into one bundle:
`artifacts/cuda-timing-repeats-combined/`.

![Combined smoke and default ARI](_static/benchmark_combined_auto_ari.svg)

![Combined smoke and default fit time](_static/benchmark_combined_fit_seconds.svg)

## Layer selection

`DiReEVoC` uses coverage-weighted silhouette by default: mean silhouette over all points,
with noise contributing 0. This penalizes layers that score well only because they throw
many points away as noise. The current CUDA run still shows that layer selection is not
solved in general: MNIST, digits, and several tiny or weak-feature datasets have much
better fixed-k substrate scores than auto-selected scores.

## Graph fragment merge

Some DiRe layouts preserve the right connected geometry but EVoC emits adjacent pieces as
separate density fragments. The pipeline now uses DiRe's original-space kNN graph to
merge selected-layer fragments only when reciprocal cross-neighbor evidence or noise
bridges support the merge. It rejects merges that collapse the selected layer to one
cluster and rejects severe coverage-weighted-silhouette losses.

In this run, moons and circles use the known two-cluster base target for both methods and
both reach `1.000` ARI. Breast cancer remains a DiReEVoC win (`0.404` vs pure EVoC's
`0.116`), though weaker than the previous fragment-merge-only run. Collapse-to-one
merges are rejected on MNIST, digits, Fashion-MNIST, DBpedia-14, Olivetti, and Forest
CoverType.

## Adaptive smoothing

The tight synthetic-blobs failure was a layer-generation failure: the DiRe layout had a
perfect fixed-k substrate, but EVoC's default smoothing could generate only degenerate
high-noise layers. The pipeline retries a small grid of larger `min_samples` and
`base_min_cluster_size` values when the base clustering is mostly noise, then re-selects
using the same coverage-weighted score.

In the broad CUDA run, blobs reach `0.968` auto ARI using `min_samples=10` and
`base_min_cluster_size=100` under the default holdout split; the smoke-only all-training
run records `0.975`. DiReEVoC can still retry larger internal smoothing values when its
initial selected layer is mostly noise; the selected values are recorded in
`selected_min_samples_` and `selected_base_min_cluster_size_`.

## Honest conclusion

This run supports a narrower claim than the old README:

- Wins are dataset-dependent. DiReEVoC wins on iris, wine, breast cancer, Fashion-MNIST,
  CoverType, and pooled CIFAR-10; ties on moons and circles under the matched feasible
  controls; and loses on blobs, MNIST, DBpedia-14, digits, and Olivetti in the auto
  metric.
- Fixed-k can disagree with auto. Several datasets have good fixed-k substrate scores but
  poor selected layers, so fixed-k and oracle numbers must not hide selector failures.
- Tiny data and weak image features are poor regimes. Olivetti, CIFAR-10, and CoverType
  are not strong quality wins in absolute ARI.
- DiRe layout construction dominates end-to-end time except where pure EVoC itself is
  slow.

## Compared with `master`

The `master` README was too broad. The CUDA repro now runs 13 datasets by default and
records all per-dataset figures plus warm-run timing fields. Fashion-MNIST remains a
DiReEVoC win (`0.440 vs 0.390`), moons and circles are feasible ties when both methods
use the known two-cluster base target, but DBpedia-14 still slightly favors pure EVoC in
this 39.2k run (`0.607 vs 0.623`). MNIST also favors pure EVoC (`0.720 vs 0.813`).
Historical 20 Newsgroups and arXiv claims are not part of the default repro unless their
loaders are enabled explicitly.

## Reproduce

On a CUDA Docker host reachable by SSH:

```bash
scripts/run_remote_cuda_repro.sh user@gpu-host
```

This syncs the working tree to the host, builds the Docker image, runs
`examples/run_cuda_repro.py`, and downloads all JSON, CSV, log, dataset, and SVG
artifacts to `artifacts/cuda/`.

The dataset suite can also be run directly. Use `smoke` for only synthetic blobs, moons,
and circles, and use `default` for the larger reproducibility suite that includes those
same three synthetic datasets plus MNIST, Fashion-MNIST, DBpedia-14, CoverType, CIFAR-10,
and the smaller sklearn/Olivetti datasets:

```bash
python examples/benchmark_dataset_suite.py --datasets smoke \
  --output-dir artifacts/smoke-suite --timing-repeats 20
python examples/benchmark_dataset_suite.py --datasets default \
  --output-dir artifacts/dataset-suite --test-fraction 0.2 --timing-repeats 20
```

With repeated timings, `fit_seconds` is the warm mean with the first run discarded by
default (`--timing-primary warm_mean`). `fit_seconds_first`,
`fit_seconds_mean_all`, and `fit_seconds_warm_mean` are recorded separately. Rebuild the
README/Sphinx figures from existing artifacts with:

```bash
python scripts/plot_benchmark_metrics.py
```

The default DiRe layout dimension is `n_components=8`. To test whether `4` is good
enough, and to avoid rebuilding DiRe while tuning EVoC-side controls, run the
cached-layout tuner on a prepared `.npz` dataset:

```bash
python examples/benchmark_cached_layout_tuning.py \
  --npz artifacts/cuda/datasets/fashion_mnist.npz \
  --label Fashion-MNIST \
  --output-dir artifacts/cached-layout/fashion_mnist \
  --n-components-grid 4,8 \
  --base-min-cluster-size-grid 25,50,100 \
  --base-n-clusters-grid none,classes
```

It writes `metrics.json` and `trials.csv`; use `dire_layout_seconds` for the one-time
layout cost, `cached_trial_seconds` for each EVoC trial on that layout, and
`full_once_seconds` for the cost of one layout plus one selected trial.

For the scale regime, do not repeat each dataset 20 times. The small-data benchmark is
expected to favor pure EVoC because pure EVoC has no DiRe layout overhead. The larger
question is whether pure EVoC's neighbour/MST cost and raw-space density quality degrade
on larger corpora. Use the one-run `scale` preset:

```bash
python examples/benchmark_dataset_suite.py \
  --datasets scale \
  --output-dir artifacts/scale-suite \
  --prefer-hf \
  --timing-repeats 1 \
  --test-fraction 0.0 \
  --continue-on-error
```

The preset uses 30k synthetic blobs/moons/circles, full MNIST and Fashion-MNIST when
OpenML or the Hugging Face train+test mirror is available, full CIFAR-10
(`60k`, not `100k`), full DBpedia-14 embeddings when materialized with `--full`, full
Forest CoverType, full Olivetti faces, full Facebook Page-Page, and full OGBN-Arxiv.
