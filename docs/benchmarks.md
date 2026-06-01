# Quality benchmarks: does DiRe->EVoC beat pure EVoC?

Reproduce with `examples/run_cuda_repro.py` or the SSH wrapper
`scripts/run_remote_cuda_repro.sh`. Each dataset is scored three ways:

- **auto** - operational, out-of-the-box: the layer the method actually selects.
- **oracle** - best ARI over layers the method produced; this removes selection error.
- **fixed-k** - the base layer forced to `k = n_classes`, isolating substrate quality.

"Pure EVoC" = `EVoC.fit_predict` on the original feature vectors. "DiReEVoC" = the same
EVoC clustering on a DiRe layout. ARI is against ground-truth labels.

## CUDA results

These numbers come from `scripts/run_remote_cuda_repro.sh` on an NVIDIA A10 using the
Dockerfile in this repository: CUDA 12.8 image, RAPIDS 26.2 Python packages. The run
completed successfully: quickstart, the 13-dataset default suite, artifact generation,
and artifact download all returned exit code 0. The local artifact tree is
`artifacts/cuda-broad-163-192-33-223-fragment-merge2/`.

Operational auto ARI vs ground truth:

| dataset | n train | d | cls | DiReEVoC auto | pure EVoC auto | delta | fixed-k substrate |
|---|---:|---:|---:|---:|---:|---:|---|
| Synthetic blobs 12k | 9,600 | 32 | 8 | 1.000 | 1.000 | +0.000 | DiRe 1.000 vs 1.000 |
| Synthetic moons 12k | 9,600 | 8 | 2 | 0.524 | 0.350 | +0.174 | DiRe 1.000 vs 1.000 |
| Synthetic circles 12k | 9,600 | 8 | 2 | 0.279 | 0.037 | +0.241 | DiRe 1.000 vs 1.000 |
| MNIST | 24,000 | 784 | 10 | 0.656 | 0.813 | -0.157 | DiRe 0.772 vs 0.848 |
| sklearn digits | 1,438 | 64 | 10 | 0.261 | 0.282 | -0.021 | DiRe 0.805 vs 0.873 |
| sklearn iris | 120 | 4 | 3 | 0.588 | 0.236 | +0.352 | DiRe 0.562 vs 0.582 |
| sklearn wine | 142 | 13 | 3 | 0.000 | 0.121 | -0.121 | DiRe 0.411 vs 0.257 |
| sklearn breast cancer | 455 | 30 | 2 | 0.630 | 0.116 | +0.514 | DiRe 0.669 vs 0.511 |
| Fashion-MNIST | 24,000 | 784 | 10 | 0.440 | 0.390 | +0.049 | DiRe 0.398 vs 0.273 |
| DBpedia-14 BGE-small | 39,200 | 384 | 14 | 0.608 | 0.623 | -0.015 | DiRe 0.560 vs 0.625 |
| Olivetti faces | 320 | 4,096 | 40 | 0.050 | 0.063 | -0.014 | DiRe 0.014 vs 0.056 |
| Forest CoverType | 40,000 | 54 | 7 | 0.016 | -0.006 | +0.022 | DiRe 0.006 vs -0.033 |
| CIFAR-10 | 1,200 | 3,072 | 10 | 0.030 | 0.023 | +0.006 | DiRe 0.010 vs 0.008 |

The code default is OpenML first, then fallback. The recorded remote artifact was rerun
with the declared Hugging Face mirror option for MNIST, Fashion-MNIST, and CIFAR-10 after
OpenML proved unreliable on that host. Forest CoverType used the sklearn `fetch_covtype`
fallback directly because it is fully available in sklearn. CIFAR-10 is intentionally
capped because raw 3072-dimensional pixel kNN is memory-heavy on an A10.

Timing from the same CUDA run:

![CUDA fit runtime comparison](_static/benchmark_fit_seconds.svg)

| dataset | DiReEVoC fit | pure EVoC fit | intrinsic dim | adaptive smoothing |
|---|---:|---:|---:|---|
| Synthetic blobs 12k | 18.36s | 11.79s | 22.51 | min_samples=10, base_min_cluster_size=100 |
| Synthetic moons 12k | 11.71s | 18.22s | 2.03 | min_samples=10, base_min_cluster_size=10 |
| Synthetic circles 12k | 8.42s | 0.28s | 2.07 | min_samples=10, base_min_cluster_size=10 |
| MNIST | 26.85s | 0.84s | 13.86 | min_samples=10, base_min_cluster_size=10 |
| sklearn digits | 10.06s | 0.19s | 8.66 | min_samples=25, base_min_cluster_size=10 |
| sklearn iris | 6.74s | 11.65s | 2.92 | min_samples=10, base_min_cluster_size=10 |
| sklearn wine | 8.78s | 0.16s | 4.05 | min_samples=10, base_min_cluster_size=10 |
| sklearn breast cancer | 7.09s | 0.18s | 3.61 | min_samples=10, base_min_cluster_size=10 |
| Fashion-MNIST | 28.04s | 0.77s | 15.56 | min_samples=50, base_min_cluster_size=10 |
| DBpedia-14 BGE-small | 37.42s | 0.77s | 25.69 | min_samples=50, base_min_cluster_size=10 |
| Olivetti faces | 7.44s | 0.34s | 5.21 | min_samples=10, base_min_cluster_size=10 |
| Forest CoverType | 25.35s | 0.62s | 3.04 | min_samples=25, base_min_cluster_size=100 |
| CIFAR-10 | 9.85s | 0.42s | 25.12 | min_samples=50, base_min_cluster_size=100 |

![CUDA fit runtime ratio](_static/benchmark_fit_ratio.svg)

## Layer selection

`DiReEVoC` uses coverage-weighted silhouette by default: mean silhouette over all points,
with noise contributing 0. This penalizes layers that score well only because they throw
many points away as noise. The current CUDA run still shows that layer selection is not
solved in general: MNIST, digits, wine, and several tiny or raw-pixel datasets have much
better fixed-k substrate scores than auto-selected scores.

## Graph fragment merge

Some DiRe layouts preserve the right connected geometry but EVoC emits adjacent pieces as
separate density fragments. The pipeline now uses DiRe's original-space kNN graph to
merge selected-layer fragments only when reciprocal cross-neighbor evidence or noise
bridges support the merge. It rejects merges that collapse the selected layer to one
cluster and rejects severe coverage-weighted-silhouette losses.

In this run, fragment merging improves moons from the old `0.045` auto ARI to `0.524`
and circles from `0.045` to `0.279`. It also improves breast cancer (`0.630` vs pure
EVoC's `0.116`). Collapse-to-one merges are rejected on MNIST, digits, Fashion-MNIST,
DBpedia-14, Olivetti, and Forest CoverType.

## Adaptive smoothing

The tight synthetic-blobs failure was a layer-generation failure: the DiRe layout had a
perfect fixed-k substrate, but EVoC's default smoothing could generate only degenerate
high-noise layers. The pipeline now retries a small grid of larger `min_samples` and
`base_min_cluster_size` values when the base clustering is mostly noise, then re-selects
using the same coverage-weighted score.

In the broad CUDA run, blobs recover to `1.000` auto ARI using `min_samples=10` and
`base_min_cluster_size=100`. The selected smoothing is recorded in
`selected_min_samples_` and `selected_base_min_cluster_size_`.

## Honest conclusion

This run supports a narrower claim than the old README:

- Wins are dataset-dependent. DiReEVoC wins on moons, circles, iris, breast cancer,
  Fashion-MNIST, CoverType, and CIFAR-10; ties on blobs; and loses on MNIST, DBpedia-14,
  digits, wine, and Olivetti in the auto metric.
- Fixed-k can disagree with auto. Several datasets have good fixed-k substrate scores but
  poor selected layers, so fixed-k and oracle numbers must not hide selector failures.
- Tiny data and raw pixels are weak regimes. Olivetti, CIFAR-10, wine, and CoverType are
  not strong quality wins in absolute ARI.
- DiRe layout construction dominates end-to-end time except where pure EVoC itself is
  slow.

## Compared with `master`

The `master` README was too broad. The CUDA repro now runs 13 datasets by default and
records all per-dataset figures. Fashion-MNIST remains a DiReEVoC win (`0.440 vs
0.390`), moons and circles now win after graph fragment merging, but DBpedia-14 still
slightly favors pure EVoC in this 39.2k run (`0.608 vs 0.623`). MNIST also favors pure
EVoC (`0.656 vs 0.813`). Historical 20 Newsgroups and arXiv claims are not part of the
default repro unless their loaders are enabled explicitly.

## Reproduce

On a CUDA Docker host reachable by SSH:

```bash
scripts/run_remote_cuda_repro.sh user@gpu-host
```

This syncs the working tree to the host, builds the Docker image, runs
`examples/run_cuda_repro.py`, and downloads all JSON, CSV, log, dataset, and SVG
artifacts to `artifacts/cuda/`.
