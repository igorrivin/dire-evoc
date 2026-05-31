# Quality benchmarks: does DiRe→EVoC beat vanilla EVoC?

Reproduce with `examples/benchmark_quality.py`. Each dataset is scored three ways, which
separate *substrate quality* (is the DiRe layout a good thing to cluster?) from *selector
quality* (does the layer-selection machinery pick the good clustering?):

- **auto** — operational, out-of-the-box: the layer the method actually selects.
- **oracle** — best ARI over the layers the method *produced* (selector removed).
- **fixed-k** — the base layer forced to `k = n_classes` on each substrate (peak-finding
  and selection both removed; pure substrate at the true granularity).

"vanilla EVoC" = `EVoC.fit_predict` (EVoC's own node embedding + clustering). "DiReEVoC" =
the same EVoC clustering on a DiRe layout. ARI is against ground-truth labels.

## CUDA results

These numbers come from `scripts/run_remote_cuda_repro.sh` on an NVIDIA A10 using the
Dockerfile in this repository. The full run completed successfully: quickstart, dataset
preparation, visual benchmarks, and quality benchmarks all returned exit code 0. After
running the repro, the local artifact tree is `artifacts/cuda/`.

Operational (auto) ARI vs ground truth:

| dataset | n | d | cls | DiReEVoC (auto) | vanilla EVoC (auto) | winner | fixed-k substrate |
|---|---|---|---|---|---|---|---|
| DBpedia-14 BGE-small | 49k | 384 | 14 | **0.612** (k=13, 13.4% noise) | 0.572 (k=14, 14.3% noise) | **DiRe +0.040** | DiRe 0.612 vs 0.572 |
| Fashion-MNIST | 60k | 784 | 10 | **0.360** (k=14, 21.0% noise) | 0.323 (k=9, 23.9% noise) | **DiRe +0.037** | DiRe 0.398 vs 0.273 |
| Olivetti faces | 400 | 4096 | 40 | **0.073** (k=5, 34.2% noise) | 0.057 (k=7, 43.0% noise) | DiRe +0.016, both poor | vanilla 0.021 vs 0.041 |
| Synthetic blobs 12k | 12k | 32 | 8 | 0.223 (k=2, 0.0% noise) | **1.000** (k=8, 0.0% noise) | vanilla +0.777 | tie 1.000 vs 1.000 |

DBpedia-14 is a labeled vector store: DBpedia ontology entities embedded with
BGE-small. Fashion-MNIST is the full 60k training split. Olivetti is a tiny-data stress
case. Synthetic blobs are deliberately easy density-separated data; they expose a
selector failure rather than a substrate failure.

Timing from the same CUDA run:

| dataset | DiReEVoC fit | vanilla EVoC fit | DiRe layout | EVoC on DiRe layout | DiRe fixed-k layer | vanilla fixed-k fit | intrinsic dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBpedia-14 BGE-small | 16.45s | 1.06s | 14.20s | 0.76s | 0.25s | 1.15s | 25.19 |
| Fashion-MNIST | 13.89s | 1.82s | 11.30s | 0.96s | 0.24s | 1.96s | 15.45 |
| Olivetti faces | 7.92s | 0.27s | 7.53s | 0.38s | 0.00s | 0.19s | 5.44 |
| Synthetic blobs 12k | 9.55s | 12.25s | 8.25s | 0.40s | 0.10s | 2.00s | 22.51 |

The GPU-accelerated EVoC phase is small in these examples; DiRe layout construction is
the dominant cost. That means quality and diagnostics are the main reasons to run the
pipeline at these sizes, not end-to-end speed.

## Layer selection

`DiReEVoC` uses coverage-weighted silhouette by default: mean silhouette over all points,
with noise contributing 0. This penalizes layers that score well only because they throw
many points away as noise. The current CUDA run still shows that layer selection is not
solved in general: synthetic blobs have perfect fixed-k substrate quality
(`1.000 vs 1.000`), but DiReEVoC's selected layer is a coarse two-cluster partition with
ARI `0.223`.

## Honest conclusion

This CUDA run supports a narrower claim than the old README:

- **Real medium/large datasets:** DiReEVoC wins on DBpedia-14 and Fashion-MNIST, and the
  fixed-k substrate comparison is also better on both.
- **Tiny data:** Olivetti is too small for a strong claim. Auto ARI is slightly higher for
  DiReEVoC, but both are poor and fixed-k favors vanilla.
- **Synthetic easy data:** the DiRe substrate can be perfect while the selected layer is
  wrong. Do not hide selector failures behind fixed-k or oracle numbers.
- **Speed:** EVoC-on-layout is fast; DiRe layout construction dominates end to end.

## Compared with `master`

The `master` README was partly true but too broad:

- It claimed a Fashion-MNIST win of `0.395 vs 0.224`. The CUDA repro still shows a
  Fashion-MNIST win, but smaller: `0.360 vs 0.323` auto ARI.
- It claimed a DBpedia-14 win of `0.623 vs 0.606`. The CUDA repro still shows a
  DBpedia-14 win, but with current numbers: `0.612 vs 0.572`.
- It claimed Olivetti was essentially a tiny-data tie. The CUDA repro agrees on the
  conclusion, but the fixed-k number is worse for DiReEVoC (`0.021 vs 0.041`).
- It included 20 Newsgroups, MNIST, and arXiv claims that were not rerun in this CUDA
  pass. They should be treated as historical notes until reproduced by
  `examples/run_cuda_repro.py`.
- It said the full hybrid path depended on an EVoC `precomputed_linkage` seam. The current
  code no longer requires that seam: when it is absent, `dire_evoc.hybrid` calls EVoC's
  existing tree utilities directly after cuML builds the single-linkage tree.

## Caveat on methodology

`base_n_clusters` extracts the requested k as the *base layer* but `EVoC.fit_predict`
returns the *selected* layer (a different k) — compare `cluster_layers_[0]` on both sides
for a fair fixed-k number (the harness does this).

## Reproduce

On a CUDA Docker host reachable by SSH:

```bash
scripts/run_remote_cuda_repro.sh user@gpu-host
```

This syncs the working tree to the host, builds the Docker image, runs
`examples/run_cuda_repro.py`, and downloads all JSON, CSV, log, dataset, and SVG artifacts
to `artifacts/cuda/`.
