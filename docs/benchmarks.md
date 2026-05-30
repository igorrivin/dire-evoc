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

## Results

| dataset | metric | DiReEVoC | vanilla EVoC | winner |
|---|---|---|---|---|
| MNIST-784 (10 cls) | auto | 0.832 (k=11) | **0.903** (k=10) | vanilla |
| | oracle | 0.832 (k=11) | **0.903** (k=10) | vanilla |
| | fixed-k=10 | 0.890 | **0.903** | ~tie (substrate comparable) |
| 20NG-BGE384 (20 cls) | auto | 0.079 (k=35) | **0.192** (k=20) | vanilla |
| | oracle | **0.211** (k=13) | 0.192 (k=20) | **DiRe** |
| | fixed-k=20 | 0.112 | **0.192** | vanilla |

(arXiv-BGE, from the prior scaling study: both methods ARI ≈ 0 with ~75% noise — a
continuous manifold where a hard partition is the wrong object; the DiRe *layout* is the
deliverable there, not a clustering.)

## Honest conclusion

**DiRe→EVoC does not beat vanilla EVoC out of the box on these datasets** — vanilla wins
the operational (auto) metric on both MNIST and 20NG. This *reverses* the earlier working
belief and is worth stating plainly.

But the breakdown shows the embedding is not the problem:

- **The DiRe layout is a competitive substrate.** At fixed k it ties vanilla on MNIST
  (0.890 vs 0.903), and on 20NG it contains a markedly better clustering than vanilla
  produces at all (oracle 0.211 at k=13 vs vanilla 0.192).
- **The layer selector is the bottleneck.** On 20NG the silhouette selector picks a
  k=35 layer (ARI 0.079) when a k=13 layer (ARI 0.211) is sitting in the same layer set —
  a 0.13-ARI self-inflicted loss. On MNIST it picks k=11 instead of k=10. The good
  clustering is *produced* and then *not selected*.

So the value lever for dire-evoc's quality story is **layer selection / regime-adaptive
granularity**, not the embedding or the GPU speed path. The substrate advantage is real
but currently squandered by over-segmenting selection. Until that is addressed, the honest
framing is: dire-evoc's wins are **throughput** (the validated GPU Phase-2 hybrid) and the
**free intrinsic dimension**, plus the layout-as-deliverable on continuous manifolds —
not a universal clustering-quality win.

## Caveat on methodology

`base_n_clusters` extracts the requested k as the *base layer* but `EVoC.fit_predict`
returns the *selected* layer (a different k) — compare `cluster_layers_[0]` on both sides
for a fair fixed-k number (the harness does this). The `--openml` path works when
OpenML's API is reachable (it was returning HTTP 504 at time of writing).
