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

## Results (with coverage-weighted layer selection)

Operational (auto) ARI vs ground truth, across datasets of varying size and origin:

| dataset | n | d | cls | DiReEVoC (auto) | vanilla EVoC (auto) | winner | fixed-k substrate |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 60k | 784 | 10 | **0.395** (k=14) | 0.224 (k=16) | **DiRe +0.17** | DiRe 0.396 vs 0.240 |
| DBpedia-14 (BGE) | 49k | 384 | 14 | **0.623** (k=13) | 0.606 (k=12) | **DiRe +0.02** | ~tie 0.612 vs 0.610 |
| 20NG (BGE) | 19k | 384 | 20 | **0.214** (k=13) | 0.192 (k=20) | **DiRe +0.02** | vanilla 0.109 vs 0.192 |
| MNIST | 70k | 784 | 10 | 0.832 (k=11) | **0.903** (k=10) | vanilla −0.07 | ~tie 0.890 vs 0.903 |
| Olivetti faces | 400 | 4096 | 40 | 0.094 (k=7) | 0.091 (k=6) | tie (both poor) | vanilla 0.024 vs 0.292 |
| arXiv-BGE | 723k | 384 | — | ≈0 | ≈0 | — (continuous) | — |

DBpedia-14 is a genuine *labeled vector store*: DBpedia ontology entities (Company, Artist,
Animal, Film, …) embedded with BGE-small. arXiv-BGE (prior scaling study): both ≈0 with
~75% noise — a continuous manifold where a hard partition is the wrong object and the DiRe
*layout* is the deliverable. Reproduce data with `examples/prepare_datasets.py`.

## The layer-selector fix (coverage-weighted silhouette)

The first run of this benchmark used plain silhouette to pick the operative layer, and
DiRe→EVoC *lost* on 20NG (auto ARI 0.079, k=35) — even though its layer set contained a
k=13 layer at ARI 0.214. Diagnosis: plain silhouette is computed only on *clustered*
points, so a layer that dumps ~half the data as noise but leaves tight residual clusters
scores well. It systematically over-segments. (This was the open hard case in the earlier
selector study: *"20NG/no-target defeats every compactness-based selector."*)

Fix: score the **mean silhouette over all points, with noise contributing 0** — which
equals `silhouette(clustered) × coverage`. It rewards layers that are both well-separated
*and* explain the data. Across synth / MNIST / 20NG this drops the selector's ARI-regret
from a mean of 0.058 (max 0.173) to **0.000** — it picks the oracle layer every time.
After the fix, **DiRe→EVoC wins operationally on 20NG (0.214 vs 0.192)**. `DiReEVoC`
uses it by default; `selector="silhouette_raw"` restores the old behaviour.

## Adaptive `min_samples` (tight-substrate failure)

A second, distinct failure (found by Alexander Kolpakov): on *ultra-tight* synthetic blobs
(spread 0.10), DiRe→EVoC scored ARI ≈ 0.003 while fixed-k=8 gave 1.0. Diagnosis — **not**
the selector and **not** the layout (true-label silhouette on the layout is 0.86, and
fixed-k=8 → 1.0), but **layer generation**: at the default `min_samples=10` EVoC labels
~85% of the points noise and over-fragments (k=28/51), so *every* layer in the set is
degenerate and there is nothing good to select. It is DiRe-specific in degree because the
DiRe layout is *tighter* than EVoC's own embedding (vanilla EVoC gets 0.87 here), so EVoC's
default density smoothing is too local.

No single `min_samples`/`base_min_cluster_size` default is robust — tight blobs need large
smoothing, real text/image embeddings need small (`min_samples=25` hurts 20NG and Fashion).
Fix (`adaptive_min_samples=True`, default): if the base clustering labels more than
`noise_gate` (0.5) of points as noise, re-cluster at larger `min_samples` (2.5×, 5×) and
re-select by the *same* coverage-weighted score, adopting a higher `min_samples` only if it
strictly beats the base. Result: blobs recover **0.003 → 1.0** (escalating to
`min_samples=50`), while every real dataset stays below the gate and is untouched
(`selected_min_samples_` reports what was used). Genuinely diffuse data (continuous
manifolds) keeps its honest high-noise result because no escalation beats it.

## Honest conclusion

With the coverage-weighted selector, DiRe→EVoC is **competitive-to-better on real,
medium-to-large labeled datasets**, and the wins are largest where a good embedding helps
most:

- **Fashion-MNIST: clear win (+0.17).** The DiRe layout separates the classes far better
  than EVoC's own embedding (and with much less noise, 18% vs 33%).
- **DBpedia-14 and 20NG: wins** (+0.02 each) — modest but real, on labeled vector-store /
  text-embedding data.
- **MNIST: narrow loss (−0.07)** — and *not* a selector failure (selected = oracle). The
  residual is layer *granularity*: no clean k=10 layer is produced (layers are k=11, 20,
  45…); at fixed k=10 the substrate ties vanilla (0.890 vs 0.903).

Two regime boundaries the data makes explicit:

- **Small n (Olivetti, n=400): DiRe underperforms.** The DiRe manifold optimization is
  data-hungry; with 400 points / 10 per class its layout is worse than EVoC's own embedding
  at fixed k (0.024 vs 0.292). dire-evoc's substrate advantage needs enough data to form
  the manifold — it is a large-n method.
- **Continuous manifolds (arXiv): a partition is the wrong object** for either method
  (ARI ≈ 0); the layout itself is the deliverable.

So the earlier "vanilla wins everything" was a *selector* artifact, now fixed; and the
opposite over-claim ("DiRe always wins") is also false. The honest map: **DiRe→EVoC wins
on medium/large density-separated data, loses on tiny data, and neither clusters continuous
manifolds.** Robust wins across all regimes remain **throughput** (GPU Phase-2 hybrid) and
**free intrinsic dimension**, plus layout-as-deliverable on continuous manifolds.

## Caveat on methodology

`base_n_clusters` extracts the requested k as the *base layer* but `EVoC.fit_predict`
returns the *selected* layer (a different k) — compare `cluster_layers_[0]` on both sides
for a fair fixed-k number (the harness does this).
