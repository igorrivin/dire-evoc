**Introducing `dire-evoc` — density clustering on GPU layouts, with honest benchmarks.**

I've been building a small bridge between two tools I admire: Leland McInnes's **EVōC**
(density-based clustering) and **DiRe-RAPIDS** (GPU dimensionality reduction, with Sasha
Kolpakov). The idea is simple — use a DiRe layout as the *substrate* that EVōC clusters,
and add the pieces that only make sense at the intersection.

**What you get:**
🔹 *A broader CUDA repro, not just a cherry-picked table.* The default run now covers 13
datasets, writes per-dataset SVG/JSON/CSV artifacts, uses matched EVoC base controls for
DiRe→EVōC and pure EVōC, and records 20-run warm-primary timings plus first-run and
first-included fields.
DiRe→EVōC wins Fashion-MNIST (**0.440 vs 0.390**), breast cancer (**0.404 vs 0.116**),
iris (**0.562 vs 0.236**), wine (**0.465 vs 0.121**), and pooled CIFAR-10
(**0.029 vs 0.012**); moons/circles are feasible ties when both methods use
`base_n_clusters=2` (**1.000 vs 1.000**); pure EVōC wins MNIST (**0.720 vs 0.813**) and
slightly wins DBpedia-14 (**0.607 vs 0.623**).
🔹 *Intrinsic dimension, essentially free* — read straight off the k-NN graph DiRe already
builds (TwoNN / Levina–Bickel), no second pass.
🔹 *Out-of-sample `predict()`* — assign new points to existing clusters (EVōC itself has none).

**And — because benchmarks should be honest — what you *don't* get:** it's not a universal
win. Tiny data and weak feature representations are weak regimes, fixed-k can still be
much better than auto selection on some datasets, and the DiRe layout cost dominates
end-to-end time at these benchmark sizes. The CIFAR benchmark uses 30k images
average-pooled to 16x16 RGB.

So: better clustering where clustering is meaningful, free intrinsic dimension, and
prediction — with a conditional speed-up at scale. Full code + the benchmarks that say
where it loses: [link]

Built on the shoulders of @Leland McInnes (EVōC/UMAP/HDBSCAN) and @Alexander Kolpakov
(DiRe-RAPIDS).
