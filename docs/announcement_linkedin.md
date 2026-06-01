**Introducing `dire-evoc` — density clustering on GPU layouts, with honest benchmarks.**

I've been building a small bridge between two tools I admire: Leland McInnes's **EVōC**
(density-based clustering) and **DiRe-RAPIDS** (GPU dimensionality reduction, with Sasha
Kolpakov). The idea is simple — use a DiRe layout as the *substrate* that EVōC clusters,
and add the pieces that only make sense at the intersection.

**What you get:**
🔹 *A broader CUDA repro, not just a cherry-picked table.* The default run now covers 13
datasets and writes per-dataset SVG/JSON/CSV artifacts. DiRe→EVōC wins Fashion-MNIST
(**0.440 vs 0.390**), breast cancer (**0.630 vs 0.116**), moons (**0.524 vs 0.350**),
and circles (**0.279 vs 0.037**), but pure EVōC wins MNIST (**0.656 vs 0.813**) and
slightly wins DBpedia-14 (**0.608 vs 0.623**).
🔹 *Intrinsic dimension, essentially free* — read straight off the k-NN graph DiRe already
builds (TwoNN / Levina–Bickel), no second pass.
🔹 *Out-of-sample `predict()`* — assign new points to existing clusters (EVōC itself has none).

**And — because benchmarks should be honest — what you *don't* get:** it's not a universal
win. Tiny data and raw pixels are weak regimes, fixed-k can still be much better than
auto selection on some datasets, and the DiRe layout cost dominates end-to-end time at
these benchmark sizes.

So: better clustering where clustering is meaningful, free intrinsic dimension, and
prediction — with a conditional speed-up at scale. Full code + the benchmarks that say
where it loses: [link]

Built on the shoulders of @Leland McInnes (EVōC/UMAP/HDBSCAN) and @Alexander Kolpakov
(DiRe-RAPIDS).
