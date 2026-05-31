**Introducing `dire-evoc` — density clustering on GPU layouts, with honest benchmarks.**

I've been building a small bridge between two tools I admire: Leland McInnes's **EVōC**
(density-based clustering) and **DiRe-RAPIDS** (GPU dimensionality reduction, with Sasha
Kolpakov). The idea is simple — use a DiRe layout as the *substrate* that EVōC clusters,
and add the pieces that only make sense at the intersection.

**What you get:**
🔹 *Better clustering on the CUDA repro's large embedding datasets.* On DBpedia-14
BGE-small, DiRe→EVōC scores ARI **0.612 vs 0.572** for EVōC on the raw vectors; on full
Fashion-MNIST it scores **0.360 vs 0.323**.
🔹 *Intrinsic dimension, essentially free* — read straight off the k-NN graph DiRe already
builds (TwoNN / Levina–Bickel), no second pass.
🔹 *Out-of-sample `predict()`* — assign new points to existing clusters (EVōC itself has none).

**And — because benchmarks should be honest — what you *don't* get:** it's not a universal
win. Tiny data is weak (Olivetti is poor for both methods, and fixed-k favors vanilla),
the synthetic blobs run exposes a real layer-selection failure even though fixed-k is
perfect, and the DiRe layout cost dominates end-to-end time at these benchmark sizes.

So: better clustering where clustering is meaningful, free intrinsic dimension, and
prediction — with a conditional speed-up at scale. Full code + the benchmarks that say
where it loses: [link]

Built on the shoulders of @Leland McInnes (EVōC/UMAP/HDBSCAN) and @Alexander Kolpakov
(DiRe-RAPIDS).
