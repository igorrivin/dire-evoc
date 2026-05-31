**Introducing `dire-evoc` — density clustering on GPU layouts, with honest benchmarks.**

I've been building a small bridge between two tools I admire: Leland McInnes's **EVōC**
(density-based clustering) and **DiRe-RAPIDS** (GPU dimensionality reduction, with Sasha
Kolpakov). The idea is simple — use a DiRe layout as the *substrate* that EVōC clusters,
and add the pieces that only make sense at the intersection.

**What you get:**
🔹 *Better clustering on large embedding datasets.* On Fashion-MNIST, DiRe→EVōC scores ARI
**0.40 vs 0.22** for EVōC on the raw vectors — and it wins on DBpedia and 20-Newsgroups too.
🔹 *Intrinsic dimension, essentially free* — read straight off the k-NN graph DiRe already
builds (TwoNN / Levina–Bickel), no second pass.
🔹 *Out-of-sample `predict()`* — assign new points to existing clusters (EVōC itself has none).

**And — because benchmarks should be honest — what you *don't* get:** it's not a universal
win. It needs enough data to form the manifold (it's *worse* than plain EVōC on 400-point
Olivetti), a hard partition is the wrong object on truly continuous manifolds (think arXiv
abstracts), and it's only *faster* than CPU EVōC at very large n on low-core machines —
otherwise the GPU layout isn't free.

So: better clustering where clustering is meaningful, free intrinsic dimension, and
prediction — with a conditional speed-up at scale. Full code + the benchmarks that say
where it loses: [link]

Built on the shoulders of @Leland McInnes (EVōC/UMAP/HDBSCAN) and @Alexander Kolpakov
(DiRe-RAPIDS).
