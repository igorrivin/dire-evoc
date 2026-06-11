**Introducing `dire-evoc` — density clustering on GPU layouts, with honest benchmarks.**

I've been building a small bridge between two tools I admire: Leland McInnes's **EVōC**
(density-based clustering) and **DiRe-RAPIDS** (GPU dimensionality reduction, with Sasha
Kolpakov). The idea is simple — use a DiRe layout as the *substrate* that EVōC clusters,
and add the pieces that only make sense at the intersection.

**What you get:**
🔹 *A broader CUDA repro, not just a cherry-picked table.* The default run now covers 18
datasets, writes per-dataset SVG/JSON/CSV artifacts, uses matched EVoC base controls for
DiRe→EVōC and pure EVōC, and records 20-run warm-primary timings plus first-run and
first-included fields.
The scale run switches to one timed fit per dataset and covers 30k synthetic controls,
full MNIST/Fashion-MNIST/CIFAR-10, full DBpedia-14 embeddings, Forest CoverType, Wiki-CS,
and OGBN-Arxiv.
The selector branch also adds an opt-in guarded graph-component selector: it recovers the
separated-spirals control at **1.000 ARI** where coverage-weighted silhouette gets
**0.151**, while falling back to silhouette on MNIST/Fashion/CIFAR/OpenML rows unless the
graph candidate passes explicit guards.
DiRe→EVōC wins Fashion-MNIST (**0.440 vs 0.390**), breast cancer (**0.404 vs 0.116**),
iris (**0.562 vs 0.236**), wine (**0.465 vs 0.121**), and pooled CIFAR-10
(**0.029 vs 0.012**). The newly added OpenML rows include strong DiRe→EVōC auto wins on
OptDigits (**0.848 vs 0.147**) and PenDigits (**0.738 vs 0.125**); moons/circles are
feasible ties when both methods use
`base_n_clusters=2` (**1.000 vs 1.000**); pure EVōC wins MNIST (**0.720 vs 0.813**) and
slightly wins DBpedia-14 (**0.607 vs 0.623**).
🔹 *Stability at a million points, on real unlabelled data.* A separate push-button pipeline
clusters 1M Cohere-Wikipedia embeddings and tracks how a shared 100k anchor's labels agree
with themselves as N grows 100k→1M. Both methods return fast (vanilla EVōC ~**26s**; DiRe→EVōC
~10× slower, the layout step), so the number to read is the agreement across N: it's **~0**
for vanilla EVōC (its own two-seed floor is 0.39) and **~0.5** for DiRe→EVōC (two-seed floor
0.57 — so adding data moves it about as much as reseeding does). Same remote-run → download →
plot flow as the supervised suite.
🔹 *Intrinsic dimension, essentially free* — read straight off the k-NN graph DiRe already
builds (TwoNN / Levina–Bickel), no second pass.
🔹 *Out-of-sample `predict()`* — assign new points to existing clusters (EVōC itself has none).

**And — because benchmarks should be honest — what you *don't* get:** it's not a universal
win. Tiny data and weak feature representations are weak regimes, fixed-k can still be
much better than auto selection on some datasets, and the DiRe layout cost dominates
end-to-end time on most recorded rows. The CIFAR benchmark uses average-pooled 16x16 RGB
features: 30k images in the repeated default repro and 60k in the scale repro.

So: better clustering where clustering is meaningful, free intrinsic dimension, and
prediction — with benchmark artifacts that say where it loses. Full code + the
reproducible metrics: [link]

Built on the shoulders of @Leland McInnes (EVōC/UMAP/HDBSCAN) and @Alexander Kolpakov
(DiRe-RAPIDS).
