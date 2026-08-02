# LinkedIn announcement

Clustering high-dimensional data is rarely a single problem.

First, you need a representation that makes the structure visible. Then you need to find
meaningful groups. And when new data arrives, you need a practical way to connect it to
what you have already found.

I’ve released dire-evoc, an open-source Python package that connects DiRe-RAPIDS and EVoC
in one GPU-oriented workflow.

It provides:

• a one-call DiRe → EVoC clustering pipeline
• intrinsic-dimension estimation that reuses DiRe’s existing nearest-neighbor graph
• a GPU handoff for the expensive single-linkage-tree stage
• out-of-sample prediction for assigning new observations
• a reusable layout for visualization and further analysis

The package is deliberately thin. It does not reimplement DiRe or EVoC; it makes their
shared boundary explicit and adds the integration pieces that are useful when the two are
used together.

Just as importantly, the benchmark story is not “it wins everywhere.” It doesn’t.

dire-evoc does more work than clustering raw vectors directly, so it is slower and earns
that cost only when the DiRe layout exposes structure that the original feature space
does not. Pure EVoC can still be the better choice when raw-space geometry already works
well, and tiny datasets or weak feature representations remain difficult regimes.

The repository includes reproducible benchmarks, including the cases where the approach
ties or loses—not only its strongest results.

The GPU path has now been tested with RAPIDS 26.06, and the CPU-facing CI matrix covers
Python 3.10–3.14.

Code, examples, and benchmark details:
https://github.com/igorrivin/dire-evoc

Built on the excellent work behind DiRe-RAPIDS and EVoC, with thanks to Sasha Kolpakov
and Leland McInnes.

#OpenSource #GPUComputing #MachineLearning #Clustering #RAPIDS
