`dire-evoc`: density clustering (EVōC) on GPU DiRe layouts.

• better clustering on big embedding sets — Fashion-MNIST ARI 0.40 vs 0.22
• intrinsic dimension, free from the kNN graph
• out-of-sample predict()

Honest: not universal — needs large n, won't partition continuous manifolds, speed-up only
at scale. [link]
