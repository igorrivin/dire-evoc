`dire-evoc`: density clustering (EVōC) on GPU DiRe layouts.

• 13-dataset CUDA repro: Fashion-MNIST 0.440 vs 0.390; MNIST and DBpedia favor pure EVōC
• graph-fragment merge fixes moons/circles: 0.524 vs 0.350; 0.279 vs 0.037
• intrinsic dimension, free from the kNN graph
• out-of-sample predict()

Honest: not universal — tiny data/raw pixels are weak, fixed-k can beat auto selection,
and DiRe layout cost dominates at these sizes. [link]
