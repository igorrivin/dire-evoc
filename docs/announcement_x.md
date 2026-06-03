`dire-evoc`: density clustering (EVōC) on GPU DiRe layouts.

• 13-dataset CUDA repro with matched EVoC base controls and 20-run timing: Fashion-MNIST 0.440 vs 0.390; MNIST and DBpedia favor pure EVōC
• moons/circles are feasible ties when both methods use base_n_clusters=2: 1.000 vs 1.000
• CIFAR-10 uses 30k images pooled to 16x16 RGB for the recorded repro
• intrinsic dimension, free from the kNN graph
• out-of-sample predict()

Honest: not universal — tiny data/weak features are weak, fixed-k can beat auto selection,
and DiRe layout cost dominates at these sizes. [link]
