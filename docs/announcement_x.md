`dire-evoc`: density clustering (EVōC) on GPU DiRe layouts.

• 18-dataset CUDA repro with matched EVoC base controls and warm-primary 20-run timing: OptDigits 0.848 vs 0.147; PenDigits 0.738 vs 0.125; MNIST and DBpedia favor pure EVōC
• scale CUDA sweep: one timed fit over 30k synthetic controls, full MNIST/Fashion/CIFAR-10, full DBpedia, CoverType, Wiki-CS, and OGBN-Arxiv
• opt-in guarded graph selector: separated spirals 1.000 ARI vs silhouette 0.151, with guards that fall back on MNIST/Fashion/CIFAR/OpenML rows
• moons/circles are feasible ties when both methods use base_n_clusters=2: 1.000 vs 1.000
• CIFAR-10 uses pooled 16x16 RGB features: 30k in the repeated default repro, 60k in the scale repro
• scaling/stability on 1M unlabelled Cohere-wiki embeddings: both return fast, so read the shared-anchor agreement across N (100k→1M) — ~0 for vanilla EVōC (two-seed floor 0.39), ~0.5 for DiRe→EVōC (two-seed floor 0.57). Same remote-run→download→plot flow as the suite
• intrinsic dimension read straight off DiRe's existing kNN graph
• out-of-sample predict()

Honest: not universal — tiny data/weak features are weak, fixed-k can beat auto selection,
and DiRe layout cost dominates at these sizes. [link]
