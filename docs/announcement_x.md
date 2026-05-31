`dire-evoc`: density clustering (EVōC) on GPU DiRe layouts.

• CUDA repro: DBpedia-14 ARI 0.612 vs 0.572; Fashion-MNIST 0.360 vs 0.323
• intrinsic dimension, free from the kNN graph
• out-of-sample predict()

Honest: not universal — tiny data is weak, layer selection can fail, and DiRe layout cost
dominates at these sizes. [link]
