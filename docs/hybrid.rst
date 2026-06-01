Hybrid GPU Clustering
=====================

The ``dire_evoc.hybrid`` module accelerates EVoC's expensive clustering phase by building
the single-linkage tree with cuML and handing that tree back to EVoC for its own
post-processing.

What is accelerated
-------------------

EVoC's layer machinery needs the HDBSCAN* single-linkage tree over mutual-reachability
distances. RAPIDS/cuML already implements that tree construction on the GPU.

``dire-evoc`` uses cuML for the tree and EVoC for everything after the tree:

1. Build the single-linkage tree with ``cuml.cluster.HDBSCAN``.
2. Convert it to EVoC's linkage matrix format.
3. Run EVoC's condense, persistence, and layer extraction post-process on that tree.

If the installed EVoC exposes ``build_cluster_layers(..., precomputed_linkage=slt)``,
``dire-evoc`` uses that seam. Current PyPI EVoC does not expose it, so the fallback calls
EVoC's own tree utility functions directly.

Usage
-----

.. code-block:: python

   from dire_evoc.hybrid import gpu_single_linkage_tree, hybrid_cluster_layers

   slt = gpu_single_linkage_tree(layout, min_samples=10)
   layers, memberships, persistence = hybrid_cluster_layers(layout, min_samples=10)

The low-level ``gpu_single_linkage_tree`` function is useful when you want the linkage
matrix directly. The higher-level ``hybrid_cluster_layers`` function runs the full EVoC
layer post-process.

Requirements
------------

The hybrid path requires:

* ``cuml`` for GPU single-linkage-tree construction.
* ``evoc`` for layer post-processing.
* ``hdbscan`` because cuML exposes ``single_linkage_tree_`` through that package.

Performance boundary
--------------------

The hybrid path is a throughput optimization for large data. It can be faster when
single-linkage-tree construction dominates and CPU parallelism is limited. On small data
or many-core CPU machines, EVoC's native CPU path can be faster.

Correctness boundary
--------------------

The single-linkage tree depends on ``min_samples``. The extracted base layer also depends
on ``base_min_cluster_size`` and, when used, ``base_n_clusters``. To compare native EVoC
and the hybrid path directly, keep these controls the same on both sides.
