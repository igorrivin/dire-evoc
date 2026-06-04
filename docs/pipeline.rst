DiReEVoC Pipeline
=================

``DiReEVoC`` is the high-level estimator in ``dire_evoc.pipeline``. It runs:

1. DiRe-RAPIDS layout construction in ``n_components`` dimensions.
2. EVoC density clustering on that layout.
3. Layer selection by coverage-weighted silhouette.
4. Intrinsic-dimension estimation from DiRe's existing kNN graph.
5. Optional prediction for new embeddings using a kNN vote in the original space.

Basic use
---------

.. code-block:: python

   from dire_evoc import DiReEVoC

   model = DiReEVoC(
       n_components=8,
       metric="cosine",
       min_samples=10,
       base_min_cluster_size=5,
       selector="silhouette",
       random_state=0,
   ).fit(X)

   labels = model.labels_
   memberships = model.membership_

Important attributes
--------------------

``layout_``
   The DiRe layout clustered by EVoC.

``labels_``
   Labels for the selected operative layer. Noise is ``-1``.

``membership_``
   Per-point membership strength for ``labels_``.

``cluster_layers_``
   All EVoC layers, ordered fine to coarse.

``selected_layer_``
   Index into ``cluster_layers_`` used for ``labels_``.

``intrinsic_dimension_``
   TwoNN intrinsic-dimension estimate read from the fitted DiRe reducer's kNN graph.
   If that graph is unavailable, this is ``None``.

Layer selection
---------------

The default selector is ``"silhouette"``. It computes silhouette on clustered points and
multiplies by coverage, treating noise as a zero-silhouette contribution. This avoids the
failure mode where a layer discards many difficult points as noise and receives an
artificially high plain silhouette score.

Before final selection, DiReEVoC can append one extra candidate layer from DiRe's stored
kNN graph. It sweeps neighbour counts around ``ceil(intrinsic_dimension) + 1`` and keeps
a connected-component labelling only when adjacent k values agree and every component is
at least ``base_min_cluster_size``. That graph candidate overrides the EVoC layer only
when the post-merge EVoC result is visibly more fragmented, or when it matches an explicit
``base_n_clusters`` control.

Available selectors:

``"silhouette"``
   Coverage-weighted silhouette plus the guarded graph-component candidate. This is the
   default.

``"silhouette_raw"``
   Plain silhouette over clustered points only.

``"base"``
   Use the finest EVoC layer.

``"none"``
   Use the coarsest EVoC layer.

Base-layer controls
-------------------

``base_min_cluster_size`` defaults to ``5``, matching EVoC's own default. Set
``base_n_clusters`` only when you intentionally want a controlled benchmark target, such
as the known two-cluster synthetic moons/circles examples. The broad benchmark suite
passes the same base controls to DiReEVoC and pure EVoC.

When to trust labels
--------------------

Density clustering is appropriate when the data has density-separated structure. On a
continuous manifold, the hard partition can be the wrong object. Inspect the layout, the
noise fraction, and ``intrinsic_dimension_`` before treating ``labels_`` as ground truth.
