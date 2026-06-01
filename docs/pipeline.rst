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

Available selectors:

``"silhouette"``
   Coverage-weighted silhouette. This is the default.

``"silhouette_raw"``
   Plain silhouette over clustered points only.

``"base"``
   Use the finest EVoC layer.

``"none"``
   Use the coarsest EVoC layer.

When to trust labels
--------------------

Density clustering is appropriate when the data has density-separated structure. On a
continuous manifold, the hard partition can be the wrong object. Inspect the layout, the
noise fraction, and ``intrinsic_dimension_`` before treating ``labels_`` as ground truth.
