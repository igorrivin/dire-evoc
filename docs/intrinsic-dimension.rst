Intrinsic Dimension
===================

The ``dire_evoc.dimension`` module estimates intrinsic dimension from sorted
nearest-neighbour distances.

Why this is cheap in the pipeline
---------------------------------

DiRe builds a k-nearest-neighbour graph before it optimizes the layout. Once that graph
exists, intrinsic dimension is an arithmetic pass over the neighbour distances rather
than a second neighbour search.

.. code-block:: python

   from dire_evoc import intrinsic_dimension_from_dire

   dim = intrinsic_dimension_from_dire(fitted_reducer, method="twonn")

For standalone use, compute neighbours on CPU:

.. code-block:: python

   from dire_evoc import intrinsic_dimension

   dim = intrinsic_dimension(X, k=20, method="twonn")

Estimators
----------

``twonn``
   The default. Uses the ratio between first and second nearest-neighbour distances.
   This is robust to most approximate-kNN error because only the nearest two neighbours
   matter.

``mle``
   Levina-Bickel maximum-likelihood estimator with the MacKay-Ghahramani
   inverse-averaging correction. This uses the first ``k`` neighbours and is useful as a
   second opinion.

Metric handling
---------------

Ratio-based intrinsic-dimension estimators need distances on the local geodesic scale.
For cosine-normalized embeddings, the local geodesic is the angle between vectors.
Raw cosine distance is approximately the square of that radius near zero, so treating
cosine distance as Euclidean distance biases the dimension downward.

Supported metric inputs:

``"euclidean"``
   Distances are already Euclidean.

``"angular"``
   Distances are already angles.

``"cosine"``
   Distances are cosine distances and are converted to angles.

``"chordal"``
   Distances are Euclidean chordal distances on the unit sphere and are converted to
   angles.

``intrinsic_dimension_from_dire`` reads ``reducer.metric`` and applies the appropriate
conversion automatically for DiRe's cosine path.
