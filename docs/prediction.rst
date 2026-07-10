Out-of-Sample Prediction
========================

EVoC and DiRe layouts are transductive: adding new points does not naturally give those
points cluster labels in the existing fitted layout. ``dire-evoc`` handles prediction by
voting in the original embedding space instead.

How it works
------------

``knn_vote_predict``:

1. Prepares reference and query embeddings with ``prepare_reference``. Cosine inputs are
   L2-normalized; Euclidean inputs retain their magnitude.
2. Finds nearest training points with the selected metric.
3. Gives each neighbour an inverse-distance vote.
4. Returns the label with the most vote weight and that label's membership share.

Noise label ``-1`` participates like any other label.
``DiReEVoC.predict`` automatically reuses the fitted model's metric. Direct callers must
pass the same ``metric`` to both preparation and voting.

Pipeline use
------------

.. code-block:: python

   model = DiReEVoC(min_samples=10, random_state=0).fit(X_train)
   labels = model.predict(X_new)
   labels, membership = model.predict(X_new, return_membership=True)

Direct use
----------

.. code-block:: python

   from dire_evoc import prepare_reference, knn_vote_predict

   reference = prepare_reference(X_train)
   query = prepare_reference(X_new)

   labels, membership = knn_vote_predict(
       query,
       reference,
       train_labels,
       n_neighbors=15,
   )

For Euclidean embeddings, preserve magnitude explicitly:

.. code-block:: python

   reference = prepare_reference(X_train, metric="euclidean")
   query = prepare_reference(X_new, metric="euclidean")
   labels, membership = knn_vote_predict(
       query, reference, train_labels, n_neighbors=15, metric="euclidean"
   )

Scaling notes
-------------

The current implementation uses exact dense matrix multiplication in chunks. That keeps
the voting logic simple and deterministic. For very large reference sets, an approximate
nearest-neighbour index can replace the neighbour search while preserving the same vote
logic.
