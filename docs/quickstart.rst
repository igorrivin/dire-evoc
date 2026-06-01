Quick Start
===========

One-call pipeline
-----------------

Use ``DiReEVoC`` when the full GPU stack is available and you want the complete workflow:
DiRe layout, EVoC cluster layers, intrinsic dimension, and out-of-sample prediction.

.. code-block:: python

   from dire_evoc import DiReEVoC

   model = DiReEVoC(min_samples=10, random_state=0).fit(X)

   labels = model.labels_
   layout = model.layout_
   dimension = model.intrinsic_dimension_

   labels_new = model.predict(X_new)

``labels`` is the selected EVoC layer. Label ``-1`` means noise.

Standalone intrinsic dimension
------------------------------

The intrinsic-dimension utilities work without the GPU stack:

.. code-block:: python

   import numpy as np
   from dire_evoc import intrinsic_dimension

   rng = np.random.default_rng(0)
   X = rng.normal(size=(5000, 12))

   dim = intrinsic_dimension(X, k=20, method="twonn")
   print(dim)

Standalone prediction
---------------------

Out-of-sample assignment is a distance-weighted kNN vote over training embeddings:

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

Run the bundled example
-----------------------

With DiRe-RAPIDS, RAPIDS/cuML, and EVoC installed:

.. code-block:: bash

   python examples/quickstart.py
