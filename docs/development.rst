Development
===========

Local checks
------------

Install development dependencies and run the CPU test suite:

.. code-block:: bash

   python -m pip install -e '.[dev]'
   python -m pytest

The pipeline smoke test imports ``cuml``, ``dire_rapids``, and ``evoc`` with
``pytest.importorskip``. It is skipped automatically when the GPU stack is not available.

Documentation checks
--------------------

.. code-block:: bash

   python -m pip install -e '.[docs]'
   python scripts/plot_benchmark_metrics.py
   python -m sphinx -b html docs docs/_build/html

GitHub Pages deployment is handled by ``.github/workflows/docs.yml``. The workflow builds
the checked-in benchmark figures, runs Sphinx with warnings treated as errors, uploads the
HTML artifact, and deploys it through the repository's Pages environment.

Package layout
--------------

``src/dire_evoc/dimension.py``
   Intrinsic-dimension estimators and DiRe reducer integration.

``src/dire_evoc/hybrid.py``
   cuML single-linkage-tree bridge into EVoC cluster-layer construction.

``src/dire_evoc/pipeline.py``
   ``DiReEVoC`` estimator tying layout, clustering, ID, and prediction together.

``src/dire_evoc/predict.py``
   Out-of-sample kNN vote assignment.

``examples/``
   Runnable quickstart and benchmark harnesses.

``examples/benchmark_visuals.py``
   Reproducible benchmark/profile runner that writes JSON, CSV, and SVG figures.

``examples/run_cuda_repro.py``
   Runs the bundled CUDA examples and benchmarks and writes one artifact tree with a
   manifest, logs, metrics JSON/CSV, datasets, and SVG figures.

``scripts/run_remote_cuda_repro.sh``
   Syncs the working tree to an SSH-accessible CUDA Docker host, builds the Docker image,
   runs ``examples/run_cuda_repro.py``, and downloads the artifact tree to
   ``artifacts/cuda/``.

``scripts/validate_hybrid_e2e.py``
   Full GPU-stack validation for correctness and throughput.
