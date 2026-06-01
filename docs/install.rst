Installation
============

Base install
------------

The base package installs only the dependencies needed by the CPU-importable utilities:

.. code-block:: bash

   python -m pip install -e .

This is enough for:

* ``dire_evoc.dimension`` intrinsic-dimension estimators.
* ``dire_evoc.predict`` out-of-sample kNN voting.
* Importing the public package API.

Full GPU pipeline
-----------------

The one-call ``DiReEVoC`` pipeline and the hybrid GPU clustering bridge need the optional
DiRe, EVoC, and RAPIDS dependencies:

.. code-block:: bash

   python -m pip install -e '.[rapids,dire,cluster]'

RAPIDS packages are tied to CUDA and Python versions. In practice, it is often easier to
create a compatible RAPIDS environment first, then install ``dire-evoc`` into that
environment.

CUDA Docker repro
-----------------

The repository also includes a Dockerfile matching the CUDA benchmark environment. To
run the full repro on an SSH-accessible CUDA Docker host and download all JSON, CSV, log,
dataset, and SVG artifacts:

.. code-block:: bash

   scripts/run_remote_cuda_repro.sh user@gpu-host

By default the wrapper builds ``dire-evoc:cuda128`` on the host, runs
``examples/run_cuda_repro.py`` inside Docker, and downloads artifacts into
``artifacts/cuda/``.

Documentation
-------------

Install the Sphinx extra and build the HTML docs:

.. code-block:: bash

   python -m pip install -e '.[docs]'
   python scripts/plot_benchmark_metrics.py
   python -m sphinx -b html docs docs/_build/html

The ``docs`` extra includes Matplotlib because the README and Sphinx benchmark pages use
checked-in SVG figures generated from ``scripts/plot_benchmark_metrics.py``.

Development
-----------

.. code-block:: bash

   python -m pip install -e '.[dev]'
   python -m pytest

The CPU test suite does not require RAPIDS. End-to-end pipeline tests are skipped unless
``cuml``, ``dire_rapids``, and ``evoc`` import successfully.
