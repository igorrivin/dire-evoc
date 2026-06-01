Benchmarks
==========

The benchmark scripts compare DiReEVoC against pure EVoC and separate three questions:

``auto``
   The layer selected by the method under the declared benchmark configuration.

``oracle``
   The best ARI among layers the method produced. This removes layer-selection error.

``fixed-k``
   The base layer forced to the known number of classes. This isolates substrate quality.

Quality summary
---------------

The current benchmark notes are in ``docs/benchmarks.md``. The headline result is
conditional, not universal. These figures summarize the CUDA Docker repro on an NVIDIA
A10, using matched dataset-specific EVoC base controls for DiReEVoC and pure EVoC:

.. image:: _static/benchmark_auto_ari.svg
   :alt: Auto ARI benchmark comparison for DiReEVoC and pure EVoC

.. image:: _static/benchmark_auto_delta.svg
   :alt: Auto ARI delta, DiReEVoC minus pure EVoC

.. list-table::
   :header-rows: 1

   * - Dataset
     - DiReEVoC auto ARI
     - Pure EVoC auto ARI
     - Result
   * - Synthetic blobs 12k
     - 1.000
     - 1.000
     - Tie
   * - Synthetic moons 12k
     - 1.000
     - 1.000
     - Tie with base_n_clusters=2
   * - Synthetic circles 12k
     - 1.000
     - 1.000
     - Tie with base_n_clusters=2
   * - MNIST
     - 0.723
     - 0.813
     - Pure EVoC win
   * - sklearn digits
     - 0.294
     - 0.305
     - Pure EVoC slight win
   * - sklearn iris
     - 0.562
     - 0.236
     - DiReEVoC win
   * - sklearn wine
     - 0.277
     - 0.121
     - DiReEVoC win
   * - sklearn breast cancer
     - 0.404
     - 0.116
     - DiReEVoC win
   * - Fashion-MNIST
     - 0.440
     - 0.390
     - DiReEVoC win
   * - DBpedia-14 BGE-small
     - 0.607
     - 0.623
     - Pure EVoC slight win
   * - Olivetti faces
     - 0.028
     - 0.063
     - Tiny-data tie; both poor
   * - Forest CoverType
     - 0.016
     - -0.026
     - DiReEVoC slight win; both poor
   * - CIFAR-10
     - 0.030
     - 0.023
     - DiReEVoC slight win; both poor

The fixed-k substrate comparison removes layer-selection and peak-granularity effects:

.. image:: _static/benchmark_fixed_k_ari.svg
   :alt: Fixed-k ARI benchmark comparison for DiReEVoC and pure EVoC

Runtime summary
---------------

The same CUDA run records fit timings for every dataset:

.. image:: _static/benchmark_fit_seconds.svg
   :alt: Fit runtime comparison for DiReEVoC and pure EVoC

.. image:: _static/benchmark_fit_ratio.svg
   :alt: Fit runtime ratio, DiReEVoC divided by pure EVoC

.. list-table::
   :header-rows: 1

   * - Dataset
     - DiReEVoC fit
     - Pure EVoC fit
   * - Synthetic blobs 12k
     - 10.33s
     - 11.70s
   * - Synthetic moons 12k
     - 5.68s
     - 0.27s
   * - Synthetic circles 12k
     - 5.66s
     - 0.26s
   * - MNIST
     - 27.92s
     - 0.74s
   * - sklearn digits
     - 9.91s
     - 0.19s
   * - sklearn iris
     - 6.63s
     - 0.23s
   * - sklearn wine
     - 7.31s
     - 0.16s
   * - sklearn breast cancer
     - 7.53s
     - 0.17s
   * - Fashion-MNIST
     - 27.54s
     - 0.76s
   * - DBpedia-14 BGE-small
     - 23.76s
     - 0.76s
   * - Olivetti faces
     - 7.38s
     - 0.28s
   * - Forest CoverType
     - 23.49s
     - 0.59s
   * - CIFAR-10
     - 10.33s
     - 0.41s

Comparative metrics
-------------------

``auto``
   The operational result: the layer selected by the method without looking at labels.
   For DiReEVoC this is the coverage-weighted silhouette layer. For pure EVoC it is
   EVoC's selected layer.

``layer``
   A flat clustering extracted from EVoC's density hierarchy. Layer 0 is the finest
   clustering; later layers are coarser and usually have fewer clusters. Each layer is a
   full label vector over the samples, with ``-1`` marking noise.

``oracle``
   The best ARI among the layers the method produced. This is not an unsupervised result;
   it uses ground-truth labels after the fact to answer whether the hierarchy contained a
   good clustering even if the selector chose another layer.

``fixed-k``
   The base layer forced toward the known number of classes. This is the cleanest
   substrate test because DiReEVoC and pure EVoC are compared at the same target
   granularity.

``ARI`` and ``NMI``
   External agreement scores against known labels. ARI is stricter about pairwise
   agreement and corrected for chance; NMI measures shared label information.

``noise`` and ``clusters``
   Diagnostics for interpretation. A higher ARI with extreme noise may be less useful
   operationally than a slightly lower ARI that covers most points.

``intrinsic dimension``
   A TwoNN estimate computed from DiRe's existing neighbour distances. It is not a
   clustering score; it tells you whether the data looks locally low-dimensional or
   manifold-like relative to the ambient embedding dimension.

``timings``
   Wall-clock phase timings. In the reproducible profile, DiRe layout time and EVoC
   clustering time are separated so it is clear which part is the bottleneck on the
   current machine.

Interpretation
--------------

The practical map is:

* DiReEVoC wins are dataset-dependent, not universal.
* Tiny datasets do not give DiRe enough data to form a reliable manifold; Olivetti is a
  weak result and fixed-k favors pure EVoC.
* Layer selection remains a real failure mode. Moons and circles are feasible when both
  methods use the known two-cluster base target, but several datasets still have much
  better fixed-k substrate quality than selected-layer quality.
* Continuous manifolds may not have a useful hard partition at all.
* CUDA accelerates the single-linkage-tree bridge, but DiRe layout construction can still
  dominate end-to-end time.

Compared with ``master``
------------------------

The old README was too broad. The CUDA repro now runs 13 datasets by default and records
all per-dataset figures. Fashion-MNIST remains a DiReEVoC win (``0.440`` vs ``0.390``),
moons and circles are now perfect ties under matched ``base_n_clusters=2`` controls, but
DBpedia-14 still slightly favors pure EVoC in this 39.2k run (``0.607`` vs ``0.623``).
MNIST also favors pure EVoC (``0.723`` vs ``0.813``). Historical 20 Newsgroups and arXiv
claims are not part of the default repro unless their loaders are enabled explicitly.

Reproduce
---------

Use the CUDA repro wrapper to run on an SSH-accessible CUDA Docker host and download all
artifacts:

.. code-block:: bash

   scripts/run_remote_cuda_repro.sh user@gpu-host

The wrapper syncs the working tree to the host, builds the Docker image, runs
``examples/run_cuda_repro.py``, and downloads ``manifest.json``, logs, datasets,
``metrics.json``, ``metrics.csv``, and SVG figures into ``artifacts/cuda/``.

For a single local benchmark with prepared data:

.. code-block:: bash

   python examples/benchmark_quality.py --npz /path/to/data.npz --label "Dataset name"

For metrics files and SVG plots, use:

.. code-block:: bash

   python examples/benchmark_visuals.py --synthetic-blobs --n-samples 12000 \
      --test-fraction 0.2 --output-dir artifacts/benchmark-visuals

The output directory contains ``metrics.json``, ``metrics.csv``, ``quality_ari.svg``,
``timing_breakdown.svg``, and ``diagnostics.svg``. Use ``--npz`` or ``--openml`` to run
the same profile on prepared data.
