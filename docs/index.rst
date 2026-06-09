dire-evoc
=========

``dire-evoc`` is a thin integration layer between DiRe-RAPIDS layouts and EVoC density
clustering. It provides a one-call pipeline, intrinsic-dimension estimates from DiRe's
existing nearest-neighbour graph, a GPU single-linkage-tree bridge, and out-of-sample
cluster assignment.

The package keeps GPU-heavy dependencies behind lazy imports. The base package can be
imported and tested on a CPU-only machine; the full pipeline requires DiRe-RAPIDS, EVoC,
and RAPIDS/cuML.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   install
   quickstart
   pipeline
   intrinsic-dimension
   hybrid
   prediction
   benchmarks
   scaling

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   development
