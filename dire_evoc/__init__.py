"""dire-evoc: DiRe-RAPIDS layouts + EVoC density clustering, GPU-accelerated.

A thin integration layer over two upstream primitives -- ``dire-rapids`` (GPU
dimensionality reduction) and ``evoc`` (density clustering) -- adding the pieces that
only make sense at their intersection:

- :mod:`dire_evoc.dimension` -- intrinsic-dimension estimation reusing DiRe's kNN graph
  (essentially free once the layout is built).
- :mod:`dire_evoc.hybrid`    -- GPU Phase-2 clustering: cuML single-linkage tree fed into
  EVoC's layer post-process (~3.3x on Phase 2 at 1M points, identical clustering).
- :mod:`dire_evoc.predict`   -- out-of-sample cluster assignment by kNN vote.

The heavy dependencies (cuml, cupy, dire-rapids, evoc) are imported lazily by the
functions that need them, so the pure-numpy estimators in this package import and test
on any machine.
"""
from .dimension import (
    intrinsic_dimension,
    intrinsic_dimension_from_dire,
    mle,
    twonn,
)
from .predict import knn_vote_predict, prepare_reference

__all__ = [
    "intrinsic_dimension",
    "intrinsic_dimension_from_dire",
    "twonn",
    "mle",
    "knn_vote_predict",
    "prepare_reference",
]

__version__ = "0.1.0"
