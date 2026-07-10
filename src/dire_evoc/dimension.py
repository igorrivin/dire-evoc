"""Intrinsic-dimension estimation, ~free from a kNN graph.

The DiRe-EVoC pipeline already builds a k-nearest-neighbour graph on the original
high-dimensional embeddings. The intrinsic dimension (ID) of those embeddings is a
pure function of that graph's *sorted neighbour distances* -- so once DiRe has run,
ID costs O(n) arithmetic, no second neighbour search. This module provides that.

Two estimators, both classical and both defined directly on neighbour distances:

TwoNN  -- Facco, d'Errico, Rodriguez & Laio, "Estimating the intrinsic dimension of
          datasets by a minimal neighborhood information", Sci. Rep. 7 (2017).
          For each point the ratio mu = r2 / r1 of its 2nd- to 1st-nearest-neighbour
          distance is Pareto(d)-distributed; d is recovered from the linear relation
          -log(1 - F(mu)) = d * log(mu). Uses only the two nearest neighbours, so it
          is robust to approximate-kNN error (1st/2nd-NN recall is ~1 in practice).

MLE    -- Levina & Bickel, "Maximum Likelihood Estimation of Intrinsic Dimension",
          NIPS 2004, with the MacKay-Ghahramani inverse-averaging correction. Uses
          the first k neighbour distances; a useful second opinion but mildly
          sensitive to the neighbourhood scale k.

METRIC -- CRITICAL. These estimators are *ratio*-based, so the distance metric must
          be (locally) the manifold geodesic. For data on the unit sphere (any
          cosine-normalised embedding -- BGE text, normalised image features, ...)
          the geodesic is the angle theta. Cosine *distance* 1 - cos(theta) ~ theta^2/2
          near 0, i.e. the SQUARE of the geodesic radius. A ratio-based estimator reads
          dimension off how neighbour count grows with radius (volume ~ r^d); squaring
          the radius (r -> r^2) sends r^d -> r^(d/2), so raw cosine distance HALVES the
          reported ID. Pass ``metric='cosine'`` (or ``'chordal'``) and the neighbour
          distances are converted to angles before estimation.
"""
import operator

import numpy as np

__all__ = ["twonn", "mle", "intrinsic_dimension", "intrinsic_dimension_from_dire"]


# ---------------------------------------------------------------------------
# metric conversion -- everything becomes a locally-geodesic distance
# ---------------------------------------------------------------------------
def _to_geodesic(neighbor_dist, metric):
    """Convert neighbour distances to a locally-Euclidean (geodesic) scale.

    metric:
      'euclidean' -- distances are already Euclidean; returned unchanged.
      'angular'   -- distances are already angles theta; returned unchanged.
      'sqeuclidean' -- squared Euclidean distances; -> Euclidean distance.
      'cosine'    -- distances are cosine distance 1 - cos(theta); -> theta.
      'chordal'   -- distances are chordal Euclidean on the unit sphere
                     ||a - b|| = 2 sin(theta/2); -> theta.
    """
    d = np.asarray(neighbor_dist, dtype=np.float64)
    if metric in ("euclidean", "angular"):
        return d
    if metric == "sqeuclidean":
        return np.sqrt(np.maximum(d, 0.0))
    if metric == "cosine":
        cos = np.clip(1.0 - d, -1.0, 1.0)
        return np.arccos(cos)
    if metric == "chordal":
        # ||a-b|| = 2 sin(theta/2)  =>  theta = 2 arcsin(d/2)
        return 2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))
    raise ValueError(
        f"unknown metric {metric!r}; expected one of "
        "'euclidean', 'sqeuclidean', 'angular', 'cosine', 'chordal'"
    )


# ---------------------------------------------------------------------------
# estimators -- operate on sorted-ascending neighbour distances (n, k)
# ---------------------------------------------------------------------------
def twonn(neighbor_dist, metric="euclidean", fraction=0.9):
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017).

    Parameters
    ----------
    neighbor_dist : array (n_points, >=2)
        Per point, distances to neighbours sorted ascending; only the 1st and 2nd
        columns are used.
    metric : {'euclidean','sqeuclidean','angular','cosine','chordal'}
        Scale of the input distances (see module docstring); converted to geodesic.
    fraction : float in (0, 1]
        Lower fraction of the mu-CDF used for the straight-line fit (Facco's tail
        trim discards the noisy upper tail). 0.9 is their recommended default.

    Returns
    -------
    dict with keys 'd' (the linear-fit estimate, primary), 'd_mle' (closed-form
    Pareto MLE d = N / sum log mu), and 'n_used'.
    """
    d = _to_geodesic(neighbor_dist, metric)
    if d.ndim != 2 or d.shape[1] < 2:
        raise ValueError("TwoNN needs a (n_points, >=2) distance array")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("neighbor distances must be finite and non-negative")
    r1, r2 = d[:, 0], d[:, 1]
    good = r1 > 0
    mu = r2[good] / r1[good]
    mu = mu[mu > 1.0 + 1e-12]
    mu = np.sort(mu)
    n = mu.size
    if n < 2:
        raise ValueError("not enough non-degenerate points for TwoNN")
    F = np.arange(1, n + 1) / n
    keep = F < fraction
    if not np.any(keep):
        raise ValueError("fraction leaves no points for the TwoNN fit")
    x = np.log(mu[keep])
    y = -np.log1p(-F[keep])
    d_fit = float(np.dot(x, y) / np.dot(x, x))   # least-squares slope through origin
    d_mle = float(n / np.sum(np.log(mu)))
    return {"d": d_fit, "d_mle": d_mle, "n_used": n}


def mle(neighbor_dist, k=None, metric="euclidean"):
    """Levina-Bickel MLE intrinsic dimension with MacKay-Ghahramani averaging.

    Uses the first ``k`` neighbours (default: all supplied). Per point,
    inv_m_i = (1/(k-1)) sum_{j=1}^{k-1} log(T_k / T_j); the estimate is
    1 / mean_i(inv_m_i) (averaging the inverse local slopes, the bias-reduced form).

    Returns a plain float.
    """
    d = _to_geodesic(neighbor_dist, metric)
    if d.ndim != 2:
        raise ValueError("MLE needs a two-dimensional distance array")
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("neighbor distances must be finite and non-negative")
    if k is None:
        k = d.shape[1]
    else:
        try:
            k = operator.index(k)
        except TypeError as exc:
            raise ValueError("k must be an integer") from exc
    if k < 3:
        raise ValueError("MLE needs k >= 3 neighbours")
    if k > d.shape[1]:
        raise ValueError(f"MLE requested k={k}, but only {d.shape[1]} neighbours are available")
    T = d[:, :k]
    good = np.all(T > 0, axis=1)
    if not np.any(good):
        raise ValueError("not enough non-degenerate points for MLE")
    logT = np.log(T[good])
    Tk = logT[:, k - 1:k]
    inv_m = np.sum(Tk - logT[:, :k - 1], axis=1) / (k - 1)
    return float(1.0 / np.mean(inv_m))


# ---------------------------------------------------------------------------
# convenience: estimate straight from a point cloud (CPU kNN)
# ---------------------------------------------------------------------------
def intrinsic_dimension(X, k=20, method="twonn", metric="auto", normalize=None):
    """Intrinsic dimension of a point cloud ``X`` (computes its own kNN, CPU).

    For the pipeline's "free" path that reuses DiRe's GPU graph, use
    :func:`intrinsic_dimension_from_dire` instead -- this function is the
    standalone / reference route.

    Parameters
    ----------
    X : array (n, d)
    k : neighbours to retrieve (TwoNN uses 2; MLE uses k).
    method : {'twonn', 'mle'}.
    metric : {'auto','euclidean','cosine'}.
        'auto' inspects X: if every row is unit-norm it treats the data as living on
        the sphere and uses the angular geodesic ('cosine'); otherwise 'euclidean'.
    normalize : bool or None
        If True, L2-normalise X first (and use the angular metric). None = leave as is.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a two-dimensional array with at least two rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if metric not in {"auto", "euclidean", "cosine"}:
        raise ValueError("metric must be one of 'auto', 'euclidean', or 'cosine'")
    if method not in {"twonn", "mle"}:
        raise ValueError(f"unknown method {method!r}")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be an integer") from exc
    min_k = 2 if method == "twonn" else 3
    if k < min_k:
        raise ValueError(f"{method} needs k >= {min_k} neighbours")
    if k >= X.shape[0]:
        raise ValueError(f"k={k} must be smaller than the number of samples ({X.shape[0]})")
    if normalize:
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
        metric = "cosine"
    if metric == "auto":
        unit = np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)
        metric = "cosine" if unit else "euclidean"
    # Use the requested geometry for neighbour search. On normalized data cosine and
    # chordal Euclidean have the same ordering, but cosine works correctly even when a
    # caller explicitly requests it for data that were not pre-normalized.
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    # Omitting X asks sklearn for neighbours of the fitted samples while excluding each
    # row's own index. This is robust to duplicate points, where self is not guaranteed
    # to be the first zero-distance result.
    dist, _ = nn.kneighbors()
    id_metric = metric
    if method == "twonn":
        return twonn(dist, metric=id_metric)["d"]
    if method == "mle":
        return mle(dist, k=k, metric=id_metric)


def _reducer_metric(reducer):
    """Return the distance scale stored in a fitted DiRe neighbour graph."""
    metric = getattr(reducer, "metric_spec", getattr(reducer, "metric", "cosine"))
    if metric is None:
        return "euclidean"
    if not isinstance(metric, str):
        raise ValueError("intrinsic dimension is unavailable for a callable DiRe metric")
    metric = metric.strip().lower()
    aliases = {"l2": "euclidean", "squared_euclidean": "sqeuclidean"}
    metric = aliases.get(metric, metric)
    if metric not in {"euclidean", "sqeuclidean", "angular", "cosine", "chordal"}:
        raise ValueError(
            f"intrinsic dimension is unavailable for DiRe metric {metric!r}; "
            "use cosine, euclidean, or sqeuclidean"
        )
    return metric


def intrinsic_dimension_from_dire(reducer, method="twonn", k=None):
    """Intrinsic dimension reusing a fitted DiRe reducer's existing kNN graph -- free.

    Reads ``reducer._knn_distances`` (DiRe stores cosine distances on its
    cosine-metric path) and converts to the angular geodesic before estimating, so
    the ID is not biased downward by treating cosine distance as a Euclidean radius.

    Parameters
    ----------
    reducer : a fitted dire_rapids reducer with ``_knn_distances`` populated.
    method : {'twonn', 'mle'}.
    k : neighbours for MLE (default: all available).
    """
    dist = getattr(reducer, "_knn_distances", None)
    if dist is None:
        raise ValueError(
            "reducer has no _knn_distances; fit the reducer (or call _compute_knn) "
            "before estimating intrinsic dimension"
        )
    dist = np.sort(np.asarray(dist, dtype=np.float64), axis=1)
    id_metric = _reducer_metric(reducer)
    if method == "twonn":
        return twonn(dist, metric=id_metric)["d"]
    if method == "mle":
        return mle(dist, k=k, metric=id_metric)
    raise ValueError(f"unknown method {method!r}")
