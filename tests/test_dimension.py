"""Correctness of the intrinsic-dimension estimators on KNOWN-dimension data.

These run on CPU with no GPU/cuml/dire dependency. The headline checks are:
 - a d-dimensional uniform cube embedded in higher ambient space is recovered as ~d;
 - the metric matters: applying the estimator to cosine *distance* on sphere data
   reports ~2x the true dimension, while the angular conversion recovers it (this is
   the bug the dire-evoc feature exists to avoid).
"""
import numpy as np
import pytest

from dire_evoc.dimension import (
    _to_geodesic,
    intrinsic_dimension,
    intrinsic_dimension_from_dire,
    mle,
    twonn,
)


def _cube(n, d, ambient, seed=0):
    """n points uniform in [0,1]^d, zero-padded into R^ambient (true ID = d)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, ambient), dtype=np.float64)
    X[:, :d] = rng.random((n, d))
    return X


def _neighbor_dist(X, k):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1:]


@pytest.mark.parametrize("d", [3, 5, 8])
def test_twonn_recovers_cube_dimension(d):
    X = _cube(8000, d, ambient=d + 10)
    dist = _neighbor_dist(X, 2)
    est = twonn(dist)["d"]
    # uniform-cube boundary effects bias TwoNN slightly low; allow a generous band
    # that still excludes the 2x-metric-bug value (which would be ~2d).
    assert d - 1.2 < est < d + 1.0, f"TwoNN d={d} got {est:.2f}"


@pytest.mark.parametrize("d", [3, 5, 8])
def test_mle_recovers_cube_dimension(d):
    X = _cube(8000, d, ambient=d + 10)
    dist = _neighbor_dist(X, 20)
    est = mle(dist, k=20)
    assert d - 1.2 < est < d + 1.2, f"MLE d={d} got {est:.2f}"


def test_metric_halving_is_real_and_corrected():
    """On the sphere, cosine-distance ~ theta^2/2 is the SQUARE of the geodesic radius,
    so a ratio-based estimator reads off d/2 (halves the ID); the angular conversion
    recovers d. (Squaring the radius r->r^2 sends volume ~ r^d to ~ r^(d/2).)"""
    rng = np.random.default_rng(1)
    d = 5
    # a d-dim patch mapped onto the unit sphere in R^(d+1) via normalization
    X = np.zeros((8000, d + 6))
    X[:, :d] = rng.normal(size=(8000, d)) * 0.15
    X[:, d] = 1.0                                    # push onto a cap, then normalize
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    euc, _ = nn.kneighbors(X)            # chordal Euclidean on the sphere
    euc = euc[:, 1:]

    id_chordal = twonn(euc, metric="chordal")["d"]   # correct geodesic
    # mimic feeding cosine distance (1 - cos) = e^2/2 for unit vectors
    cosine_dist = euc ** 2 / 2.0
    id_cosine = twonn(cosine_dist, metric="euclidean")["d"]   # WRONG: treats as flat

    assert abs(id_chordal - d) < 1.5, f"chordal ID {id_chordal:.2f} should be ~{d}"
    # treating cosine-distance as flat-Euclidean HALVES the ID (squared radius)
    assert id_cosine < 0.65 * id_chordal, (
        f"expected ~0.5x halving, got chordal={id_chordal:.2f} cosine={id_cosine:.2f}"
    )
    # and the conversion inside _to_geodesic('cosine') undoes it
    id_fixed = twonn(cosine_dist, metric="cosine")["d"]
    assert abs(id_fixed - id_chordal) < 0.6


def test_intrinsic_dimension_convenience():
    X = _cube(6000, 6, ambient=16)
    est = intrinsic_dimension(X, k=20, method="twonn")
    assert 4.8 < est < 7.0


def test_to_geodesic_conversions():
    # cosine distance of two unit vectors at 60 degrees: 1 - cos(60) = 0.5 -> theta=pi/3
    theta = _to_geodesic(np.array([[0.5]]), "cosine")[0, 0]
    assert abs(theta - np.pi / 3) < 1e-9
    # chordal at 60 deg: 2 sin(30) = 1.0 -> theta = pi/3
    theta2 = _to_geodesic(np.array([[1.0]]), "chordal")[0, 0]
    assert abs(theta2 - np.pi / 3) < 1e-9


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        _to_geodesic(np.array([[1.0, 2.0]]), "haversine")


def test_intrinsic_dimension_from_dire_uses_metric_spec():
    distances = np.array(
        [[0.10, 0.21], [0.12, 0.20], [0.16, 0.25], [0.20, 0.36], [0.25, 0.48]]
    )

    class Reducer:
        _knn_distances = distances
        metric = "cosine"  # legacy attribute must not override current DiRe metadata
        metric_spec = "euclidean"

    expected = twonn(distances, metric="euclidean")["d"]
    assert intrinsic_dimension_from_dire(Reducer()) == pytest.approx(expected)


def test_intrinsic_dimension_from_dire_handles_squared_euclidean():
    distances = np.array(
        [[0.01, 0.04], [0.04, 0.09], [0.09, 0.25], [0.16, 0.36], [0.25, 0.64]]
    )

    class Reducer:
        _knn_distances = distances
        metric_spec = "sqeuclidean"

    expected = twonn(np.sqrt(distances), metric="euclidean")["d"]
    assert intrinsic_dimension_from_dire(Reducer()) == pytest.approx(expected)


def test_explicit_cosine_uses_cosine_neighbor_search_for_unnormalized_data():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(300, 6)) * rng.lognormal(size=(300, 1))

    from sklearn.neighbors import NearestNeighbors

    dist, _ = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X).kneighbors(X)
    expected = twonn(dist[:, 1:], metric="cosine")["d"]
    assert intrinsic_dimension(X, k=5, metric="cosine") == pytest.approx(expected)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_twonn_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        twonn(np.array([[1.0, 2.0], [1.0, 3.0]]), fraction=fraction)
