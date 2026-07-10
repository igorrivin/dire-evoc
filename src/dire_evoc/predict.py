"""Out-of-sample cluster assignment by distance-weighted kNN vote.

Carried in dire-evoc (rather than imported from evoc) so the pipeline does not block
on an upstream evoc release. A new point inherits the distance-weighted majority label
of its nearest *training* points in the same geometry used to fit DiRe; noise (label -1)
competes as an ordinary label. This sidesteps embedding a new point into the transductive
DiRe layout: the vote happens in the original embedding space.
"""

import operator

import numpy as np

__all__ = ["prepare_reference", "knn_vote_predict"]

_COSINE_METRICS = {"cosine"}
_EUCLIDEAN_METRICS = {"euclidean", "l2", "sqeuclidean", "squared_euclidean"}


def _normalize_metric(metric):
    if not isinstance(metric, str):
        raise ValueError("metric must be a string")
    metric = metric.strip().lower()
    if metric in _COSINE_METRICS:
        return "cosine"
    if metric in _EUCLIDEAN_METRICS - {"sqeuclidean", "squared_euclidean"}:
        return "euclidean"
    if metric in {"sqeuclidean", "squared_euclidean"}:
        return "sqeuclidean"
    raise ValueError(
        f"unsupported prediction metric {metric!r}; expected cosine, euclidean, "
        "or sqeuclidean"
    )


def prepare_reference(data, metric="cosine"):
    """Prepare float embeddings for exact kNN voting.

    Cosine-like metrics produce L2-normalized vectors so cosine similarity is a single
    BLAS matrix multiplication. Euclidean metrics preserve vector magnitudes. In all
    cases the result is a finite, C-contiguous float32 copy.
    """
    metric = _normalize_metric(metric)
    data = np.array(data, dtype=np.float32, order="C", copy=True)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")
    if metric == "cosine":
        norms = np.sqrt(np.einsum("ij,ij->i", data, data))
        norms[norms == 0.0] = 1.0
        data /= norms[:, np.newaxis]
    return data


def _validate_vote_inputs(query, reference, reference_labels, n_neighbors, chunk_size):
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    labels = np.asarray(reference_labels)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("query and reference must be two-dimensional arrays")
    if query.shape[1] != reference.shape[1]:
        raise ValueError(
            f"query has {query.shape[1]} features, but reference has {reference.shape[1]}"
        )
    if reference.shape[0] == 0:
        raise ValueError("reference must contain at least one point")
    if labels.ndim != 1 or labels.shape[0] != reference.shape[0]:
        raise ValueError("reference_labels must be one-dimensional and aligned with reference")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("reference_labels must contain integer labels")
    if np.any(labels < -1):
        raise ValueError("reference_labels may use -1 for noise but no smaller values")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(reference)):
        raise ValueError("query and reference must contain only finite values")
    if isinstance(n_neighbors, (bool, np.bool_)):
        raise ValueError("n_neighbors must be a positive integer")
    try:
        k_requested = operator.index(n_neighbors)
    except TypeError as exc:
        raise ValueError("n_neighbors must be a positive integer") from exc
    if k_requested < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if chunk_size is not None:
        if isinstance(chunk_size, (bool, np.bool_)):
            raise ValueError("chunk_size must be a positive integer")
        try:
            chunk_size_int = operator.index(chunk_size)
        except TypeError as exc:
            raise ValueError("chunk_size must be a positive integer") from exc
        if chunk_size_int < 1:
            raise ValueError("chunk_size must be a positive integer")
        chunk_size = chunk_size_int
    return query, reference, labels.astype(np.int64, copy=False), k_requested, chunk_size


def knn_vote_predict(
    query,
    reference,
    reference_labels,
    n_neighbors,
    chunk_size=None,
    metric="cosine",
):
    """Assign query points by a distance-weighted vote over training points.

    The search is exact and chunked, using dense matrix multiplication. ``query`` and
    ``reference`` should be prepared with :func:`prepare_reference` using the same
    ``metric``. Arbitrary non-negative cluster ids are supported without allocating a
    dense array up to the largest id; noise ``-1`` participates in the vote normally.

    Returns
    -------
    labels : int64 array, shape (n_query,)
        Winning cluster label for each query point; -1 denotes noise.
    membership : float64 array, shape (n_query,)
        Winning label's share of total neighbour weight, in (0, 1].
    """
    metric = _normalize_metric(metric)
    query, reference, reference_labels, n_neighbors, chunk_size = _validate_vote_inputs(
        query, reference, reference_labels, n_neighbors, chunk_size
    )
    n_query = query.shape[0]
    n_reference = reference.shape[0]
    k = min(n_neighbors, n_reference)

    if chunk_size is None:
        chunk_size = int(np.clip(256_000_000 // (n_reference * 4), 64, 8192))

    unique_labels, encoded_labels = np.unique(reference_labels, return_inverse=True)
    labels = np.empty(n_query, dtype=np.int64)
    membership = np.empty(n_query, dtype=np.float64)
    reference_norm_sq = None
    if metric != "cosine":
        reference_norm_sq = np.einsum("ij,ij->i", reference, reference)

    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        query_chunk = query[start:stop]
        products = query_chunk @ reference.T
        if metric == "cosine":
            topk = np.argpartition(products, n_reference - k, axis=1)[:, n_reference - k :]
            rows = np.arange(stop - start)[:, None]
            distances = np.maximum(1.0 - products[rows, topk], 0.0)
        else:
            query_norm_sq = np.einsum("ij,ij->i", query_chunk, query_chunk)[:, None]
            distances_sq = np.maximum(query_norm_sq + reference_norm_sq - 2.0 * products, 0.0)
            topk = np.argpartition(distances_sq, k - 1, axis=1)[:, :k]
            rows = np.arange(stop - start)[:, None]
            distances = distances_sq[rows, topk]
            if metric == "euclidean":
                distances = np.sqrt(distances)

        weights = 1.0 / (distances + 1e-6)
        topk_labels = encoded_labels[topk]
        votes = np.zeros((stop - start, unique_labels.size), dtype=np.float64)
        vote_rows = np.arange(stop - start)
        for j in range(k):
            np.add.at(votes, (vote_rows, topk_labels[:, j]), weights[:, j])

        winning = np.argmax(votes, axis=1)
        labels[start:stop] = unique_labels[winning]
        total = votes.sum(axis=1)
        membership[start:stop] = votes[vote_rows, winning] / total

    return labels, membership
