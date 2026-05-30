"""Out-of-sample cluster assignment by distance-weighted kNN vote.

Carried in dire-evoc (rather than imported from evoc) so the pipeline does not block
on an upstream evoc release -- the same logic also lives in ``evoc.EVoC.predict`` as a
goodwill contribution. A new point inherits the distance-weighted majority label of its
nearest *training* points under the cosine metric the clustering was built on; noise
(label -1) competes as an ordinary label. This sidesteps embedding a new point into the
transductive DiRe layout: the vote happens in the original embedding space.
"""
import numpy as np

__all__ = ["prepare_reference", "knn_vote_predict"]


def prepare_reference(data):
    """L2-normalise float data to unit vectors (cosine -> inner product).

    Returns a C-contiguous float32 copy so a single BLAS matmul serves the vote.
    """
    data = np.array(data, dtype=np.float32, order="C", copy=True)
    norms = np.sqrt(np.einsum("ij,ij->i", data, data))
    norms[norms == 0.0] = 1.0
    data /= norms[:, np.newaxis]
    return data


def knn_vote_predict(query, reference, reference_labels, n_neighbors, chunk_size=None):
    """Assign each ``query`` point a label by a distance-weighted vote over its nearest
    ``reference`` (training) points, under cosine distance on unit-normalised data.

    Exact (chunked dense inner-product matmul). For very large reference sets an ANN
    index (pynndescent / cuVS) is the drop-in replacement for the matmul; the vote
    logic is unchanged.

    Parameters
    ----------
    query, reference : float32 arrays, unit-normalised (see :func:`prepare_reference`).
    reference_labels : int array (n_reference,), cluster label per training point.
    n_neighbors : int, neighbours to vote over.

    Returns
    -------
    labels : int array (n_query,); -1 denotes noise.
    membership : float array (n_query,), winning label's share of neighbour weight, (0, 1].
    """
    n_query = query.shape[0]
    n_reference = reference.shape[0]
    k = int(min(n_neighbors, n_reference))

    if chunk_size is None:
        chunk_size = int(np.clip(256_000_000 // (n_reference * 4), 64, 8192))

    n_label_cols = int(reference_labels.max()) + 2 if n_reference else 1
    labels = np.empty(n_query, dtype=np.int64)
    membership = np.empty(n_query, dtype=np.float64)

    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        sims = query[start:stop] @ reference.T              # cosine sim (unit vecs)
        topk = np.argpartition(sims, n_reference - k, axis=1)[:, n_reference - k:]
        rows = np.arange(stop - start)[:, None]
        topk_sims = sims[rows, topk]
        topk_labels = reference_labels[topk]

        weights = 1.0 / (1.0 - topk_sims + 1e-6)            # inverse cosine distance
        votes = np.zeros((stop - start, n_label_cols), dtype=np.float64)
        cols = topk_labels + 1                              # noise(-1) -> column 0
        for j in range(k):
            np.add.at(votes, (np.arange(stop - start), cols[:, j]), weights[:, j])

        winning = np.argmax(votes, axis=1)
        labels[start:stop] = winning - 1
        total = votes.sum(axis=1)
        membership[start:stop] = votes[np.arange(stop - start), winning] / total

    return labels, membership
