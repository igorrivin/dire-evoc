"""The shared disjoint-set / union-find (CPU)."""
import numpy as np

from dire_evoc._unionfind import UnionFind, components_from_edges


def test_union_and_find():
    uf = UnionFind(5)
    assert uf.union(0, 1) is True
    assert uf.union(1, 2) is True
    assert uf.union(0, 2) is False  # already connected
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)
    assert uf.n_components() == 3  # {0,1,2}, {3}, {4}


def test_labels_compacted_and_deterministic():
    uf = UnionFind(4)
    uf.union(2, 3)
    labels = uf.labels()
    assert set(np.unique(labels)) == {0, 1, 2}  # compacted 0..c-1
    assert labels[2] == labels[3]
    assert UnionFind(4).labels().tolist() == [0, 1, 2, 3]  # all singletons


def test_components_from_edges_transitive_chain():
    labels = components_from_edges(5, np.array([[0, 1], [1, 2], [3, 4]]))
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert np.unique(labels).size == 2


def test_components_from_edges_empty():
    labels = components_from_edges(3, np.empty((0, 2), dtype=np.int64))
    assert labels.tolist() == [0, 1, 2]


def test_partition_invariant_to_union_order():
    a = UnionFind(6)
    for i, j in [(0, 1), (1, 2), (3, 4)]:
        a.union(i, j)
    b = UnionFind(6)
    for i, j in [(4, 3), (2, 1), (1, 0)]:
        b.union(i, j)
    # same partition regardless of order/direction (labels compacted the same way)
    assert a.labels().tolist() == b.labels().tolist()
