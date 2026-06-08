"""One disjoint-set (union-find) implementation for the whole package.

Path halving + union by size, so a sequence of m operations over n elements costs
O(m * alpha(n)) -- effectively linear. This replaces the inline ``find``/``union`` closures
that used to be duplicated across the pipeline, and backs the sparse consensus voting.

It works directly on an edge stream, so callers never materialize an n x n matrix; the
connected components are read off at the end with :meth:`labels`.
"""
from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return int(x)

    def union(self, a, b):
        """Merge the sets of ``a`` and ``b``; return True if they were separate."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra  # smaller tree under larger
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def n_components(self):
        return len({self.find(i) for i in range(self.parent.shape[0])})

    def labels(self):
        """Component labels compacted to 0..c-1 (deterministic via sorted roots)."""
        n = self.parent.shape[0]
        roots = np.fromiter((self.find(i) for i in range(n)), dtype=np.int64, count=n)
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int64, copy=False)


def components_from_edges(n_nodes, edges):
    """Connected-component labels for ``n_nodes`` given undirected ``edges`` (m, 2).

    Empty ``edges`` yields all singletons (0..n_nodes-1).
    """
    uf = UnionFind(n_nodes)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    for a, b in edges:
        uf.union(int(a), int(b))
    return uf.labels()
