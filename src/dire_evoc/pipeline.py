"""One-call DiRe -> EVoC pipeline: layout, clustering, intrinsic dimension, predict.

``DiReEVoC`` runs the whole integration end to end:

  1. build a DiRe layout of the data (GPU, cuVS) -- the common substrate;
  2. cluster the layout with EVoC's density machinery, using the GPU single-linkage
     tree when RAPIDS is available (``dire_evoc.hybrid``) and EVoC's CPU MST otherwise;
  3. select the operative layer by coverage-weighted silhouette on the layout;
  4. read the intrinsic dimension off DiRe's kNN graph for free;
  5. expose ``predict`` for out-of-sample points via a kNN vote in the ORIGINAL space.

Quality note carried from the regime study: density clustering is the right object only
when the data is density-separated. On continuous / high-noise manifolds the DiRe layout
itself (``.layout_``) is the deliverable, not a hard partition -- inspect
``intrinsic_dimension_`` and the noise fraction before trusting ``labels_``.
"""
import numpy as np

from .dimension import intrinsic_dimension_from_dire
from .predict import knn_vote_predict, prepare_reference

__all__ = ["DiReEVoC"]


class DiReEVoC:
    """DiRe layout + EVoC clustering, with intrinsic dimension and out-of-sample predict.

    Parameters
    ----------
    n_components : int, default=8
        Dimension of the DiRe layout that EVoC clusters.
    metric : str, default="cosine"
        Metric for DiRe's kNN graph (and the intrinsic-dimension geodesic conversion).
    min_samples : int, default=10
        EVoC / HDBSCAN* core-distance parameter. Shared by the CPU and GPU paths.
    base_min_cluster_size : int, default=5
        Finest-granularity cluster size for EVoC's base layer.
    base_n_clusters : int or None, default=None
        Optional target number of clusters for the base layer. This is a benchmark/control
        knob, not the default unsupervised mode.
    use_gpu : {"auto", True, False}, default="auto"
        Whether to build the single-linkage tree on the GPU (cuML). "auto" uses the GPU
        iff cuML imports. The DiRe layout can still use its own acceleration when this
        is false; this flag controls the EVoC-phase hybrid.
    selector : {"silhouette", "silhouette_raw", "base", "none"}, default="silhouette"
        How to pick the operative layer for ``labels_``. "silhouette" uses
        COVERAGE-WEIGHTED silhouette (mean silhouette over all points, noise scored 0) --
        the validated default; it avoids the over-segmentation that plain silhouette
        suffers when a layer discards much of the data as noise. "silhouette_raw" is the
        old non-coverage-weighted behaviour; "base" uses the finest layer; "none" the
        coarsest.
    adaptive_min_samples : bool, default=True
        If the base clustering labels more than ``noise_gate`` of the points as noise --
        the signature of a layout too tight for EVoC's default density smoothing (the
        whole layer set comes out over-fragmented) -- re-cluster at larger ``min_samples``
        and re-select by the same coverage-weighted score, adopting the result only if it
        strictly beats the base. Genuinely diffuse data (continuous manifolds) keeps its
        honest high-noise result because nothing beats it. Only active for the default
        "silhouette" selector.
    noise_gate : float, default=0.5
        Noise fraction above which the adaptive ``min_samples`` search is triggered.
    merge_fragments : bool, default=False
        If True, post-process the selected layer by unioning clusters that share strong
        reciprocal kNN evidence in DiRe's ORIGINAL-space graph (the signature of adjacent
        pieces of one continuous component). This is a heuristic with hand-tuned thresholds
        (``fragment_merge_*``) and a loose acceptance guard; it is **off by default pending
        review** and only active for the "silhouette" selector. Validated to no-op on
        well-separated data, but it can change clusterings on other inputs.
    n_neighbors_predict : int or None
        Neighbours for the out-of-sample vote; defaults to the DiRe ``n_neighbors``.
    random_state : int or None
    **dire_kwargs
        Forwarded to ``dire_rapids.create_dire`` (e.g. ``n_neighbors``, ``max_iter_layout``).

    Attributes
    ----------
    layout_ : ndarray (n, n_components)        the DiRe layout
    labels_ : ndarray (n,)                     operative-layer cluster labels (-1 = noise)
    membership_ : ndarray (n,)                 per-point membership strength of that layer
    cluster_layers_ : list of ndarray          all EVoC layers (fine -> coarse)
    intrinsic_dimension_ : float               TwoNN ID read off DiRe's kNN graph
    reducer_ : the fitted DiRe reducer
    selected_layer_ : int                      index into ``cluster_layers_`` chosen for
                                               ``labels_``
    selected_min_samples_ : int                min_samples actually used (> min_samples if
                                               the adaptive search escalated it)
    selected_base_min_cluster_size_ : int      base_min_cluster_size actually used
    """

    def __init__(self, n_components=8, metric="cosine", min_samples=10,
                 base_min_cluster_size=5, base_n_clusters=None,
                 use_gpu="auto", selector="silhouette",
                 adaptive_min_samples=True, noise_gate=0.5,
                 merge_fragments=False, fragment_merge_min_edges=3,
                 fragment_merge_edge_fraction=0.01,
                 fragment_merge_min_score_ratio=0.25,
                 gpu_cluster_min_samples=1000,
                 n_neighbors_predict=None, random_state=None, **dire_kwargs):
        self.n_components = n_components
        self.metric = metric
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.base_n_clusters = base_n_clusters
        self.use_gpu = use_gpu
        self.selector = selector
        self.adaptive_min_samples = adaptive_min_samples
        self.noise_gate = noise_gate
        self.merge_fragments = merge_fragments
        self.fragment_merge_min_edges = fragment_merge_min_edges
        self.fragment_merge_edge_fraction = fragment_merge_edge_fraction
        self.fragment_merge_min_score_ratio = fragment_merge_min_score_ratio
        self.gpu_cluster_min_samples = gpu_cluster_min_samples
        self.n_neighbors_predict = n_neighbors_predict
        self.random_state = random_state
        self.dire_kwargs = dire_kwargs

    # -- internals -----------------------------------------------------------
    def _resolve_gpu(self):
        if self.use_gpu is True or self.use_gpu is False:
            return self.use_gpu
        try:
            import cuml  # noqa: F401
            return True
        except Exception:
            return False

    def _build_layout(self, X):
        from dire_rapids import create_dire
        reducer = create_dire(
            n_components=self.n_components, metric=self.metric,
            random_state=self.random_state, **self.dire_kwargs,
        )
        layout = np.ascontiguousarray(reducer.fit_transform(X), dtype=np.float32)
        return layout, reducer

    def _cluster(self, layout, on_gpu, min_samples=None, base_min_cluster_size=None):
        ms = self.min_samples if min_samples is None else min_samples
        bmc = (
            self.base_min_cluster_size
            if base_min_cluster_size is None
            else base_min_cluster_size
        )
        if on_gpu and layout.shape[0] >= self.gpu_cluster_min_samples:
            from .hybrid import hybrid_cluster_layers
            return hybrid_cluster_layers(
                layout, min_samples=ms,
                base_min_cluster_size=bmc,
                base_n_clusters=self.base_n_clusters,
            )
        from evoc.clustering import build_cluster_layers
        return build_cluster_layers(
            layout, min_samples=ms,
            base_min_cluster_size=bmc,
            base_n_clusters=self.base_n_clusters,
        )

    def _select_layer(self, layout, layers):
        """Pick the operative layer by COVERAGE-WEIGHTED silhouette on the layout.

        Plain silhouette is computed only on clustered points, so a layer that dumps
        half the data as noise but leaves tight residual clusters scores well -- it
        systematically over-segments (e.g. picks k=51 on 20NG where k=8 is right).
        The fix: score the mean silhouette over *all* sampled points, with noise points
        contributing 0. That equals ``silhouette(clustered) * coverage`` (coverage =
        fraction non-noise), since silhouette_score is the mean of the per-point values.
        It rewards layers that are both well-separated AND explain the data. It is still
        a heuristic: the CUDA benchmark suite includes a synthetic case where the layout
        contains the right fixed-k clustering but the selector chooses too coarse a layer.
        ``selector="silhouette_raw"`` keeps the old behaviour.
        """
        return self._best_layer(layout, layers)[0]

    def _layer_score(self, Xs, lab_sample):
        """Coverage-weighted silhouette of one layer's labels on the sample.

        ``silhouette(clustered) * coverage`` = mean silhouette over all points with noise
        scored 0. Returns -inf for a degenerate layer (<2 clusters among the sample)."""
        from sklearn.metrics import silhouette_score
        mask = lab_sample >= 0
        if np.unique(lab_sample[mask]).size < 2:
            return -np.inf
        s = silhouette_score(Xs[mask], lab_sample[mask])
        if self.selector != "silhouette_raw":
            s *= mask.mean()
        return float(s)

    def _best_layer(self, layout, layers):
        """Return (index, score) of the operative layer under the configured selector.

        ``score`` is the coverage-weighted silhouette of the chosen layer, used both to
        pick within a layer set and to compare candidate sets across smoothing params."""
        rng = np.random.default_rng(self.random_state)
        n = layout.shape[0]
        sample = (rng.choice(n, 10_000, replace=False) if n > 10_000 else np.arange(n))
        Xs = layout[sample]
        if self.selector == "base" or len(layers) == 1:
            i = 0
        elif self.selector == "none":
            i = len(layers) - 1
        else:
            scores = [self._layer_score(Xs, lab[sample]) for lab in layers]
            i = int(np.argmax(scores))
            return i, scores[i]
        return i, self._layer_score(Xs, layers[i][sample])

    def _adaptive_smoothing_candidates(self):
        """Smoothing candidates for tight layouts that over-fragment at defaults."""
        min_samples_values = {
            self.min_samples,
            int(round(self.min_samples * 2.5)),
            int(round(self.min_samples * 5.0)),
        }
        base_size_values = {
            self.base_min_cluster_size,
            int(round(self.base_min_cluster_size * 5.0)),
            int(round(self.base_min_cluster_size * 10.0)),
        }
        candidates = []
        seen = set()
        for ms in sorted(value for value in min_samples_values if value > 0):
            for bmc in sorted(value for value in base_size_values if value > 0):
                key = (ms, bmc)
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
        return candidates

    def _maybe_adapt_smoothing(
        self,
        layout,
        on_gpu,
        layers,
        membership_layers,
        persistence,
        selected_layer,
        score,
    ):
        """Escalate EVoC smoothing if the selected layer is mostly noise.

        Tight DiRe layouts can make EVoC's default density smoothing under-shoot, so the
        generated hierarchy contains only fragmented high-noise layers even when the
        fixed-k substrate is excellent. Search a small grid over ``min_samples`` and
        ``base_min_cluster_size`` and keep the candidate with the best unsupervised
        coverage-weighted score. Diffuse data keeps its high-noise result when no
        candidate improves the score.
        """
        selected_min_samples = self.min_samples
        selected_base_min_cluster_size = self.base_min_cluster_size
        if not (
            self.adaptive_min_samples
            and self.selector == "silhouette"
            and (layers[0] < 0).mean() > self.noise_gate
        ):
            return (
                score,
                selected_layer,
                layers,
                membership_layers,
                persistence,
                selected_min_samples,
                selected_base_min_cluster_size,
            )

        best = (
            score,
            selected_layer,
            layers,
            membership_layers,
            persistence,
            selected_min_samples,
            selected_base_min_cluster_size,
        )
        for ms, bmc in self._adaptive_smoothing_candidates():
            if ms == self.min_samples and bmc == self.base_min_cluster_size:
                continue
            try:
                ls, lm, lp = self._cluster(
                    layout,
                    on_gpu,
                    min_samples=ms,
                    base_min_cluster_size=bmc,
                )
            except Exception:
                continue
            si, sc = self._best_layer(layout, ls)
            if sc > best[0]:
                best = (sc, si, ls, lm, lp, ms, bmc)
        return best

    def _fragment_merge_labels(self, labels, knn_indices):
        """Merge selected-layer fragments using DiRe's original-space kNN graph.

        This is deliberately graph-local rather than a generic penalty for high ``k``:
        many separated blobs in low intrinsic dimension should stay separate. We only
        union two selected clusters when they have enough *reciprocal* cross-neighbour
        evidence in the existing kNN graph, the signature of adjacent pieces of one
        continuous component.
        """
        labels = np.asarray(labels)
        if knn_indices is None:
            return labels, {"merged": False, "reason": "missing_knn"}
        knn_indices = np.asarray(knn_indices)
        if knn_indices.ndim != 2 or knn_indices.shape[0] != labels.shape[0]:
            return labels, {"merged": False, "reason": "bad_knn_shape"}

        active = labels >= 0
        unique = np.unique(labels[active])
        if unique.size < 3:
            return labels, {"merged": False, "reason": "too_few_clusters"}

        label_to_pos = {int(label): i for i, label in enumerate(unique)}
        pos_to_label = {i: int(label) for label, i in label_to_pos.items()}
        sizes = np.array([np.sum(labels == label) for label in unique], dtype=np.int64)
        directed = {}
        noise_bridges = {}
        for i, lab in enumerate(labels):
            if lab < 0:
                neighbour_labels = []
                for j in knn_indices[i]:
                    if j < 0 or j >= labels.shape[0] or j == i:
                        continue
                    other = labels[int(j)]
                    if other >= 0:
                        neighbour_labels.append(label_to_pos[int(other)])
                for offset, a in enumerate(sorted(set(neighbour_labels))):
                    for b in sorted(set(neighbour_labels))[offset + 1:]:
                        key = (a, b) if a < b else (b, a)
                        noise_bridges[key] = noise_bridges.get(key, 0) + 1
                continue
            a = label_to_pos[int(lab)]
            for j in knn_indices[i]:
                if j < 0 or j >= labels.shape[0] or j == i:
                    continue
                other = labels[int(j)]
                if other < 0 or other == lab:
                    continue
                b = label_to_pos[int(other)]
                directed[(a, b)] = directed.get((a, b), 0) + 1

        parent = np.arange(unique.size)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        merged_edges = 0
        for (a, b), count_ab in list(directed.items()):
            if a >= b:
                continue
            reciprocal_edges = min(count_ab, directed.get((b, a), 0))
            if reciprocal_edges < self.fragment_merge_min_edges:
                continue
            smaller = max(1, int(min(sizes[a], sizes[b])))
            if reciprocal_edges / smaller >= self.fragment_merge_edge_fraction:
                union(a, b)
                merged_edges += 1

        bridge_edges = 0
        for (a, b), bridge_count in noise_bridges.items():
            if bridge_count >= self.fragment_merge_min_edges:
                union(a, b)
                bridge_edges += 1

        roots = np.array([find(i) for i in range(unique.size)])
        if np.unique(roots).size == unique.size:
            return labels, {
                "merged": False,
                "reason": "no_strong_adjacencies",
                "candidate_edges": int(len(directed)),
                "candidate_bridges": int(len(noise_bridges)),
            }

        root_to_new = {
            root: new_label
            for new_label, root in enumerate(sorted(np.unique(roots)))
        }
        merged_labels = labels.copy()
        for old_pos, old_label in pos_to_label.items():
            merged_labels[labels == old_label] = root_to_new[find(old_pos)]
        return merged_labels, {
            "merged": True,
            "merged_edges": int(merged_edges),
            "bridge_edges": int(bridge_edges),
            "clusters_before": int(unique.size),
            "clusters_after": int(np.unique(merged_labels[merged_labels >= 0]).size),
        }

    def _maybe_merge_fragments(
        self,
        layout,
        layers,
        membership_layers,
        persistence,
        selected_layer,
        reducer,
    ):
        if not (self.merge_fragments and self.selector == "silhouette"):
            return layers, membership_layers, persistence, selected_layer, {
                "merged": False,
                "reason": "disabled",
            }

        merged_labels, info = self._fragment_merge_labels(
            layers[selected_layer],
            getattr(reducer, "_knn_indices", None),
        )
        if not info.get("merged"):
            return layers, membership_layers, persistence, selected_layer, info

        base_score = self._layer_score(layout, layers[selected_layer])
        merged_score = self._layer_score(layout, merged_labels)
        clusters_after = int(np.unique(merged_labels[merged_labels >= 0]).size)
        if clusters_after < 2:
            info = dict(info)
            info.update(
                {
                    "merged": False,
                    "reason": "collapsed_to_one_cluster",
                    "base_score": float(base_score),
                    "merged_score": float(merged_score),
                }
            )
            return layers, membership_layers, persistence, selected_layer, info
        if (
            np.isfinite(base_score)
            and base_score > 0
            and (
                not np.isfinite(merged_score)
                or merged_score < base_score * self.fragment_merge_min_score_ratio
            )
        ):
            info = dict(info)
            info.update(
                {
                    "merged": False,
                    "reason": "score_loss_too_large",
                    "base_score": float(base_score),
                    "merged_score": float(merged_score),
                }
            )
            return layers, membership_layers, persistence, selected_layer, info

        info = dict(info)
        info.update({"base_score": float(base_score), "merged_score": float(merged_score)})
        layers = list(layers) + [merged_labels]
        membership_layers = list(membership_layers) + [membership_layers[selected_layer]]
        persistence = list(persistence) + [np.nan]
        return layers, membership_layers, persistence, len(layers) - 1, info

    # -- API -----------------------------------------------------------------
    def fit(self, X, y=None):
        X = np.ascontiguousarray(X)
        on_gpu = self._resolve_gpu()
        self.layout_, self.reducer_ = self._build_layout(X)
        layers, membership_layers, persistence = self._cluster(self.layout_, on_gpu)
        sel, score = self._best_layer(self.layout_, layers)
        (
            score,
            sel,
            layers,
            membership_layers,
            persistence,
            self.selected_min_samples_,
            self.selected_base_min_cluster_size_,
        ) = self._maybe_adapt_smoothing(
            self.layout_,
            on_gpu,
            layers,
            membership_layers,
            persistence,
            sel,
            score,
        )
        (
            layers,
            membership_layers,
            persistence,
            sel,
            self.fragment_merge_,
        ) = self._maybe_merge_fragments(
            self.layout_,
            layers,
            membership_layers,
            persistence,
            sel,
            self.reducer_,
        )

        self.cluster_layers_ = layers
        self.membership_strength_layers_ = membership_layers
        self.persistence_scores_ = persistence
        self.selected_layer_ = sel
        self.labels_ = layers[sel]
        self.membership_ = membership_layers[sel]

        # free intrinsic dimension off DiRe's existing kNN graph
        try:
            self.intrinsic_dimension_ = intrinsic_dimension_from_dire(
                self.reducer_, method="twonn")
        except Exception:
            self.intrinsic_dimension_ = None

        # retain a metric-prepared reference for out-of-sample predict
        if X.dtype in (np.float32, np.float64):
            self._reference_ = prepare_reference(X)
            self._reference_labels_ = self.labels_
        else:
            self._reference_ = None
        self._n_neighbors_ = (
            self.n_neighbors_predict
            if self.n_neighbors_predict is not None
            else getattr(self.reducer_, "n_neighbors", 15)
        )
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def predict(self, X, n_neighbors=None, return_membership=False):
        """Assign cluster labels to new points by a kNN vote over the training points in
        the original embedding space (noise competes as a label)."""
        if getattr(self, "_reference_", None) is None:
            raise NotImplementedError(
                "predict supports float embeddings (cosine metric); the model was fit "
                "on non-float data."
            )
        query = prepare_reference(np.ascontiguousarray(X))
        if query.shape[1] != self._reference_.shape[1]:
            raise ValueError(
                f"X has {query.shape[1]} features, but the model was fit on "
                f"{self._reference_.shape[1]}."
            )
        k = self._n_neighbors_ if n_neighbors is None else n_neighbors
        labels, membership = knn_vote_predict(
            query, self._reference_, self._reference_labels_, k)
        return (labels, membership) if return_membership else labels
