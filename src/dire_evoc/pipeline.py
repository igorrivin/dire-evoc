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

from ._unionfind import components_from_edges
from .consensus import consensus_labels
from .dimension import intrinsic_dimension_from_dire
from .predict import knn_vote_predict, prepare_reference
from .selectors import best_layer as select_best_layer
from .selectors import (
    coverage_silhouette_score,
    guarded_graph_component_decision,
)

__all__ = ["DiReEVoC"]


class DiReEVoC:
    """DiRe layout + EVoC clustering, with intrinsic dimension and out-of-sample predict.

    Parameters
    ----------
    n_components : int, default=8
        Dimension of the DiRe layout that EVoC clusters.
    metric : str, default="cosine"
        Metric for DiRe's kNN graph, intrinsic-dimension conversion, and out-of-sample
        voting. Prediction supports cosine, euclidean, and sqeuclidean.
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
    selector : {"silhouette", "auto", "auto_ensemble", "silhouette_raw", "base", "none"}, default="silhouette"
        How to pick the operative layer for ``labels_``. "silhouette" uses
        COVERAGE-WEIGHTED silhouette (mean silhouette over all points, noise scored 0) --
        the validated default; it avoids the over-segmentation that plain silhouette
        suffers when a layer discards much of the data as noise. "silhouette_raw" is the
        old non-coverage-weighted behaviour. "auto" uses the same silhouette fallback
        plus guarded stable graph-component overrides. "auto_ensemble" additionally
        records a noisy-or confidence combination and requires graph confidence to beat
        silhouette confidence. "base" uses the finest layer; "none" the coarsest.
    adaptive_min_samples : bool, default=True
        If the base clustering labels more than ``noise_gate`` of the points as noise --
        the signature of a layout too tight for EVoC's default density smoothing (the
        whole layer set comes out over-fragmented) -- re-cluster at larger ``min_samples``
        and re-select by the same coverage-weighted score, adopting the result only if it
        strictly beats the base. Genuinely diffuse data (continuous manifolds) keeps its
        honest high-noise result because nothing beats it. Active for the silhouette-based
        selectors: "silhouette", "auto", and "auto_ensemble".
    noise_gate : float, default=0.5
        Noise fraction above which the adaptive ``min_samples`` search is triggered.
    graph_component_candidates : bool, default=True
        If DiRe's stored kNN graph has stable connected components across a small
        neighbour-count sweep, append that labelling as a candidate auto layer.
    graph_component_min_stability : float, default=0.95
        Adjusted-Rand agreement required between adjacent kNN-component labellings.
    graph_component_k_radius : int, default=4
        Sweep radius around ``ceil(intrinsic_dimension) + 1`` for the graph-component
        candidate.
    graph_component_min_cluster_ratio : float, default=1.5
        The EVoC-selected layer must have at least this many clusters per graph component
        before the graph candidate overrides it, unless ``base_n_clusters`` is matched.
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
    graph_component_candidate_ : dict          diagnostics for the graph-component gate
    """

    def __init__(self, n_components=8, metric="cosine", min_samples=10,
                 base_min_cluster_size=5, base_n_clusters=None,
                 use_gpu="auto", selector="silhouette",
                 adaptive_min_samples=True, noise_gate=0.5,
                 graph_component_candidates=True,
                 graph_component_min_stability=0.95,
                 graph_component_k_radius=4,
                 graph_component_min_cluster_ratio=1.5,
                 gpu_cluster_min_samples=1000,
                 restarts=1, restart_combine="consensus",
                 restart_threshold=0.5, restart_min_cluster_size=1,
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
        self.graph_component_candidates = graph_component_candidates
        self.graph_component_min_stability = graph_component_min_stability
        self.graph_component_k_radius = graph_component_k_radius
        self.graph_component_min_cluster_ratio = graph_component_min_cluster_ratio
        self.gpu_cluster_min_samples = gpu_cluster_min_samples
        self.restarts = restarts
        self.restart_combine = restart_combine
        self.restart_threshold = restart_threshold
        self.restart_min_cluster_size = restart_min_cluster_size
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

    def _build_layout(self, X, seed=None):
        from dire_rapids import create_dire
        reducer = create_dire(
            n_components=self.n_components, metric=self.metric,
            random_state=self.random_state if seed is None else seed,
            **self.dire_kwargs,
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
        return coverage_silhouette_score(
            Xs,
            lab_sample,
            raw=self.selector == "silhouette_raw",
        )

    def _best_layer(self, layout, layers):
        """Return (index, score) of the operative layer under the configured selector.

        ``score`` is the coverage-weighted silhouette of the chosen layer, used both to
        pick within a layer set and to compare candidate sets across smoothing params."""
        i, score, _ = select_best_layer(
            layout,
            layers,
            selector=self.selector,
            random_state=self.random_state,
        )
        return i, score

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

    def _knn_component_labels(self, knn_indices, k):
        """Connected components of the undirected first-k neighbour graph."""
        knn_indices = np.asarray(knn_indices)
        n = knn_indices.shape[0]
        k = int(min(max(1, k), knn_indices.shape[1]))
        src = np.repeat(np.arange(n), k)
        dst = knn_indices[:, :k].reshape(-1)
        valid = (dst >= 0) & (dst < n) & (dst != src)
        edges = np.stack([src[valid], dst[valid]], axis=1)
        return components_from_edges(n, edges)

    def _graph_component_candidate(self, reducer, min_component_size):
        """Return a stable small-k graph-component labelling, if one exists.

        This targets the case where the DiRe neighbour graph has already separated the
        data before EVoC's hierarchy/selector sees the right layer. It is deliberately
        gated by adjacent-k stability and component size so a connected manifold is not
        split just because one small graph happened to fragment.
        """
        if not self.graph_component_candidates:
            return None, {"enabled": False}

        knn_indices = getattr(reducer, "_knn_indices", None)
        if knn_indices is None:
            return None, {"enabled": True, "accepted": False, "reason": "missing_knn"}
        knn_indices = np.asarray(knn_indices)
        if knn_indices.ndim != 2 or knn_indices.shape[0] == 0 or knn_indices.shape[1] == 0:
            return None, {"enabled": True, "accepted": False, "reason": "bad_knn_shape"}

        from sklearn.metrics import adjusted_rand_score

        n = knn_indices.shape[0]
        max_k = knn_indices.shape[1]
        if hasattr(self, "intrinsic_dimension_"):
            intrinsic_dim = self.intrinsic_dimension_
        else:
            try:
                intrinsic_dim = intrinsic_dimension_from_dire(reducer, method="twonn")
            except Exception:
                intrinsic_dim = None
        center = 3 if intrinsic_dim is None else int(np.ceil(intrinsic_dim)) + 1
        radius = int(max(0, self.graph_component_k_radius))
        k_values = [
            k for k in range(center - radius, center + radius + 1)
            if 1 <= k <= max_k
        ]
        if not k_values:
            k_values = [min(max(1, center), max_k)]

        records = []
        by_k = {}
        for k in sorted(set(k_values)):
            labels = self._knn_component_labels(knn_indices, k)
            counts = np.bincount(labels)
            component_count = int(counts.size)
            min_size = int(counts.min()) if counts.size else 0
            large_count = int(np.sum(counts >= min_component_size))
            valid = (
                2 <= component_count
                and component_count < n
                and min_size >= min_component_size
            )
            by_k[k] = labels
            records.append(
                {
                    "k": int(k),
                    "components": component_count,
                    "min_size": min_size,
                    "large_components": large_count,
                    "valid": bool(valid),
                }
            )

        best = None
        for left, right in zip(records, records[1:]):
            if not (left["valid"] and right["valid"]):
                continue
            stability = float(
                adjusted_rand_score(by_k[left["k"]], by_k[right["k"]])
            )
            if stability < self.graph_component_min_stability:
                continue
            labels = by_k[left["k"]]
            score_key = (
                stability,
                -abs(left["k"] - center),
                -left["components"],
            )
            if best is None or score_key > best[0]:
                best = (
                    score_key,
                    labels,
                    {
                        "enabled": True,
                        "accepted": True,
                        "k": int(left["k"]),
                        "next_k": int(right["k"]),
                        "stability": stability,
                        "components": int(left["components"]),
                        "min_size": int(left["min_size"]),
                        "intrinsic_dimension": None
                        if intrinsic_dim is None
                        else float(intrinsic_dim),
                        "records": records,
                    },
                )

        if best is None:
            return None, {
                "enabled": True,
                "accepted": False,
                "reason": "no_stable_large_components",
                "intrinsic_dimension": None
                if intrinsic_dim is None
                else float(intrinsic_dim),
                "records": records,
            }
        return best[1], best[2]

    def _maybe_add_graph_component_layer(
        self,
        reducer,
        layers,
        membership_layers,
        persistence,
    ):
        labels, info = self._graph_component_candidate(
            reducer,
            min_component_size=max(2, int(self.base_min_cluster_size)),
        )
        self.graph_component_candidate_ = info
        self.graph_component_layer_ = None
        if labels is None:
            return layers, membership_layers, persistence
        layers = list(layers) + [labels]
        membership_layers = list(membership_layers) + [
            np.ones(labels.shape[0], dtype=np.float32)
        ]
        persistence = list(persistence) + [float(info["stability"])]
        self.graph_component_layer_ = len(layers) - 1
        return layers, membership_layers, persistence

    def _should_select_graph_component_layer(
        self,
        evoc_labels,
        graph_labels,
        silhouette_score=None,
    ):
        """Choose the graph candidate only when it fixes clear EVoC fragmentation."""
        decision = guarded_graph_component_decision(
            evoc_labels=evoc_labels,
            graph_labels=graph_labels,
            graph_info=getattr(self, "graph_component_candidate_", {}) or {},
            base_n_clusters=self.base_n_clusters,
            min_cluster_ratio=self.graph_component_min_cluster_ratio,
            policy=(
                "auto_ensemble"
                if self.selector == "auto_ensemble"
                else "guarded_override"
            ),
            silhouette_score=silhouette_score,
        )
        return decision.selected, decision.diagnostics

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
            and self.selector in {"silhouette", "auto", "auto_ensemble"}
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

    # -- API -----------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the pipeline.

        ``restarts == 1`` (default) is a single run -- unchanged behaviour. With
        ``restarts > 1`` and ``restart_combine="consensus"`` the pipeline runs ``restarts``
        independently seeded fits and combines them by SPARSE CO-ASSOCIATION VOTING: a pair
        of points is linked when it is a kNN-neighbour and shares a cluster in at least
        ``restart_threshold`` of the runs, and the consensus clusters are the connected
        components of that graph (see ``dire_evoc.consensus``). This targets small,
        run-variant datasets; it costs ``restarts``x the per-run wall-clock. ``labels_`` is
        the consensus labelling, ``membership_`` the per-point non-noise vote fraction, and
        ``restart_regime_`` records the regime.
        """
        X = np.ascontiguousarray(X)
        on_gpu = self._resolve_gpu()
        restarts = max(1, int(self.restarts))
        if restarts == 1:
            self._fit_once(X, on_gpu, self.random_state)
            self.restart_regime_ = "single_run"
            return self
        if self.restart_combine != "consensus":
            raise ValueError(
                f"restart_combine={self.restart_combine!r} is not supported; only "
                "'consensus' (sparse co-association voting) is implemented for restarts>1."
            )
        labelings, neighbor_lists, non_noise = [], [], []
        for i in range(restarts):
            seed = None if self.random_state is None else int(self.random_state) + i
            self._fit_once(X, on_gpu, seed)  # leaves self.* set to this (carrier) run
            labelings.append(np.asarray(self.labels_).copy())
            neighbor_lists.append(np.asarray(getattr(self.reducer_, "_knn_indices")).copy())
            non_noise.append(np.asarray(self.labels_) >= 0)
        consensus = consensus_labels(
            labelings, neighbor_lists,
            threshold=self.restart_threshold,
            min_cluster_size=self.restart_min_cluster_size,
        )
        # keep the last run's layout_/reducer_/cluster_layers_ as the carrier for
        # inspection; override the operative labelling with the consensus vote.
        self.labels_ = consensus
        self.membership_ = np.mean(np.stack(non_noise, axis=0), axis=0).astype(np.float64)
        self.selected_layer_ = None
        self.restart_regime_ = f"consensus_of_{restarts}"
        if getattr(self, "_reference_", None) is not None:
            self._reference_labels_ = consensus
        return self

    def _fit_once(self, X, on_gpu, seed):
        self.graph_component_layer_ = None
        graph_enabled = bool(
            self.graph_component_candidates and self.selector in {"auto", "auto_ensemble"}
        )
        if graph_enabled:
            graph_reason = "not_evaluated"
        elif self.graph_component_candidates:
            graph_reason = "disabled_for_selector"
        else:
            graph_reason = "disabled"
        self.graph_component_candidate_ = {
            "enabled": graph_enabled,
            "accepted": False,
            "reason": graph_reason,
        }
        self.layout_, self.reducer_ = self._build_layout(X, seed=seed)
        # Compute the graph-derived ID once. Auto selectors reuse this value when choosing
        # graph-component k values; the ordinary silhouette path avoids that graph sweep.
        try:
            self.intrinsic_dimension_ = intrinsic_dimension_from_dire(
                self.reducer_, method="twonn"
            )
            self.intrinsic_dimension_error_ = None
        except Exception as exc:
            self.intrinsic_dimension_ = None
            self.intrinsic_dimension_error_ = exc
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
        if graph_enabled:
            layers, membership_layers, persistence = self._maybe_add_graph_component_layer(
                self.reducer_,
                layers,
                membership_layers,
                persistence,
            )
        if getattr(self, "graph_component_layer_", None) is not None:
            use_graph, graph_info = self._should_select_graph_component_layer(
                layers[sel],
                layers[self.graph_component_layer_],
                silhouette_score=score,
            )
            self.graph_component_candidate_ = graph_info
            if use_graph:
                sel = self.graph_component_layer_

        self.cluster_layers_ = layers
        self.membership_strength_layers_ = membership_layers
        self.persistence_scores_ = persistence
        self.selected_layer_ = sel
        self.labels_ = layers[sel]
        self.membership_ = membership_layers[sel]

        # retain a metric-prepared reference for out-of-sample predict
        if X.dtype in (np.float32, np.float64):
            try:
                self._reference_ = prepare_reference(X, metric=self.metric)
                self._reference_labels_ = self.labels_
                self._prediction_metric_error_ = None
            except ValueError as exc:
                self._reference_ = None
                self._prediction_metric_error_ = exc
        else:
            self._reference_ = None
            self._prediction_metric_error_ = None
        self._n_neighbors_ = (
            self.n_neighbors_predict
            if self.n_neighbors_predict is not None
            else getattr(self.reducer_, "n_neighbors", 15)
        )

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def predict(self, X, n_neighbors=None, return_membership=False):
        """Assign cluster labels to new points by a kNN vote over the training points in
        the original embedding space (noise competes as a label)."""
        if getattr(self, "_reference_", None) is None:
            raise NotImplementedError(
                "predict supports float embeddings with cosine, euclidean, or "
                f"sqeuclidean metrics; this model uses {self.metric!r}."
            )
        query = prepare_reference(np.ascontiguousarray(X), metric=self.metric)
        if query.shape[1] != self._reference_.shape[1]:
            raise ValueError(
                f"X has {query.shape[1]} features, but the model was fit on "
                f"{self._reference_.shape[1]}."
            )
        k = self._n_neighbors_ if n_neighbors is None else n_neighbors
        labels, membership = knn_vote_predict(
            query,
            self._reference_,
            self._reference_labels_,
            k,
            metric=self.metric,
        )
        return (labels, membership) if return_membership else labels
