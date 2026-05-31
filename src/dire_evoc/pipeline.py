"""One-call DiRe -> EVoC pipeline: layout, clustering, intrinsic dimension, predict.

``DiReEVoC`` runs the whole integration end to end:

  1. build a DiRe layout of the data (GPU, cuVS) -- the common substrate;
  2. cluster the layout with EVoC's density machinery, using the GPU single-linkage
     tree when RAPIDS is available (``dire_evoc.hybrid``) and EVoC's CPU MST otherwise
     -- the clustering is identical either way (validated ARI = 1.0 at fixed k);
  3. select the operative layer by silhouette on the layout -- the validated default
     selector (markedly lower ARI-regret than EVoC's persistence default);
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
    base_min_cluster_size : int, default=10
        Finest-granularity cluster size for EVoC's base layer.
    use_gpu : {"auto", True, False}, default="auto"
        Whether to build the single-linkage tree on the GPU (cuML). "auto" uses the GPU
        iff cuML imports. The two paths give the same clustering; GPU wins at large n.
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
    selected_layer_ : int                      index into cluster_layers_ chosen for labels_
    selected_min_samples_ : int                min_samples actually used (> min_samples if
                                               the adaptive search escalated it)
    """

    def __init__(self, n_components=8, metric="cosine", min_samples=10,
                 base_min_cluster_size=10, use_gpu="auto", selector="silhouette",
                 adaptive_min_samples=True, noise_gate=0.5,
                 n_neighbors_predict=None, random_state=None, **dire_kwargs):
        self.n_components = n_components
        self.metric = metric
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.use_gpu = use_gpu
        self.selector = selector
        self.adaptive_min_samples = adaptive_min_samples
        self.noise_gate = noise_gate
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

    def _cluster(self, layout, on_gpu, min_samples=None):
        ms = self.min_samples if min_samples is None else min_samples
        if on_gpu:
            from .hybrid import hybrid_cluster_layers
            return hybrid_cluster_layers(
                layout, min_samples=ms,
                base_min_cluster_size=self.base_min_cluster_size,
            )
        from evoc.clustering import build_cluster_layers
        return build_cluster_layers(
            layout, min_samples=ms,
            base_min_cluster_size=self.base_min_cluster_size,
        )

    def _select_layer(self, layout, layers):
        """Pick the operative layer by COVERAGE-WEIGHTED silhouette on the layout.

        Plain silhouette is computed only on clustered points, so a layer that dumps
        half the data as noise but leaves tight residual clusters scores well -- it
        systematically over-segments (e.g. picks k=51 on 20NG where k=8 is right).
        The fix: score the mean silhouette over *all* sampled points, with noise points
        contributing 0. That equals ``silhouette(clustered) * coverage`` (coverage =
        fraction non-noise), since silhouette_score is the mean of the per-point values.
        It rewards layers that are both well-separated AND explain the data, and drops
        ARI-regret to ~0 across synth/MNIST/20NG (the last being the case plain
        silhouette could not solve). ``selector="silhouette_raw"`` keeps the old behaviour.
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
        pick within a layer set and to compare candidate sets across ``min_samples``."""
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

    # -- API -----------------------------------------------------------------
    def fit(self, X, y=None):
        X = np.ascontiguousarray(X)
        on_gpu = self._resolve_gpu()
        self.layout_, self.reducer_ = self._build_layout(X)
        layers, membership_layers, persistence = self._cluster(self.layout_, on_gpu)
        sel, score = self._best_layer(self.layout_, layers)
        self.selected_min_samples_ = self.min_samples

        # Adaptive smoothing. If the base clustering calls more than `noise_gate` of the
        # points noise, the DiRe layout is likely tighter than EVoC's default min_samples
        # expects, so the whole layer set comes out over-fragmented (diagnosed on
        # ultra-tight synthetic blobs: 85% noise at ms=10, ARI 1.0 at ms>=50). Re-cluster
        # at larger min_samples and re-select by the SAME coverage-weighted score; adopt a
        # higher min_samples only if it strictly beats the base. Genuinely diffuse data
        # (continuous manifolds) keeps its honest high-noise result -- nothing beats it.
        if (self.adaptive_min_samples and self.selector == "silhouette"
                and (layers[sel] < 0).mean() > self.noise_gate):
            best = (score, sel, layers, membership_layers, persistence, self.min_samples)
            for factor in (2.5, 5.0):
                ms = int(round(self.min_samples * factor))
                if ms <= self.min_samples:
                    continue
                try:
                    ls, lm, lp = self._cluster(self.layout_, on_gpu, min_samples=ms)
                except Exception:
                    continue
                si, sc = self._best_layer(self.layout_, ls)
                if sc > best[0]:
                    best = (sc, si, ls, lm, lp, ms)
            score, sel, layers, membership_layers, persistence, self.selected_min_samples_ = best

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
