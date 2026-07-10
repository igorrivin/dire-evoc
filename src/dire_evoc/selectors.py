"""Internal selector utilities for choosing DiRe/EVoC cluster layers.

The public estimator keeps the familiar string selector API, while this module holds
the reusable scoring and arbitration pieces. The hard gates are intentional: smooth
confidence combinations are useful diagnostics, but they should not rescue graph
candidates that are structurally unsafe.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectorCandidate:
    """A candidate labelling produced by one selector family."""

    name: str
    labels: np.ndarray
    score: float = -np.inf
    confidence: float = 0.0
    accepted: bool = True
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SelectorDecision:
    """Decision returned by an arbitration policy."""

    selected: bool
    reason: str
    confidence: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def cluster_count(labels) -> int:
    labels = np.asarray(labels)
    return int(np.unique(labels[labels >= 0]).size)


def noise_fraction(labels) -> float:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels < 0))


def sample_indices(n: int, random_state=None, max_samples: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if n > max_samples:
        return rng.choice(n, max_samples, replace=False)
    return np.arange(n)


def coverage_silhouette_score(Xs, lab_sample, *, raw: bool = False) -> float:
    """Silhouette score, optionally coverage-weighted by non-noise fraction."""
    from sklearn.metrics import silhouette_score

    lab_sample = np.asarray(lab_sample)
    mask = lab_sample >= 0
    n_clustered = int(mask.sum())
    n_labels = np.unique(lab_sample[mask]).size
    if n_labels < 2 or n_labels >= n_clustered:
        return -np.inf
    score = float(silhouette_score(Xs[mask], lab_sample[mask]))
    if not raw:
        score *= float(mask.mean())
    return score


def silhouette_confidence(score: float) -> float:
    """Map a silhouette-like score from [-1, 1] into [0, 1]."""
    if not np.isfinite(score):
        return 0.0
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def or_probability(*scores: float) -> float:
    """Noisy-or combination: x + y - x*y for two inputs."""
    miss = 1.0
    for score in scores:
        miss *= 1.0 - float(np.clip(score, 0.0, 1.0))
    return float(1.0 - miss)


def best_layer(
    layout,
    layers: Sequence[np.ndarray],
    *,
    selector: str,
    random_state=None,
) -> tuple[int, float, list[float]]:
    """Return the selected EVoC layer index, its score, and all layer scores."""
    if len(layers) == 0:
        raise ValueError("layers must not be empty")

    sample = sample_indices(layout.shape[0], random_state=random_state)
    Xs = layout[sample]
    if selector == "base" or len(layers) == 1:
        scores = [
            coverage_silhouette_score(Xs, layers[0][sample], raw=False)
        ]
        return 0, scores[0], scores
    if selector == "none":
        i = len(layers) - 1
        scores = [
            coverage_silhouette_score(Xs, np.asarray(layer)[sample], raw=False)
            for layer in layers
        ]
        return i, scores[i], scores

    raw = selector == "silhouette_raw"
    scores = [
        coverage_silhouette_score(Xs, np.asarray(layer)[sample], raw=raw)
        for layer in layers
    ]
    i = int(np.argmax(scores))
    return i, scores[i], scores


def graph_component_confidence(
    *,
    graph_info: dict,
    evoc_clusters: int,
    graph_clusters: int,
    base_n_clusters: int | None,
    min_cluster_ratio: float,
    hard_gate_passed: bool,
) -> float:
    if not hard_gate_passed:
        return 0.0
    stability = float(graph_info.get("stability", 0.0) or 0.0)
    target_match = (
        1.0
        if base_n_clusters is not None and graph_clusters == base_n_clusters
        else 0.0
    )
    ratio = evoc_clusters / max(1, graph_clusters)
    fragmentation = min(1.0, ratio / max(float(min_cluster_ratio), 1e-12))
    return float(np.clip(stability * max(target_match, fragmentation), 0.0, 1.0))


def guarded_graph_component_decision(
    *,
    evoc_labels,
    graph_labels,
    graph_info: dict,
    base_n_clusters: int | None,
    min_cluster_ratio: float,
    policy: str = "guarded_override",
    silhouette_score: float | None = None,
) -> SelectorDecision:
    """Arbitrate a stable graph-component candidate against the EVoC-selected layer."""
    info = dict(graph_info or {})
    if not info.get("accepted"):
        return SelectorDecision(False, info.get("reason", "not_accepted"), 0.0, info)

    evoc_labels = np.asarray(evoc_labels)
    graph_labels = np.asarray(graph_labels)
    evoc_clusters = cluster_count(evoc_labels)
    graph_clusters = cluster_count(graph_labels)
    info.update(
        {
            "evoc_selected_clusters": evoc_clusters,
            "graph_selected_clusters": graph_clusters,
            "evoc_selected_noise": noise_fraction(evoc_labels),
        }
    )

    selected = False
    reason = "evoc_not_fragmented_relative_to_graph"
    hard_gate_passed = False
    if graph_clusters < 2:
        reason = "too_few_graph_clusters"
    elif base_n_clusters is not None and graph_clusters == base_n_clusters:
        selected = True
        hard_gate_passed = True
        reason = "matches_base_n_clusters"
    elif graph_clusters < 3:
        reason = "too_few_graph_clusters_without_base_n_clusters"
    else:
        cluster_ratio = evoc_clusters / max(1, graph_clusters)
        info["evoc_to_graph_cluster_ratio"] = float(cluster_ratio)
        if cluster_ratio >= min_cluster_ratio:
            selected = True
            hard_gate_passed = True
            reason = "evoc_fragmented_relative_to_graph"

    graph_conf = graph_component_confidence(
        graph_info=info,
        evoc_clusters=evoc_clusters,
        graph_clusters=graph_clusters,
        base_n_clusters=base_n_clusters,
        min_cluster_ratio=min_cluster_ratio,
        hard_gate_passed=hard_gate_passed,
    )
    if policy == "auto_ensemble":
        sil_conf = silhouette_confidence(
            -np.inf if silhouette_score is None else silhouette_score
        )
        info["silhouette_confidence"] = sil_conf
        info["graph_confidence"] = graph_conf
        info["combined_confidence"] = or_probability(sil_conf, graph_conf)
        selected = bool(selected and graph_conf >= sil_conf)
        if not selected and hard_gate_passed:
            reason = "graph_confidence_not_above_silhouette"

    info["selection_reason"] = reason
    info["graph_confidence"] = graph_conf
    return SelectorDecision(selected, reason, graph_conf, info)
