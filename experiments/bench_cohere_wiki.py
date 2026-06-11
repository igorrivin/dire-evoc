"""Benchmark DiReEVoC on streamed Cohere Wikipedia embeddings.

Streams N rows of precomputed embeddings from a Hugging Face dataset (no full
download), runs the DiReEVoC pipeline on the high-dimensional vectors, and reports
data-load timing, fit timing, and the structural clustering metrics (there is no
ground-truth labelling for Wikipedia, so quality is reported as intrinsic dimension,
cluster count, noise fraction, and silhouette of the operative layer rather than ARI).

Example:
    LD_PRELOAD=$CONDA_PREFIX/lib/libnvJitLink.so.13.2.78 \\
        python experiments/bench_cohere_wiki.py \\
            --dataset Cohere/wikipedia-2023-11-embed-multilingual-v3 \\
            --config en --n-rows 20000
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def find_emb_field(row: dict) -> str:
    """Identify the embedding field of a streamed row (Cohere uses 'emb')."""
    for cand in ("emb", "embedding", "embeddings", "vector"):
        if cand in row:
            return cand
    # fall back to the longest list-like numeric field
    best, best_len = None, -1
    for k, v in row.items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > best_len:
            best, best_len = k, len(v)
    if best is None:
        raise KeyError(f"no embedding-like field in row keys={list(row.keys())}")
    return best


def stream_embeddings(dataset: str, config: str | None, n_rows: int):
    from datasets import load_dataset

    t0 = time.perf_counter()
    ds = load_dataset(dataset, config, split="train", streaming=True)
    it = iter(ds)
    first = next(it)
    t_first = time.perf_counter() - t0
    field = find_emb_field(first)
    dim = len(first[field])

    buf = np.empty((n_rows, dim), dtype=np.float32)
    buf[0] = first[field]
    got = 1
    report_every = max(1, n_rows // 10)
    for row in it:
        if got >= n_rows:
            break
        buf[got] = row[field]
        got += 1
        if got % report_every == 0:
            print(f"  streamed {got}/{n_rows} rows "
                  f"({got / (time.perf_counter() - t0):.0f} rows/s)", flush=True)
    X = buf[:got]
    t_total = time.perf_counter() - t0
    return X, field, t_first, t_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cohere/wikipedia-2023-11-embed-multilingual-v3")
    ap.add_argument("--config", default="en")
    ap.add_argument("--n-rows", type=int, default=20000)
    ap.add_argument("--n-components", type=int, default=8)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--use-gpu", default="auto")
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    print(f"[stream] {args.dataset} config={args.config} n_rows={args.n_rows}", flush=True)
    X, field, t_first, t_stream = stream_embeddings(args.dataset, args.config, args.n_rows)
    print(f"[stream] field='{field}' X.shape={X.shape} dtype={X.dtype}", flush=True)
    print(f"[stream] first-row latency={t_first:.1f}s  total stream={t_stream:.1f}s "
          f"({X.shape[0] / t_stream:.0f} rows/s)", flush=True)

    from dire_evoc import DiReEVoC

    use_gpu = {"auto": "auto", "true": True, "false": False}[str(args.use_gpu).lower()]
    model = DiReEVoC(n_components=args.n_components, metric=args.metric,
                     use_gpu=use_gpu, random_state=args.random_state)

    t0 = time.perf_counter()
    model.fit(X)
    t_fit = time.perf_counter() - t0

    labels = np.asarray(model.labels_)
    n = labels.shape[0]
    noise = int((labels == -1).sum())
    uniq = np.unique(labels[labels != -1])
    n_clusters = uniq.shape[0]
    sizes = np.array([(labels == c).sum() for c in uniq]) if n_clusters else np.array([])

    print("\n==================== RESULTS ====================")
    print(f"dataset            : {args.dataset} [{args.config}]")
    print(f"n_points x dim     : {n} x {X.shape[1]}")
    print(f"use_gpu (resolved) : {model._resolve_gpu()}")
    print(f"--- timings ---")
    print(f"stream load (s)    : {t_stream:.2f}  (first row {t_first:.1f}s)")
    print(f"fit total (s)      : {t_fit:.2f}")
    print(f"fit throughput     : {n / t_fit:.0f} points/s")
    print(f"--- structure ---")
    print(f"intrinsic dim      : {model.intrinsic_dimension_:.3f}")
    print(f"n_clusters         : {n_clusters}")
    print(f"noise fraction     : {noise / n:.3f}  ({noise} pts)")
    print(f"selected layer     : {model.selected_layer_} of {len(model.cluster_layers_)}")
    print(f"selected min_samp  : {model.selected_min_samples_}")
    if n_clusters:
        print(f"cluster size       : min={sizes.min()} median={int(np.median(sizes))} "
              f"max={sizes.max()} mean={sizes.mean():.1f}")
    print("=================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
