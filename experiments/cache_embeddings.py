"""Stream Cohere Wikipedia embeddings to disk once, in resumable shards.

Streaming the HF dataset costs ~2000 rows/s, so re-streaming for every scale point is
wasteful. This caches the first N embeddings to fixed-size .npy shards under a cache dir.
Re-running resumes: it counts existing shards and uses ``IterableDataset.skip`` to continue
where it left off (streaming cannot seek, but ``skip`` replays cheaply from the shard list).

Usage:
    LD_PRELOAD=$CONDA_PREFIX/lib/libnvJitLink.so.13.2.78 \\
        python experiments/cache_embeddings.py --n-rows 1000000 --cache-dir /tmp/cohere_en_emb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

SHARD = 100_000  # rows per shard


def emb_field(row: dict) -> str:
    for c in ("emb", "embedding", "embeddings", "vector"):
        if c in row:
            return c
    raise KeyError(f"no embedding field in {list(row.keys())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cohere/wikipedia-2023-11-embed-multilingual-v3")
    ap.add_argument("--config", default="en")
    ap.add_argument("--n-rows", type=int, default=1_000_000)
    ap.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache.glob("emb_*.npy"))
    done_shards = len(existing)
    cached_rows = done_shards * SHARD
    target_shards = (args.n_rows + SHARD - 1) // SHARD
    if cached_rows >= args.n_rows:
        print(f"[cache] already have {cached_rows} rows ({done_shards} shards) >= {args.n_rows}")
        return 0

    from datasets import load_dataset

    print(f"[cache] {args.dataset}[{args.config}] -> {cache} "
          f"(have {done_shards} shards, need {target_shards})", flush=True)
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    if cached_rows:
        ds = ds.skip(cached_rows)
        print(f"[cache] skipping {cached_rows} already-cached rows...", flush=True)

    t0 = time.perf_counter()
    field = None
    buf, sidx, written = [], done_shards, 0
    for row in ds:
        if field is None:
            field = emb_field(row)
        buf.append(np.asarray(row[field], dtype=np.float32))
        if len(buf) == SHARD:
            np.save(cache / f"emb_{sidx:04d}.npy", np.stack(buf))
            written += len(buf)
            sidx += 1
            rate = (written) / (time.perf_counter() - t0)
            print(f"[cache] wrote shard {sidx} ({sidx*SHARD} rows total, {rate:.0f} rows/s)",
                  flush=True)
            buf = []
            if sidx >= target_shards:
                break
    if buf and sidx < target_shards:  # final partial shard
        np.save(cache / f"emb_{sidx:04d}.npy", np.stack(buf))
        print(f"[cache] wrote final partial shard {sidx} ({len(buf)} rows)", flush=True)

    print(f"[cache] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
