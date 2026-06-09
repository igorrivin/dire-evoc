"""Scale comparison: vanilla EVoC vs DiReEVoC on cached high-dim embeddings.

Question: does *vanilla* EVoC (its own NN-descent kNN + Boruvka MST on the raw 1024-dim
vectors) ever return at millions of points, and how does it compare to DiReEVoC (which
first reduces to a low-dim DiRe layout on the GPU)?

Each (method, N) fit runs in an isolated subprocess under a wall-clock watchdog so a
non-returning vanilla run is killed and recorded as TIMEOUT rather than hanging the ladder.
A tiny warmup fit precedes the timed fit so numba / cuVS JIT compilation is excluded.
Results checkpoint to JSON after every cell (resumable).

Orchestrator:
    LD_PRELOAD=$CONDA_PREFIX/lib/libnvJitLink.so.13.2.78 python experiments/scale_compare.py \\
        --cache-dir /tmp/cohere_en_emb --sizes 100000 250000 500000 1000000 \\
        --methods vanilla dire --timeout 1800 --out experiments/scale_results.json
Worker (internal): ... --worker --method vanilla --n 100000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SHARD = 100_000


def load_n(cache_dir: str, n: int) -> np.ndarray:
    shards = sorted(Path(cache_dir).glob("emb_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no cached shards in {cache_dir}")
    need = (n + SHARD - 1) // SHARD
    parts = [np.load(s) for s in shards[:need]]
    X = np.concatenate(parts, axis=0)[:n]
    if X.shape[0] < n:
        raise ValueError(f"cache has only {X.shape[0]} rows, need {n}")
    return np.ascontiguousarray(X, dtype=np.float32)


def metrics(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n = labels.shape[0]
    noise = int((labels == -1).sum())
    uniq = np.unique(labels[labels != -1])
    return {"n_clusters": int(uniq.shape[0]),
            "noise_frac": round(noise / n, 4)}


def run_worker(method: str, n: int, cache_dir: str) -> int:
    X = load_n(cache_dir, n)

    if method == "vanilla":
        from evoc import EVoC
        # warmup to exclude numba JIT
        EVoC(random_state=0).fit(X[:1000])
        t0 = time.perf_counter()
        model = EVoC(random_state=0).fit(X)
        dt = time.perf_counter() - t0
        out = {"fit_s": round(dt, 2), **metrics(model.labels_)}
    elif method == "dire":
        from dire_evoc import DiReEVoC
        DiReEVoC(use_gpu="auto", random_state=0).fit(X[:1000])  # warmup
        t0 = time.perf_counter()
        model = DiReEVoC(use_gpu="auto", random_state=0).fit(X)
        dt = time.perf_counter() - t0
        out = {"fit_s": round(dt, 2),
               "intrinsic_dim": round(float(model.intrinsic_dimension_), 3),
               **metrics(model.labels_)}
    else:
        raise ValueError(method)

    print("RESULT " + json.dumps(out), flush=True)
    return 0


def orchestrate(args) -> int:
    out_path = Path(args.out)
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    punted = set()          # methods abandoned because they blew up at a smaller N
    history = {}            # method -> list of (n, fit_s) for extrapolation

    for n in sorted(args.sizes):
        for method in args.methods:
            key = f"{method}@{n}"
            if key in results and results[key].get("status") == "ok":
                history.setdefault(method, []).append((n, results[key]["fit_s"]))
                print(f"[skip] {key} already done: {results[key]}", flush=True)
                continue
            if method in punted:
                rec = {"status": "punted", "reason": "method abandoned at smaller N"}
                results[key] = rec
                print(f"[punt] {key}: {rec}", flush=True)
                out_path.write_text(json.dumps(results, indent=2))
                continue

            # Extrapolate from the two most recent points; punt before running if the
            # projected fit time already exceeds the timeout (don't even start).
            hist = history.get(method, [])
            if len(hist) >= 2:
                (n0, t0_), (n1, t1_) = hist[-2], hist[-1]
                if t1_ > t0_ > 0 and n1 > n0:
                    p = np.log(t1_ / t0_) / np.log(n1 / n0)        # power-law exponent
                    proj = t1_ * (n / n1) ** p
                    print(f"[proj] {method}: exponent~{p:.2f}, "
                          f"projected fit@{n}={proj:.0f}s", flush=True)
                    if proj > args.timeout:
                        punted.add(method)
                        rec = {"status": "punted", "reason": "projection exceeds timeout",
                               "exponent": round(float(p), 2), "projected_fit_s": round(proj)}
                        results[key] = rec
                        print(f"[punt] {key}: {rec}", flush=True)
                        out_path.write_text(json.dumps(results, indent=2))
                        continue

            print(f"\n[run ] {key} (timeout {args.timeout}s)...", flush=True)
            cmd = [sys.executable, __file__, "--worker", "--method", method,
                   "--n", str(n), "--cache-dir", args.cache_dir]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=args.timeout, start_new_session=True)
                wall = time.perf_counter() - t0
                line = next((l for l in proc.stdout.splitlines()
                             if l.startswith("RESULT ")), None)
                if proc.returncode == 0 and line:
                    rec = {"status": "ok", "wall_s": round(wall, 2),
                           **json.loads(line[len("RESULT "):])}
                    history.setdefault(method, []).append((n, rec["fit_s"]))
                else:
                    rec = {"status": "error", "wall_s": round(wall, 2),
                           "returncode": proc.returncode,
                           "stderr_tail": proc.stderr.strip().splitlines()[-3:]}
            except subprocess.TimeoutExpired:
                rec = {"status": "timeout", "wall_s": args.timeout}
                punted.add(method)   # timed out here -> hopeless at larger N
            results[key] = rec
            print(f"[done] {key}: {rec}", flush=True)
            out_path.write_text(json.dumps(results, indent=2))

    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method")
    ap.add_argument("--n", type=int)
    ap.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100_000, 250_000, 500_000, 1_000_000])
    ap.add_argument("--methods", nargs="+", default=["vanilla", "dire"])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="experiments/scale_results.json")
    args = ap.parse_args()

    if args.worker:
        return run_worker(args.method, args.n, args.cache_dir)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
