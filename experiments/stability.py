"""Stability of vanilla EVoC vs DiReEVoC under growing N (documentation-grade).

The first 100k cached rows (the "anchor set") are a subset of every larger prefix, so we
can ask the question that actually matters: as N grows 100k -> 1M, do those same anchor
points keep their cluster assignment, or does the labelling churn? Stability = ARI of the
anchor labels at one N against the anchor labels at the next N. A stable method scores
ARI ~ 1 against itself; an unstable one decays. This is selector-agnostic: each method is
judged against itself.

Two confounds are handled explicitly:

* Selector vs algorithm. We save not just the default ``labels_`` but every layer of
  ``cluster_layers_``. If even the best layer-to-layer match (an oracle selector) is
  unstable, the instability is algorithmic; if a fixed layer is stable while the default
  ``labels_`` churns, the chaos was merely EVoC's default layer-picker.

* Run-to-run nondeterminism. We refit at the SAME N (100k) twice and report that ARI as a
  noise floor. An across-N drop only counts as real instability if it falls below this floor.

Each fit runs in an isolated subprocess under a watchdog; anchor labels are persisted so the
analysis is a cheap re-run. Orchestrator runs all fits then prints the ARI tables.

    LD_PRELOAD=$CONDA_PREFIX/lib/libnvJitLink.so.13.2.78 python experiments/stability.py \\
        --cache-dir /tmp/cohere_en_emb --sizes 100000 250000 500000 1000000 \\
        --methods vanilla dire --out-dir experiments/stability_anchors --timeout 900
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SHARD = 100_000
ANCHOR = 100_000  # shared-point set = first ANCHOR rows


def load_n(cache_dir: str, n: int) -> np.ndarray:
    shards = sorted(Path(cache_dir).glob("emb_*.npy"))
    need = (n + SHARD - 1) // SHARD
    X = np.concatenate([np.load(s) for s in shards[:need]], axis=0)[:n]
    return np.ascontiguousarray(X, dtype=np.float32)


def ncl(labels: np.ndarray) -> int:
    return int(np.unique(labels[labels != -1]).shape[0])


def run_worker(method: str, n: int, seed: int, cache_dir: str, out_dir: str) -> int:
    X = load_n(cache_dir, n)
    if method == "vanilla":
        from evoc import EVoC
        EVoC(random_state=seed).fit(X[:1000])                       # warmup JIT
        m = EVoC(random_state=seed).fit(X)
    elif method == "dire":
        from dire_evoc import DiReEVoC
        DiReEVoC(use_gpu="auto", random_state=seed).fit(X[:1000])   # warmup
        m = DiReEVoC(use_gpu="auto", random_state=seed).fit(X)
    else:
        raise ValueError(method)

    layers = [np.asarray(l)[:ANCHOR] for l in m.cluster_layers_]
    default = np.asarray(m.labels_)[:ANCHOR]
    out = Path(out_dir) / f"{method}_{n}_s{seed}.npz"
    np.savez(out,
             default=default,
             layer_ncl=np.array([ncl(l) for l in layers]),
             **{f"layer_{i}": l for i, l in enumerate(layers)})
    print(f"RESULT saved {out.name} default_ncl={ncl(default)} "
          f"n_layers={len(layers)}", flush=True)
    return 0


def load_anchor(out_dir: str, method: str, n: int, seed: int = 0):
    return np.load(Path(out_dir) / f"{method}_{n}_s{seed}.npz")


def fit_all(args):
    jobs = [(m, n, 0) for n in sorted(args.sizes) for m in args.methods]
    # nondeterminism floor: a second seed at the anchor size
    jobs += [(m, ANCHOR, 1) for m in args.methods]
    for method, n, seed in jobs:
        out = Path(args.out_dir) / f"{method}_{n}_s{seed}.npz"
        if out.exists():
            print(f"[skip] {out.name} exists", flush=True)
            continue
        print(f"\n[fit ] {method}@{n} seed={seed} (timeout {args.timeout}s)...", flush=True)
        cmd = [sys.executable, __file__, "--worker", "--method", method, "--n", str(n),
               "--seed", str(seed), "--cache-dir", args.cache_dir, "--out-dir", args.out_dir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout, start_new_session=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
            if line:
                status = line
            elif proc.stderr.strip():
                status = "FAILED " + proc.stderr.strip().splitlines()[-1]
            else:
                status = "FAILED (no RESULT line)"
            print(f"[fit ] {method}@{n} s{seed}: {status}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[fit ] {method}@{n} s{seed}: TIMEOUT", flush=True)


def analyze(args):
    from sklearn.metrics import adjusted_rand_score as ari
    sizes = sorted(args.sizes)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("\n================ STABILITY (ARI on shared 100k anchor) ================")
    print("Noise floor = same data (N=100k), two seeds — instability only counts below this.\n")

    for method in args.methods:
        print(f"---- {method} ----")
        # noise floor
        try:
            a0 = load_anchor(args.out_dir, method, ANCHOR, 0)["default"]
            a1 = load_anchor(args.out_dir, method, ANCHOR, 1)["default"]
            floor = ari(a0, a1)
            print(f"  noise floor (100k seed0 vs seed1) default-labels ARI = {floor:.3f}")
        except FileNotFoundError:
            floor = None
            print("  noise floor: <missing>")

        # default-labels stability across consecutive N
        print("  default-labels ARI across N (consecutive):")
        for ni, nj in zip(sizes[:-1], sizes[1:]):
            try:
                li = load_anchor(args.out_dir, method, ni, 0)
                lj = load_anchor(args.out_dir, method, nj, 0)
                a = ari(li["default"], lj["default"])
                print(f"    {ni:>8} -> {nj:<8}  ARI={a:.3f}   "
                      f"(ncl {ncl(li['default'])} -> {ncl(lj['default'])})")
            except FileNotFoundError:
                print(f"    {ni:>8} -> {nj:<8}  <missing>")

        # oracle layer-matched stability: best ARI over all layer pairings
        print("  oracle best-layer-matched ARI across N (confound check):")
        for ni, nj in zip(sizes[:-1], sizes[1:]):
            try:
                fi, fj = load_anchor(args.out_dir, method, ni, 0), load_anchor(args.out_dir, method, nj, 0)
            except FileNotFoundError:
                print(f"    {ni:>8} -> {nj:<8}  <missing>")
                continue
            li = [fi[k] for k in fi.files if k.startswith("layer_") and k != "layer_ncl"]
            lj = [fj[k] for k in fj.files if k.startswith("layer_") and k != "layer_ncl"]
            best = max((ari(a, b) for a, b in itertools.product(li, lj)), default=float("nan"))
            print(f"    {ni:>8} -> {nj:<8}  best-ARI={best:.3f}  "
                  f"(layers {len(li)} x {len(lj)})")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    ap.add_argument("--out-dir", default="experiments/stability_anchors")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100_000, 250_000, 500_000, 1_000_000])
    ap.add_argument("--methods", nargs="+", default=["vanilla", "dire"])
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if args.worker:
        return run_worker(args.method, args.n, args.seed, args.cache_dir, args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        fit_all(args)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
