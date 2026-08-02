"""Decompose the DiReEVoC / EVoC stability floor into its causes (cheap diagnosis).

Two distinct quantities are conflated by a single "floor" number:

* same-seed determinism -- refit with the SAME random_state and compare. ARI < 1 here is
  true nondeterminism (GPU atomics, approximate ANN k-NN), not seed choice.
* seed sensitivity -- refit across DIFFERENT seeds and compare. ARI < 1 here is expected
  for any stochastic method; it measures robustness to the seed, not nondeterminism.

A lucky boundary makes the determinism test clean for free: DiRe uses exact PyTorch
brute-force k-NN below its cuVS threshold (200k points for d>=200) and approximate cuVS
IVF-Flat above it. So N=100k uses EXACT k-NN and N=250k uses APPROXIMATE k-NN -- comparing
same-seed determinism at the two sizes isolates the ANN contribution without forcing any
backend.

Reports, on the shared 100k anchor:
  - same-seed determinism ARI at N=100k (exact kNN) vs N=250k (ANN kNN), default + oracle
  - seed-sensitivity ARI across 5 seeds at N=100k (mean +/- std), default + oracle, both methods

    python experiments/reproducibility_diag.py --cache-dir /tmp/cohere_en_emb \\
        --out-dir experiments/diag_anchors --timeout 600
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

SHARD = 100_000
ANCHOR = 100_000
SEEDS = [0, 1, 2, 3, 4]


def load_n(cache_dir: str, n: int) -> np.ndarray:
    shards = sorted(Path(cache_dir).glob("emb_*.npy"))
    need = (n + SHARD - 1) // SHARD
    X = np.concatenate([np.load(s) for s in shards[:need]], axis=0)[:n]
    return np.ascontiguousarray(X, dtype=np.float32)


def ncl(lbl: np.ndarray) -> int:
    return int(np.unique(lbl[lbl != -1]).shape[0])


def run_worker(method, n, seed, rep, cache_dir, out_dir):
    X = load_n(cache_dir, n)
    if method == "vanilla":
        from evoc import EVoC
        m = EVoC(random_state=seed).fit(X)
    elif method == "dire":
        from dire_evoc import DiReEVoC
        m = DiReEVoC(use_gpu="auto", random_state=seed).fit(X)
    else:
        raise ValueError(method)
    layers = [np.asarray(l)[:ANCHOR] for l in m.cluster_layers_]
    default = np.asarray(m.labels_)[:ANCHOR]
    np.savez(Path(out_dir) / f"{method}_{n}_s{seed}_r{rep}.npz",
             default=default, **{f"layer_{i}": l for i, l in enumerate(layers)})
    print(f"RESULT {method}@{n} s{seed} r{rep} ncl={ncl(default)} layers={len(layers)}",
          flush=True)
    return 0


def _layers(f):
    return [f[k] for k in f.files if k.startswith("layer_")]


def oracle_ari(fa, fb, ari):
    return max((ari(a, b) for a, b in itertools.product(_layers(fa), _layers(fb))),
              default=float("nan"))


def load(out_dir, method, n, seed, rep):
    return np.load(Path(out_dir) / f"{method}_{n}_s{seed}_r{rep}.npz")


def orchestrate(args):
    jobs = []
    # same-seed determinism: dire, exact (100k) vs ANN (250k), seed 0, two reps
    for n in (100_000, 250_000):
        jobs += [("dire", n, 0, 0), ("dire", n, 0, 1)]
    # seed sensitivity at 100k: both methods, 5 seeds (rep 0). seed0/rep0 reused above.
    for method in ("dire", "vanilla"):
        for s in SEEDS:
            jobs.append((method, 100_000, s, 0))
    # de-dup
    seen, uniq = set(), []
    for j in jobs:
        if j not in seen:
            seen.add(j); uniq.append(j)

    for method, n, seed, rep in uniq:
        out = Path(args.out_dir) / f"{method}_{n}_s{seed}_r{rep}.npz"
        if out.exists():
            print(f"[skip] {out.name}", flush=True)
            continue
        print(f"[fit ] {method}@{n} s{seed} r{rep}...", flush=True)
        cmd = [sys.executable, __file__, "--worker", "--method", method, "--n", str(n),
               "--seed", str(seed), "--rep", str(rep),
               "--cache-dir", args.cache_dir, "--out-dir", args.out_dir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout, start_new_session=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
            print(f"       {line or 'FAILED: ' + (proc.stderr.strip().splitlines()[-1:] or [''])[0]}",
                  flush=True)
        except subprocess.TimeoutExpired:
            print("       TIMEOUT", flush=True)


def analyze(args):
    from sklearn.metrics import adjusted_rand_score as ari

    print("\n========= REPRODUCIBILITY DECOMPOSITION (ARI on 100k anchor) =========\n")
    print("1) Same-seed determinism (refit at random_state=0, compare runs):")
    print("   N=100k uses EXACT k-NN; N=250k uses APPROXIMATE cuVS IVF-Flat.\n")
    for n, knn in ((100_000, "exact"), (250_000, "ANN")):
        try:
            a, b = load(args.out_dir, "dire", n, 0, 0), load(args.out_dir, "dire", n, 0, 1)
            print(f"   dire @{n:>7} ({knn:>5} kNN):  default ARI={ari(a['default'], b['default']):.3f}"
                  f"   oracle ARI={oracle_ari(a, b, ari):.3f}")
        except FileNotFoundError:
            print(f"   dire @{n:>7} ({knn:>5} kNN):  <missing>")

    print("\n2) Seed sensitivity at N=100k (pairwise ARI over 5 seeds, mean +/- std):")
    for method in ("dire", "vanilla"):
        try:
            fs = [load(args.out_dir, method, 100_000, s, 0) for s in SEEDS]
        except FileNotFoundError:
            print(f"   {method}: <missing>")
            continue
        d = [ari(fa["default"], fb["default"]) for fa, fb in itertools.combinations(fs, 2)]
        o = [oracle_ari(fa, fb, ari) for fa, fb in itertools.combinations(fs, 2)]
        print(f"   {method:8}: default {np.mean(d):.3f} +/- {np.std(d):.3f}   "
              f"oracle {np.mean(o):.3f} +/- {np.std(o):.3f}   (n_pairs={len(d)})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method"); ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    ap.add_argument("--out-dir", default="experiments/diag_anchors")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return run_worker(args.method, args.n, args.seed, args.rep, args.cache_dir, args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        orchestrate(args)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
