"""Test whether FP16 k-NN tie-breaking is the cause of DiReEVoC's small-N nondeterminism.

The reproducibility diagnosis (experiments/reproducibility_diag.py) found that dire is only
~0.58 same-seed reproducible at N=100k (exact kNN) but ~0.96 at N=250k (ANN kNN) -- the
opposite of an ANN-noise story. Prime suspect: at n_dims>=500 DiRe runs its brute-force
k-NN in FP16 (``use_fp16 = n_dims >= 500 or n_samples >= 100000``). FP16 near-ties on the
1024-d cosine graph can break nondeterministically on the GPU, perturbing the neighbor graph
and tipping a 5-vs-6 layer bifurcation.

This forces FP32 k-NN (monkeypatching ``_compute_knn``) and re-measures same-seed
determinism at N=100k. If same-seed ARI jumps from ~0.58 toward ~0.96, FP16 tie-breaking is
the root cause and the fix is to force FP32 k-NN at small N.

    python experiments/fp16_probe.py --cache-dir /tmp/cohere_en_emb --reps 3
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


def load_n(cache_dir, n):
    shards = sorted(Path(cache_dir).glob("emb_*.npy"))
    need = (n + SHARD - 1) // SHARD
    return np.ascontiguousarray(
        np.concatenate([np.load(s) for s in shards[:need]], axis=0)[:n], dtype=np.float32)


def force_fp32_knn():
    """Monkeypatch DiRe k-NN to FP32, forwarding all other args untouched."""
    import dire_rapids.dire_pytorch as dp
    targets = [dp.DiRePyTorch]
    try:
        import dire_rapids.dire_cuvs as dc
        targets.append(dc.DiReCuVS)
    except Exception:
        pass
    for cls in targets:
        orig = cls._compute_knn

        def make(o):
            def wrapped(self, X, *a, **k):
                k["use_fp16"] = False
                return o(self, X, *a, **k)
            return wrapped
        cls._compute_knn = make(orig)


def run_worker(n, seed, rep, fp32, cache_dir, out_dir):
    if fp32:
        force_fp32_knn()
    X = load_n(cache_dir, n)
    from dire_evoc import DiReEVoC
    m = DiReEVoC(use_gpu="auto", random_state=seed).fit(X)
    layers = [np.asarray(l)[:ANCHOR] for l in m.cluster_layers_]
    default = np.asarray(m.labels_)[:ANCHOR]
    tag = "fp32" if fp32 else "fp16"
    np.savez(Path(out_dir) / f"dire_{tag}_{n}_s{seed}_r{rep}.npz",
             default=default, **{f"layer_{i}": l for i, l in enumerate(layers)})
    ncl = int(np.unique(default[default != -1]).shape[0])
    print(f"RESULT {tag} dire@{n} s{seed} r{rep} ncl={ncl} layers={len(layers)}", flush=True)
    return 0


def _layers(f):
    return [f[k] for k in f.files if k.startswith("layer_")]


def analyze(out_dir, n, seed, reps, ari):
    files = [np.load(Path(out_dir) / f"dire_fp32_{n}_s{seed}_r{r}.npz") for r in range(reps)]
    d = [ari(a["default"], b["default"]) for a, b in itertools.combinations(files, 2)]
    o = [max(ari(x, y) for x, y in itertools.product(_layers(a), _layers(b)))
         for a, b in itertools.combinations(files, 2)]
    ncls = [int(np.unique(f["default"][f["default"] != -1]).shape[0]) for f in files]
    print("\n===== FP32 k-NN same-seed determinism at N=100k (seed 0) =====")
    print(f"  cluster counts across reps : {ncls}")
    print(f"  default-label ARI  : {np.mean(d):.3f} +/- {np.std(d):.3f}  (pairs={len(d)})")
    print(f"  oracle  layer  ARI : {np.mean(o):.3f} +/- {np.std(o):.3f}")
    print(f"  --- compare FP16 baseline: default 0.577 / oracle 0.731 (from diag) ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    ap.add_argument("--out-dir", default="experiments/fp16_probe_anchors")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if args.worker:
        return run_worker(args.n, args.seed, args.rep, args.fp32, args.cache_dir, args.out_dir)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for rep in range(args.reps):
        out = Path(args.out_dir) / f"dire_fp32_{args.n}_s{args.seed}_r{rep}.npz"
        if out.exists():
            print(f"[skip] {out.name}", flush=True); continue
        print(f"[fit ] fp32 dire@{args.n} s{args.seed} r{rep}...", flush=True)
        cmd = [sys.executable, __file__, "--worker", "--fp32", "--n", str(args.n),
               "--seed", str(args.seed), "--rep", str(rep),
               "--cache-dir", args.cache_dir, "--out-dir", args.out_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout, start_new_session=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
        fp32_log = "Using FP32 for k-NN" in proc.stderr
        print(f"       {line or 'FAILED'}   [FP32 kNN confirmed in log: {fp32_log}]", flush=True)

    from sklearn.metrics import adjusted_rand_score as ari
    analyze(args.out_dir, args.n, args.seed, args.reps, ari)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
