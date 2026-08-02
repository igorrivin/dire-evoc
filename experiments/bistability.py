"""Characterize DiReEVoC's same-seed bistability at fixed N (honest determinism study).

Earlier single-pair "determinism" numbers were unreliable: at fixed seed, DiReEVoC's
selected clustering appears to land on one of a few discrete granularity *branches*
(e.g. ~700 vs ~1250 clusters at N=100k), so two runs agree highly when they match branch
and poorly when they do not. A single pair therefore estimates nothing. This runs many
same-seed reps and reports the actual distribution.

Note: contrary to an earlier assumption, DiReCuVS uses cuVS k-NN for n_samples >= 10,000,
so BOTH 100k and 250k use cuVS (not exact PyTorch) -- the k-NN method actually used is
scraped from the DiRe log per rep and reported, rather than assumed.

Per rep it records the selected cluster count, the selected-layer index, the layer count,
the per-layer cluster counts, and the k-NN backend log lines. It then reports, per N:
  - the branch distribution (selected granularities across reps)
  - pairwise ARI over all reps (default labels and oracle best-layer), mean +/- std
  - the distinct k-NN method lines observed

    python experiments/bistability.py --cache-dir /tmp/cohere_en_emb --reps 10 \\
        --sizes 100000 250000 --out-dir experiments/bistability_anchors
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SHARD = 100_000
ANCHOR = 100_000
KNN_LOG = re.compile(r"(k-?NN|brute|IVF|flat|cuVS|FP16|FP32|exact)", re.IGNORECASE)


def load_n(cache_dir, n):
    shards = sorted(Path(cache_dir).glob("emb_*.npy"))
    need = (n + SHARD - 1) // SHARD
    return np.ascontiguousarray(
        np.concatenate([np.load(s) for s in shards[:need]], axis=0)[:n], dtype=np.float32)


def ncl(lbl):
    return int(np.unique(lbl[lbl != -1]).shape[0])


def run_worker(n, seed, rep, cache_dir, out_dir, restarts=1):
    X = load_n(cache_dir, n)
    from dire_evoc import DiReEVoC
    m = DiReEVoC(use_gpu="auto", random_state=seed, restarts=restarts).fit(X)
    layers = [np.asarray(l)[:ANCHOR] for l in m.cluster_layers_]
    default = np.asarray(m.labels_)[:ANCHOR]
    np.savez(Path(out_dir) / f"dire_R{restarts}_{n}_s{seed}_r{rep}.npz",
             default=default,
             layer_ncl=np.array([ncl(l) for l in layers]),
             selected_layer=int(getattr(m, "selected_layer_", -1)),
             **{f"layer_{i}": l for i, l in enumerate(layers)})
    print(f"RESULT n={n} rep={rep} sel_ncl={ncl(default)} "
          f"sel_layer={getattr(m, 'selected_layer_', -1)} n_layers={len(layers)}", flush=True)
    return 0


def _layers(f):
    return [f[k] for k in f.files if k.startswith("layer_") and k != "layer_ncl"]


def orchestrate(args):
    knn_methods = {}
    for n in args.sizes:
        knn_methods[n] = set()
        for rep in range(args.reps):
            out = Path(args.out_dir) / f"dire_R{args.restarts}_{n}_s{args.seed}_r{rep}.npz"
            if out.exists():
                print(f"[skip] {out.name}", flush=True)
                continue
            print(f"[fit ] dire@{n} s{args.seed} r{rep} (restarts={args.restarts})...", flush=True)
            cmd = [sys.executable, __file__, "--worker", "--n", str(n), "--seed",
                   str(args.seed), "--rep", str(rep), "--cache-dir", args.cache_dir,
                   "--out-dir", args.out_dir, "--restarts", str(args.restarts)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=args.timeout, start_new_session=True)
                line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
                knn = sorted({l.split("- ")[-1].strip() for l in proc.stderr.splitlines()
                              if "k-NN" in l or "brute-force search" in l})
                knn_methods[n].update(knn)
                print(f"       {line or 'FAILED'}", flush=True)
            except subprocess.TimeoutExpired:
                print("       TIMEOUT", flush=True)
        Path(args.out_dir, "knn_methods.json").write_text(
            json.dumps({str(k): sorted(v) for k, v in knn_methods.items()}, indent=2))


def analyze(args):
    from sklearn.metrics import adjusted_rand_score as ari
    km = {}
    kmf = Path(args.out_dir, "knn_methods.json")
    if kmf.exists():
        km = json.loads(kmf.read_text())

    print("\n================ SAME-SEED BISTABILITY (seed 0) ================\n")
    for n in args.sizes:
        fs = []
        for rep in range(args.reps):
            p = Path(args.out_dir) / f"dire_R{args.restarts}_{n}_s{args.seed}_r{rep}.npz"
            if p.exists():
                fs.append(np.load(p))
        if len(fs) < 2:
            print(f"N={n}: <insufficient reps>")
            continue
        sel = [ncl(f["default"]) for f in fs]
        nlay = [len(_layers(f)) for f in fs]
        sl = [int(f["selected_layer"]) for f in fs]
        d = [ari(a["default"], b["default"]) for a, b in itertools.combinations(fs, 2)]
        o = [max(ari(x, y) for x, y in itertools.product(_layers(a), _layers(b)))
             for a, b in itertools.combinations(fs, 2)]
        print(f"N={n}  ({len(fs)} reps)")
        print(f"  selected cluster count : {sorted(sel)}")
        print(f"  n_layers / sel_layer   : {nlay} / {sl}")
        print(f"  default-label ARI      : {np.mean(d):.3f} +/- {np.std(d):.3f}  "
              f"[min {np.min(d):.3f}, max {np.max(d):.3f}]  (pairs={len(d)})")
        print(f"  oracle  best-layer ARI : {np.mean(o):.3f} +/- {np.std(o):.3f}  "
              f"[min {np.min(o):.3f}, max {np.max(o):.3f}]")
        print(f"  k-NN method(s) observed: {km.get(str(n), '?')}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--n", type=int); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    ap.add_argument("--out-dir", default="experiments/bistability_anchors")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 250_000])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return run_worker(args.n, args.seed, args.rep, args.cache_dir, args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        orchestrate(args)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
