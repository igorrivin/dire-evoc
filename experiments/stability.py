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
import json
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


def _layers_of(fields) -> list:
    return [fields[k] for k in fields.files
            if k.startswith("layer_") and k != "layer_ncl"]


def _range(values: list[float]) -> dict:
    """Summarize a list of per-step ARIs as a representative value plus its range."""
    clean = [float(v) for v in values if v == v]  # drop NaN
    if not clean:
        return {"value": None, "min": None, "max": None, "per_step": []}
    return {
        "value": round(sum(clean) / len(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "per_step": [round(v, 3) for v in clean],
    }


def measure(args) -> dict:
    """Compute the four stability measurements per method into a structured dict.

    Mirrors the printed tables: a same-N two-seed noise floor (default labels and the
    oracle best-matching layer) and the across-N agreement of the shared anchor (default
    labels and oracle best layer). This is the dict serialized to stability_results.json,
    which scripts/plot_scaling_metrics.py turns into the documentation figure.
    """
    from sklearn.metrics import adjusted_rand_score as ari
    sizes = sorted(args.sizes)
    by_method = {}
    for method in args.methods:
        rec = {"floor_default": None, "floor_best": None,
               "across_default": _range([]), "across_oracle": _range([])}
        # same-N two-seed noise floor (default labels + oracle best layer)
        try:
            f0 = load_anchor(args.out_dir, method, ANCHOR, 0)
            f1 = load_anchor(args.out_dir, method, ANCHOR, 1)
            rec["floor_default"] = round(float(ari(f0["default"], f1["default"])), 3)
            l0, l1 = _layers_of(f0), _layers_of(f1)
            rec["floor_best"] = round(float(max(
                (ari(a, b) for a, b in itertools.product(l0, l1)),
                default=float("nan"))), 3)
        except FileNotFoundError:
            pass
        # across-N agreement of the shared anchor (consecutive steps)
        defaults, oracles = [], []
        for ni, nj in zip(sizes[:-1], sizes[1:]):
            try:
                fi = load_anchor(args.out_dir, method, ni, 0)
                fj = load_anchor(args.out_dir, method, nj, 0)
            except FileNotFoundError:
                continue
            defaults.append(float(ari(fi["default"], fj["default"])))
            li, lj = _layers_of(fi), _layers_of(fj)
            oracles.append(float(max(
                (ari(a, b) for a, b in itertools.product(li, lj)),
                default=float("nan"))))
        rec["across_default"] = _range(defaults)
        rec["across_oracle"] = _range(oracles)
        by_method[method] = rec

    def _summary_row(key, label, short, field, attr):
        row = {"key": key, "label": label, "short": short}
        for method, plot_key in (("vanilla", "vanilla"), ("dire", "dire")):
            rec = by_method.get(method, {})
            if field in ("floor_default", "floor_best"):
                row[f"{plot_key}_value"] = rec.get(field)
            else:
                rng = rec.get(field, {})
                row[f"{plot_key}_value"] = rng.get("value")
                row[f"{plot_key}_range"] = [rng.get("min"), rng.get("max")]
        return row

    summary = {
        "_about": ("Stability of vanilla EVoC vs DiReEVoC under growing N, measured as the "
                   "ARI of the shared 100k-point anchor as N grows. Wikipedia embeddings "
                   "have no ground truth, so ARI here is self-consistency (each method judged "
                   "against itself), not agreement with class labels. Produced by "
                   "experiments/stability.py."),
        "dataset": "Cohere/wikipedia-2023-11-embed-multilingual-v3 (en)",
        "anchor": ANCHOR,
        "sizes": sizes,
        "metric": "adjusted_rand_index",
        "methods": by_method,
        "measurements": [
            _summary_row("floor_default",
                         "Noise floor: default labels\n(same N=100k, two seeds)",
                         "floor (default)", "floor_default", None),
            _summary_row("floor_best",
                         "Noise floor: best layer\n(same N=100k, two seeds)",
                         "floor (best layer)", "floor_best", None),
            _summary_row("across_default",
                         "Across N: default labels\n(100k -> 1M)",
                         "across N (default)", "across_default", None),
            _summary_row("across_oracle",
                         "Across N: oracle best layer\n(100k -> 1M)",
                         "across N (oracle layer)", "across_oracle", None),
        ],
    }
    return summary


def analyze(args):
    sizes = sorted(args.sizes)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary = measure(args)

    print("\n================ STABILITY (ARI on shared 100k anchor) ================")
    print("Noise floor = same data (N=100k), two seeds — instability only counts below this.\n")
    for method in args.methods:
        rec = summary["methods"][method]
        print(f"---- {method} ----")
        print(f"  noise floor (100k seed0 vs seed1) default-labels ARI = {rec['floor_default']}")
        print(f"  noise floor (100k seed0 vs seed1) best-layer    ARI = {rec['floor_best']}")
        print(f"  default-labels ARI across N: {rec['across_default']['per_step']} "
              f"(value {rec['across_default']['value']})")
        print(f"  oracle best-layer ARI across N: {rec['across_oracle']['per_step']} "
              f"(value {rec['across_oracle']['value']})")
        print()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[stability] wrote {out}")
    return summary


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
    ap.add_argument("--json-out", default="experiments/stability_results.json",
                    help="write the structured stability summary here (artifact for plotting)")
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
