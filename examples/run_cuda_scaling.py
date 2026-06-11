#!/usr/bin/env python3
"""Run the scaling + stability study inside the CUDA benchmark environment.

This is the scaling counterpart to ``examples/run_cuda_repro.py``. It answers a different
question from the supervised dataset suite: on real, *unlabelled* embeddings, does vanilla
EVoC return at a million points, and is its clustering *stable* as data is added? It runs
three stages and assembles a standard artifact tree under ``--output-dir`` so the local
plotting step (``scripts/plot_scaling_metrics.py``) can rebuild the documentation figures
from the downloaded tree, exactly like the general benchmark pipeline:

1. ``experiments/cache_embeddings.py`` — stream + cache the corpus once, in stream order so
   the first 100k rows are a subset of every larger prefix.
2. ``experiments/scale_compare.py`` — timing, cluster counts, and intrinsic dimension for
   vanilla EVoC vs DiReEVoC across the size ladder -> ``scale_results.json``.
3. ``experiments/stability.py`` — ARI of the shared 100k anchor as N grows (self-consistency,
   not ground truth) -> ``stability_results.json`` plus per-run anchor labels.

The cuVS/cuML imports need conda's ``libnvJitLink`` to load first; set ``LD_PRELOAD`` in the
environment (or Docker image) before invoking, e.g.
``LD_PRELOAD=$CONDA_PREFIX/lib/libnvJitLink.so.13.*``.

    python3 examples/run_cuda_scaling.py --output-dir artifacts/cuda-scaling \
        --sizes 100000 250000 500000 1000000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_command(name: str, command: list[str], output_dir: Path) -> dict:
    run_dir = output_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"{name}.stdout.txt"
    stderr_path = run_dir / f"{name}.stderr.txt"

    t0 = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    dt = time.perf_counter() - t0

    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "seconds": dt,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/cuda-scaling")
    parser.add_argument("--cache-dir", default="/tmp/cohere_en_emb")
    parser.add_argument("--dataset", default="Cohere/wikipedia-2023-11-embed-multilingual-v3")
    parser.add_argument("--config", default="en")
    parser.add_argument("--n-rows", type=int, default=1_000_000,
                        help="rows to cache; must be >= max(--sizes)")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[100_000, 250_000, 500_000, 1_000_000])
    parser.add_argument("--methods", nargs="+", default=["vanilla", "dire"])
    parser.add_argument("--scale-timeout", type=int, default=1800)
    parser.add_argument("--stability-timeout", type=int, default=900)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-scale", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [str(n) for n in sorted(args.sizes)]
    scale_results = output_dir / "scale_results.json"
    stability_results = output_dir / "stability_results.json"
    anchors_dir = output_dir / "stability_anchors"

    manifest = {
        "output_dir": str(output_dir),
        "dataset": args.dataset,
        "config": args.config,
        "n_rows": int(args.n_rows),
        "sizes": [int(n) for n in sizes],
        "methods": list(args.methods),
        "cache_dir": args.cache_dir,
        "runs": [],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.skip_cache:
        manifest["runs"].append(run_command(
            "cache_embeddings",
            [sys.executable, "experiments/cache_embeddings.py",
             "--dataset", args.dataset, "--config", args.config,
             "--n-rows", str(args.n_rows), "--cache-dir", args.cache_dir],
            output_dir))

    if not args.skip_scale:
        manifest["runs"].append(run_command(
            "scale_compare",
            [sys.executable, "experiments/scale_compare.py",
             "--cache-dir", args.cache_dir, "--sizes", *sizes,
             "--methods", *args.methods, "--timeout", str(args.scale_timeout),
             "--out", str(scale_results)],
            output_dir))

    if not args.skip_stability:
        manifest["runs"].append(run_command(
            "stability",
            [sys.executable, "experiments/stability.py",
             "--cache-dir", args.cache_dir, "--sizes", *sizes,
             "--methods", *args.methods, "--timeout", str(args.stability_timeout),
             "--out-dir", str(anchors_dir), "--json-out", str(stability_results)],
            output_dir))

    failures = [run for run in manifest["runs"] if run["returncode"] != 0]
    manifest["success"] = not failures
    manifest["failed_runs"] = [run["name"] for run in failures]
    manifest["scale_results"] = str(scale_results) if scale_results.exists() else None
    manifest["stability_results"] = str(stability_results) if stability_results.exists() else None
    (output_dir / "cuda_scaling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
