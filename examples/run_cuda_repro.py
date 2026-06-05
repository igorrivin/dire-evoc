#!/usr/bin/env python3
"""Run the CUDA-backed example/benchmark suite and collect reproducible artifacts.

The default CUDA repro runs the quickstart plus the broad dataset visual benchmark suite
from ``examples/benchmark_dataset_suite.py``. That suite compares DiReEVoC against pure
EVoC for each dataset and writes per-dataset metrics/figures plus suite-level CSV/SVG
summaries.

Typical Docker run:

    python3 examples/run_cuda_repro.py --output-dir artifacts/cuda

Full-scale one-run regime:

    python3 examples/run_cuda_repro.py --datasets scale --prefer-hf \
        --timing-repeats 1 --continue-on-error --output-dir artifacts/scale
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


def dataset_suite_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "examples/benchmark_dataset_suite.py",
        "--datasets",
        args.datasets,
        "--output-dir",
        str(output_dir),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--min-samples",
        str(args.min_samples),
        "--n-components",
        str(args.n_components),
        "--n-neighbors",
        str(args.n_neighbors),
        "--predict-neighbors",
        str(args.predict_neighbors),
        "--max-iter-layout",
        str(args.max_iter_layout),
        "--timing-repeats",
        str(args.timing_repeats),
        "--timing-primary",
        args.timing_primary,
        "--use-gpu",
        args.use_gpu,
        "--selector",
        args.selector,
    ]
    if args.base_min_cluster_size is not None:
        command.extend(["--base-min-cluster-size", str(args.base_min_cluster_size)])
    if args.base_n_clusters is not None:
        command.extend(["--base-n-clusters", str(args.base_n_clusters)])
    if args.include_graphs:
        command.append("--include-graphs")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.skip_existing:
        command.append("--skip-existing")
    if args.force_fetch:
        command.append("--force-fetch")
    if args.prefer_hf:
        command.append("--prefer-hf")
    if args.full:
        command.append("--full")
    if args.subsample is not None:
        command.extend(["--subsample", str(args.subsample)])
    if args.skip_fixed_k:
        command.append("--skip-fixed-k")
    return command


def timing_policy(timing_repeats: int, timing_primary: str) -> str:
    if max(1, int(timing_repeats)) == 1:
        return "single_run"
    if timing_primary == "mean_all":
        return "mean_all_first_included"
    return "warm_mean_first_discarded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/cuda")
    parser.add_argument(
        "--datasets",
        default="default",
        help="dataset suite selection: smoke, default, scale, all, graphs, or comma-separated slugs",
    )
    parser.add_argument("--include-graphs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument(
        "--prefer-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="prefer declared mirrors for OpenML datasets; default is OpenML first, then sklearn/HF fallback",
    )
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="holdout fraction; defaults to 0.0 for scale and 0.2 otherwise",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--base-min-cluster-size", type=int, default=None)
    parser.add_argument("--base-n-clusters", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--predict-neighbors", type=int, default=15)
    parser.add_argument("--max-iter-layout", type=int, default=100)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help=(
            "repeat each timed benchmark this many times; suite fit_seconds follows "
            "--timing-primary and metrics also record first-run, mean-all, and warm-only summaries"
        ),
    )
    parser.add_argument(
        "--timing-primary",
        default="warm_mean",
        choices=["warm_mean", "mean_all"],
        help="which repeated timing summary is reported in suite fit_seconds",
    )
    parser.add_argument("--use-gpu", default="auto", choices=["auto", "true", "false"])
    parser.add_argument(
        "--selector",
        default="silhouette",
        choices=["silhouette", "silhouette_raw", "base", "none"],
    )
    parser.add_argument("--skip-fixed-k", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.timing_repeats = max(1, int(args.timing_repeats))
    if args.test_fraction is None:
        args.test_fraction = 0.0 if args.datasets == "scale" else 0.2
    if args.datasets == "scale":
        args.full = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "output_dir": str(output_dir),
        "datasets": args.datasets,
        "include_graphs": args.include_graphs,
        "prefer_hf": args.prefer_hf,
        "timing_repeats": int(args.timing_repeats),
        "timing_primary": args.timing_primary,
        "timing_policy": timing_policy(args.timing_repeats, args.timing_primary),
        "runs": [],
    }

    manifest["runs"].append(
        run_command("quickstart", [sys.executable, "examples/quickstart.py"], output_dir)
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    suite_command = dataset_suite_command(args, output_dir)
    manifest["runs"].append(
        run_command("benchmark_dataset_suite", suite_command, output_dir)
    )

    suite_manifest_path = output_dir / "manifest.json"
    suite_manifest = None
    if suite_manifest_path.exists():
        suite_manifest = json.loads(suite_manifest_path.read_text())

    failures = [run for run in manifest["runs"] if run["returncode"] != 0]
    manifest["success"] = not failures
    manifest["failed_runs"] = [run["name"] for run in failures]
    manifest["suite_manifest"] = suite_manifest
    manifest["summary_csv"] = str(output_dir / "suite_metrics.csv")
    manifest["summary_svgs"] = [
        str(output_dir / "suite_auto_ari.svg"),
        str(output_dir / "suite_auto_delta.svg"),
        str(output_dir / "suite_fit_seconds.svg"),
        str(output_dir / "suite_fit_seconds_warm.svg"),
        str(output_dir / "suite_fit_seconds_mean_all.svg"),
    ]
    (output_dir / "cuda_repro_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
