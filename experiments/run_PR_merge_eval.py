#!/usr/bin/env python3
"""Merge-decision evidence for the restart-consensus PR.

Runs the full README dataset suite twice -- single-run baseline (restarts=1) and
8-restart consensus voting -- so we can compare apples-to-apples and decide whether
``restart_combine="consensus"`` is worth merging to master. Pure orchestration around
``examples/benchmark_dataset_suite.py``; intended to run inside the CUDA container.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

README_DATASETS = (
    "blobs30k,circles30k,moons30k,spirals_clean30k,mnist,fashion_mnist,"
    "cifar10,optdigits,pendigits,satimage,segment,vehicle"
)


def suite_cmd(output_dir, extra):
    return [
        sys.executable, "examples/benchmark_dataset_suite.py",
        "--datasets", README_DATASETS,
        "--selector", "auto", "--prefer-hf",
        "--timing-repeats", "1", "--test-fraction", "0.0",
        "--full", "--skip-fixed-k", "--continue-on-error",
        "--output-dir", str(output_dir),
        *extra,
    ]


def run(label, cmd):
    print(f"\n########## {label} ##########", flush=True)
    print(time.strftime("%H:%M:%SZ", time.gmtime()), "  ".join(cmd), flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    print(f"[{label}] returncode={rc} in {time.perf_counter() - t0:.1f}s", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--restart-threshold", type=float, default=0.5)
    ap.add_argument("--restart-min-cluster-size", type=int, default=2)
    ap.add_argument("--single-dir", default="artifacts/suite-single")
    ap.add_argument("--consensus-dir", default="artifacts/suite-consensus")
    args = ap.parse_args()

    rc1 = run("SINGLE-RUN baseline", suite_cmd(args.single_dir, ["--restarts", "1"]))
    rc2 = run(
        f"CONSENSUS ({args.restarts} restarts)",
        suite_cmd(args.consensus_dir, [
            "--restarts", str(args.restarts),
            "--restart-combine", "consensus",
            "--restart-threshold", str(args.restart_threshold),
            "--restart-min-cluster-size", str(args.restart_min_cluster_size),
        ]),
    )
    print(f"\n########## DONE (single rc={rc1}, consensus rc={rc2}) ##########", flush=True)
    print("single   :", Path(args.single_dir, "suite_metrics.csv"))
    print("consensus:", Path(args.consensus_dir, "suite_metrics.csv"))
    return rc1 or rc2


if __name__ == "__main__":
    raise SystemExit(main())
