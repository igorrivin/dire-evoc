#!/usr/bin/env python3
"""Run the standard README benchmark with the size-aware restart-consensus policy.

This is the one-command entry for the full usual benchmark on this PR's branch: the same
12-dataset selector="auto" sweep as the README, but with multiple-restart consensus voting
applied ONLY to small, run-variant datasets (n <= --small-max-n); large datasets run once.
It is the evidence for whether restart-consensus is worth merging: small sets pick up the
consensus gain, large sets are unchanged single runs. Intended to run in the CUDA container.
"""
import argparse
import subprocess
import sys
import time

README_DATASETS = (
    "blobs30k,circles30k,moons30k,spirals_clean30k,mnist,fashion_mnist,"
    "cifar10,optdigits,pendigits,satimage,segment,vehicle"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=8, help="restarts for SMALL datasets")
    ap.add_argument("--small-max-n", type=int, default=3000)
    ap.add_argument("--restart-threshold", type=float, default=0.5)
    ap.add_argument("--restart-min-cluster-size", type=int, default=2)
    ap.add_argument("--output-dir", default="artifacts/selector-auto-consensus")
    args = ap.parse_args()

    cmd = [
        sys.executable, "examples/benchmark_dataset_suite.py",
        "--datasets", README_DATASETS,
        "--selector", "auto", "--prefer-hf",
        "--timing-repeats", "1", "--test-fraction", "0.0",
        "--full", "--skip-fixed-k", "--continue-on-error",
        "--restarts", str(args.restarts),
        "--restart-combine", "consensus",
        "--restart-threshold", str(args.restart_threshold),
        "--restart-min-cluster-size", str(args.restart_min_cluster_size),
        "--small-max-n", str(args.small_max_n),
        "--output-dir", args.output_dir,
    ]
    print(time.strftime("%H:%M:%SZ", time.gmtime()), " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    print(f"\nstandard size-aware consensus benchmark: rc={rc} in {time.perf_counter()-t0:.1f}s")
    print("metrics:", f"{args.output_dir}/suite_metrics.csv  (restart_regime column marks small vs single)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
