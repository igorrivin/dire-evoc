#!/usr/bin/env python3
"""Run the CUDA-backed example/benchmark suite and collect reproducible artifacts.

This is intentionally a thin orchestrator around the public examples. It records the
commands, stdout/stderr, runtime, return code, dataset paths, metrics JSON, and SVG
figures in one output directory.

Typical Docker run:

    python3 examples/run_cuda_repro.py --output-dir artifacts/cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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


def write_synthetic_npz(path: Path, n=12_000, k=8, d=32, spread=0.10, seed=0) -> None:
    from benchmark_visuals import gaussian_blobs

    X, y = gaussian_blobs(n=n, k=k, d=d, spread=spread, seed=seed)
    np.savez(path, X=X, y=y)


def prepare_datasets(output_dir: Path, include_dbpedia: bool) -> dict:
    from prepare_datasets import dbpedia14, fashion_mnist, olivetti

    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "synthetic": dataset_dir / "synthetic_blobs_12k.npz",
        "fashion_mnist": dataset_dir / "fashion_mnist.npz",
        "olivetti": dataset_dir / "olivetti.npz",
    }
    write_synthetic_npz(paths["synthetic"])
    fashion_mnist(str(paths["fashion_mnist"]))
    olivetti(str(paths["olivetti"]))

    if include_dbpedia:
        paths["dbpedia14"] = dataset_dir / "dbpedia14_bge_small.npz"
        dbpedia14(str(paths["dbpedia14"]))

    return {key: str(path) for key, path in paths.items()}


def benchmark_visuals_command(
    dataset: str,
    label: str,
    output_dir: Path,
    *,
    npz: str | None = None,
    synthetic: bool = False,
    subsample: int | None = None,
    test_fraction: float = 0.0,
) -> list[str]:
    command = [
        sys.executable,
        "examples/benchmark_visuals.py",
        "--label",
        label,
        "--output-dir",
        str(output_dir / dataset),
        "--test-fraction",
        str(test_fraction),
    ]
    if synthetic:
        command.extend(["--synthetic-blobs", "--n-samples", "12000"])
    else:
        command.extend(["--npz", str(npz)])
    if subsample is not None:
        command.extend(["--subsample", str(subsample)])
    return command


def benchmark_quality_command(
    dataset: str,
    label: str,
    output_dir: Path,
    npz: str,
    *,
    subsample: int | None = None,
) -> tuple[str, list[str]]:
    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "examples/benchmark_quality.py",
        "--npz",
        npz,
        "--label",
        label,
    ]
    if subsample is not None:
        command.extend(["--subsample", str(subsample)])
    return f"benchmark_quality_{dataset}", command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/cuda")
    parser.add_argument("--skip-dbpedia", action="store_true")
    parser.add_argument("--fashion-subsample", type=int, default=None)
    parser.add_argument("--dbpedia-subsample", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "output_dir": str(output_dir),
        "datasets": {},
        "runs": [],
    }

    manifest["runs"].append(
        run_command("quickstart", [sys.executable, "examples/quickstart.py"], output_dir)
    )

    t0 = time.perf_counter()
    datasets = prepare_datasets(output_dir, include_dbpedia=not args.skip_dbpedia)
    manifest["datasets"] = datasets
    manifest["prepare_datasets_seconds"] = time.perf_counter() - t0

    visual_runs = [
        (
            "benchmark_visuals_synthetic",
            benchmark_visuals_command(
                "synthetic_blobs_12k",
                "Synthetic blobs 12k",
                output_dir,
                synthetic=True,
                test_fraction=0.2,
            ),
        ),
        (
            "benchmark_visuals_olivetti",
            benchmark_visuals_command(
                "olivetti",
                "Olivetti faces",
                output_dir,
                npz=datasets["olivetti"],
            ),
        ),
        (
            "benchmark_visuals_fashion_mnist",
            benchmark_visuals_command(
                "fashion_mnist",
                "Fashion-MNIST",
                output_dir,
                npz=datasets["fashion_mnist"],
                subsample=args.fashion_subsample,
            ),
        ),
    ]
    if "dbpedia14" in datasets:
        visual_runs.append(
            (
                "benchmark_visuals_dbpedia14",
                benchmark_visuals_command(
                    "dbpedia14",
                    "DBpedia-14 BGE-small",
                    output_dir,
                    npz=datasets["dbpedia14"],
                    subsample=args.dbpedia_subsample,
                ),
            )
        )

    for name, command in visual_runs:
        manifest["runs"].append(run_command(name, command, output_dir))
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    quality_runs = [
        benchmark_quality_command(
            "synthetic", "Synthetic blobs 12k", output_dir, datasets["synthetic"]
        ),
        benchmark_quality_command("olivetti", "Olivetti faces", output_dir, datasets["olivetti"]),
        benchmark_quality_command(
            "fashion_mnist",
            "Fashion-MNIST",
            output_dir,
            datasets["fashion_mnist"],
            subsample=args.fashion_subsample,
        ),
    ]
    if "dbpedia14" in datasets:
        quality_runs.append(
            benchmark_quality_command(
                "dbpedia14",
                "DBpedia-14 BGE-small",
                output_dir,
                datasets["dbpedia14"],
                subsample=args.dbpedia_subsample,
            )
        )

    for name, command in quality_runs:
        manifest["runs"].append(run_command(name, command, output_dir))
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    failures = [run for run in manifest["runs"] if run["returncode"] != 0]
    manifest["success"] = not failures
    manifest["failed_runs"] = [run["name"] for run in failures]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
