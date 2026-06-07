#!/usr/bin/env bash
# Runs INSIDE the container (repo at /workspace/dire-evoc, cwd here). Re-runs the
# benchmark suites that back the README + docs on the remove-merge-fragments branch.
set -uo pipefail
git config --global --add safe.directory /workspace/dire-evoc 2>/dev/null || true
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR=/workspace/dire-evoc/.cache/numba
export HF_HOME=/workspace/dire-evoc/.cache/huggingface
mkdir -p "$NUMBA_CACHE_DIR" "$HF_HOME"

step() { echo; echo "########## $* ##########"; date -u +%H:%M:%SZ; }

case "${1:-all}" in
  selector)
    step "selector-auto-full sweep (README headline table + SVGs)"
    python3 examples/benchmark_dataset_suite.py \
      --datasets blobs30k,circles30k,moons30k,spirals_clean30k,mnist,fashion_mnist,cifar10,optdigits,pendigits,satimage,segment,vehicle \
      --selector auto --output-dir artifacts/selector-auto-full \
      --prefer-hf --timing-repeats 1 --test-fraction 0.0 --full --skip-fixed-k --continue-on-error
    ;;
  phases)
    step "per-phase timing on MNIST full (README timing table)"
    python3 examples/benchmark_visuals.py --openml mnist_784 --subsample 70000 \
      --selector auto --max-iter-layout 128 --test-fraction 0.0 \
      --output-dir artifacts/phase-timing-mnist
    ;;
  default)
    step "default suite, repeated timing (docs comparative + fit figures)"
    python3 examples/run_cuda_repro.py --datasets default --prefer-hf \
      --timing-repeats 5 --timing-primary warm_mean --continue-on-error \
      --output-dir artifacts/cuda-timing-repeats-default
    ;;
  smoke)
    step "smoke suite, repeated timing (combined figures)"
    python3 examples/run_cuda_repro.py --datasets smoke --prefer-hf \
      --timing-repeats 20 --timing-primary warm_mean --continue-on-error \
      --output-dir artifacts/cuda-timing-repeats-smoke
    ;;
  scale)
    step "scale suite (docs scale figures)"
    python3 examples/run_cuda_repro.py --datasets scale --prefer-hf \
      --timing-repeats 1 --continue-on-error \
      --output-dir artifacts/cuda-scale
    ;;
esac
echo; echo "########## DONE ${1:-all} ##########"; date -u +%H:%M:%SZ
