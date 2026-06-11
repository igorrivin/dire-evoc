#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the scaling + stability study on a remote CUDA host, download the artifact tree, and
rebuild the documentation figures locally — the scaling counterpart to
scripts/run_remote_cuda_repro.sh.

Usage:
  scripts/run_remote_cuda_scaling.sh [user@host] [extra run_cuda_scaling.py args...]

Remote target:
  pass user@gpu-host as the first argument, or set REMOTE=user@gpu-host
  remote dir:   $HOME/dire-evoc on the remote host
  image:        dire-evoc:cuda128
  remote out:   artifacts/cuda-scaling
  local out:    artifacts/cuda-scaling

After download this runs, unless PLOT=0:
  python scripts/plot_scaling_metrics.py --artifacts <local out>
which regenerates docs/_static/benchmark_scaling_*.svg from the downloaded tree.

Environment overrides:
  REMOTE=user@gpu-host
  REMOTE_DIR=/custom/remote/path
  IMAGE=dire-evoc:cuda128
  REMOTE_OUTPUT=artifacts/cuda-scaling
  LOCAL_OUTPUT=artifacts/cuda-scaling
  DOCKER="sudo docker"
  LD_PRELOAD_GLOB='$CONDA_PREFIX/lib/libnvJitLink.so.13.*'  (preloaded in-container)
  PLOT=0          # skip the local figure rebuild
  SKIP_SYNC=1
  SKIP_BUILD=1
  SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

Examples:
  scripts/run_remote_cuda_scaling.sh user@gpu-host
  scripts/run_remote_cuda_scaling.sh user@gpu-host --sizes 100000 250000 500000 1000000
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REMOTE="${REMOTE:-}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  REMOTE="$1"
  shift
fi

if [[ -z "${REMOTE}" ]]; then
  echo "error: provide a remote CUDA host, e.g. scripts/run_remote_cuda_scaling.sh user@gpu-host" >&2
  echo >&2
  usage >&2
  exit 2
fi

SSH_OPTS="${SSH_OPTS:-}"
ssh_cmd=(ssh)
rsync_cmd=(rsync)
if [[ -n "${SSH_OPTS}" ]]; then
  # shellcheck disable=SC2206
  ssh_opts_array=(${SSH_OPTS})
  ssh_cmd+=("${ssh_opts_array[@]}")
  rsync_cmd+=(-e "ssh ${SSH_OPTS}")
fi

REMOTE_DIR="${REMOTE_DIR:-}"
if [[ -z "${REMOTE_DIR}" ]]; then
  REMOTE_DIR="$("${ssh_cmd[@]}" "${REMOTE}" 'printf "%s/dire-evoc" "$HOME"')"
fi
IMAGE="${IMAGE:-dire-evoc:cuda128}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-artifacts/cuda-scaling}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-artifacts/cuda-scaling}"
DOCKER="${DOCKER:-sudo docker}"
LD_PRELOAD_GLOB="${LD_PRELOAD_GLOB:-\$CONDA_PREFIX/lib/libnvJitLink.so.13.*}"
RUN_ARGS=("$@")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "remote:       ${REMOTE}"
echo "remote dir:   ${REMOTE_DIR}"
echo "docker image: ${IMAGE}"
echo "remote out:   ${REMOTE_OUTPUT}"
echo "local out:    ${LOCAL_OUTPUT}"

"${ssh_cmd[@]}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  echo "syncing working tree to remote..."
  "${rsync_cmd[@]}" -az \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude '.mypy_cache' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.egg-info' \
    --exclude 'build' \
    --exclude 'dist' \
    --exclude 'docs/_build' \
    --exclude '.venv' \
    --exclude 'artifacts' \
    "${ROOT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "building Docker image on remote..."
  "${ssh_cmd[@]}" "${REMOTE}" "cd '${REMOTE_DIR}' && ${DOCKER} build -t '${IMAGE}' ."
fi

remote_args=()
if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
  for arg in "${RUN_ARGS[@]}"; do
    remote_args+=("$(printf '%q' "${arg}")")
  done
fi
remote_args_string=""
if [[ ${#remote_args[@]} -gt 0 ]]; then
  remote_args_string="${remote_args[*]}"
fi

echo "running CUDA scaling + stability study..."
set +e
"${ssh_cmd[@]}" "${REMOTE}" "cd '${REMOTE_DIR}' && ${DOCKER} run --rm --gpus all \
  -e LD_PRELOAD=\"\$(ls ${LD_PRELOAD_GLOB} 2>/dev/null | head -n1)\" \
  -v '${REMOTE_DIR}:/workspace/dire-evoc' \
  -w /workspace/dire-evoc \
  '${IMAGE}' \
  python3 examples/run_cuda_scaling.py --output-dir '${REMOTE_OUTPUT}' ${remote_args_string}"
run_status=$?
set -e

echo "downloading artifacts..."
mkdir -p "${LOCAL_OUTPUT}"
"${rsync_cmd[@]}" -az "${REMOTE}:${REMOTE_DIR}/${REMOTE_OUTPUT}/" "${LOCAL_OUTPUT}/"

echo "done"
echo "manifest:  ${LOCAL_OUTPUT}/cuda_scaling_manifest.json"
echo "scale:     ${LOCAL_OUTPUT}/scale_results.json"
echo "stability: ${LOCAL_OUTPUT}/stability_results.json"

if [[ "${PLOT:-1}" != "0" ]]; then
  echo "rebuilding documentation figures from the downloaded tree..."
  python "${ROOT_DIR}/scripts/plot_scaling_metrics.py" --artifacts "${LOCAL_OUTPUT}"
fi

exit "${run_status}"
