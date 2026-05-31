#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the CUDA repro suite on a remote Docker host and download all artifacts.

Usage:
  scripts/run_remote_cuda_repro.sh [user@host] [extra run_cuda_repro.py args...]

Remote target:
  pass user@gpu-host as the first argument, or set REMOTE=user@gpu-host
  remote dir:   $HOME/dire-evoc on the remote host
  image:        dire-evoc:cuda128
  remote out:   artifacts/cuda
  local out:    artifacts/cuda

Environment overrides:
  REMOTE=user@gpu-host
  REMOTE_DIR=/custom/remote/path
  IMAGE=dire-evoc:cuda128
  REMOTE_OUTPUT=artifacts/cuda
  LOCAL_OUTPUT=artifacts/cuda
  DOCKER="sudo docker"
  SKIP_SYNC=1
  SKIP_BUILD=1

Examples:
  scripts/run_remote_cuda_repro.sh user@gpu-host --skip-dbpedia
  LOCAL_OUTPUT=artifacts/a10-full scripts/run_remote_cuda_repro.sh user@gpu-host
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
  echo "error: provide a remote CUDA host, e.g. scripts/run_remote_cuda_repro.sh user@gpu-host" >&2
  echo >&2
  usage >&2
  exit 2
fi

REMOTE_DIR="${REMOTE_DIR:-}"
if [[ -z "${REMOTE_DIR}" ]]; then
  REMOTE_DIR="$(ssh "${REMOTE}" 'printf "%s/dire-evoc" "$HOME"')"
fi
IMAGE="${IMAGE:-dire-evoc:cuda128}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-artifacts/cuda}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-artifacts/cuda}"
DOCKER="${DOCKER:-sudo docker}"
REPRO_ARGS=("$@")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "remote:       ${REMOTE}"
echo "remote dir:   ${REMOTE_DIR}"
echo "docker image: ${IMAGE}"
echo "remote out:   ${REMOTE_OUTPUT}"
echo "local out:    ${LOCAL_OUTPUT}"

ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  echo "syncing working tree to remote..."
  rsync -az \
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
  ssh "${REMOTE}" "cd '${REMOTE_DIR}' && ${DOCKER} build -t '${IMAGE}' ."
fi

remote_args=()
for arg in "${REPRO_ARGS[@]}"; do
  remote_args+=("$(printf '%q' "${arg}")")
done

echo "running CUDA repro suite..."
ssh "${REMOTE}" "cd '${REMOTE_DIR}' && ${DOCKER} run --rm --gpus all \
  -v '${REMOTE_DIR}:/workspace/dire-evoc' \
  -w /workspace/dire-evoc \
  '${IMAGE}' \
  python3 examples/run_cuda_repro.py --output-dir '${REMOTE_OUTPUT}' ${remote_args[*]}"

echo "downloading artifacts..."
mkdir -p "${LOCAL_OUTPUT}"
rsync -az "${REMOTE}:${REMOTE_DIR}/${REMOTE_OUTPUT}/" "${LOCAL_OUTPUT}/"

echo "done"
echo "manifest: ${LOCAL_OUTPUT}/manifest.json"
echo "figures:  ${LOCAL_OUTPUT}/*/*.svg"
echo "metrics:  ${LOCAL_OUTPUT}/*/metrics.json"
