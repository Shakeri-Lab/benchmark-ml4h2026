#!/usr/bin/env bash
# Create the conda environment using Miniforge on the HPC.
# Defaults to a local prefix under the current working directory to avoid $HOME quotas.

set -euo pipefail

ENV_NAME="bg-benchmark"
PYTHON_VER="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PREFIX="${REPO_ROOT}/.conda_envs/${ENV_NAME}"
ENV_PREFIX="${DEFAULT_PREFIX}"
USE_HOME=0

usage() {
  cat <<'EOF'
Usage: setup_env.sh [--prefix /path/to/env] [--use-home]
  --prefix   Install env at the given prefix (default: $PWD/.conda_envs/bg-benchmark)
  --use-home Use the default conda env location in $HOME (env name: bg-benchmark)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --use-home)
      USE_HOME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "[+] Loading miniforge module..."
module load miniforge/24.3.0-py3.11

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ $USE_HOME -eq 1 ]]; then
  TARGET="${ENV_NAME}"
  CREATE_CMD=(conda create -y -n "${ENV_NAME}" "python=${PYTHON_VER}")
else
  TARGET="${ENV_PREFIX}"
  CREATE_CMD=(conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VER}")
fi

if conda env list | awk '{print $1}' | grep -qx "${TARGET}"; then
  echo "[=] Conda env '${TARGET}' already exists; skipping creation."
else
  echo "[+] Creating conda env at '${TARGET}' (python ${PYTHON_VER})..."
  "${CREATE_CMD[@]}"
fi

echo "[+] Activating env and installing requirements..."
if [[ $USE_HOME -eq 1 ]]; then
  conda activate "${ENV_NAME}"
else
  conda activate "${ENV_PREFIX}"
fi

pip install -r "${REPO_ROOT}/requirements.txt"

echo "[✓] Environment ready at '${TARGET}'."
