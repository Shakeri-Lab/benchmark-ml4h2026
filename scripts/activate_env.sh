#!/usr/bin/env bash
# Activate the conda environment created via setup_env.sh.
# Defaults to the local prefix under the current working directory.
# Exports PYTHONNOUSERSITE=1 to avoid picking up ~/.local packages on HPC.

# Save and restore caller shell options to avoid leaving -e/-u enabled in interactive shells.
__OLD_SHELL_OPTS="$(set +o)"

ENV_NAME="bg-benchmark"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PREFIX="${REPO_ROOT}/.conda_envs/${ENV_NAME}"
ENV_PREFIX="${DEFAULT_PREFIX}"
USE_HOME=0

usage() {
  cat <<'EOF'
Usage: activate_env.sh [--prefix /path/to/env] [--use-home]
  --prefix   Activate env at the given prefix (default: $PWD/.conda_envs/bg-benchmark)
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
  conda activate "${ENV_NAME}"
  export PYTHONNOUSERSITE=1
  echo "[✓] Activated '${ENV_NAME}' (home env)."
else
  conda activate "${ENV_PREFIX}"
  export PYTHONNOUSERSITE=1
  echo "[✓] Activated '${ENV_PREFIX}'."
fi

# Restore caller shell options
eval "${__OLD_SHELL_OPTS}"
