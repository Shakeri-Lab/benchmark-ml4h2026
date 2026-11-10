#!/usr/bin/env bash
#SBATCH --job-name=bg-hyper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CONFIG_DIR=${1:-}
ARCH=${2:-}
DATASET=${3:-}

if [[ -z "${CONFIG_DIR}" || -z "${ARCH}" || -z "${DATASET}" ]]; then
  echo "Usage: $0 <config_dir> <arch> <dataset>" >&2
  exit 1
fi

mkdir -p logs

cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark || {
  echo "Cannot cd into project directory!" >&2
  exit 1
}

module load miniforge/24.3.0-py3.11
_OLD_ARGS=("$@")
set --
source ./scripts/activate_env.sh
set -- "${_OLD_ARGS[@]}"

extra_args=()
if [[ "${HYPER_FAIL_IF_EXISTS:-0}" == "1" ]]; then
  extra_args+=(--fail-if-exists)
fi
if [[ -n "${HYPER_OUTPUT_ROOT:-}" ]]; then
  extra_args+=(--output-root "${HYPER_OUTPUT_ROOT}")
fi

if [[ "${DATASET}" == "action_cf_benchmark" || "${DATASET}" == "action_cf_benchmark_controliq" ]]; then
  python exp/run_hyperparameter_search_cf.py \
    --config-dir "${CONFIG_DIR}" \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    "${extra_args[@]}"
else
  python exp/run_hyperparameter_search.py \
    --config-dir "${CONFIG_DIR}" \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    "${extra_args[@]}"
fi
