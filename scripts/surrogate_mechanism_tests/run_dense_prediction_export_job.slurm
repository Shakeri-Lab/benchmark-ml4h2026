#!/usr/bin/env bash
#SBATCH --account=shakeri-lab
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/surrogate/mechanism_frontier_tests/slurm_logs/%x-%j.out
#SBATCH --error=outputs/surrogate/mechanism_frontier_tests/slurm_logs/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda_envs/bg-benchmark}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/surrogate/mechanism_frontier_tests/dense_predictions}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OVERWRITE="${OVERWRITE:-0}"
SPLITS="${SPLITS:-test}"

cd "${ROOT}"
mkdir -p outputs/surrogate/mechanism_frontier_tests/slurm_logs "${OUTPUT_ROOT}"

args=(exp/export_mechanism_frontier_dense_predictions.py --run-dir "${RUN_DIR}" --output-root "${OUTPUT_ROOT}" --run-id "${RUN_ID}" --splits "${SPLITS}")
if [[ "${OVERWRITE}" == "1" ]]; then args+=(--overwrite); fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  "${ENV_PREFIX}/bin/python" "${args[@]}"
else
  module load miniforge/24.3.0-py3.11 || true
  conda run -p "${ENV_PREFIX}" python "${args[@]}"
fi
