#!/usr/bin/env bash
#SBATCH --job-name=bg-polreg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

TRAIN_CFG=${1:-}
EVAL_CFG=${2:-}
RUN_ID=${3:-}

if [[ -z "${TRAIN_CFG}" || -z "${EVAL_CFG}" || -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <train_config> <eval_config> <run_id>" >&2
  exit 1
fi

mkdir -p logs

cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark || {
  echo "Cannot cd into project directory!" >&2
  exit 1
}

module load miniforge/24.3.0-py3.11
export PYTHONNOUSERSITE=1
ENV_PREFIX="/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/.conda_envs/bg-benchmark"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/results/policy_regret}"
RUN_DIR="${OUTPUT_ROOT%/}/${RUN_ID}"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="$(pwd)/${RUN_DIR}"
fi
if [[ "${ALLOW_EXISTING:-0}" != "1" && -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 1
fi

conda run -p "${ENV_PREFIX}" python exp/run_policy_regret.py \
  --train --test \
  --train-config "${TRAIN_CFG}" \
  --eval-config "${EVAL_CFG}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}"
