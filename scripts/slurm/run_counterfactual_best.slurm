#!/usr/bin/env bash
#SBATCH --job-name=bg-cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${TRAIN_CFG:?TRAIN_CFG is required}"
: "${EVAL_CFG:?EVAL_CFG is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${ENV_PREFIX:?ENV_PREFIX is required}"

N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
HORIZONS="${HORIZONS:-120}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/results}"

mkdir -p logs

cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark || {
  echo "Cannot cd into project directory!" >&2
  exit 1
}

RUN_DIR="${OUTPUT_ROOT%/}/${RUN_ID}"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="$(pwd)/${RUN_DIR}"
fi
if [[ "${ALLOW_EXISTING:-0}" != "1" && -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 1
fi

module load miniforge/24.3.0-py3.11
export PYTHONNOUSERSITE=1

conda run -p "${ENV_PREFIX}" python exp/run_counterfactual_with_metrics.py \
  --train --test \
  --train-config "${TRAIN_CFG}" \
  --eval-config "${EVAL_CFG}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --horizons "${HORIZONS}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"
