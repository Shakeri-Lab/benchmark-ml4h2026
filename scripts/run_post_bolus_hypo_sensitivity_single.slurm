#!/usr/bin/env bash
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/post_bolus_hypo_sensitivity_%j.out
#SBATCH --error=logs/post_bolus_hypo_sensitivity_%j.err

set -euo pipefail

ROOT="${ROOT:-/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda_envs/bg-benchmark}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR is required}"
OUTPUT_CSV="${OUTPUT_CSV:?OUTPUT_CSV is required}"
THRESHOLD_MINUTES="${THRESHOLD_MINUTES:-15 30 46 60}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"

cd "${ROOT}"
mkdir -p logs "$(dirname "${OUTPUT_CSV}")"

read -r -a THRESHOLDS <<< "${THRESHOLD_MINUTES}"

"${ENV_PREFIX}/bin/python" analysis/post_bolus_hypo_threshold_sensitivity.py \
  --model-dir "${MODEL_DIR}" \
  --output-csv "${OUTPUT_CSV}" \
  --threshold-minutes "${THRESHOLDS[@]}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"
