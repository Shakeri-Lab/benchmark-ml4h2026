#!/usr/bin/env bash
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/post_bolus_hypo_sensitivity_aggregate_%j.out
#SBATCH --error=logs/post_bolus_hypo_sensitivity_aggregate_%j.err

set -euo pipefail

ROOT="${ROOT:-/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda_envs/bg-benchmark}"
PARTS_DIR="${PARTS_DIR:?PARTS_DIR is required}"
OUTPUT_CSV="${OUTPUT_CSV:?OUTPUT_CSV is required}"

cd "${ROOT}"
mkdir -p logs "$(dirname "${OUTPUT_CSV}")"
"${ENV_PREFIX}/bin/python" analysis/aggregate_post_bolus_hypo_threshold_sensitivity.py \
  --parts-dir "${PARTS_DIR}" \
  --output-csv "${OUTPUT_CSV}"
