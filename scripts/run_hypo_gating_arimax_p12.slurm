#!/bin/bash
#SBATCH --job-name=hypo_arimax_p12
#SBATCH --output=logs/hypo_arimax_p12_%j.out
#SBATCH --error=logs/hypo_arimax_p12_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard

set -euo pipefail

cd /project/shakeri-lab/Alireza_timeseries/benchmark
mkdir -p logs
POST_BOLUS_THRESHOLD="${POST_BOLUS_THRESHOLD:-}"

if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "[!] MODEL_DIR is not set. Submit with: sbatch --export=MODEL_DIR=path scripts/run_hypo_gating_arimax_p12.slurm"
  exit 1
fi

echo "[+] Evaluating ${MODEL_DIR}"
module load miniforge/24.3.0-py3.11
ENV_PREFIX="/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/.conda_envs/bg-benchmark"
export PYTHONNOUSERSITE=1

ARGS=(--models_dir "${MODEL_DIR}" --split test)
if [[ -n "$POST_BOLUS_THRESHOLD" ]]; then
  ARGS+=(--post_bolus_threshold "$POST_BOLUS_THRESHOLD")
fi
conda run -p "${ENV_PREFIX}" python analysis/evaluate_hypo_gating.py "${ARGS[@]}"
