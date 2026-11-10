#!/bin/bash
#SBATCH --job-name=hypo_gating
#SBATCH --output=logs/hypo_gating_%j.out
#SBATCH --error=logs/hypo_gating_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard

cd /project/shakeri-lab/Alireza_timeseries/benchmark
source ./scripts/activate_env.sh
set -euo pipefail
mkdir -p logs
POST_BOLUS_THRESHOLD="${POST_BOLUS_THRESHOLD:-}"

if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "[!] MODEL_DIR is not set. Submit with: sbatch --export=MODEL_DIR=path scripts/run_hypo_gating_single.slurm"
  exit 1
fi

echo "[+] Evaluating ${MODEL_DIR}"
ARGS=(--models_dir "${MODEL_DIR}" --split test)
if [[ -n "$POST_BOLUS_THRESHOLD" ]]; then
  ARGS+=(--post_bolus_threshold "$POST_BOLUS_THRESHOLD")
fi
python analysis/evaluate_hypo_gating.py "${ARGS[@]}"
