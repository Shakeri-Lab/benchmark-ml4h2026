#!/bin/bash
#SBATCH --job-name=hypo_gating_best
#SBATCH --output=logs/hypo_gating_best_%A_%a.out
#SBATCH --error=logs/hypo_gating_best_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-0

set -euo pipefail
mkdir -p logs
source ./scripts/activate_env.sh

LIST_FILE="${LIST_FILE:-outputs/results/best_models.txt}"
POST_BOLUS_THRESHOLD="${POST_BOLUS_THRESHOLD:-}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "[!] Missing $LIST_FILE"
  echo "    Create it with: ls -d outputs/results/best* > $LIST_FILE"
  exit 1
fi

MODEL_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST_FILE")
if [[ -z "$MODEL_DIR" ]]; then
  echo "[!] No model dir for task id ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "[+] Evaluating $MODEL_DIR"
ARGS=(--models_dir "$MODEL_DIR" --split test)
if [[ -n "$POST_BOLUS_THRESHOLD" ]]; then
  ARGS+=(--post_bolus_threshold "$POST_BOLUS_THRESHOLD")
fi
python analysis/evaluate_hypo_gating.py "${ARGS[@]}"
