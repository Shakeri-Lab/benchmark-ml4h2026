#!/bin/bash
set -euo pipefail

mkdir -p logs

SLURM_SCRIPT="scripts/run_hypo_gating_arimax_p6.slurm"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "[!] Missing $SLURM_SCRIPT"
  exit 1
fi

for d in \
  outputs/results/best_arimax_dclp3_p6 \
  outputs/results/best_arimax_dclp5_p6 \
  outputs/results/best_arimax_pedap_p6 \
  outputs/results/best_arimax_shanghait1dm_p6
do
  if [[ -d "$d" ]]; then
    echo "[+] Submitting $d"
    sbatch --export=MODEL_DIR="$d" "$SLURM_SCRIPT"
  else
    echo "[!] Skipping missing directory $d"
  fi
done
