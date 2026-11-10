#!/bin/bash
set -euo pipefail

mkdir -p logs

SLURM_SCRIPT="scripts/run_hypo_gating_single.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "[!] Missing $SLURM_SCRIPT"
  exit 1
fi

shopt -s nullglob
for d in outputs/results/best*; do
  if [[ -d "$d" ]]; then
    echo "[+] Submitting $d"
    sbatch --export=MODEL_DIR="$d" "$SLURM_SCRIPT"
  fi
done
shopt -u nullglob
