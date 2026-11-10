#!/usr/bin/env bash
#SBATCH --job-name=bggatepass
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
bash scripts/slurm/submit_gating_th.sh 6
bash scripts/slurm/submit_gating_th.sh 12
