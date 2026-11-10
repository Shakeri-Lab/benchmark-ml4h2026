#!/usr/bin/env bash
#SBATCH --job-name=bgcount
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
module load miniforge/24.3.0-py3.11
export PYTHONNOUSERSITE=1
set --
source scripts/activate_env.sh
python count_windows_tmp.py
