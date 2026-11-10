#!/usr/bin/env bash
#SBATCH --job-name=bghyp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --output=logs/%x-%A_%a.out
# Sharded hyperparameter search.
# Usage: sbatch --array=0-N%K array_job.slurm <config_dir> <arch> <dataset> <nshards>
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
CFG=${1:?}; ARCH=${2:?}; DS=${3:?}; NSHARDS=${4:?}
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/.conda_envs/bg-benchmark/bin/python"
"$PY" -c "import numpy" >/dev/null 2>&1 || { echo "[env] FATAL" >&2; exit 1; }
"$PY" exp/run_hyperparameter_search.py \
  --config-dir "$CFG" --arch "$ARCH" --dataset "$DS" \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "$NSHARDS"
