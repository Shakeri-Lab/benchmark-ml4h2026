#!/usr/bin/env bash
#SBATCH --job-name=bghypc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=cdt_computing
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --output=logs/%x-%A_%a.out
# Sharded GPU hyperparameter search.
# Usage: sbatch --array=0-N array_job_gpu.slurm <config_dir> <arch> <dataset> <nshards>
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
CFG=${1:?}; ARCH=${2:?}; DS=${3:?}; NSHARDS=${4:?}
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/.conda_envs/bg-benchmark/bin/python"
"$PY" -c "import numpy, torch" >/dev/null 2>&1 || { echo "[env] FATAL" >&2; exit 1; }
case "$DS" in
  action_cf_benchmark|action_cf_benchmark_controliq) RUNNER=exp/run_hyperparameter_search_cf.py ;;
  *)                                                 RUNNER=exp/run_hyperparameter_search.py ;;
esac
echo "[shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}] $ARCH / $DS via $RUNNER"
"$PY" "$RUNNER" \
  --config-dir "$CFG" --arch "$ARCH" --dataset "$DS" \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "$NSHARDS"
