#!/usr/bin/env bash
#SBATCH --job-name=gru2hv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
mkdir -p logs
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark

module load miniforge/24.3.0-py3.11
source ./scripts/activate_env.sh

# GRU arm-2 hyperparameter re-search at the 2h protocol (pred_len=24), meal-free.
# The original grid searched hidden_size in {16,32} and its optimum sat on the
# ceiling; this grid spans {256,512,768,1024}. Output label is *_v2 so the
# original search directory is never touched.
python exp/run_hyperparameter_search_cf.py \
  --config-dir configs/hyperparameter/gru_action_cf_benchmark_controliq_v2 \
  --arch gru \
  --dataset action_cf_benchmark_controliq_v2 \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --nshards 8
