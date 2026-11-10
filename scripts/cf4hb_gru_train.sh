#!/usr/bin/env bash
#SBATCH --job-name=cf4hbgru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
module load miniforge/24.3.0-py3.11
source ./scripts/activate_env.sh
# Train only: the eval pass compares 48-step predictions against the 2h dataset's
# 24-step suffixes and errors out, discarding the checkpoint.
python exp/run_counterfactual_experiment.py --train \
  --train-config configs/counterfactual_4h/train/gru.yaml \
  --eval-config  configs/counterfactual_4h/eval/gru.yaml \
  --output-root outputs/results --run-id cf4hb_gru
