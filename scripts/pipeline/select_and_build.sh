#!/usr/bin/env bash
#SBATCH --job-name=bgselect
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
PY=/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/.conda_envs/bg-benchmark/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "search completeness before selection:"
for ds in dclp3 dclp5 pedap iobp2 action_cf_benchmark_controliq; do
  ok=$(find outputs/hyperparameters/arimax_${ds} -name val_rmse.json 2>/dev/null | wc -l)
  bad=$(find outputs/hyperparameters/arimax_${ds} -name config_failed.json 2>/dev/null | wc -l)
  echo "  arimax_${ds}: ok=${ok} failed=${bad}"
done
"$PY" scripts/select_best_hyperparams.py
"$PY" scripts/build_best_lag_configs.py
echo "selection complete"
