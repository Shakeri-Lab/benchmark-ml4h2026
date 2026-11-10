#!/usr/bin/env bash
#SBATCH --job-name=bgafinal
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
n=0
for ds in dclp3 dclp5 pedap; do for h in p6 p12; do
  name="arimax_${ds}_${h}"
  [ -f "configs/best_lag/train/${name}.yaml" ] || { echo "  missing $name"; continue; }
  rm -rf "outputs/results/bestck_${name}"
  sbatch --parsable --time=12:00:00 scripts/slurm/train_best_cpu_srcenv.sh \
    "configs/best_lag/train/${name}.yaml" "configs/best_lag/eval/${name}.yaml" "bestck_${name}" >/dev/null
  n=$((n+1))
done; done
echo "submitted $n ARIMAX final runs"
if [ "$n" -eq 0 ]; then echo "ERROR: no ARIMAX final runs submitted -- missing configs/best_lag entries"; exit 1; fi
