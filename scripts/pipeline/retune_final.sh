#!/usr/bin/env bash
#SBATCH --job-name=bgretune
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
IDS=()
for m in gru attengluco; do
  for ds in dclp3 dclp5 pedap; do
    for h in p6 p12; do
      name="${m}_${ds}_${h}"
      [ -f "configs/best_lag/train/${name}.yaml" ] || { echo "  missing train cfg $name"; continue; }
      [ -f "configs/best_lag/eval/${name}.yaml" ]  || { echo "  missing eval cfg $name";  continue; }
      rm -rf "outputs/results/bestck_${name}"
      jid=$(sbatch --parsable --time=12:00:00 scripts/slurm/train_best_gpu_srcenv.sh \
        "configs/best_lag/train/${name}.yaml" "configs/best_lag/eval/${name}.yaml" "bestck_${name}")
      IDS+=("$jid")
      n=$((n+1))
    done
  done
done
echo "submitted $n retuned final runs"
if [ "$n" -eq 0 ]; then echo "ERROR: no retuned runs submitted -- missing configs/best_lag entries"; exit 1; fi
DEP=$(IFS=:; echo "${IDS[*]}")
G=$(sbatch --parsable --dependency=afterany:$DEP scripts/pipeline/gating_pass.sh)
echo "chained gating pass = $G"
