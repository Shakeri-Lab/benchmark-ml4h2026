#!/usr/bin/env bash
#SBATCH --job-name=bgiobp2
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
for m in arimax dlinear gru itransformer patchmlp patchmlp_glimmer attengluco zoh; do
  for h in p6 p12; do
    name="${m}_iobp2_${h}"
    [ -f "configs/best_lag/train/${name}.yaml" ] || { echo "  missing train cfg $name"; continue; }
    [ -f "configs/best_lag/eval/${name}.yaml" ]  || { echo "  missing eval cfg $name"; continue; }
    [ -d "outputs/results/bestck_${name}" ] && continue
    case "$m" in
      arimax|zoh)       J=scripts/slurm/train_best_cpu_srcenv.sh ;;
      patchmlp_glimmer) J=scripts/slurm/train_best_gpu_long_srcenv.sh ;;
      *)                J=scripts/slurm/train_best_gpu_srcenv.sh ;;
    esac
    sbatch --parsable --time=12:00:00 "$J" \
      "configs/best_lag/train/${name}.yaml" "configs/best_lag/eval/${name}.yaml" "bestck_${name}" >/dev/null
    n=$((n+1))
  done
done
echo "submitted $n IOBP2 final runs"
if [ "$n" -eq 0 ]; then echo "ERROR: no IOBP2 final runs submitted -- missing configs/best_lag entries"; exit 1; fi
