#!/usr/bin/env bash
#SBATCH --job-name=bgarm2post
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
PY="$(pwd)/.conda_envs/bg-benchmark/bin/python"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

MODELS="arimax,attengluco,dlinear,gru,patchmlp,itransformer,patchmlp_glimmer"
POL_ROOT=outputs/results/policy_regret
POL_PREFIX=action_cf_benchmark_bolus_controliq_
CF_PREFIX=action_cf_controliq_best_
TABLE=policy_regret_controliq_amr_regret_ci.tex
CONF_DIR="${POL_ROOT}/${POL_PREFIX%_}_action_confusion"
STAMP=20260828

echo "=== archiving previous arm-2 table/confusion ==="
mkdir -p "outputs/archive_arm2_stale_${STAMP}"
[ -e "$TABLE" ]    && mv "$TABLE"    "outputs/archive_arm2_stale_${STAMP}/" && echo "  archived $TABLE"
[ -e "$CONF_DIR" ] && mv "$CONF_DIR" "outputs/archive_arm2_stale_${STAMP}/" && echo "  archived $CONF_DIR"

echo "=== postprocess: bootstrap, confusion, LaTeX ==="
export POLICY_ROOT="$POL_ROOT" POLICY_RUN_PREFIX="$POL_PREFIX"
export CF_ROOT=outputs/results CF_RUN_PREFIX="$CF_PREFIX"
export POLICY_MODELS="$MODELS" TABLE_MODELS="$MODELS"
export TABLE_OUTPUT="$TABLE" CONFUSION_OUTPUT_DIR="$CONF_DIR"
export DATASET_LABEL='action\_cf\_benchmark\_bolus\_controliq'
export N_BOOTSTRAP=1000 BOOTSTRAP_SEED=42 HORIZON=120 ALLOW_EXISTING=1
bash scripts/slurm/policy_regret_postprocess.slurm

echo "=== reference baselines (null-effect, best-constant-dose) ==="
mkdir -p outputs/analysis
CF_RUNS=(); POL_RUNS=()
IFS=',' read -r -a ms <<< "$MODELS"
for m in "${ms[@]}"; do
  [ -d "outputs/results/${CF_PREFIX}${m}" ] && CF_RUNS+=("outputs/results/${CF_PREFIX}${m}")
  [ -d "${POL_ROOT}/${POL_PREFIX}${m}" ]    && POL_RUNS+=("${POL_ROOT}/${POL_PREFIX}${m}")
done
echo "  cf runs: ${#CF_RUNS[@]}   policy runs: ${#POL_RUNS[@]}"
"$PY" analysis/null_effect_baseline.py --runs "${CF_RUNS[@]}" \
  --horizons 120 --out-csv outputs/analysis/null_effect_controliq.csv
"$PY" analysis/constant_action_baseline.py --runs "${POL_RUNS[@]}" \
  --out-csv outputs/analysis/constant_action_controliq.csv
echo "arm-2 postprocess complete"
