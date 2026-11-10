#!/usr/bin/env bash
#SBATCH --job-name=bgarm2
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --account=shakeri-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
# Re-run arm 2 for the models whose hyperparameter searches were invalid:
#   arimax     - searched with the broken exog handling and the wrong CF regime
#   gru        - never searched (48h serial job died after config 000)
#   attengluco - searched in March against the pre-normalization model
# Unchanged models (dlinear, itransformer, patchmlp, patchmlp_glimmer) are
# re-selected identically and their existing runs are reused.
set -euo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
REPO=$(pwd)
PY="${REPO}/.conda_envs/bg-benchmark/bin/python"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

SRC=action_cf_benchmark_controliq
TGT=action_cf_benchmark_bolus_controliq
CF_CFG=configs/counterfactual_best_controliq
POL_CFG=configs/policy_regret_best_controliq
CF_PREFIX=action_cf_controliq_best_
POL_ROOT=outputs/results/policy_regret
POL_PREFIX=action_cf_benchmark_bolus_controliq_
STALE="arimax gru attengluco"
STAMP=20260828

echo "=== [1/4] re-selecting arm-2 configs (overwrite) ==="
"$PY" scripts/select_best_counterfactual_configs.py \
  --dataset "$SRC" --configs-root "$CF_CFG" --outputs-root outputs/hyperparameters \
  --glimmer-reference-config configs/policy_regret_best/train/patchmlp_glimmer_action_cf_benchmark_bolus.yaml
"$PY" scripts/select_best_policy_regret_configs.py \
  --source-dataset "$SRC" --target-dataset "$TGT" \
  --configs-root "$POL_CFG" --outputs-root outputs/hyperparameters \
  --hyperparam-config-root configs/hyperparameter

echo "=== [2/4] archiving stale arm-2 results ==="
mkdir -p "outputs/archive_arm2_stale_${STAMP}"
for m in $STALE; do
  for d in "outputs/results/${CF_PREFIX}${m}" "${POL_ROOT}/${POL_PREFIX}${m}"; do
    [ -d "$d" ] && mv "$d" "outputs/archive_arm2_stale_${STAMP}/" && echo "  archived $d"
  done
done

echo "=== [3/4] submitting arm-2 finals ==="
IDS=()
for m in $STALE; do
  tr="${REPO}/${CF_CFG}/train/${m}_${SRC}.yaml"; ev="${REPO}/${CF_CFG}/eval/${m}_${SRC}.yaml"
  [ -f "$tr" ] && [ -f "$ev" ] || { echo "  MISSING cf cfg for $m"; continue; }
  if [ "$m" = arimax ]; then P=(--partition=standard --time=12:00:00 --mem=64G)
  else                       P=(--partition=gpu --time=12:00:00 --mem=64G --gres=gpu:1); fi
  j=$(sbatch --parsable "${P[@]}" --account=shakeri-lab \
      --export=ALL,TRAIN_CFG="$tr",EVAL_CFG="$ev",RUN_ID="${CF_PREFIX}${m}",ENV_PREFIX="${REPO}/.conda_envs/bg-benchmark",OUTPUT_ROOT=outputs/results,N_BOOTSTRAP=1000,BOOTSTRAP_SEED=42,HORIZONS=120,ALLOW_EXISTING=0 \
      scripts/slurm/run_counterfactual_best.slurm)
  IDS+=("$j"); echo "  cf     $m = $j"
done
for m in $STALE; do
  tr="${REPO}/${POL_CFG}/train/${m}_${TGT}.yaml"; ev="${REPO}/${POL_CFG}/eval/${m}_${TGT}.yaml"
  [ -f "$tr" ] && [ -f "$ev" ] || { echo "  MISSING policy cfg for $m"; continue; }
  if [ "$m" = arimax ]; then JS=scripts/slurm/train_policy_regret_cpu.sh
  else                       JS=scripts/slurm/train_policy_regret_gpu.sh; fi
  j=$(sbatch --parsable --export=ALL,OUTPUT_ROOT="$POL_ROOT",ALLOW_EXISTING=0 "$JS" "$tr" "$ev" "${POL_PREFIX}${m}")
  IDS+=("$j"); echo "  policy $m = $j"
done
[ "${#IDS[@]}" -gt 0 ] || { echo "ERROR: no arm-2 finals submitted"; exit 1; }

echo "=== [4/4] chaining postprocess + reference baselines ==="
DEP=$(IFS=:; echo "${IDS[*]}")
P=$(sbatch --parsable --dependency=afterok:$DEP scripts/pipeline/arm2_post.sh)
echo "  postprocess = $P"
