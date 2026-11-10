#!/usr/bin/env bash
# Full downstream chain for the extended basal action menu.
# Fails loudly at every stage; never silently proceeds on partial data.
set -uo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
PY=.conda_envs/bg-benchmark/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
RAW=/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/sim-prep/UVA-T1D-Simulator/TestAID/action_cf_benchmark_controliq_basalA
DS=action_cf_controliq_basal8
MODELS="arimax,dlinear,gru,itransformer,patchmlp,patchmlp_glimmer,attengluco"
PR_ROOT=outputs/results/policy_regret
PR_PREFIX=basal8_pr_
step() { echo; echo "############ $* ############"; }

step "1/5 verify + preprocess + merge"
$PY scripts/basal8_prepare.py --raw "$RAW" --require-all \
    --new-proc datasets_processed/action_cf_controliq_basalA \
    --out      datasets_processed/$DS || { echo "STAGE 1 FAILED"; exit 1; }

step "2/5 counterfactual inference (test-only, existing checkpoints)"
$PY scripts/basal8_infer.py --dataset $DS --cfg-out configs/counterfactual_basal8 \
    --run-prefix basal8_ --models "$MODELS" --horizons 30,60,120 \
    || { echo "STAGE 2 FAILED"; exit 1; }

step "3/5 basal policy regret"
IFS=',' read -r -a MS <<< "$MODELS"
fail=0
for m in "${MS[@]}"; do
  rd=$PR_ROOT/${PR_PREFIX}${m}
  [ -f "$rd/policy_regret_summary.json" ] && { echo "  $m: done"; continue; }
  rm -rf "$rd"
  $PY exp/run_policy_regret.py --test \
     --train-config configs/counterfactual_basal8/train/${m}.yaml \
     --eval-config  configs/counterfactual_basal8/eval/${m}.yaml \
     --run-id ${PR_PREFIX}${m} \
     --checkpoint-dir outputs/results/action_cf_controliq_best_${m} \
     --perturb-family basal > logs/basal8_pr_${m}.log 2>&1 \
     && echo "  $m: ok" || { echo "  $m: FAILED (see logs/basal8_pr_${m}.log)"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "STAGE 3 had failures"; exit 1; }

step "4/5 action confusion matrices (basal)"
rm -rf $PR_ROOT/${PR_PREFIX%_}_action_confusion
$PY analysis/policy_regret_action_confusion.py \
   --policy-regret-root $PR_ROOT --run-prefix $PR_PREFIX \
   --output-dir $PR_ROOT/${PR_PREFIX%_}_action_confusion || { echo "STAGE 4 FAILED"; exit 1; }

step "5/5 reference baselines"
mkdir -p outputs/analysis
CF=(); PR=()
for m in "${MS[@]}"; do
  [ -d "outputs/results/basal8_${m}" ] && CF+=("outputs/results/basal8_${m}")
  [ -d "$PR_ROOT/${PR_PREFIX}${m}" ]   && PR+=("$PR_ROOT/${PR_PREFIX}${m}")
done
# --test-scenarios MUST point at the extended dataset: the default is the old
# 4-scale scenarios file, which silently filters the new scales back out.
$PY analysis/null_effect_baseline.py --runs "${CF[@]}" --horizons 120 \
   --test-scenarios datasets_processed/$DS/test_scenarios.csv \
   --out-csv outputs/analysis/null_effect_basal8.csv || echo "  null-effect baseline FAILED"
$PY analysis/constant_action_baseline.py --runs "${PR[@]}" \
   --out-csv outputs/analysis/constant_action_basal8.csv || echo "  constant-action baseline FAILED"

step "DONE"
echo "  merged dataset : datasets_processed/$DS"
echo "  cf runs        : outputs/results/basal8_*"
echo "  policy regret  : $PR_ROOT/${PR_PREFIX}*"
echo "  confusion      : $PR_ROOT/${PR_PREFIX%_}_action_confusion"
echo "  baselines      : outputs/analysis/{null_effect,constant_action}_basal8.csv"
