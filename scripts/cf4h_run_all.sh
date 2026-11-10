#!/usr/bin/env bash
# Full downstream chain for the 4-hour counterfactual regeneration.
# Fails loudly at every stage; never silently proceeds on partial data.
set -uo pipefail
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
PY=.conda_envs/bg-benchmark/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib

DS=action_cf_controliq_4h
MODELS="${MODELS:-arimax,dlinear,gru,itransformer,patchmlp,patchmlp_glimmer,attengluco}"
HORIZONS=30,60,120,240
SRC_RUN='cf4hb_{m}'
PR_ROOT=outputs/results/policy_regret
step() { echo; echo "############ $* ############"; }

step "1/5 verify + preprocess + assemble 4h dataset"
$PY scripts/cf4h_prepare.py --require-all || { echo "STAGE 1 FAILED"; exit 1; }

step "2/5 counterfactual inference (test-only, 4h checkpoints)"
$PY scripts/basal8_infer.py --dataset $DS --cfg-out configs/counterfactual_4h_eval \
    --run-prefix cf4h_ --source-run "$SRC_RUN" \
    --src-config-dir configs/counterfactual_4h \
    --models "$MODELS" --horizons $HORIZONS || { echo "STAGE 2 FAILED"; exit 1; }

step "3/5 policy regret, both action families"
IFS=',' read -r -a MS <<< "$MODELS"
fail=0
for fam in basal bolus; do
  for m in "${MS[@]}"; do
    rd=$PR_ROOT/cf4h_pr_${fam}_${m}
    [ -f "$rd/policy_regret_summary.json" ] && { echo "  $fam/$m: done"; continue; }
    rm -rf "$rd"
    $PY exp/run_policy_regret.py --test \
       --train-config configs/counterfactual_4h_eval/train/${m}.yaml \
       --eval-config  configs/counterfactual_4h_eval/eval/${m}.yaml \
       --run-id cf4h_pr_${fam}_${m} \
       --checkpoint-dir outputs/results/cf4hb_${m} \
       --perturb-family $fam > logs/cf4h_pr_${fam}_${m}.log 2>&1 \
       && echo "  $fam/$m: ok" || { echo "  $fam/$m: FAILED (logs/cf4h_pr_${fam}_${m}.log)"; fail=1; }
  done
done
[ "$fail" -eq 0 ] || { echo "STAGE 3 had failures"; exit 1; }

step "4/5 action confusion matrices, both families"
for fam in basal bolus; do
  out=$PR_ROOT/cf4h_${fam}_action_confusion
  rm -rf "$out"
  # --family sets axis labels, column names and filenames; without it a basal run
  # would be written out labelled "bolus".
  $PY analysis/policy_regret_action_confusion.py \
     --policy-regret-root $PR_ROOT --run-prefix cf4h_pr_${fam}_ \
     --family $fam --output-dir "$out" || { echo "STAGE 4 FAILED ($fam)"; exit 1; }
done

step "5/5 reference baselines"
mkdir -p outputs/analysis
CF=(); PRB=(); PRO=()
for m in "${MS[@]}"; do
  [ -d "outputs/results/cf4h_${m}" ]           && CF+=("outputs/results/cf4h_${m}")
  [ -d "$PR_ROOT/cf4h_pr_basal_${m}" ]         && PRB+=("$PR_ROOT/cf4h_pr_basal_${m}")
  [ -d "$PR_ROOT/cf4h_pr_bolus_${m}" ]         && PRO+=("$PR_ROOT/cf4h_pr_bolus_${m}")
done
# --test-scenarios MUST point at the 4h dataset: the default is the old scenarios
# file, which silently filters the new scales and the 4h horizon back out.
$PY analysis/null_effect_baseline.py --runs "${CF[@]}" --horizons 240 \
   --test-scenarios datasets_processed/$DS/test_scenarios.csv \
   --out-csv outputs/analysis/null_effect_cf4h.csv || echo "  null-effect baseline FAILED"
$PY analysis/constant_action_baseline.py --runs "${PRB[@]}" \
   --out-csv outputs/analysis/constant_action_cf4h_basal.csv || echo "  constant-action (basal) FAILED"
$PY analysis/constant_action_baseline.py --runs "${PRO[@]}" \
   --out-csv outputs/analysis/constant_action_cf4h_bolus.csv || echo "  constant-action (bolus) FAILED"

step "DONE"
echo "  dataset       : datasets_processed/$DS"
echo "  cf runs       : outputs/results/cf4h_*"
echo "  policy regret : $PR_ROOT/cf4h_pr_{basal,bolus}_*"
echo "  confusion     : $PR_ROOT/cf4h_{basal,bolus}_action_confusion"
echo "  baselines     : outputs/analysis/{null_effect_cf4h,constant_action_cf4h_*}.csv"
