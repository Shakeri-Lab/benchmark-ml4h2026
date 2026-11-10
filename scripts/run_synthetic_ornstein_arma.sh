#!/usr/bin/env bash
set -euo pipefail

# Synthetic datasets
SYNTH_OU="synth_ornstein"
SYNTH_ARMA="synth_arma"

# Common generation params
TIME_LENGTH=288
N_SAMPLES=100
N_FEATURES=4
BATCH_SIZE=1
SEED=42

echo "[1/5] Generate synthetic datasets..."
python scripts/generate_synthetic_dataset.py \
  --name "${SYNTH_OU}" \
  --component-model ou \
  --n-features "${N_FEATURES}" \
  --n-samples "${N_SAMPLES}" \
  --time-length "${TIME_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --random-seed "${SEED}"

python scripts/generate_synthetic_dataset.py \
  --name "${SYNTH_ARMA}" \
  --component-model arma \
  --component-kwargs '{"p": 2, "q": 2}' \
  --n-features "${N_FEATURES}" \
  --n-samples "${N_SAMPLES}" \
  --time-length "${TIME_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --random-seed "${SEED}"

echo "[2/5] Generate split specs..."
python generate_splits.py \
  --dataset "${SYNTH_OU}" \
  --timeseries "datasets_processed/${SYNTH_OU}/timeseries.csv" \
  --seed "${SEED}" \
  --ratios 0.7 0.1 0.2 \
  --outdir configs/preprocessing

python generate_splits.py \
  --dataset "${SYNTH_ARMA}" \
  --timeseries "datasets_processed/${SYNTH_ARMA}/timeseries.csv" \
  --seed "${SEED}" \
  --ratios 0.7 0.1 0.2 \
  --outdir configs/preprocessing

echo "[3/5] Apply splits..."
python apply_splits.py \
  --spec-file "configs/preprocessing/${SYNTH_OU}_patients_70_10_20_seed42.json" \
  --dataset "${SYNTH_OU}" \
  --timeseries-root datasets_processed \
  --out-root datasets_processed

python apply_splits.py \
  --spec-file "configs/preprocessing/${SYNTH_ARMA}_patients_70_10_20_seed42.json" \
  --dataset "${SYNTH_ARMA}" \
  --timeseries-root datasets_processed \
  --out-root datasets_processed

echo "[4/5] Train + test PatchMLP..."
python exp/run_experiment.py \
  --train --test \
  --train-config "configs/training/${SYNTH_OU}_patchmlp_mv.yaml" \
  --eval-config "configs/evaluation/${SYNTH_OU}_patchmlp_mv.yaml" \
  --run-id "${SYNTH_OU}_patchmlp_mv"

python exp/run_experiment.py \
  --train --test \
  --train-config "configs/training/${SYNTH_ARMA}_patchmlp_mv.yaml" \
  --eval-config "configs/evaluation/${SYNTH_ARMA}_patchmlp_mv.yaml" \
  --run-id "${SYNTH_ARMA}_patchmlp_mv"

echo "[5/5] Train + test ARIMAX..."
python exp/run_experiment.py \
  --train --test \
  --train-config "configs/training/${SYNTH_OU}_arimax_mv.yaml" \
  --eval-config "configs/evaluation/${SYNTH_OU}_arimax_mv.yaml" \
  --run-id "${SYNTH_OU}_arimax_mv"

python exp/run_experiment.py \
  --train --test \
  --train-config "configs/training/${SYNTH_ARMA}_arimax_mv.yaml" \
  --eval-config "configs/evaluation/${SYNTH_ARMA}_arimax_mv.yaml" \
  --run-id "${SYNTH_ARMA}_arimax_mv"

echo "[✓] Done."
