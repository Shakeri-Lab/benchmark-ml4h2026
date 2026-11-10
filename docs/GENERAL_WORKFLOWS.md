# Blood Glucose Forecasting Benchmark

This project standardizes multiple CGM/insulin/meal datasets into a common format and will benchmark forecasting models along three axes: (1) raw accuracy (RMSE, SEG, Clarke error grid), (2) robustness to distribution shift (patient or temporal), and (3) mechanistic accuracy.

Repository: https://github.com/Shakeri-Lab/benchmark-ml4h2026

## Repository Layout
- `preprocess.py`: CLI entrypoint to preprocess datasets (`--dataset DCLP5` or `--dataset all`).
- `preprocessing/`: dataset-specific preprocessors and shared helpers.
- `data/adapters.py`: runtime adapters for unified CSV and DIAX per-subject JSON sources.
- `datasets_raw/`: raw data (not tracked).
- `datasets_processed/`: outputs (not tracked); each dataset folder will contain a unified CSV and metadata.
- `configs/preprocessing/`: frozen split specs (train/val/test patient lists).
- `models/`: baseline model implementations and external wrappers (Torch + TF).
- `exp/`: training/evaluation harness.
- `analysis/`: plotting/analysis scripts (e.g., rank boxplot).
- `third_party/`: vendored upstream baselines (iTransformer, DLinear, PatchMLP, AttenGluco, GARNN, GLIMMER, etc.).
- `requirements.txt`: Python dependencies.
- `scripts/setup_env.sh`, `scripts/activate_env.sh`: conda environment bootstrap/activation using Miniforge on HPC.
- `scripts/setup_glimmer_env.sh`, `scripts/activate_glimmer_env.sh`: TF 2.18.0 env for GLIMMER.
- `scripts/run_glimmer_paper.py`: paper-faithful GLIMMER runner (patient-wise temporal split).
- `scripts/generate_synthetic_dataset.py`: synthetic dataset generator from `preprocessing/synthetic`.
- `scripts/slurm/`: batch training job submission scripts.
- `PROJECT_TRACKER.md`: milestone tracker.
- `configs/training`, `configs/evaluation`: YAML configs for training/evaluation (supports univariate/multivariate runs).

## Environment Setup (HPC, Miniforge)
Envs default to the current working directory to avoid $HOME quotas.

1) Create the env (local prefix: `./.conda_envs/bg-benchmark`):
   ```bash
   bash scripts/setup_env.sh
   ```

   Options:
   - Use a custom prefix: `bash setup_env.sh --prefix /scratch/$USER/bg-benchmark`
   - Use $HOME’s default conda location: `bash setup_env.sh --use-home`

2) Activate later in new shells (matches the creation location):
   ```bash
   source scripts/activate_env.sh            # activates ./.conda_envs/bg-benchmark
   source scripts/activate_env.sh --prefix /scratch/$USER/bg-benchmark
   source scripts/activate_env.sh --use-home
   ```
   (Activation exports `PYTHONNOUSERSITE=1` so Python ignores `~/.local` packages.)

3) If you created an older env in $HOME and want to remove it:
   ```bash
   module load miniforge/24.3.0-py3.11
   conda env remove -n bg-benchmark
   ```

## Preprocessing Datasets
Preprocessed outputs land in `datasets_processed/<dataset>/`.

Raw diabetes datasets can be downloaded from `https://public.jaeb.org/datasets/diabetes`. Place the downloaded files under `datasets_raw/` using the dataset-specific directory structure expected by the preprocessors before running the commands below.

Example for DCLP5:
```bash
python preprocess.py --dataset DCLP5
```
The DCLP5 pipeline currently:
- Reads CGM from `TandemCGMDATAGXB.txt`, `DCLP5TandemCGMDATAGXB_b.txt`, `DexcomClarityCGM.txt`.
- Normalizes to `pat_id`, `date`, `cgm`, drops invalid rows.
- Deduplicates per patient within ~5-minute (configurable) windows keeping the latest record (progress bar per patient).
- Splits sequences on CGM gaps >30 minutes, resamples CGM to 5-minute grid, and linearly interpolates inside gaps ≤30 minutes; assigns globally unique `seq_id` across all patients.
- Attaches basal from `DCLP5TandemBASALRATECHG_b.txt` using the latest basal in [-3h, +30s] around each CGM timestamp (progress bar per patient).
- Attaches bolus (default 0) from `DCLP5TandemBolus_Completed_Combined_b.txt` in a configurable window; drops spans with no bolus beyond a configurable gap (default 12h); tracks Standard vs Extended counts.
- Attaches weight (kg) using closest/interpolated values from available weight sources (screening/follow-up/phys exam/AE) when present.
- Filters out sequences shorter than the configured minimum length (default 312 steps = 24h history + 2h horizon at 5-minute steps).
- Writes `datasets_processed/DCLP5/timeseries.csv` with columns `pat_id, seq_id, date, cgm, basal, bolus, weight_kg, meal` and `metadata.json`.

DCLP3 follows the same pipeline/config pattern (data under `datasets_raw/DCLP3/Data Files/`); current patient count after preprocessing is ~114, close to the ~112 reported in literature (differences likely due to cleaning/ID handling). A private, preprocessed variant can be placed at `datasets_preprocessed/DCLP3-private/timeseries.csv`; generate its metadata with `scripts/gen_metadata_dclp3_private.py` (kept untracked), generate splits with `generate_splits.py` pointing to that CSV, and apply them with `apply_splits.py`. Private data should remain out of version control.
PEDAP follows the same pattern (data under `datasets_raw/PEDAP/Data Files/`), using Dexcom + Tandem CGM sources, Tandem basal/bolus, and weights from screening/follow-up/phys-exam/AE when available.

### Loading DIAX datasets directly

The forecasting dataloader can read a DIAX cohort directory directly, without
first writing `timeseries.csv`. The adapter aligns the per-subject JSON channels
to the benchmark's five-minute schema, splits CGM at gaps longer than 30 minutes,
and supplies zero/NaN defaults for optional channels. CSV remains the default
when a `timeseries.csv` file is present.

Point `processed_root` at the parent of the DIAX cohort directories and select
the format in the config:

```yaml
split:
  split_name: dclp3_patients_70_10_20_seed42
  split_root: configs/preprocessing
  processed_root: /standard/CDT_storage/dataset_processing/dataset_DIAX
  source_format: diax
```

Alternatively, set `source_path` (or `data_path`) to one cohort directory or one
DIAX JSON file. Patient lists in the split spec are matched to DIAX
`source_subject_id` values. Non-numeric identifiers are encoded as stable integer
tensors; the reversible mapping is available as `loader.dataset.patient_id_map`.
`source_format: auto` detects either `timeseries.csv` or DIAX JSON files.

## Dataset Splits
- Frozen patient-level splits live in `configs/preprocessing/` and are named `<dataset>_patients_70_10_20_seed<seed>.json`.
- Current splits use a 70/10/20 train/val/test ratio with a fixed seed (42), generated by shuffling sorted patient IDs per dataset. Specs include the seed, ratios, and explicit patient lists for reproducibility.
- Regenerate (if ever needed) by running an ad-hoc script against `datasets_processed/<dataset>/timeseries.csv` to extract `pat_id`, shuffle with the desired seed, and write a new spec file; keep the generator script untracked (`generate_splits.py` is ignored).

## Experiments, Checkpoints, and Metrics
- Training/eval are driven by YAML configs (nested sections):
  - Training: `split` (split spec + directories), `training` (optimizer/scheduler/epochs/loss), `model_data` (seed + model + dataloader).
  - Eval: `split` (split spec + directories), `model_data` (seed + model + dataloader), `evaluation` (metrics + event thresholds).
- Run both train and test (defaults to both if no flags are given):
  ```bash
  python exp/run_experiment.py \
    --train --test \
    --train-config configs/training/dclp3_private_linear.yaml \
    --eval-config configs/evaluation/dclp3_private_linear.yaml \
    --run-id dclp3_private_lin
  ```
- Eval-only from a checkpoint:
  ```bash
  python exp/run_experiment.py \
    --test \
    --eval-config configs/evaluation/dclp3_private_linear.yaml \
    --checkpoint-dir outputs/results/<run_id_with_checkpoint> \
    --run-id eval_from_ckpt
  ```
- Outputs live under `outputs/results/<run_id>/`: `metrics.json`, `config.yaml`, `preds.npz`, `GT.npz`, `event.npz` (per-sample event distances), `checkpoint.pt`, `logs.log`.
- Metrics computed: `rmse`, `mae`, `seg`, `gmse`, `grmse`, `ceg_ratio`, `peg_ratio`, `seg_z_ratio`, plus optional `event_conditioned` metrics when `event_threshold` is set.

## Running Experiments
The repo supports four main workflows:
- hyperparameter search
- standard forecasting
- counterfactual perturbation benchmarks
- policy regret on bolus counterfactuals

### 1. Hyperparameter Search
Hyperparameter configs are materialized under `configs/hyperparameter/`, and search outputs are written to `outputs/hyperparameters/`.

Generate the 64-config search grids:
```bash
python scripts/hyperparameters/generate_configs.py
```

Run a single real-data search locally:
```bash
python exp/run_hyperparameter_search.py \
  --config-dir configs/hyperparameter/arimax_dclp3 \
  --arch arimax \
  --dataset dclp3
```

Run a single action-counterfactual search locally:
```bash
python exp/run_hyperparameter_search_cf.py \
  --config-dir configs/hyperparameter/arimax_action_cf_benchmark \
  --arch arimax \
  --dataset action_cf_benchmark
```

Submit the full real-data search sweep on Slurm:
```bash
bash scripts/hyperparameters/submit_all_hyper.sh
```

Submit the full action-counterfactual search sweep on Slurm:
```bash
bash scripts/hyperparameters/submit_action_cf_hyper.sh
```

After searches finish, select the best configs:
```bash
python scripts/select_best_hyperparams.py
python scripts/build_best_lag_configs.py
python scripts/select_best_counterfactual_configs.py
python scripts/select_best_policy_regret_configs.py
```

This produces:
- `configs/hyperparameter_best/` for one best config per model-dataset pair
- `configs/best_lag/{train,eval}/` for the standard forecasting benchmark at multiple horizons
- `configs/counterfactual_best/{train,eval}/` for action-counterfactual runs
- `configs/policy_regret_best/{train,eval}/` for policy-regret runs

### 2. Standard Forecasting
For a single train/test run, use `exp/run_experiment.py` directly:
```bash
python exp/run_experiment.py \
  --train --test \
  --train-config configs/best_lag/train/gru_dclp3_p6.yaml \
  --eval-config configs/best_lag/eval/gru_dclp3_p6.yaml \
  --run-id best_gru_dclp3_p6
```

The standard benchmark convention is:
- run id: `best_<architecture>_<dataset>_p<num_steps>`
- output directory: `outputs/results/best_<architecture>_<dataset>_p<num_steps>`

To submit the full best-lag forecasting sweep on Slurm:
```bash
bash scripts/slurm/submit_best_all.sh
```

Useful horizon conventions:
- `p3` = 15 minutes
- `p6` = 30 minutes
- `p9` = 45 minutes
- `p12` = 60 minutes

### 3. Counterfactual Perturbation Benchmark
Counterfactual train/test runs use `exp/run_counterfactual_experiment.py`, and the metric wrapper `exp/run_counterfactual_with_metrics.py` additionally writes `cf_metrics.{json,csv}` and patient-level bootstrap summaries.

Run a single counterfactual experiment with metrics:
```bash
python exp/run_counterfactual_with_metrics.py \
  --train --test \
  --train-config configs/counterfactual_best/train/arimax_action_cf_benchmark.yaml \
  --eval-config configs/counterfactual_best/eval/arimax_action_cf_benchmark.yaml \
  --run-id action_cf_best_arimax \
  --output-root outputs/results \
  --horizons 30,60,90,120 \
  --n-bootstrap 1000 \
  --bootstrap-seed 42
```

The standard counterfactual naming convention is:
- run id: `action_cf_best_<architecture>`
- output directory: `outputs/results/action_cf_best_<architecture>`

Submit the full counterfactual benchmark on Slurm:
```bash
bash scripts/slurm/submit_counterfactual_best.sh
```

### 4. Policy Regret
Policy regret is computed on the bolus-only counterfactual benchmark and writes outputs under `outputs/results/policy_regret/`.

Run a single policy-regret experiment locally:
```bash
python exp/run_policy_regret.py \
  --train --test \
  --force-base-run \
  --train-config configs/policy_regret_best/train/arimax_action_cf_benchmark_bolus.yaml \
  --eval-config configs/policy_regret_best/eval/arimax_action_cf_benchmark_bolus.yaml \
  --run-id action_cf_benchmark_bolus_arimax \
  --output-root outputs/results/policy_regret
```

Bootstrap the saved patient-level policy-regret metrics:
```bash
python analysis/policy_regret_bootstrap.py \
  --run-dir outputs/results/policy_regret/action_cf_benchmark_bolus_arimax \
  --n-bootstrap 1000 \
  --seed 42
```

Submit the full policy-regret sweep on Slurm:
```bash
bash scripts/slurm/submit_policy_regret_all.sh
```

The standard policy-regret naming convention is:
- run id: `action_cf_benchmark_bolus_<architecture>`
- output directory: `outputs/results/policy_regret/action_cf_benchmark_bolus_<architecture>`

## Event Handling
- Dataloader returns event distance per sample: steps since the last non-zero value in any feature column except `{cgm, pat_id/pid/patient_id/subject_id, seq_id, basal, date}`; NaNs count as zero; `-1` if no event in the context window.
- `event.npz` stores these distances; `event_threshold` in eval config enables event-conditioned metrics.

## Baselines (Updated)
Supported baselines include:
- **Torch**: ZOH, GRU, GRUPP/GRUPPP, TCN, Linear, ARIMAX (statsmodels).
- **Vendored**: iTransformer, DLinear, PatchMLP (TFB), AttenGluco, GARNN (gatv2_exp), GLIMMER (TF 2.18.0).
- **Zero-shot**: Parrot, TimesFM.

Each model is configured via `configs/training/*` and `configs/evaluation/*` with a univariate/multivariate switch.
External models live under `third_party/` and are wrapped in `models/` with thin adapters.

## SHAP Explainability
`exp/run_shap.py` runs permutation-SHAP with modality × time-window “players”, and saves:
- `shap_values`, `pred_values`, `base_values` per sample
- Slice-partitioned outputs based on post-meal/post-bolus/nocturnal events
Configs live in `configs/analysis/` (e.g., `shap_default.yaml`).

## Synthetic Dataset Generation
Generate synthetic data from the third-party mixture model:
```bash
python scripts/generate_synthetic_dataset.py --name synth_ar3 --n-features 4 --n-samples 100
```
This writes:
```
datasets_processed/synth_ar3/timeseries.csv
datasets_processed/synth_ar3/metadata.json
```
Then generate a split spec and apply:
```bash
python generate_splits.py --dataset synth_ar3 --timeseries datasets_processed/synth_ar3/timeseries.csv \
  --seed 42 --ratios 0.7 0.1 0.2 --outdir configs/preprocessing

python apply_splits.py --spec-file configs/preprocessing/synth_ar3_patients_70_10_20_seed42.json \
  --dataset synth_ar3 --timeseries-root datasets_processed --out-root datasets_processed
```

## GLIMMER (TensorFlow 2.18.0)
GLIMMER uses a separate TF environment:
```bash
bash scripts/setup_glimmer_env.sh
source scripts/activate_glimmer_env.sh
```
Paper-faithful (patient-wise temporal split):
```bash
python scripts/run_glimmer_paper.py --dataset DCLP3 --variant transformer --train-window 72 --pred-len 12 \
  --ma-window 200 --epochs 30 --batch-size 48 --repeat 3 --run-id dclp3_glimmer_paper
```

## SLURM Batch Training (30-minute horizon)
`scripts/slurm/submit_all_train_test.sh` submits training jobs for all core models/datasets using *_p6 configs.

## Adding Models
1. Implement the model under `models/` (e.g., `models/my_model.py`) as a `torch.nn.Module` with a `forward` method (the harness calls `forward`, passing `pred_len` if provided).
2. Register it in `models/registry.py` (add to `_MODEL_REGISTRY`).
3. Expose hyperparameters via the training config (`model_data.model.params`); the harness passes `input_dim`, `history_len`, and `pred_len` automatically when constructing the model.
4. If you need a custom loss, add it to `models/losses.py` and reference it by name in the training config (`loss:`).

## Notes on Private Data
- Keep private preprocessed datasets in `datasets_preprocessed/<name>/timeseries.csv`.
- Generate metadata locally (untracked) via helper scripts.
- Generate/apply splits with `generate_splits.py` and `apply_splits.py` (tools already fall back to `datasets_preprocessed`).
- Do not commit private data or helper scripts that expose private paths.
