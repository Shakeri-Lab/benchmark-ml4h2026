# SHAP Usage Guide

This repo includes a permutation/Monte‑Carlo SHAP implementation for CGM forecasting with “players” defined as **(modality × recency window)**. The SHAP runner is `exp/run_shap.py` and uses the same dataset/windowing pipeline as the evaluation harness.

## Quick start

```bash
python exp/run_shap.py \
  --shap_config configs/analysis/shap_dclp3_private_gru_mv.yaml \
  --checkpoint_dir outputs/results/dclp3_private_gru_mv \
  --out_dir outputs/shap/dclp3_private_gru_mv \
  --model gru
```

Notes:
- `--checkpoint_dir` should contain `checkpoint.pt`.
- `--model` can be omitted if the checkpoint includes `model_name`.
- `--split_path` can override the config’s split CSV path.

At the end of the run, the script prints **RMSE over the selected horizons** so you can sanity‑check predictions.

## Config structure

A SHAP config is a YAML file with sections:

- `data`: dataloader settings and split CSV path
- `shap`: SHAP hyperparameters (permutations, background draws, horizons)
- `background`: background pool sampling policy
- `output`: output format and optional sample cap
- `model`: torch.compile settings for SHAP inference
- `modality_channels`: mapping of feature columns to modalities
- `windows_steps_by_modality` (optional): custom windows by modality

Example (abridged):

```yaml
data:
  split_path: datasets_processed/DCLP3-private/splits/dclp3-private_patients_70_10_20_seed42/test.csv
  split_part: test
  history_len: 48
  pred_len: 12
  stride: 12
  batch_size: 128
  univariate: false
  processed_root: datasets_processed
  input_layout: BTC
  nocturnal_start_hour: 0
  nocturnal_end_hour: 6
  subsample_size: 2000        # optional: random subset of samples
  # subsample_fraction: 0.2
  subsample_seed: 42

shap:
  horizons_steps: [3, 6, 12]   # steps (1 step = 5 min)
  n_permutations: 16
  n_background_draws: 16
  microbatch_size: 64
  perm_batch_size: 4         # batch multiple permutations per model call
  seed: 42

background:
  prefer_same_slice: true
  min_pool_size_same_slice: 25
  allow_global_fallback: false

output:
  partition_by_slice: true
  diagnostics_subset_size: 128

model:
  compile: false

modality_channels:
  cgm: ["cgm"]
  meal: ["meal"]
  bolus: ["bolus"]
```

### Horizons are in **steps**, not minutes
If your step is 5 minutes, `horizons_steps=[3,6,12]` corresponds to 15, 30, 60 minutes.

## Reconstruction (masked‑player inpainting)

After masking players, the SHAP runner performs deterministic, modality‑specific reconstruction **before** model inference:

- **CGM**: linear interpolation for interior masked segments; ZOH at edges; 0 if fully masked.
- **Basal**: stepwise constant; if left/right differ, a single step at the midpoint.
- **Meal / Bolus**: sparse signals default to 0. If one is masked and the other has an event within ±k steps (default k=3), an impulse is imputed in the masked channel. Meal magnitude is inferred as binary vs grams; bolus uses median positive bolus.

Optional reconstruction overrides (not required; defaults used if omitted):

```yaml
reconstruction:
  meal_bolus_k: 3
  meal_type: null          # "binary" | "grams" | null(auto)
  meal_binary_max: 1.5
  meal_binary_ratio: 0.95
```

## Outputs

Outputs are saved under the `--out_dir` you provide:

- `shap_<slice>.npz`: SHAP values per sample
- `metadata_<slice>.csv`: per‑sample metadata
- `diagnostics.json`: efficiency diagnostic statistics
- `players.json`: player definitions used
- `config_resolved.json`: resolved config
- `logs.log`: run log (including RMSE at end)

Slices are based on event indicators:
- `both`, `post_meal`, `post_bolus`, `nocturnal`, `neither`

## Tips

- For large models, set `shap.microbatch_size` to avoid OOM.
- For faster runs, reduce `n_permutations` or `n_background_draws`.
- You can cap run size with `output.max_eval_samples`.
