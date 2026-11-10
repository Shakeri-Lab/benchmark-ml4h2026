# Project Tracker

## Milestones
- [x] Standardize raw datasets into a unified schema (`pat_id`, `seq_id`, `date`, `cgm`, `basal`, `bolus_standard`, `bolus_extended`, `meal`, `weight_kg`, dataset-specific extras).
- [x] Implement quality controls for CGM, insulin, and meal data (dedup, resample, basal/bolus windows, bolus gap filtering, sequence-length filtering).
- [x] Generate benchmark-ready splits and metadata (patient-level 70/10/20 with seed=42; stored in `preprocessing/splits/` and applied to `datasets_processed/<dataset>/splits/`).
- [x] Implement evaluation harness for accuracy (RMSE, SEG, Clarke/Parkes error grids, gMSE/gRMSE, SEG-Z).
- [ ] Add robustness evaluations (patient shift, temporal shift).
- [ ] Add mechanistic accuracy checks (physiology-consistent behaviors).

## Current Focus
- Preprocessing: DCLP5, DCLP3, PEDAP, Shanghai_T1DM standardized with sequence splitting, resampling, basal/bolus attachment (including extended/IV for Shanghai), weight attachment, metadata; private DCLP3 variant supported under `datasets_preprocessed/`.
- Splits: Patient-level 70/10/20 (seed=42) generated/applied for all datasets; outputs live under `datasets_processed/<dataset>/splits/<spec>/` (or `datasets_preprocessed` for private data).
- Data loading: PyTorch sliding-window dataloader using split specs; returns X, Y, pat_id, seq_id, last history timestamp, feature metadata, and per-sample event distance.
- Metrics: Clinical metrics module (RMSE/MAE, SEG, CEG/PEG, gMSE/gRMSE, SEG-Z) plus streaming aggregator; supports event-conditioned metrics via event distances.
- Experiments: Harness supports train/eval configs, optimizers/schedulers/early stopping/AMP, checkpoint save/load (`--checkpoint-dir`), event-conditioned evaluation, and configurable losses (MSE/MAE/gMSE).

## Next Up
- Add public sanity-check tests (currently in `internal_tests/`).
- Add robustness evaluations (patient/temporal shift), mechanistic checks.
- Expand model zoo and expose hyperparameters via configs; polish documentation for private dataset handling and event-conditioned metrics.
