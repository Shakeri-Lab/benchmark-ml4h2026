# PatchMLP + GLIMMER-Style Loss

This baseline adapts the **GLIMMER** idea (region-weighted loss with GA-tuned hypo/hyper weights) to the
existing **PatchMLP** backbone in this repo:

- **Concept taken from GLIMMER**: prioritize hypo/hyper regions by weighting the loss for glucose targets
  below/above thresholds; optimize `w_hypo` and `w_hyper` with a genetic algorithm on validation data.
- **Changes for this repo**: implemented **entirely in PyTorch**, uses **PatchMLP** as the backbone,
  trains a **global model** with the standard train/val/test splits, and exposes the GA settings via config.

## How to run
Example (private DCLP3, multivariate):

```bash
python exp/run_experiment.py \
  --train --test \
  --train-config configs/training/dclp3_private_patchmlp_glimmer_mv.yaml \
  --eval-config configs/evaluation/dclp3_private_patchmlp_glimmer_mv.yaml \
  --run-id dclp3_private_patchmlp_glimmer_mv
```

GA settings (population size, generations, search ranges, inner epochs) live under
`training.ga_search` in the training config. The loss uses `training.loss_params` for
thresholds and defaults.
