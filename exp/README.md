# Experiment Harness

This folder contains utilities for running a single experiment (training/eval) on a specified split.

Key entrypoint:
- `run_experiment.py`: loads data via the split specs, instantiates a model from `models/`, runs evaluation with `metrics/clinical.py`, and writes outputs to `outputs/results/<run_id>/` (config.yaml, metrics.json, preds.npz, GT.npz, logs.log).
- `run_counterfactual_experiment.py`: counterfactual harness for action-conditional evaluation (scenario-aware metrics + saved X/Y/Y_hat with scenario labels).

Model registry:
- Defined in `models/registry.py` (e.g., `zoh` baseline). Extend this with new architectures as needed.

Usage example (train + eval):
```bash
python exp/run_experiment.py \
  --train --test \
  --train-config configs/training/dclp3_private_linear.yaml \
  --eval-config configs/evaluation/dclp3_private_linear.yaml \
  --run-id dclp3_private_lin
```
