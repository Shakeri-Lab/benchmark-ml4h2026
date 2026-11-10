This repository vendors the official GARNN implementation under `third_party/garnn_public_upstream` and wraps
the `gatv2_exp` variant for the benchmark.

Upstream repository: https://github.com/ChengzhePiao/garnn_public.git  
Vendored commit: daad52cde3fd5f6de132de0c24cc4b36deb7319d

Wrapper: `models/garnn_gatv2_exp.py`  
- Uses the upstream `MLP_GATV2_exp_MLP` model directly.  
- The wrapper predicts a single horizon value; use the dataloader `target_horizon` to select which step within the context-to-horizon window (e.g., `-1` for the last step).  
- Multi/univariate behavior is controlled via the `univariate` flag in the model config.  
