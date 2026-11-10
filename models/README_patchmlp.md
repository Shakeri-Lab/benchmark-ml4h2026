This repository vendors the official TFB PatchMLP implementation under `third_party/tfb_upstream` and wraps
only the PatchMLP model for the benchmark.

Upstream repository: https://github.com/decisionintelligence/TFB.git  
Vendored commit: 1284925fd48d62f39dcb96aeddfa46dbb2905ec3

Wrapper: `models/patchmlp.py`  
- Uses the upstream `ts_benchmark/baselines/patchmlp/models/patchmlp_model.py::PatchMLPModel` directly.  
- Outputs the CGM channel only to match the benchmark target shape `[B, pred_len]`.  
- Multi/univariate behavior controls the input channels; output remains univariate.  
