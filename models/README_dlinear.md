This repository vendors the official DLinear implementation under `third_party/dlinear_upstream` and wraps
the DLinear model for the benchmark.

Upstream repository: https://github.com/vivva/DLinear.git  
Vendored commit: da9e67442b95af76488b8e4e1806cc3185723dd9

Wrapper: `models/dlinear.py`  
- Uses the upstream `models/DLinear.py::Model` directly.  
- Outputs the CGM channel only to match the benchmark target shape `[B, pred_len]`.  
- Multi/univariate behavior is controlled via the `univariate` flag in the model config.  
