This directory vendors the official GARNN implementation.

Upstream URL: https://github.com/ChengzhePiao/garnn_public.git  
Upstream SSH: git@github.com:ChengzhePiao/garnn_public.git  
Commit: daad52cde3fd5f6de132de0c24cc4b36deb7319d
License: Not found in upstream repository (no LICENSE file detected).

Modifications:
- Guard optional dependencies (dgl, torchvision) in `util.py` to allow importing without those packages.
- Guard optional dependency `positional_encodings` in `gat_mem_org_log/method.py` (PE-only variants).
- Guard optional dependency `pytorch_forecasting` in `gat_mem_org_log/method.py` (NHiTS/NBEATS variants).
