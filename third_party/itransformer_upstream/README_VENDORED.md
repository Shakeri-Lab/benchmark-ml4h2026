This directory vendors the official iTransformer implementation.

Upstream URL: https://github.com/thuml/iTransformer.git
Commit: c2426e68ca13f74aaec08045c5c724d8ad328124
License: MIT (see `LICENSE`)

Modifications:
- `layers/SelfAttention_Family.py`: add a guarded import for `reformer_pytorch` so iTransformer can run without optional Reformer deps.
