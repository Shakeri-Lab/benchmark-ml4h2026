## iTransformer baseline

This repository vendors the official iTransformer implementation under:
`third_party/itransformer_upstream/`

Upstream URL: https://github.com/thuml/iTransformer.git  
Vendored commit: `c2426e68ca13f74aaec08045c5c724d8ad328124`  
License: MIT (see `third_party/itransformer_upstream/LICENSE`)

The wrapper in `models/itransformer.py` instantiates the upstream model and adapts
the input/output to the benchmark’s `[B, T, C] -> [B, pred_len]` interface.
