# AttenGluco (reference)

This wrapper uses the official AttenGluco implementation vendored under:
`third_party/attengluco_upstream`.

- Upstream: https://github.com/rzarhmi/AttenGluco
- Commit: ff01834e9cd477360f80c7974e1d137a037f6422

Notes:
- The upstream model expects 3 input streams (A/B/C). The wrapper maps our
  channels to these streams while keeping the upstream computation unchanged.
- Output is univariate CGM forecast with shape `[B, pred_len]`.
