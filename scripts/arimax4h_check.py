#!/usr/bin/env python3
"""Quick validation-split inference test for ARIMAX at pred_len=48.

ARIMAX is not a gradient model: it carries fitted SARIMAX params plus buffers
(exog_keep, exog_is_rate). This checks the whole path actually runs and produces
sane numbers before the 4h simulation lands.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, yaml
from data import make_cf_dataloader_from_cfg
from models import build_model_from_cfg

RUN = sys.argv[1] if len(sys.argv) > 1 else "cf4h_arimax"
CFG = sys.argv[2] if len(sys.argv) > 2 else "configs/counterfactual_4h/train/arimax.yaml"
MAXB = int(sys.argv[3]) if len(sys.argv) > 3 else 8      # cap: SARIMAX filtering is slow

cfg = yaml.safe_load(Path(CFG).read_text())
md, split = cfg["model_data"], cfg["split"]
dl_cfg = dict(md["dataloader"]); dl_cfg["batch_size"] = 16
pred, hist = int(dl_cfg["pred_len"]), int(dl_cfg["history_len"])
ld = make_cf_dataloader_from_cfg(split, dl_cfg, split_part="val", seed=42)
feats = ld.dataset.feature_cols
print(f"  features: {feats}  (n_windows={len(ld.dataset)})", flush=True)

model = build_model_from_cfg(md["model"], input_dim=len(feats), history_len=hist, pred_len=pred)
sd = torch.load(f"outputs/results/{RUN}/checkpoint.pt", map_location="cpu", weights_only=False)
if isinstance(sd, dict):
    for k in ("state_dict","model_state","model"):
        if k in sd and isinstance(sd[k], dict): sd = sd[k]; break
r = model.load_state_dict(sd, strict=False)
print(f"  checkpoint: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
if r.missing_keys: print("   missing:", r.missing_keys[:5], flush=True)
model.eval()

se = np.zeros(pred); n = 0; t0 = time.time()
with torch.no_grad():
    for bi, batch in enumerate(ld):
        if bi >= MAXB: break
        xb, yb = batch[0], batch[1]
        p = model(xb.float())
        p = p.squeeze(-1) if p.dim() == 3 else p
        y = yb.float().squeeze(-1) if yb.dim() == 3 else yb.float()
        if not np.isfinite(p.numpy()).all():
            print(f"  WARNING: non-finite predictions in batch {bi}", flush=True)
        se += ((p - y) ** 2).sum(0).numpy(); n += y.shape[0]
        print(f"   batch {bi+1}/{MAXB}  n={n}  elapsed={time.time()-t0:.1f}s", flush=True)
mse = se / max(n, 1)
print(f"\n  ARIMAX ({RUN}) on val, {n} windows:")
print(f"    RMSE 0-2h = {np.sqrt(mse[:24].mean()):8.2f}")
print(f"    RMSE 0-4h = {np.sqrt(mse.mean()):8.2f}")
print(f"    step24    = {np.sqrt(mse[23]):8.2f}   step48 = {np.sqrt(mse[47]):8.2f}")
