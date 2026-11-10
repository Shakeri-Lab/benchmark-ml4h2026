#!/usr/bin/env python3
"""Per-horizon RMSE for the retrained 4-hour arm-2 models.

Uses make_cf_dataloader_from_cfg -- the SAME builder training used -- so the
insulin left-shift (_shift_insulin_cols) is applied identically. An earlier
version used TimeSeriesWindowDataset, omitted the shift, and produced numbers
3-7x worse than training for every model whose inputs depend on it.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, yaml
from data import make_cf_dataloader_from_cfg
from models import build_model_from_cfg

MODELS = ["dlinear","gru","itransformer","patchmlp","patchmlp_glimmer","attengluco"]

def main():
    print("  %-18s %10s %10s %10s %10s %7s" % ("model","0-2h","0-4h","step24","step48","n"), flush=True)
    for m in MODELS:
        cfg = yaml.safe_load(Path(f"configs/counterfactual_4h/train/{m}.yaml").read_text())
        md, split = cfg["model_data"], cfg["split"]
        dl_cfg = md["dataloader"]
        pred = int(dl_cfg["pred_len"]); hist = int(dl_cfg["history_len"])
        try:
            ld = make_cf_dataloader_from_cfg(split, dl_cfg, split_part="val", seed=42)
        except Exception as e:
            print(f"  {m}: loader failed: {e}", flush=True); continue
        input_dim = len(ld.dataset.feature_cols)
        model = build_model_from_cfg(md["model"], input_dim=input_dim, history_len=hist, pred_len=pred)
        sd = torch.load(f"outputs/results/cf4h_{m}/checkpoint.pt", map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for k in ("state_dict","model_state","model"):
                if k in sd and isinstance(sd[k], dict): sd = sd[k]; break
        r = model.load_state_dict(sd, strict=False)
        if r.missing_keys:
            print(f"  {m}: SKIP -- {len(r.missing_keys)} missing keys", flush=True); continue
        model.eval()
        se = np.zeros(pred); n = 0
        with torch.no_grad():
            for batch in ld:
                xb, yb = batch[0], batch[1]
                p = model(xb.float())
                p = p.squeeze(-1) if p.dim() == 3 else p
                y = yb.float().squeeze(-1) if yb.dim() == 3 else yb.float()
                se += ((p - y) ** 2).sum(0).numpy(); n += y.shape[0]
        mse = se / max(n, 1)
        print("  %-18s %10.2f %10.2f %10.2f %10.2f %7d  (feat=%s)" % (
            m, np.sqrt(mse[:24].mean()), np.sqrt(mse.mean()),
            np.sqrt(mse[23]), np.sqrt(mse[47]), n, ld.dataset.feature_cols), flush=True)

if __name__ == "__main__":
    main()
