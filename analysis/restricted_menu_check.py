#!/usr/bin/env python3
"""Does the basal reversal survive inside a clinically plausible action range?

Reviewers may reasonably ask whether the inverted ordering is driven by extreme
out-of-distribution commands (0x, 4x). This recomputes sign agreement and Kendall
tau_b on restricted menus that a clinician might actually consider, and reports the
per-action response curve so the effect can be read directly.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "patchmlp_glimmer", "attengluco"]


def kendall_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x); c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            if a == 0 and b == 0: tx += 1; ty += 1
            elif a == 0: tx += 1
            elif b == 0: ty += 1
            elif np.sign(a) == np.sign(b): c += 1
            else: d += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - tx) * (n0 - ty))
    return (c - d) / den if den > 0 else np.nan


def load(model):
    z = np.load(f"outputs/results/cf4h_{model}/cf_artifacts.npz", allow_pickle=True)
    name = np.array([str(x) for x in np.asarray(z["scenario_name"]).reshape(-1)])
    styp = np.array([str(x) for x in np.asarray(z["scenario_type"]).reshape(-1)])
    return pd.DataFrame({
        "pat_id": np.asarray(z["pat_id"]).reshape(-1), "seq_id": np.asarray(z["seq_id"]).reshape(-1),
        "name": name, "type": styp,
        # the shipped counterfactual metrics evaluate at a single horizon index,
        # not the trajectory mean; match that so numbers are comparable to Table 4.
        "true": np.asarray(z["gts"])[..., -1], "pred": np.asarray(z["preds"])[..., -1]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path, default=Path("outputs/analysis/restricted_menu.csv"))
    a = ap.parse_args()
    MENUS = {
        ("basal", "full"):       [0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        ("basal", "plausible"):  [0.5, 1.5, 2.0],
        ("bolus", "full"):       [round(x, 2) for x in np.arange(0, 3.001, 0.25)],
        ("bolus", "plausible"):  [0.5, 0.75, 1.0, 1.25, 1.5],
    }
    rows, curves = [], []
    for m in MODELS:
        df = load(m)
        fac = df[df.type == "factual"].set_index(["pat_id", "seq_id"])
        cf = df[df.type == "counterfactual"].copy()
        cf["scale"] = [float(g.group(1)) if (g := re.search(r"_([0-9.]+)x$", n)) else np.nan for n in cf.name]
        cf["fam"] = np.where(cf.name.str.startswith("basal_"), "basal",
                     np.where(cf.name.str.startswith("bolus_add_avg_"), "bolus", "other"))
        cf = cf[cf.fam != "other"]
        # effects are relative to the factual arm on the same episode
        cf["d_true"] = cf.true.to_numpy() - fac.loc[list(zip(cf.pat_id, cf.seq_id)), "true"].to_numpy()
        cf["d_pred"] = cf.pred.to_numpy() - fac.loc[list(zip(cf.pat_id, cf.seq_id)), "pred"].to_numpy()
        for (fam, lab), scales in MENUS.items():
            sub = cf[(cf.fam == fam) & (cf.scale.round(2).isin(scales))]
            if sub.empty: continue
            sa = float((np.sign(sub.d_pred) == np.sign(sub.d_true)).mean())
            tb = [kendall_b(g.d_pred, g.d_true) for _, g in sub.groupby(["pat_id", "seq_id"])
                  if g.scale.nunique() >= 3]
            rows.append({"model": m, "family": fam, "menu": lab, "n_actions": len(scales),
                         "n_pairs": len(sub), "sign_agreement": sa,
                         "kendall_tau_b": float(np.nanmean(tb)) if tb else np.nan})
            if lab == "full":
                for s, g in sub.groupby(sub.scale.round(2)):
                    curves.append({"model": m, "family": fam, "scale": float(s),
                                   "true_effect": float(g.d_true.mean()),
                                   "pred_effect": float(g.d_pred.mean())})
    t = pd.DataFrame(rows); t.to_csv(a.out_csv, index=False)
    pd.DataFrame(curves).to_csv(str(a.out_csv).replace(".csv", "_curves.csv"), index=False)
    for fam in ("basal", "bolus"):
        print(f"\n=== {fam} ===")
        p = t[t.family == fam].pivot_table(index="model", columns="menu",
                                           values=["sign_agreement", "kendall_tau_b"])
        print(p.round(3).to_string())
    print("\nwrote", a.out_csv)


if __name__ == "__main__":
    main()
