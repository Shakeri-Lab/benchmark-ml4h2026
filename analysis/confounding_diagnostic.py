#!/usr/bin/env python3
"""Measure the observational insulin-glucose association and contrast it with the
interventional one, separately for basal and bolus.

The paper argues that deep forecasters invert the basal effect because, under a
closed-loop behavior policy, insulin is delivered *because* glucose is rising, so
the observational association carries the opposite sign to the causal effect. That
claim is testable directly on this dataset, which contains both arms on identical
episodes:

  observational slope : across episodes, regress subsequent glucose change on the
                        insulin the controller actually delivered (factual arm).
                        This is the signal a model trained observationally sees.
  interventional slope: within episode, regress subsequent glucose change on the
                        commanded multiplier (counterfactual arm). This is truth.

A confounded channel shows a non-negative observational slope against a strongly
negative interventional one. The size of that gap is the confound.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

DS = Path("datasets_processed/action_cf_controliq_4h")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def partial(x, y, z):
    """Correlation of x and y after linearly removing z (pre-window glucose)."""
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 4:
        return np.nan
    Z = np.c_[np.ones_like(z), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearson(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("outputs/analysis/confounding_diagnostic.json"))
    a = ap.parse_args()

    sfx = pd.read_csv(DS / "test_suffix.csv")
    pre = pd.read_csv(DS / "test_prefix.csv")
    key = ["pat_id", "seq_id"]
    g0 = pre.sort_values("date").groupby(key)["cgm"].last().rename("g0")

    fac = sfx[sfx.scenario_type == "factual"].copy()
    per = fac.groupby(key).agg(basal_U=("basal", "sum"), bolus_U=("bolus", "sum"),
                               g_mean=("cgm", "mean")).join(g0)
    per["dg"] = per.g_mean - per.g0
    out = {"n_episodes": int(len(per))}
    print(f"[obs] {len(per)} factual episodes")

    # ---- observational: across episodes, what the controller actually delivered ----
    for chan, col in (("basal", "basal_U"), ("bolus", "bolus_U")):
        r = pearson(per[col], per.dg)
        rp = partial(per[col], per.dg, per.g0)
        sl = np.polyfit(per[col][np.isfinite(per[col])], per.dg[np.isfinite(per[col])], 1)[0]
        out[f"obs_{chan}"] = {"pearson_r": r, "partial_r_given_g0": rp, "slope_mgdl_per_U": float(sl)}
        print(f"[obs] {chan:5s}: r={r:+.3f}  partial r|g0={rp:+.3f}  slope={sl:+.2f} mg/dL per U")

    # ---- interventional: within episode, across the commanded menu ----
    cf = sfx[sfx.scenario_type == "counterfactual"].copy()
    cf["scale"] = [float(m.group(1)) if (m := re.search(r"_([0-9.]+)x$", str(n))) else np.nan
                   for n in cf.scenario_name]
    for chan, pat in (("basal", r"^basal_"), ("bolus", r"^bolus_add_avg_")):
        sub = cf[cf.scenario_name.str.match(pat)]
        ep = sub.groupby(key + ["scenario_name", "scale"])["cgm"].mean().reset_index()
        rs, sls, deliv = [], [], []
        for (p, s), g in ep.groupby(key):
            if g.scale.nunique() < 3:
                continue
            rs.append(pearson(g.scale, g.cgm))
            sls.append(np.polyfit(g.scale, g.cgm, 1)[0])
        # convert per-multiplier slope to per-U using the mean delivered insulin
        u = per["basal_U" if chan == "basal" else "bolus_U"].mean()
        per_U = float(np.nanmean(sls) / u) if u else float("nan")
        out[f"int_{chan}"] = {"mean_within_episode_r": float(np.nanmean(rs)),
                             "mean_slope_mgdl_per_multiplier": float(np.nanmean(sls)),
                             "mean_delivered_U": float(u),
                             "slope_mgdl_per_U": per_U,
                             "n_episodes": len(rs)}
        print(f"[int] {chan:5s}: mean within-episode r={np.nanmean(rs):+.3f}  "
              f"slope={np.nanmean(sls):+.2f} mg/dL per multiplier  "
              f"({per_U:+.2f} per U, mean delivered {u:.2f} U, n={len(rs)})")

    for chan in ("basal", "bolus"):
        gap = out[f"obs_{chan}"]["pearson_r"] - out[f"int_{chan}"]["mean_within_episode_r"]
        out[f"gap_{chan}"] = gap
        print(f"[gap] {chan:5s}: observational r minus interventional r = {gap:+.3f}")
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2))
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
