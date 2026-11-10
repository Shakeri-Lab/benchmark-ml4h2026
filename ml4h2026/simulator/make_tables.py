"""Create new Table 4/5 fragments; never edit the paper or existing artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from analysis.behavior_policy_baselines import load, bgri, episode_table, boot, MODELS

ORDER = ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "attengluco", "patchmlp_glimmer"]
LABELS = dict(arimax="ARIMAX", dlinear="DLinear", gru="GRU", itransformer="iTransformer", patchmlp="PatchMLP",
              attengluco="AttenGluco", patchmlp_glimmer=r"Glimmer\textsuperscript{*}")


def get_metric(df, family, metric, allow_missing=False):
    s = df[(df.perturbation_family == family) & (df.horizon_minutes == 240) & (df.metric_name == metric)]
    s = s[s.ordinal_group == family + "_intensity"] if metric == "kendall_rank_corr" else s[s.perturbation_type == "__ALL__"]
    if len(s) == 0 and allow_missing:
        return pd.Series(dict(metric_value=np.nan, ci_low=np.nan, ci_high=np.nan))
    assert len(s) == 1, (family, metric, s)
    return s.iloc[0]


def endpoint_null(z, family):
    names = z["scenario_name"].astype(str).reshape(-1)
    key = list(zip(z["pat_id"].reshape(-1), z["seq_id"].reshape(-1)))
    factual = {k: pr for k,n,pr in zip(key, names, z["preds"]) if n == "factual"}
    differences = [float(pr[-1] - factual[k][-1]) for k,n,pr in zip(key, names, z["preds"])
                   if n.startswith("basal_" if family == "basal" else "bolus_add_avg_")]
    return bool(np.max(np.abs(differences)) <= 1e-4)


def selection_summary(df):
    ep, oracle, actions, modal = episode_table(df)
    r = boot(ep, "regret").rename(columns={"mean": "regret", "lo": "regret_lo", "hi": "regret_hi"})
    m = boot(ep, "match").rename(columns={"mean": "AMR", "lo": "AMR_lo", "hi": "AMR_hi"})
    t = r.merge(m, on="policy")
    const = t[t.policy.str.startswith("__const__")].sort_values(["regret", "policy"])
    best = const.iloc[0].copy(); best["policy"] = "Best constant"
    t = pd.concat([t[~t.policy.str.startswith("__const__")], best.to_frame().T], ignore_index=True)
    return t, len(oracle), len(actions)


def make(out, variant):
    run = out / variant
    tables = run / "tables_v2"; tables.mkdir(exist_ok=False)
    label = "ARIMAX"
    z = np.load(run / "cf_artifacts.npz", allow_pickle=True)
    dfs = {m: pd.read_csv(Path("outputs/review_revision_20261007/table4") / (m + ".csv")) for m in ORDER}
    original = dfs["arimax"].copy(); dfs["arimax"] = pd.read_csv(run / "table4_metrics.csv")
    rows4 = []; comparisons = []
    for family in ("basal", "bolus"):
        for m in ORDER:
            for metric in ("rmse_effect", "sign_agreement", "kendall_rank_corr"):
                undefined = (metric != "rmse_effect" and (m == "dlinear" or (m == "arimax" and endpoint_null(z, family))))
                v = get_metric(dfs[m], family, metric, allow_missing=undefined)
                row = dict(family=family, model=label if m == "arimax" else LABELS[m], model_id=m, metric=metric,
                           value=float(v.metric_value), lo=float(v.ci_low), hi=float(v.ci_high), displayed_as_dash=undefined)
                rows4.append(row)
                if m == "arimax":
                    old = get_metric(original, family, metric)
                    comparisons.append(dict(table=4, family=family, metric=metric, original=float(old.metric_value), new=float(v.metric_value),
                                            original_lo=float(old.ci_low), original_hi=float(old.ci_high),
                                            new_lo=float(v.ci_low), new_hi=float(v.ci_high), displayed_as_dash=undefined))
    t4 = pd.DataFrame(rows4); t4.to_csv(tables / "table4.csv", index=False)
    tex4 = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
            r"\caption{\textbf{Counterfactual performance.} Patient-level mean [95\% CI] over basal and bolus menus; dashes denote null endpoint responses.}",
            r"\label{tab:cf_menu_isolated_" + variant + "}", r"\begin{tabular}{llccc}\toprule",
            r"& Model & eRMSE $\downarrow$ & SA $\uparrow$ & $\tau_b$ $\uparrow$ \\\midrule"]
    for family in ("basal", "bolus"):
        tex4.append(r"\multirow{7}{*}{\rotatebox[origin=c]{90}{\small " + family.capitalize() + " pert.}}")
        for m in ORDER:
            cells = []
            for metric in ("rmse_effect", "sign_agreement", "kendall_rank_corr"):
                current = t4[(t4.family == family) & (t4.model_id == m) & (t4.metric == metric)].iloc[0]
                if current.displayed_as_dash:
                    cells.append("---"); continue
                eligible = t4[(t4.family == family) & (t4.metric == metric) & ~t4.displayed_as_dash]
                vals = eligible.value.round(2)
                best = vals.min() if metric == "rmse_effect" else vals.max()
                val = f"{current.value:.2f} [{current.lo:.2f}, {current.hi:.2f}]"
                cells.append(r"\textbf{" + val + "}" if round(current.value, 2) == best else val)
            tex4.append("& " + (label if m == "arimax" else LABELS[m]) + " & " + " & ".join(cells) + r" \\")
        if family == "basal": tex4.append(r"\midrule")
    tex4.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    with (tables / "table4.tex").open("x") as f: f.write("\n".join(tex4) + "\n")

    rows5 = []; checks = []
    for family, regex in (("basal", r"^basal_[0-9.]+x$"), ("bolus", r"^bolus_add_avg_[0-9.]+x$")):
        df = load(family, regex)
        names = z["scenario_name"].astype(str).reshape(-1)
        keep = np.array([bool(re.match(regex, n)) or n == "factual" for n in names])
        model_rows = df.model == "arimax"
        existing = df.loc[model_rows]
        np.testing.assert_array_equal(existing.name.to_numpy(), names[keep])
        np.testing.assert_array_equal(existing.pat_id.to_numpy(), z["pat_id"].reshape(-1)[keep])
        np.testing.assert_array_equal(existing.seq_id.to_numpy(), z["seq_id"].reshape(-1)[keep])
        np.testing.assert_allclose(existing.actual_bgri, bgri(z["gts"][keep]), rtol=0, atol=0)
        df.loc[model_rows, "pred_bgri"] = bgri(z["preds"][keep])
        table, episodes, actions = selection_summary(df)
        saved = pd.read_csv(f"outputs/analysis/behavior_policy_{family}.csv").set_index("policy")
        for r in table.to_dict("records"):
            policy = r["policy"]
            if policy in ORDER or policy in ("Control-IQ (behavior policy)", "Best constant"):
                rows5.append(dict(family=family, n_episodes=episodes, n_actions=actions, **r))
            if policy in ORDER and policy != "arimax" or policy == "Control-IQ (behavior policy)":
                for metric in ("regret", "regret_lo", "regret_hi", "AMR", "AMR_lo", "AMR_hi"):
                    difference = abs(float(r[metric]) - float(saved.loc[policy, metric]))
                    checks.append(dict(family=family, model=policy, metric=metric, absolute_difference=difference))
                    assert difference <= 1e-10, (policy, family, metric, difference)
            if policy == "arimax":
                for metric in ("AMR", "regret"):
                    comparisons.append(dict(table=5, family=family, metric=metric, original=float(saved.loc[policy, metric]), new=float(r[metric]),
                                            original_lo=float(saved.loc[policy, metric+"_lo"]), original_hi=float(saved.loc[policy, metric+"_hi"]),
                                            new_lo=float(r[metric+"_lo"]), new_hi=float(r[metric+"_hi"]), displayed_as_dash=False))
    t5 = pd.DataFrame(rows5); t5.to_csv(tables / "table5.csv", index=False)
    pd.DataFrame(checks).to_csv(tables / "unchanged_model_checks.csv", index=False)
    pd.DataFrame(comparisons).to_csv(tables / "original_vs_new.csv", index=False)
    tex5 = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{8pt}",
            r"\caption{\textbf{Insulin action selection.} AMR: action match rate.}",
            r"\label{tab:policy_isolated_" + variant + "}", r"\begin{tabular}{clrr}\toprule",
            r"& Model & AMR $\uparrow$ & Regret $\downarrow$ \\\midrule"]
    for family in ("basal", "bolus"):
        sub = t5[t5.family == family].set_index("policy")
        tex5.append(r"\multirow{9}{*}{\rotatebox[origin=c]{90}{\small " + family.capitalize() + " selection}}")
        for policy in ["Best constant", "Control-IQ (behavior policy)", *ORDER]:
            row = sub.loc[policy]
            shown = label if policy == "arimax" else (LABELS[policy] if policy in ORDER else
                    r"\emph{" + ("Best constant" if policy == "Best constant" else "Control-IQ") + "}")
            cells = []
            for metric in ("AMR", "regret"):
                val = float(row[metric]); text = f"{val:.3f}"
                eligible = sub.loc[ORDER, metric].astype(float).round(3)
                best = eligible.max() if metric == "AMR" else eligible.min()
                if policy in ORDER and round(val, 3) == best:
                    text = r"\textbf{" + text + "}"
                cells.append(text)
            tex5.append("& " + shown + " & " + " & ".join(cells) + r" \\")
            if policy == "Control-IQ (behavior policy)": tex5.append(r"\cmidrule(l){2-4}")
        if family == "basal": tex5.append(r"\midrule")
    tex5.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    with (tables / "table5.tex").open("x") as f: f.write("\n".join(tex5) + "\n")
    with (tables / "table_generation_provenance.json").open("x") as f:
        json.dump({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "variant": variant, "original_files_not_modified": True,
                   "unchanged_model_and_controller_checks": len(checks),
                   "metric_conventions": "Original BGRI clipping at 1 mg/dL, exact ties, patient bootstrap and factual-menu inclusion retained",
                   "label_note": "Recursive ARX is a different formulation, not a relabeled SARIMAX result"}, f, indent=2)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=["recursive_arx"], required=True)
    a = p.parse_args(); make(a.out, a.variant)
