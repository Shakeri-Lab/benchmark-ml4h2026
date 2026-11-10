"""Restricted-menu selection from frozen corrected ARIMAX predictions only.

Run from the HPC repository root. Preserve source artifacts and score all
policies against the same restricted menu, including its restricted oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from analysis.behavior_policy_baselines import load, bgri, episode_table, boot


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(frame, family, setting):
    episodes, oracle, actions, _ = episode_table(frame)
    r = boot(episodes, "regret").rename(columns={"mean": "regret", "lo": "regret_lo", "hi": "regret_hi"})
    a = boot(episodes, "match").rename(columns={"mean": "AMR", "lo": "AMR_lo", "hi": "AMR_hi"})
    table = r.merge(a, on="policy")
    constants = table[table.policy.str.startswith("__const__")].sort_values(["regret", "policy"])
    best = constants.iloc[0].copy()
    selected_constant = best.policy.removeprefix("__const__")
    best["policy"] = "Retrospective best constant"
    table = pd.concat([table[~table.policy.str.startswith("__const__")], best.to_frame().T], ignore_index=True)
    table = table.assign(family=family, setting=setting, n_episodes=len(oracle), n_actions=len(actions))
    return table, episodes, selected_constant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "restricted_selection.csv"
    if target.exists():
        raise FileExistsError(target)
    corrected = Path("outputs/review_revision_20261007/arimax_isolated_20261007/run/selected_refits/recursive_arx")
    paths = [corrected / "cf_artifacts.npz", corrected / "tables_v2/table5.csv",
             Path("outputs/review_revision_20261007/actions/action_selection.csv")]
    paths += [Path(f"outputs/results/cf4h_{m}/cf_artifacts.npz") for m in
              ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "patchmlp_glimmer", "attengluco"]]
    before = {str(p): sha(p) for p in paths}
    z = np.load(corrected / "cf_artifacts.npz", allow_pickle=True)
    names = z["scenario_name"].astype(str).reshape(-1)
    reference = pd.read_csv(corrected / "tables_v2/table5.csv")
    previous = pd.read_csv("outputs/review_revision_20261007/actions/action_selection.csv")
    tables, records, checks, constants = [], [], [], []
    for family, regex in [("basal", r"^basal_[0-9.]+x$"), ("bolus", r"^bolus_add_avg_[0-9.]+x$")]:
        frame = load(family, regex)
        keep = np.array([bool(re.match(regex, n)) or n == "factual" for n in names])
        old = frame[frame.model == "arimax"]
        for col, values in [("name", names[keep]), ("pat_id", z["pat_id"].reshape(-1)[keep]),
                            ("seq_id", z["seq_id"].reshape(-1)[keep])]:
            np.testing.assert_array_equal(old[col].to_numpy(), values)
        np.testing.assert_allclose(old.actual_bgri, bgri(z["gts"][keep]), rtol=0, atol=0)
        frame.loc[frame.model == "arimax", "pred_bgri"] = bgri(z["preds"][keep])
        for setting in ["full", "up_to_2x"]:
            subset = frame if setting == "full" else frame[(frame.name == "factual") | frame.scale.between(0, 2)].copy()
            table, episodes, constant = summarize(subset, family, setting)
            assert int(table.n_episodes.iloc[0]) == 80
            assert episodes.pat_id.nunique() == 20
            constants.append(dict(family=family, setting=setting, constant_action=constant))
            now = table.set_index("policy")
            for policy in ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "patchmlp_glimmer", "attengluco", "Control-IQ (behavior policy)"]:
                saved = (reference[(reference.family == family) & (reference.policy == policy)].iloc[0]
                         if setting == "full" else
                         previous[(previous.family == family) & (previous.setting == setting) & (previous.policy == policy)].iloc[0])
                if setting == "up_to_2x" and policy == "arimax":
                    continue
                for metric in ["regret", "regret_lo", "regret_hi", "AMR", "AMR_lo", "AMR_hi"]:
                    error = abs(float(now.loc[policy, metric]) - float(saved[metric]))
                    assert error <= 1e-10, (family, setting, policy, metric, error)
                    checks.append(dict(family=family, setting=setting, policy=policy, metric=metric, absolute_error=error))
            tables.append(table)
            records.append(episodes.assign(family=family, setting=setting))
    after = {str(p): sha(p) for p in paths}
    assert before == after
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(target, index=False)
    pd.concat(records, ignore_index=True).to_csv(args.out / "episode_selection.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out / "reproduction_checks.csv", index=False)
    pd.DataFrame(constants).to_csv(args.out / "selected_constants.csv", index=False)
    provenance = dict(source_hashes=before, original_artifacts_unchanged=before == after,
                      script_sha256=sha(Path(__file__)), n_reproduction_checks=len(checks),
                      conventions="Original BGRI clipping, exact ties, 2000 subject bootstrap draws seed 0; menu and oracle both restricted; factual retained",
                      bolus_limit="Added bolus at most 2x reference bolus, not total delivered insulin <=2x")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(result[(result.setting == "up_to_2x") & result.policy.isin(["arimax", "attengluco", "Retrospective best constant"])].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
