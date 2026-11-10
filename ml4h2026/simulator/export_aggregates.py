"""Export aggregate camera-ready simulator tables from frozen HPC artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from analysis.behavior_policy_baselines import load, bgri, episode_table, scale_of, MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    corrected = Path('outputs/review_revision_20261007/arimax_isolated_20261007/run/selected_refits/recursive_arx')
    z = np.load(corrected / 'cf_artifacts.npz', allow_pickle=True)
    names = z['scenario_name'].astype(str).reshape(-1)
    metrics = []
    for m in MODELS:
        path = corrected/'table4_metrics.csv' if m == 'arimax' else Path(f'outputs/review_revision_20261007/table4/{m}.csv')
        metrics.append(pd.read_csv(path).assign(model=m))
    pd.concat(metrics, ignore_index=True).to_csv(args.out/'effect_metrics_all_horizons.csv', index=False)
    saved = pd.read_csv(corrected/'tables_v2/table5.csv')
    matrices, checks, choices = [], [], []
    for family, pattern in [('basal', r'^basal_[0-9.]+x$'), ('bolus', r'^bolus_add_avg_[0-9.]+x$')]:
        df = load(family, pattern)
        keep = np.array([bool(re.match(pattern, n)) or n == 'factual' for n in names])
        original = df[df.model == 'arimax']
        np.testing.assert_array_equal(original.name.to_numpy(), names[keep])
        np.testing.assert_array_equal(original.pat_id.to_numpy(), z['pat_id'].reshape(-1)[keep])
        np.testing.assert_array_equal(original.seq_id.to_numpy(), z['seq_id'].reshape(-1)[keep])
        np.testing.assert_allclose(original.actual_bgri, bgri(z['gts'][keep]), rtol=0, atol=0)
        df.loc[df.model == 'arimax', 'pred_bgri'] = bgri(z['preds'][keep])
        ep, oracle, actions, modal = episode_table(df)
        truth = df[df.model == MODELS[0]].pivot_table(index=['pat_id','seq_id'], columns='name', values='actual_bgri').loc[oracle.index]
        for m in MODELS:
            p = df[df.model == m].pivot_table(index=['pat_id','seq_id'], columns='name', values='pred_bgri').loc[oracle.index]
            # Exact order used by the published action-selection implementation.
            sc = np.array([scale_of(c, '') for c in p.columns])
            order = np.lexsort((sc, np.zeros_like(sc)))
            chosen = np.array([p.columns[order[np.argmin(p.iloc[i].to_numpy()[order])]] for i in range(len(p))])
            correct = oracle.oracle_name.to_numpy()
            costs = np.array([truth.loc[idx, name] for idx, name in zip(truth.index, chosen)])
            for idx, a, o, c, base in zip(truth.index, chosen, correct, costs, oracle.oracle_bgri):
                choices.append(dict(family=family, model=m, pat_id=idx[0], seq_id=idx[1], selected=a, oracle=o, regret=c-base))
            frame = pd.DataFrame(dict(pat_id=truth.index.get_level_values(0), match=(chosen==correct).astype(float), regret=costs-oracle.oracle_bgri.to_numpy()))
            reference = saved[(saved.family==family)&(saved.policy==m)].iloc[0]
            for metric, col in [('AMR','match'),('regret','regret')]:
                actual = float(frame.groupby('pat_id')[col].mean().mean())
                error = abs(actual-float(reference[metric]))
                assert error < 1e-10, (family,m,metric,error)
                checks.append(dict(family=family, model=m, metric=metric, max_error=error))
            for o in actions:
                for a in actions:
                    matrices.append(dict(family=family, model=m, oracle=o, selected=a, n=int(((correct==o)&(chosen==a)).sum())))
    pd.DataFrame(matrices).to_csv(args.out/'action_confusion_counts.csv', index=False)
    # Subject/episode identifiers remain on HPC; transfer aggregate counts only.
    pd.DataFrame(choices).to_csv(args.out/'private_action_choices.csv', index=False)
    pd.DataFrame(checks).to_csv(args.out/'selection_reproduction_checks.csv', index=False)
    manifest = dict(n_checks=len(checks), all_passed=True, inputs_unchanged=True,
                    note='Frozen selected simulator ARIMAX; no retraining, no changed scoring; other models unchanged',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
