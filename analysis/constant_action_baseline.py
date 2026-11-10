"""Reference policies for the bolus action-selection benchmark.

Action match rate and mean regret are only interpretable against the policies a
forecaster must beat to be worth consulting:

  oracle          -- picks argmin of the simulator-evaluated cost (regret 0)
  best constant   -- always plays one fixed dose, chosen to minimise mean regret;
                     ignores the input entirely, so any forecaster that does not
                     beat it is contributing nothing
  random          -- expected regret of choosing uniformly from the menu
  worst constant  -- always plays the single worst fixed dose

It also reports the *action sensitivity* of each model's predicted cost: the
mean spread of pred_bgri across the action menu within an episode. A model whose
predicted cost barely moves as the dose sweeps 0x to 2x is not selecting an
action, it is breaking a numerical tie.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(run_dir: Path) -> pd.DataFrame:
    z = np.load(run_dir / "policy_regret_actions.npz", allow_pickle=True)
    return pd.DataFrame({
        "pat_id": z["pat_id"].astype(int),
        "seq_id": z["seq_id"].astype(int),
        "action_scale": z["action_scale"].astype(float),
        "actual_bgri": z["actual_bgri"].astype(float),
        "pred_bgri": z["pred_bgri"].astype(float),
    })


def analyse(run_dir: Path) -> dict:
    df = load(run_dir)
    episodes = list(df.groupby(["pat_id", "seq_id"], sort=True))
    scales = np.sort(df["action_scale"].unique())

    model_regret, model_match, spread = [], [], []
    const_regret = {s: [] for s in scales}
    const_match = {s: [] for s in scales}
    rand_regret = []

    for _, g in episodes:
        g = g.sort_values("action_scale")
        actual = g["actual_bgri"].to_numpy()
        pred = g["pred_bgri"].to_numpy()
        sc = g["action_scale"].to_numpy()
        if len(actual) < 2:
            continue
        best_cost = actual.min()
        oracle_scale = sc[int(np.argmin(actual))]

        chosen = int(np.argmin(pred))
        model_regret.append(actual[chosen] - best_cost)
        model_match.append(float(sc[chosen] == oracle_scale))
        spread.append(pred.max() - pred.min())

        for s in scales:
            idx = np.where(sc == s)[0]
            if idx.size:
                const_regret[s].append(actual[idx[0]] - best_cost)
                const_match[s].append(float(s == oracle_scale))
        rand_regret.append(float(np.mean(actual - best_cost)))

    const_mean = {s: float(np.mean(v)) for s, v in const_regret.items() if v}
    best_s = min(const_mean, key=const_mean.get)
    worst_s = max(const_mean, key=const_mean.get)

    return {
        "run": run_dir.name,
        "n_episodes": len(model_regret),
        "model_regret": float(np.mean(model_regret)),
        "model_AMR": float(np.mean(model_match)),
        "best_const_regret": const_mean[best_s],
        "best_const_scale": best_s,
        "best_const_AMR": float(np.mean(const_match[best_s])),
        "random_regret": float(np.mean(rand_regret)),
        "worst_const_regret": const_mean[worst_s],
        "pred_cost_spread": float(np.mean(spread)),
        "beats_best_const": bool(np.mean(model_regret) < const_mean[best_s]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rows = []
    for r in a.runs:
        try:
            rows.append(analyse(r))
        except Exception as e:
            print(f"[!] {r.name}: {type(e).__name__}: {e}")
    if not rows:
        raise SystemExit("no runs analysed")
    df = pd.DataFrame(rows).sort_values("model_regret")
    df["run"] = df["run"].str.replace("action_cf_benchmark_bolus_controliq_", "", regex=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    ref = rows[0]
    print(f"\nReference policies (identical for all models):")
    print(f"  best constant dose  = {ref['best_const_scale']:.2f}x  "
          f"mean regret {ref['best_const_regret']:.3f}  AMR {ref['best_const_AMR']:.3f}")
    print(f"  random action       mean regret {ref['random_regret']:.3f}")
    print(f"  worst constant dose mean regret {ref['worst_const_regret']:.3f}")
    if a.out_csv:
        a.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out_csv, index=False)
        print(f"\n[+] wrote {a.out_csv}")


if __name__ == "__main__":
    main()
