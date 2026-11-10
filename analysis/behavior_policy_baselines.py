#!/usr/bin/env python3
"""Score the deployed controller (Control-IQ) as a policy in the action-selection
benchmark, alongside input-independent references, with patient-level bootstrap CIs.

The controller's delivered action is the factual arm, so it is a policy like any
other: it selects one action per episode without consulting a forecaster. Adding
it answers "does the deployed system itself choose well?", and supplies the
behavior-policy baseline that raw action-match rates need in order to be read.
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "patchmlp_glimmer", "attengluco"]
FACTUAL = "factual"


def bgri(traj: np.ndarray) -> np.ndarray:
    g = np.clip(np.asarray(traj, dtype=np.float64), 1.0, None)
    f = 1.509 * (np.power(np.log(g), 1.084) - 5.381)
    return np.mean(10.0 * np.square(f), axis=-1)


def scale_of(name: str, fam: str) -> float:
    if name == FACTUAL:
        # Control-IQ's own delivery: 1x basal; for the add-average bolus menu,
        # adding nothing is the factual action.
        return 1.0 if fam == "basal" else 0.0
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)x$", name)
    return float(m.group(1)) if m else float("nan")


def load(fam: str, menu_re: str) -> pd.DataFrame:
    pat = re.compile(menu_re)
    frames = []
    for m in MODELS:
        z = np.load(f"outputs/results/cf4h_{m}/cf_artifacts.npz", allow_pickle=True)
        name = np.array([str(x) for x in np.asarray(z["scenario_name"]).reshape(-1)])
        keep = np.array([bool(pat.match(n)) or n == FACTUAL for n in name])
        d = pd.DataFrame({
            "pat_id": np.asarray(z["pat_id"]).reshape(-1)[keep],
            "seq_id": np.asarray(z["seq_id"]).reshape(-1)[keep],
            "name": name[keep],
            "actual_bgri": bgri(np.asarray(z["gts"])[keep]),
            "pred_bgri": bgri(np.asarray(z["preds"])[keep]),
        })
        d["model"] = m
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["scale"] = [scale_of(n, fam) for n in df["name"]]
    return df


def episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (episode, policy) with the action chosen and the regret incurred."""
    truth = df[df.model == MODELS[0]][["pat_id", "seq_id", "name", "scale", "actual_bgri"]]
    counts = truth.groupby(["pat_id", "seq_id"]).size()
    full = counts.max()
    keep = set(counts[counts == full].index)
    truth = truth[[(p, s) in keep for p, s in zip(truth.pat_id, truth.seq_id)]]
    actions = sorted(truth.name.unique(), key=lambda n: (np.isnan(truth[truth.name == n].scale.iloc[0]),
                                                         truth[truth.name == n].scale.iloc[0]))
    # oracle per episode: lowest cost, ties broken toward less insulin
    orc = (truth.sort_values(["pat_id", "seq_id", "actual_bgri", "scale"])
                .groupby(["pat_id", "seq_id"]).first().reset_index()
                .rename(columns={"name": "oracle_name", "scale": "oracle_scale",
                                 "actual_bgri": "oracle_bgri"}))
    modal = Counter(orc.oracle_name).most_common(1)[0][0]
    cost = truth.pivot_table(index=["pat_id", "seq_id"], columns="name", values="actual_bgri")
    orc = orc.set_index(["pat_id", "seq_id"])
    rows = []

    def add(policy, chosen):
        sel = np.array([cost.loc[i, c] for i, c in zip(cost.index, chosen)])
        rows.append(pd.DataFrame({
            "policy": policy, "pat_id": [i[0] for i in cost.index],
            "seq_id": [i[1] for i in cost.index],
            "regret": sel - orc.loc[cost.index, "oracle_bgri"].to_numpy(),
            "match": (np.array(chosen) == orc.loc[cost.index, "oracle_name"].to_numpy()).astype(float)}))

    add("Control-IQ (behavior policy)", [FACTUAL] * len(cost))
    add("Modal oracle action", [modal] * len(cost))
    for a in actions:
        add(f"__const__{a}", [a] * len(cost))
    # uniform random: exact expectation over the menu, no sampling noise
    rr = (cost[actions].to_numpy() - orc.loc[cost.index, "oracle_bgri"].to_numpy()[:, None]).mean(axis=1)
    mm = np.array([(np.array(actions) == o).mean() for o in orc.loc[cost.index, "oracle_name"]])
    rows.append(pd.DataFrame({"policy": "Uniform random action", "pat_id": [i[0] for i in cost.index],
                              "seq_id": [i[1] for i in cost.index], "regret": rr, "match": mm}))
    for m in MODELS:
        g = df[df.model == m]
        p = g.pivot_table(index=["pat_id", "seq_id"], columns="name", values="pred_bgri").loc[cost.index]
        sc = np.array([scale_of(c, "") for c in p.columns])
        order = np.lexsort((sc, np.zeros_like(sc)))
        chosen = [p.columns[order[np.argmin(p.iloc[i].to_numpy()[order])]] for i in range(len(p))]
        add(m, chosen)
    return pd.concat(rows, ignore_index=True), orc, actions, modal


def boot(ep: pd.DataFrame, col: str, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    per = ep.groupby(["policy", "pat_id"])[col].mean().reset_index()
    pats = sorted(per.pat_id.unique())
    wide = per.pivot(index="pat_id", columns="policy", values=col).loc[pats]
    idx = rng.integers(0, len(pats), size=(n, len(pats)))
    samp = np.stack([wide.to_numpy()[i].mean(axis=0) for i in idx])
    return (pd.DataFrame({"policy": wide.columns, "mean": wide.to_numpy().mean(axis=0),
                          "lo": np.percentile(samp, 2.5, axis=0),
                          "hi": np.percentile(samp, 97.5, axis=0)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["basal", "bolus"])
    ap.add_argument("--menu-regex", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    a = ap.parse_args()
    df = load(a.family, a.menu_regex)
    ep, orc, actions, modal = episode_table(df)
    n_ep = ep[ep.policy == "Control-IQ (behavior policy)"].shape[0]
    print(f"[{a.family}] {n_ep} episodes, action set = {len(actions)} "
          f"({len(actions)-1} menu + factual), modal oracle action = {modal}")
    print(f"[{a.family}] uniform-random match rate = {1/len(actions):.3f}")
    r = boot(ep, "regret").rename(columns={"mean": "regret", "lo": "regret_lo", "hi": "regret_hi"})
    m = boot(ep, "match").rename(columns={"mean": "AMR", "lo": "AMR_lo", "hi": "AMR_hi"})
    t = r.merge(m, on="policy")
    const = t[t.policy.str.startswith("__const__")].sort_values("regret")
    t = t[~t.policy.str.startswith("__const__")]
    best, worst = const.iloc[0], const.iloc[-1]
    for lab, row in (("Best constant action", best), ("Worst constant action", worst)):
        row = row.copy(); row["policy"] = f"{lab} ({row.policy.replace('__const__','')})"
        t = pd.concat([t, row.to_frame().T], ignore_index=True)
    t.to_csv(a.out_csv, index=False)
    print(t.sort_values("regret").to_string(index=False, float_format=lambda x: f"{x:7.3f}"))


if __name__ == "__main__":
    main()
