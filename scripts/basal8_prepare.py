#!/usr/bin/env python3
"""Verify the extended-basal simulator output, preprocess it, and merge it with the
existing Control-IQ counterfactual dataset to form an 8-scale basal action menu.

Nothing existing is modified: the merge writes a new processed dataset.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for preprocessing.*
import pandas as pd

TEST_PATIENTS = [2,4,12,21,25,33,41,44,45,48,68,71,72,76,80,86,87,91,97,99]
NEW_SCALES = ["basal_2.5x","basal_3x","basal_3.5x","basal_4x"]
OLD_SCALES = ["basal_0x","basal_0.5x","basal_1.5x","basal_2x"]
KEY = ["pat_id","seq_id","episode_id"]


def verify_raw(raw: Path, require_all: bool) -> list[int]:
    ok, missing = [], []
    for pid in TEST_PATIENTS:
        d = raw / f"patient_{pid:03d}"
        n = len(list(d.glob("*/suffix_basal_*.csv"))) if d.is_dir() else 0
        (ok if n == 16 else missing).append(pid)
    print(f"[verify] complete patients: {len(ok)}/20", flush=True)
    if missing:
        print(f"[verify] incomplete: {missing}", flush=True)
        if require_all:
            sys.exit(f"[verify] FAIL: {len(missing)} patients incomplete")
    return ok


def verify_scaling(raw: Path, pids: list[int]) -> None:
    """Delivered basal must rise with the commanded multiplier."""
    print("[verify] commanded-vs-delivered basal (in perturbation window):", flush=True)
    for pid in pids[:3]:
        ep = sorted((raw / f"patient_{pid:03d}").glob("episode_*"))[0]
        vals = {}
        for s in NEW_SCALES:
            f = ep / f"suffix_{s}.csv"
            if f.exists():
                df = pd.read_csv(f)
                w = df[df.get("in_perturb_window", 1) == 1] if "in_perturb_window" in df else df
                vals[s] = float(w["basal"].sum())
        seq = [vals.get(s, float("nan")) for s in NEW_SCALES]
        mono = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
        print(f"    pat {pid:3d} ep0  " +
              "  ".join(f"{s.replace('basal_',''):>5}={v:8.2f}" for s, v in vals.items()) +
              f"   monotonic={mono}", flush=True)


def _seq_map(new_proc: Path, old_proc: Path) -> dict:
    """seq_id is a global running counter assigned at preprocessing time, so the same
    episode carries different seq_ids in the two datasets. Map on (pat_id, episode_id)."""
    n = pd.read_csv(new_proc / "test_scenarios.csv")[["pat_id","seq_id","episode_id"]].drop_duplicates()
    o = pd.read_csv(old_proc / "test_scenarios.csv")[["pat_id","seq_id","episode_id"]].drop_duplicates()
    o = o.rename(columns={"seq_id": "old_seq"})
    m = n.merge(o, on=["pat_id","episode_id"], how="inner")
    return dict(zip(zip(m.pat_id, m.seq_id), m.old_seq))


def verify_alignment(new_proc: Path, old_proc: Path, smap: dict) -> None:
    """The remapped episodes must be the SAME rollouts: factual prefixes identical."""
    a = pd.read_csv(new_proc / "test_prefix.csv")
    b = pd.read_csv(old_proc / "test_prefix.csv")
    a = a.copy()
    a["seq_id"] = [smap.get((p, s), -1) for p, s in zip(a.pat_id, a.seq_id)]
    ka = a[a.seq_id >= 0].groupby(["pat_id","seq_id"])["cgm"].agg(["count","sum"])
    kb = b.groupby(["pat_id","seq_id"])["cgm"].agg(["count","sum"])
    common = ka.index.intersection(kb.index)
    # CSV float round-trip loses ~1e-9 relative, which on sums of ~7e5 is ~1e-3 absolute:
    # compare relatively, not with a fixed number of decimals.
    import numpy as np
    ca, cb = ka.loc[common], kb.loc[common]
    rel = (ca["sum"] - cb["sum"]).abs() / cb["sum"].abs().clip(lower=1.0)
    # Episodes are re-simulated from identical seeds; residual differences are
    # floating-point non-associativity across node hardware. 1e-4 relative is
    # ~0.02 mg/dL on these trajectories, against effect sizes of 6-15 mg/dL.
    same = int(((ca["count"] == cb["count"]) & (rel < 1e-4)).sum())
    print(f"[verify] max relative prefix difference: {rel.max():.2e} "
          f"(counts identical: {int((ca['count'] == cb['count']).sum())}/{len(common)})", flush=True)
    print(f"[verify] alignment: {len(common)} shared episodes, {same} with identical factual prefix",
          flush=True)
    if len(common) == 0 or same != len(common):
        sys.exit(f"[verify] FAIL: {len(common)-same} episodes differ -- not the same rollouts")


def merge(new_proc: Path, old_proc: Path, out: Path, smap: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in ["train.csv", "val.csv", "test_prefix.csv"]:
        shutil.copy(old_proc / f, out / f)
    for name in ["test_scenarios.csv", "test_suffix.csv"]:
        old = pd.read_csv(old_proc / name)
        new = pd.read_csv(new_proc / name)
        new = new[new["scenario_name"].isin(NEW_SCALES)].copy()
        new["seq_id"] = [smap.get((p, s), -1) for p, s in zip(new.pat_id, new.seq_id)]
        dropped = int((new.seq_id < 0).sum())
        new = new[new.seq_id >= 0]
        both = pd.concat([old, new], ignore_index=True).drop_duplicates()
        both.to_csv(out / name, index=False)
        print(f"[merge] {name}: {len(old)} + {len(new)} -> {len(both)}"
              + (f"  (dropped {dropped} unmapped)" if dropped else ""), flush=True)
    meta = json.loads((old_proc / "metadata.json").read_text())
    meta["notes"] = (meta.get("notes", "") +
                     " EXTENDED: basal action menu widened to {0,0.5,1.5,2,2.5,3,3.5,4}x "
                     "using additional simulator rollouts on the identical episodes.")
    meta["basal_menu"] = [s.replace("basal_", "") for s in OLD_SCALES + NEW_SCALES]
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    sc = pd.read_csv(out / "test_scenarios.csv")
    b = sc[(sc.perturb_family == "basal") & (sc.scenario_type == "counterfactual")]
    per = b.groupby(KEY).size()
    print(f"[merge] basal episodes: {per.size}, scales per episode: {sorted(per.unique())}", flush=True)
    full = int((per == 8).sum())
    print(f"[merge] episodes with the full 8-scale menu: {full}/{per.size}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--old-proc", type=Path, default=Path("datasets_processed/action_cf_benchmark_controliq"))
    p.add_argument("--new-proc", type=Path, default=Path("datasets_processed/action_cf_controliq_basalA"))
    p.add_argument("--out", type=Path, default=Path("datasets_processed/action_cf_controliq_basal8"))
    p.add_argument("--require-all", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    a = p.parse_args()

    pids = verify_raw(a.raw, a.require_all)
    if not pids:
        sys.exit("[verify] FAIL: no complete patients")
    verify_scaling(a.raw, pids)

    if not a.skip_preprocess:
        # The collector walks every patient listed in splits.csv. This raw tree holds
        # only the test-split patients, so write a restricted split naming exactly the
        # ones present. train/val are taken from the existing dataset at merge time.
        sp = a.raw / "splits.csv"
        pd.DataFrame({"pat_id": pids, "split": ["test"] * len(pids)}).to_csv(sp, index=False)
        print(f"[preprocess] restricted splits.csv -> {len(pids)} test patients", flush=True)
        from preprocessing.action_cf_benchmark import _preprocess_action_cf_like
        print(f"[preprocess] {a.raw} -> {a.new_proc}", flush=True)
        _preprocess_action_cf_like(raw_dir=a.raw, output_dir=a.new_proc,
                                   dataset_name="action_cf_controliq_basalA",
                                   notes="Extended basal scales (2.5-4x) on the Control-IQ episodes.")
    smap = _seq_map(a.new_proc, a.old_proc)
    verify_alignment(a.new_proc, a.old_proc, smap)
    merge(a.new_proc, a.old_proc, a.out, smap)
    print("[done] merged dataset at", a.out, flush=True)


if __name__ == "__main__":
    main()
