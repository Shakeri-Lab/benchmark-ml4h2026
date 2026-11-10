"""Train/validate/evaluate new models without writing to existing repository paths."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path.cwd()))
from data import make_cf_dataloader_from_cfg
from analysis.counterfactual_metrics import _load_artifacts, _load_valid_scenarios, _filter_valid, build_results
from recursive_arx import MAX_LAG, Specification, RecursiveARX, design_rows, fit_moments

DATA = Path("datasets_processed/action_cf_controliq_4h")
MODELS = ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "attengluco", "patchmlp_glimmer"]


def write_json(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_config():
    return yaml.safe_load(Path("outputs/results/cf4h_arimax/config.yaml").read_text())


def raw(part):
    return pd.read_csv(DATA / (part + ".csv")).sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path("models/arimax/model.py"), Path("data/cf_dataloader.py")]
    for m in MODELS:
        paths.extend(Path(f"outputs/results/cf4h_{m}/{s}") for s in
                     ("config.yaml", "checkpoint.pt", "cf_artifacts.npz"))
    paths.extend(DATA / s for s in ("train.csv", "val.csv", "test_prefix.csv", "test_suffix.csv", "test_scenarios.csv"))
    write_json(out / "original_hashes.json", {str(p): sha(p) for p in paths})
    code = Path(__file__).resolve().parent
    write_json(out / "code_hashes.json", {p.name: sha(p) for p in sorted(code.glob("*.py"))})
    write_json(out / "protocol.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arx_grid": {"glucose_lags": [2, 3, 6, 12, 24, 48],
                     "insulin_lags": [1, 6, 12, 24, 48],
                     "difference_order": [0, 1], "ridge": [0., .0001, .01, 1.]},
        "arx_n_candidates": 240,
        "arx_fit": "one-step regression on all contiguous factual training rows; train-only standardization; unpenalized intercept",
        "arx_selection": "lowest full 48-step recursive factual-validation RMSE among finite stationary-AR candidates; deterministic candidate-index ties",
        "validation_origins": "all original 96-history, stride-48 CF validation origins, using raw aligned insulin; any discontinuities excluded",
        "test": "same 48 targets and known insulin as frozen artifacts; raw prefix provides observed history; no future glucose input",
        "timestamp_convention": "predict G[t] from G[t-1...] and known insulin u[t],u[t-1...], aligned by recorded timestamp",
        "family_selection": "one factual-validation-selected simulator model, evaluated on BOTH basal and bolus; no counterfactual-test tuning",
        "metrics": "unchanged original patient-macro eRMSE/SA/tau and BGRI action-selection code, clipping and tie conventions",
        "test_uses": "no test metrics consulted before model selection; only the published one-step simulator ARIMAX is exposed",
    })
    torch.manual_seed(42)
    cfg = source_config()["train"]
    dc = dict(cfg["model_data"]["dataloader"], shuffle=False, history_len=96, stride=48)
    ds = make_cf_dataloader_from_cfg(cfg["split"], dc, split_part="val", seed=42).dataset
    df = raw("val")
    a = df[["cgm", "basal", "bolus"]].to_numpy(dtype=np.float32)
    keys = df[["pat_id", "seq_id"]].to_numpy()
    dates = pd.to_datetime(df.date).astype("int64").to_numpy()
    histories, plans, targets = [], [], []
    for ix in ds.indices.numpy():
        t = int(ix) + 96
        if not np.all(keys[t - 96:t + 48] == keys[t]):
            raise AssertionError("Original validation window crosses patient/sequence")
        if not np.all(np.diff(dates[t - 96:t + 48]) == 300_000_000_000):
            continue
        histories.append(a[t - 96:t]); plans.append(a[t:t + 48, 1:]); targets.append(a[t:t + 48, 0])
    hist, plan, target = map(np.asarray, (histories, plans, targets))
    # Check targets and insulin shift against the existing loader, independently.
    if len(hist) == len(ds):
        for k in (0, len(ds) // 2, len(ds) - 1):
            batch = ds[k]
            np.testing.assert_array_equal(target[k], batch[1].numpy())
            np.testing.assert_array_equal(plan[k], batch[0].numpy()[-48:, 1:])
            np.testing.assert_array_equal(hist[k, -48:, 1:], batch[0].numpy()[:48, 1:])
    np.savez_compressed(out / "validation_windows.npz", history=hist, plan=plan, target=target)
    write_json(out / "validation_support.json", {"n_original_windows": len(ds), "n_contiguous_windows": len(hist),
                                               "n_patients": int(df.pat_id.nunique()), "history": 96, "horizon": 48})
    print(f"Prepared {len(hist)} validation origins; 240 one-step ARIMAX candidates", flush=True)


def training_moments(out):
    df = raw("train")
    width = 4 * MAX_LAG
    m = dict(n=0, sum_x=np.zeros(width), sum_y=np.zeros(2), xtx=np.zeros((width, width)), xty=np.zeros((width, 2)))
    discarded = 0
    for _, g in df.groupby(["pat_id", "seq_id"], sort=False):
        a = g[["cgm", "basal", "bolus"]].to_numpy(dtype=float)
        times = pd.to_datetime(g.date).astype("int64").to_numpy()
        # No lags can cross a missing five-minute sample, patient or sequence.
        boundaries = np.r_[0, np.flatnonzero(np.diff(times) != 300_000_000_000) + 1, len(a)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            seg = a[start:end]
            if len(seg) <= MAX_LAG + 1:
                discarded += len(seg); continue
            x, y = design_rows(seg, np.arange(MAX_LAG + 1, len(seg)))
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Nonfinite factual training input")
            m["n"] += len(x); m["sum_x"] += x.sum(0); m["sum_y"] += y.sum(0)
            m["xtx"] += x.T @ x; m["xty"] += x.T @ y
    np.savez_compressed(out / "training_moments.npz", **m)
    write_json(out / "training_support.json", {"n_one_step_targets": int(m["n"]), "n_training_patients": int(df.pat_id.nunique()),
                                              "n_raw_rows": len(df), "short_segment_rows_discarded": discarded,
                                              "common_initial_lag_rows_discarded_per_segment": MAX_LAG + 1})
    return m


def arx_search(out):
    run = out / "recursive_arx"; run.mkdir(exist_ok=False)
    m = training_moments(out)
    val = np.load(out / "validation_windows.npz")
    rows = []; best = None; start = time.time()
    grid = itertools.product([2, 3, 6, 12, 24, 48], [1, 6, 12, 24, 48], [0, 1], [0., .0001, .01, 1.])
    for candidate, (p, q, d, ridge) in enumerate(grid):
        spec = Specification(p, q, d, ridge)
        model = fit_moments(spec, m)
        radius = model.spectral_radius()
        pred = model.rollout(val["history"], val["plan"])
        rmse = float(np.sqrt(np.mean((pred - val["target"]) ** 2)))
        eligible = bool(np.isfinite(rmse) and radius <= 1.000001)
        row = dict(candidate=candidate, **asdict(spec), spectral_radius=radius,
                   val_rmse=rmse if np.isfinite(rmse) else None, eligible=eligible)
        rows.append(row)
        if eligible and (best is None or rmse < best[0]):
            best = (rmse, candidate, model)
        if candidate % 20 == 0:
            print(f"ARX {candidate}/240 elapsed={time.time()-start:.1f}s best={best[:2] if best else None}", flush=True)
    pd.DataFrame(rows).to_csv(run / "search.csv", index=False)
    if best is None:
        raise RuntimeError("No finite stationary-AR candidate; no test evaluation permitted")
    rmse, candidate, model = best
    checkpoint = dict(specification=asdict(model.spec), coefficients=model.coefficients.tolist(), intercept=model.intercept)
    write_json(run / "checkpoint.json", checkpoint)
    write_json(run / "selection.json", dict(candidate=candidate, val_rmse=rmse, spectral_radius=model.spectral_radius(),
                                           n_candidates=len(rows), n_eligible=sum(r["eligible"] for r in rows),
                                           specification=asdict(model.spec), elapsed_seconds=time.time()-start))
    # Prove reloading the checkpoint leaves recursive predictions unchanged.
    saved = json.loads((run / "checkpoint.json").read_text())
    reloaded = RecursiveARX(Specification(**saved["specification"]), saved["coefficients"], saved["intercept"])
    np.testing.assert_array_equal(model.rollout(val["history"][:8], val["plan"][:8]),
                                  reloaded.rollout(val["history"][:8], val["plan"][:8]))
    infer_arx(out, model)


def infer_arx(out, model):
    # Selection has been persisted before any test predictions or metrics.
    run = out / "recursive_arx"
    assert (run / "selection.json").exists()
    frozen = np.load("outputs/results/cf4h_arimax/cf_artifacts.npz", allow_pickle=True)
    prefix = raw("test_prefix"); suffix = raw("test_suffix")
    pref = {(int(p), int(s)): g for (p, s), g in prefix.groupby(["pat_id", "seq_id"], sort=False)}
    suf = {(int(p), int(s), str(n)): g for (p, s, n), g in suffix.groupby(["pat_id", "seq_id", "scenario_name"], sort=False)}
    h, u = [], []
    for p, s, n, gt in zip(frozen["pat_id"].reshape(-1), frozen["seq_id"].reshape(-1),
                           frozen["scenario_name"].reshape(-1), frozen["gts"]):
        pg, sg = pref[(int(p), int(s))].iloc[-96:], suf[(int(p), int(s), str(n))].iloc[:48]
        times = pd.to_datetime(pd.concat([pg.date, sg.date])).astype("int64").to_numpy()
        if not np.all(np.diff(times) == 300_000_000_000):
            raise AssertionError("Test history/plan is not contiguous")
        np.testing.assert_array_equal(np.asarray(gt).reshape(-1), sg.cgm.to_numpy(dtype=np.float32))
        h.append(pg[["cgm", "basal", "bolus"]].to_numpy(dtype=float))
        u.append(sg[["basal", "bolus"]].to_numpy(dtype=float))
    history, plan = np.asarray(h), np.asarray(u)
    predictions = model.rollout(history, plan)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nonfinite test rollout")
    arrays = {k: frozen[k] for k in frozen.files}
    arrays["preds"] = predictions.astype(np.float32)
    np.savez_compressed(run / "cf_artifacts.npz", **arrays)
    # Matched direct and wrapper-free rollout; target corruption cannot affect output.
    write_json(run / "inference_checks.json", {
        "n_scenarios": len(predictions), "n_episodes": len(pref), "n_subjects": int(prefix.pat_id.nunique()),
        "exact_ground_truth_match_to_frozen_artifact": True,
        "future_glucose_not_an_input": True, "no_prediction_clipping": True,
        "min_prediction": float(predictions.min()), "max_prediction": float(predictions.max()),
        "factual_rmse": float(np.sqrt(np.mean((predictions[np.asarray(frozen["scenario_name"]).reshape(-1) == "factual"] -
                                               np.asarray(frozen["gts"])[np.asarray(frozen["scenario_name"]).reshape(-1) == "factual"]) ** 2))),
    })
    score_effect(run)
    impulse = []
    base = model.rollout(history[:1], plan[:1])
    for label, channel, sustained in (("basal_impulse", 0, False), ("bolus_impulse", 1, False),
                                      ("basal_sustained", 0, True), ("bolus_sustained", 1, True)):
        changed = plan[:1].copy()
        changed[:, slice(None) if sustained else 0, channel] += 1.
        delta = model.rollout(history[:1], changed) - base
        impulse.extend(dict(probe=label, horizon_min=5*(k+1), delta_glucose=float(v)) for k, v in enumerate(delta[0]))
    pd.DataFrame(impulse).to_csv(run / "impulse_response.csv", index=False)


def score_effect(run):
    df = _filter_valid(_load_artifacts(run / "cf_artifacts.npz"), _load_valid_scenarios(DATA / "test_scenarios.csv"))
    result = build_results(df, [30, 60, 120, 240], 5, 1e-6, 1e-3, 1e-3, 3, 1000, 42)
    pd.DataFrame(result["records"]).to_csv(run / "table4_metrics.csv", index=False)


def integrity(out):
    hashes = json.loads((out / "original_hashes.json").read_text())
    changed = [p for p,h in hashes.items() if sha(p) != h]
    if changed:
        raise RuntimeError(f"Original files changed: {changed}")
    write_json(out / "original_integrity_verified.json", {"unchanged_files": len(hashes), "all_hashes_match": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "arx", "integrity"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"prepare": prepare, "arx": arx_search, "integrity": integrity}[a.stage](a.out)
