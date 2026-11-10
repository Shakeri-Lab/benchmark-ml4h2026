"""Fit a glucose forecaster on HPC; export no patient-level artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from iob_forecaster import IOBLinearForecaster, features_from_sequence, valid_origins

BASE = Path("outputs/review_revision_20261011/iob_forecaster")
REFERENCE = Path("outputs/review_revision_20261009/warning_lead_budget")
ALPHAS = (0., 1e-6, 1e-4, .01, .1, 1., 10., 100.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)


def examples(raw, patients, stride):
    chunks = {k: [] for k in ("x", "y", "pid", "sid", "idx", "slot")}
    frame = raw[raw.pat_id.isin(patients)].dropna(subset=["cgm"])
    for (pid, sid), group in frame.groupby(["pat_id", "seq_id"], sort=False):
        cgm = group.cgm.to_numpy(dtype=float)
        ns = pd.to_datetime(group.date).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        if (ns % (300*10**9) != 0).any():
            raise RuntimeError("Dates not on the five-minute grid.")
        slots = ns // (300*10**9)
        f = features_from_sequence(cgm, slots, group.basal.to_numpy(dtype=float),
            group[["bolus_standard", "bolus_extended"]].to_numpy(dtype=float).sum(axis=1))
        idx = valid_origins(slots, stride=stride)
        # The existing dataset tests the sequence id at one extra boundary row
        # after the target. Reproduce that boundary eligibility without changing
        # the 12-step target or requiring a 65-minute contiguous forecast window.
        idx = idx[idx+13 < len(cgm)]
        if not len(idx):
            continue
        chunks["x"].append(f[idx])
        chunks["y"].append(cgm[idx[:, None] + np.arange(1, 13)])
        for key, value in (("pid", np.full(len(idx), int(pid))), ("sid", np.full(len(idx), int(sid))),
                           ("idx", idx), ("slot", slots[idx])):
            chunks[key].append(value)
    if not chunks["x"]:
        raise RuntimeError("No eligible windows.")
    return {k: np.concatenate(v) for k, v in chunks.items()}


def patient_rmse(prediction, target, ids):
    return np.asarray([np.sqrt(np.mean((prediction[ids == pid]-target[ids == pid])**2))
                       for pid in np.unique(ids)])


def ci(values):
    rng = np.random.default_rng(42)
    draws = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)), n_patients=len(values))


def run(cohort):
    start = time.monotonic()
    private = BASE / "private" / cohort
    out = BASE / "aggregate/forecasting" / cohort
    if private.exists() or out.exists():
        raise RuntimeError("Existing experiment; refuse to replace artifacts.")
    run_dir = Path("outputs/results") / f"bestck_patchmlp_glimmer_{cohort}_p12"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())["eval"]
    split = cfg["split"]
    spec_path = Path(split["split_root"]) / (split["split_name"] + ".json")
    spec = json.loads(spec_path.read_text())
    sets = [set(spec["patients"][p]) for p in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError("Patient split leakage.")
    source = Path(split["processed_root"]) / spec["dataset"] / "timeseries.csv"
    raw = pd.read_csv(source).sort_values(["pat_id", "seq_id", "date"])
    train = examples(raw, spec["patients"]["train"], stride=12)
    val = examples(raw, spec["patients"]["val"], stride=12)
    candidates = []
    best = None
    for alpha in ALPHAS:
        model = IOBLinearForecaster.fit(train["x"], train["y"], alpha)
        loss = float(patient_rmse(model.predict(val["x"]), val["y"], val["pid"]).mean())
        candidates.append(dict(alpha=alpha, validation_patient_macro_trajectory_rmse=loss))
        if best is None or loss < best[0]:
            best = loss, model
    model = best[1]
    configuration = dict(model="iob_linear", cohort=cohort, horizon_min=60, history_steps=288,
        training_stride=12, validation_model_selection_stride=12, inference_stride=1,
        tau_minutes=100., iob="Lee2009_two_compartment_remaining_fraction",
        iob_source="https://doi.org/10.1177/193229680900300511",
        insulin_channels=["basal", "bolus_standard", "bolus_extended"],
        basal_units="U/hr; multiply by 5/60 to obtain a timestamped five-minute dose",
        bolus_units="U; standard plus extended delivered amounts as recorded",
        history_truncation_min=1440, slope="(CGM[t]-CGM[t-1])/5 mg/dL/min",
        model_selection="validation patient-macro glucose trajectory RMSE only",
        alpha_candidates=list(ALPHAS), selected_alpha=model.alpha,
        train_patients=spec["patients"]["train"], val_patients=spec["patients"]["val"],
        test_patients=spec["patients"]["test"], split_source=str(spec_path),
        source_sha256=sha(source), split_sha256=sha(spec_path),
        code_sha256=sha(__file__), model_code_sha256=sha(Path(__file__).with_name("iob_forecaster.py")))
    write_json(private / "config.json", configuration)
    write_json(private / "model.json", model.to_dict())
    write_json(out / "selection.json", dict(candidates=candidates, selected_alpha=model.alpha,
        n_training_windows=len(train["x"]), n_validation_selection_windows=len(val["x"]),
        no_hypoglycemia_labels_in_model_training=True, no_test_data_in_fitting_or_model_selection=True))
    del train, val
    forecast_rows = []
    for part in ("val", "test"):
        data = examples(raw, spec["patients"][part], stride=1)
        prediction = model.predict(data["x"])
        # Check identical evaluated origin support, including gap filtering, against
        # the existing dense reference. No reference predictions/labels enter fitting.
        ref_path = REFERENCE / "private/caches" / f"patchmlp_glimmer_{cohort}_p12" / part / "dense_min_predictions.npz"
        with np.load(ref_path) as cache:
            ref = {k: cache[k] for k in ("pid", "sid", "idx")}
        changes = np.r_[0, np.flatnonzero((np.diff(ref["pid"]) != 0) |
                                         (np.diff(ref["sid"]) != 0))+1, len(ref["idx"])]
        lookup = {(int(ref["pid"][lo]), int(ref["sid"][lo])): (lo, hi)
                  for lo, hi in zip(changes[:-1], changes[1:])}
        valid_ref = []
        for (pid, sid), group in raw[raw.pat_id.isin(spec["patients"][part])].dropna(subset=["cgm"]).groupby(["pat_id", "seq_id"], sort=False):
            if (int(pid), int(sid)) not in lookup:
                continue
            slots = pd.to_datetime(group.date).to_numpy(dtype="datetime64[ns]").astype(np.int64) // (300*10**9)
            lo, hi = lookup[int(pid), int(sid)]
            ref_idx = ref["idx"][lo:hi]
            keep = ((slots[ref_idx] - slots[ref_idx-287] == 287) &
                    (slots[ref_idx+12] - slots[ref_idx] == 12))
            valid_ref.extend((int(pid), int(sid), int(i)) for i in ref_idx[keep])
        if not np.array_equal(np.asarray(valid_ref), np.column_stack((data["pid"], data["sid"], data["idx"]))):
            raise RuntimeError("Baseline evaluated origin grid differs from existing models.")
        directory = private / part
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "dense_trajectory_predictions.npz", **data, prediction=prediction)
        np.savez_compressed(directory / "dense_min_predictions.npz",
            **{k: data[k] for k in ("pid", "sid", "idx", "slot")}, min_pred=prediction.min(axis=1))
        write_json(directory / "provenance.json", dict(model="iob_linear", cohort=cohort,
            split_part=part, split_patients=spec["patients"][part], source_sha256=sha(source),
            checkpoint_sha256=sha(private/"model.json"), config_sha256=sha(private/"config.json"),
            common_evaluated_grid_verified=True, n_origins=len(data["x"])))
        if part == "test":
            for steps in (6, 12):
                for metric, pred, target in (
                    ("trajectory_rmse", prediction[:, :steps], data["y"][:, :steps]),
                    ("endpoint_rmse", prediction[:, steps-1:steps], data["y"][:, steps-1:steps])):
                    forecast_rows.append(dict(model="iob_linear", cohort=cohort, horizon_min=steps*5,
                                               metric=metric, **ci(patient_rmse(pred, target, data["pid"]))))
        print(f"{cohort}/{part}: {len(data['x'])} glucose trajectories", flush=True)
        del data, prediction
    write_json(out / "metrics.json", forecast_rows)
    write_json(out / "completion.json", dict(model="iob_linear", cohort=cohort,
        elapsed_seconds=time.monotonic()-start, selected_alpha=model.alpha,
        forecast_trajectory_not_event_classifier=True, common_evaluated_grid_verified=True))
    print(f"DONE fit/inference {cohort}: {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("dclp3", "dclp5", "pedap", "iobp2"), required=True)
    run(parser.parse_args().cohort)
