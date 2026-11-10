from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Ensure local imports work when run from repo root
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


def _infer_run_id(train_cfg: Dict[str, Any], eval_cfg: Dict[str, Any]) -> str:
    split_name = (
        train_cfg.get("split", {}).get("split_name")
        or eval_cfg.get("split", {}).get("split_name")
        or "action_cf_benchmark_bolus"
    )
    model_name = (
        train_cfg.get("model_data", {}).get("model", {}).get("name")
        or eval_cfg.get("model_data", {}).get("model", {}).get("name")
        or "model"
    )
    return f"{split_name}_{model_name}"


def _bgri_cost(traj: np.ndarray) -> np.ndarray:
    g = np.asarray(traj, dtype=np.float64)
    g = np.clip(g, 1.0, None)
    f = 1.509 * (np.power(np.log(g), 1.084) - 5.381)
    return np.mean(10.0 * np.square(f), axis=-1)


def _parse_action_scale(name: str) -> float | None:
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)x$", str(name))
    return None if m is None else float(m.group(1))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _run_base_counterfactual(
    *,
    train: bool,
    test: bool,
    train_config: Path,
    eval_config: Path,
    output_root: Path,
    run_id: str,
    checkpoint_dir: Path | None,
) -> None:
    python_exec = _resolve_python_executable()
    cmd = [
        python_exec,
        "exp/run_counterfactual_experiment.py",
        "--train-config",
        str(train_config),
        "--eval-config",
        str(eval_config),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
    ]
    if train:
        cmd.append("--train")
    if test:
        cmd.append("--test")
    if checkpoint_dir is not None:
        cmd.extend(["--checkpoint-dir", str(checkpoint_dir)])
    logging.info("Launching base counterfactual runner: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_python_executable() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    prefix_candidate = Path(sys.prefix) / "bin" / "python"
    if prefix_candidate.exists():
        return str(prefix_candidate)

    return sys.executable


def _normalize_event_thresholds(eval_cfg: Dict[str, Any]) -> Dict[str, int | None]:
    evaluation = dict((eval_cfg or {}).get("evaluation") or {})
    raw = dict(evaluation.get("event_thresholds") or {})
    defaults = {"post_meal": 3, "post_bolus": 3, "nocturnal": 0}
    out: Dict[str, int | None] = {}
    for key, default in defaults.items():
        val = raw.get(key, default)
        if val in (None, "null", "None", -1):
            out[key] = None
        else:
            out[key] = int(val)
    return out


def _in_slice(distance: int, threshold: int | None) -> bool:
    if threshold is None:
        return False
    return int(distance) >= 0 and int(distance) <= int(threshold)


def _build_episode_slice_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    slice_masks = {
        "overall": np.ones(len(episodes), dtype=bool),
        "nocturnal": episodes["is_nocturnal"].to_numpy(dtype=bool),
        "post_bolus": episodes["is_post_bolus"].to_numpy(dtype=bool),
        "post_meal": episodes["is_post_meal"].to_numpy(dtype=bool),
    }
    for slice_name, mask in slice_masks.items():
        if not np.any(mask):
            continue
        subset = episodes.loc[mask].copy()
        subset.insert(0, "slice", slice_name)
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=["slice", *episodes.columns.tolist()])
    return pd.concat(rows, ignore_index=True)


def _aggregate_patient_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "pat_id",
                "n_episodes",
                "mean_regret",
                "median_regret",
                "mean_oracle_actual_bgri",
                "mean_selected_actual_bgri",
                "mean_selected_pred_bgri",
                "action_match_rate",
                "mean_selected_actual_rank",
                "mean_abs_scale_error",
                "mean_oracle_scale",
                "mean_selected_scale",
            ]
        )
    return (
        episodes.groupby("pat_id", sort=True)
        .agg(
            n_episodes=("seq_id", "count"),
            mean_regret=("regret", "mean"),
            median_regret=("regret", "median"),
            mean_oracle_actual_bgri=("oracle_actual_bgri", "mean"),
            mean_selected_actual_bgri=("selected_actual_bgri", "mean"),
            mean_selected_pred_bgri=("selected_pred_bgri", "mean"),
            action_match_rate=("action_match", "mean"),
            mean_selected_actual_rank=("selected_actual_rank", "mean"),
            mean_abs_scale_error=("abs_scale_error", "mean"),
            mean_oracle_scale=("oracle_scale", "mean"),
            mean_selected_scale=("selected_scale", "mean"),
        )
        .reset_index()
    )


def _build_policy_regret_tables(
    cf_npz: Path,
    *,
    perturb_family: str = "bolus",
    scenario_include: str | None = None,
    event_npz: Path | None = None,
    event_thresholds: Dict[str, int | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = np.load(cf_npz, allow_pickle=True)
    if event_thresholds is None:
        event_thresholds = {"post_meal": 3, "post_bolus": 3, "nocturnal": 0}
    df = pd.DataFrame(
        {
            "pat_id": data["pat_id"].astype(int),
            "seq_id": data["seq_id"].astype(int),
            "scenario_name": data["scenario_name"].astype(object),
            "scenario_type": data["scenario_type"].astype(object),
            "perturb_family": data["perturb_family"].astype(object),
        }
    )
    preds = np.asarray(data["preds"], dtype=np.float64)
    gts = np.asarray(data["gts"], dtype=np.float64)
    df["pred_bgri"] = _bgri_cost(preds)
    df["actual_bgri"] = _bgri_cost(gts)
    df["action_scale"] = [_parse_action_scale(x) for x in df["scenario_name"]]
    if event_npz is not None and event_npz.exists():
        event_data = np.load(event_npz, allow_pickle=True)
        df["post_meal_dist"] = np.asarray(event_data["post_meal"], dtype=np.int64)
        df["post_bolus_dist"] = np.asarray(event_data["post_bolus"], dtype=np.int64)
        df["nocturnal_dist"] = np.asarray(event_data["nocturnal"], dtype=np.int64)
    else:
        df["post_meal_dist"] = -1
        df["post_bolus_dist"] = -1
        df["nocturnal_dist"] = -1

    mask = (df["perturb_family"] == perturb_family) & (df["scenario_type"] == "counterfactual")
    df = df.loc[mask].reset_index(drop=True)
    preds = preds[mask.to_numpy()]
    gts = gts[mask.to_numpy()]

    if scenario_include:
        import re as _re
        _pat = _re.compile(scenario_include)
        _keep = df["scenario_name"].astype(str).str.match(_pat)
        n_before = df["scenario_name"].nunique()
        df = df[_keep.to_numpy()]
        preds = preds[_keep.to_numpy()]
        gts = gts[_keep.to_numpy()]
        logging.info("scenario filter %r: %s -> %s distinct actions",
                     scenario_include, n_before, df["scenario_name"].nunique())
        if df.empty:
            raise ValueError(f"scenario filter {scenario_include!r} matched no rows")

    expected_actions = sorted(
        [x for x in df["scenario_name"].dropna().unique().tolist()],
        key=lambda x: (_parse_action_scale(x) is None, _parse_action_scale(x) or 0.0, str(x)),
    )
    expected_action_set = set(expected_actions)

    episode_rows: list[dict[str, Any]] = []
    skipped = 0
    for (pat_id, seq_id), g in df.groupby(["pat_id", "seq_id"], sort=True):
        present = set(g["scenario_name"].tolist())
        if present != expected_action_set:
            skipped += 1
            logging.warning(
                "Skipping patient=%s seq=%s because actions are incomplete (have %s, expected %s)",
                pat_id,
                seq_id,
                len(present),
                len(expected_action_set),
            )
            continue

        g = g.reset_index(drop=True)
        oracle_order = g.sort_values(["actual_bgri", "action_scale", "scenario_name"]).reset_index(drop=True)
        pred_order = g.sort_values(["pred_bgri", "action_scale", "scenario_name"]).reset_index(drop=True)
        oracle = oracle_order.iloc[0]
        selected = pred_order.iloc[0]
        post_meal_vals = g["post_meal_dist"].to_numpy(dtype=np.int64)
        post_bolus_vals = g["post_bolus_dist"].to_numpy(dtype=np.int64)
        nocturnal_vals = g["nocturnal_dist"].to_numpy(dtype=np.int64)
        if not (
            np.all(post_meal_vals == post_meal_vals[0])
            and np.all(post_bolus_vals == post_bolus_vals[0])
            and np.all(nocturnal_vals == nocturnal_vals[0])
        ):
            logging.warning(
                "Episode patient=%s seq=%s has inconsistent slice/event labels across actions; using first values.",
                pat_id,
                seq_id,
            )
        post_meal_dist = int(post_meal_vals[0])
        post_bolus_dist = int(post_bolus_vals[0])
        nocturnal_dist = int(nocturnal_vals[0])

        actual_order = oracle_order
        actual_rank = int(actual_order.index[actual_order["scenario_name"] == selected["scenario_name"]][0]) + 1

        episode_rows.append(
            {
                "pat_id": int(pat_id),
                "seq_id": int(seq_id),
                "n_actions": int(len(g)),
                "oracle_scenario": str(oracle["scenario_name"]),
                "selected_scenario": str(selected["scenario_name"]),
                "oracle_scale": float(oracle["action_scale"]),
                "selected_scale": float(selected["action_scale"]),
                "oracle_actual_bgri": float(oracle["actual_bgri"]),
                "selected_actual_bgri": float(selected["actual_bgri"]),
                "selected_pred_bgri": float(selected["pred_bgri"]),
                "oracle_pred_bgri": float(oracle["pred_bgri"]),
                "regret": float(selected["actual_bgri"] - oracle["actual_bgri"]),
                "action_match": float(str(selected["scenario_name"]) == str(oracle["scenario_name"])),
                "selected_actual_rank": actual_rank,
                "abs_scale_error": float(abs(float(selected["action_scale"]) - float(oracle["action_scale"]))),
                "post_meal_dist": post_meal_dist,
                "post_bolus_dist": post_bolus_dist,
                "nocturnal_dist": nocturnal_dist,
                "is_post_meal": _in_slice(post_meal_dist, event_thresholds.get("post_meal")),
                "is_post_bolus": _in_slice(post_bolus_dist, event_thresholds.get("post_bolus")),
                "is_nocturnal": _in_slice(nocturnal_dist, event_thresholds.get("nocturnal")),
            }
        )

    if not episode_rows:
        raise ValueError(f"No complete {perturb_family} counterfactual episodes were found for policy regret.")

    episodes = pd.DataFrame(episode_rows)
    patient = _aggregate_patient_metrics(episodes)
    episode_by_slice = _build_episode_slice_table(episodes)
    patient_by_slice_parts: list[pd.DataFrame] = []
    for slice_name, subset in episode_by_slice.groupby("slice", sort=False):
        patient_slice = _aggregate_patient_metrics(subset.drop(columns=["slice"]))
        if patient_slice.empty:
            continue
        patient_slice.insert(0, "slice", str(slice_name))
        patient_by_slice_parts.append(patient_slice)
    patient_by_slice = (
        pd.concat(patient_by_slice_parts, ignore_index=True)
        if patient_by_slice_parts
        else pd.DataFrame(columns=["slice", *patient.columns.tolist()])
    )

    summary = {
        "n_patients": int(patient["pat_id"].nunique()),
        "n_complete_episodes": int(len(episodes)),
        "n_skipped_incomplete_episodes": int(skipped),
        "expected_actions": expected_actions,
        "event_thresholds": dict(event_thresholds),
        "macro_avg_patient_metrics": {
            col: float(patient[col].mean()) for col in patient.columns if col != "pat_id"
        },
        "macro_avg_patient_metrics_by_slice": {
            slice_name: {
                col: float(sub[col].mean()) for col in sub.columns if col not in {"slice", "pat_id"}
            }
            for slice_name, sub in patient_by_slice.groupby("slice", sort=False)
        },
    }
    return episodes, patient, patient_by_slice, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run policy-regret evaluation on action_cf_benchmark_bolus.")
    parser.add_argument("--train", action="store_true", help="Train the base counterfactual forecaster before regret eval")
    parser.add_argument("--test", action="store_true", help="Run base counterfactual test inference before regret eval")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint dir to use for base test-only runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/results/policy_regret"))
    parser.add_argument("--force-base-run", action="store_true", help="Always rerun the base counterfactual experiment")
    parser.add_argument("--scenario-include", type=str, default=None,
                        help="Regex; keep only scenarios whose name matches. Use to score a "
                             "single ordered action menu, e.g. '^bolus_add_avg_[0-9.]+x$'.")
    parser.add_argument("--perturb-family", type=str, default="bolus",
                        help="Which perturbation family defines the action menu (default: bolus).")
    args = parser.parse_args()
    perturb_family = args.perturb_family

    train_cfg = yaml.safe_load(args.train_config.read_text()) if args.train_config.exists() else {}
    eval_cfg = yaml.safe_load(args.eval_config.read_text()) if args.eval_config.exists() else {}

    run_id = args.run_id or _infer_run_id(train_cfg or {}, eval_cfg or {})
    out_root = Path(args.output_root)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "policy_regret.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )
    logging.info("Policy-regret run ID: %s", run_id)

    cf_artifacts = run_dir / "cf_artifacts.npz"
    checkpoint_in_run = run_dir / "checkpoint.pt"

    do_train = bool(args.train)
    do_test = bool(args.test)
    if not do_train and not do_test:
        if cf_artifacts.exists():
            logging.info("Found existing cf_artifacts.npz; skipping base model run.")
        elif (args.checkpoint_dir is not None and (args.checkpoint_dir / "checkpoint.pt").exists()) or checkpoint_in_run.exists():
            do_test = True
        else:
            do_train = True
            do_test = True
    elif do_train and not do_test:
        logging.info("Policy regret requires test predictions; enabling --test as well.")
        do_test = True
    elif do_test and not do_train and not cf_artifacts.exists():
        if args.checkpoint_dir is None and not checkpoint_in_run.exists():
            logging.info("No checkpoint available for test-only mode; enabling training.")
            do_train = True

    if args.force_base_run or do_train or do_test or not cf_artifacts.exists():
        checkpoint_dir = args.checkpoint_dir
        if checkpoint_dir is None and checkpoint_in_run.exists() and not do_train:
            checkpoint_dir = run_dir
        _run_base_counterfactual(
            train=do_train,
            test=do_test or not cf_artifacts.exists(),
            train_config=args.train_config,
            eval_config=args.eval_config,
            output_root=out_root,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
        )

    if not cf_artifacts.exists():
        raise FileNotFoundError(f"Expected base artifacts at {cf_artifacts}")

    event_npz = run_dir / "event.npz"
    event_thresholds = _normalize_event_thresholds(eval_cfg or {})
    episodes, patient, patient_by_slice, summary = _build_policy_regret_tables(
        cf_artifacts,
        perturb_family=perturb_family,
        scenario_include=args.scenario_include,
        event_npz=event_npz if event_npz.exists() else None,
        event_thresholds=event_thresholds,
    )

    episodes_path = run_dir / "policy_regret_episode_metrics.csv"
    episodes_by_slice_path = run_dir / "policy_regret_episode_metrics_by_slice.csv"
    patient_path = run_dir / "policy_regret_patient_metrics.csv"
    patient_by_slice_path = run_dir / "policy_regret_patient_metrics_by_slice.csv"
    summary_path = run_dir / "policy_regret_summary.json"
    actions_npz_path = run_dir / "policy_regret_actions.npz"

    episodes.to_csv(episodes_path, index=False)
    _build_episode_slice_table(episodes).to_csv(episodes_by_slice_path, index=False)
    patient.to_csv(patient_path, index=False)
    patient_by_slice.to_csv(patient_by_slice_path, index=False)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2))

    cf = np.load(cf_artifacts, allow_pickle=True)
    df_actions = pd.DataFrame(
        {
            "pat_id": cf["pat_id"].astype(int),
            "seq_id": cf["seq_id"].astype(int),
            "scenario_name": cf["scenario_name"].astype(object),
            "scenario_type": cf["scenario_type"].astype(object),
            "perturb_family": cf["perturb_family"].astype(object),
        }
    )
    mask = (df_actions["perturb_family"] == perturb_family) & (df_actions["scenario_type"] == "counterfactual")
    np.savez(
        actions_npz_path,
        preds=np.asarray(cf["preds"])[mask.to_numpy()],
        gts=np.asarray(cf["gts"])[mask.to_numpy()],
        X=np.asarray(cf["X"])[mask.to_numpy()],
        pat_id=np.asarray(cf["pat_id"])[mask.to_numpy()],
        seq_id=np.asarray(cf["seq_id"])[mask.to_numpy()],
        scenario_name=np.asarray(cf["scenario_name"])[mask.to_numpy()],
        scenario_type=np.asarray(cf["scenario_type"])[mask.to_numpy()],
        perturb_family=np.asarray(cf["perturb_family"])[mask.to_numpy()],
        action_scale=np.asarray([_parse_action_scale(x) for x in np.asarray(cf["scenario_name"])[mask.to_numpy()]], dtype=np.float32),
        actual_bgri=_bgri_cost(np.asarray(cf["gts"])[mask.to_numpy()]),
        pred_bgri=_bgri_cost(np.asarray(cf["preds"])[mask.to_numpy()]),
    )

    logging.info("Saved policy-regret episode metrics to %s", episodes_path)
    logging.info("Saved policy-regret episode metrics by slice to %s", episodes_by_slice_path)
    logging.info("Saved policy-regret patient metrics to %s", patient_path)
    logging.info("Saved policy-regret patient metrics by slice to %s", patient_by_slice_path)
    logging.info("Saved policy-regret summary to %s", summary_path)
    logging.info(
        "Macro mean regret=%.6f over %s patients and %s complete episodes",
        summary["macro_avg_patient_metrics"]["mean_regret"],
        summary["n_patients"],
        summary["n_complete_episodes"],
    )


if __name__ == "__main__":
    main()
