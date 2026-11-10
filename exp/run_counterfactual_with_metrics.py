from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


# Ensure local imports work when run from repo root
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


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


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


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
    logging.info("Launching counterfactual experiment: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _infer_test_scenarios(eval_cfg: Dict[str, Any]) -> Path:
    split = dict(eval_cfg.get("split") or {})
    processed_root = Path(split.get("processed_root", "datasets_processed"))
    split_name = split.get("split_name")
    if not split_name:
        raise ValueError("eval_config.split.split_name is required")
    path = processed_root / str(split_name) / "test_scenarios.csv"
    return path


def _run_metrics(
    *,
    run_dir: Path,
    test_scenarios: Path,
    horizons: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> None:
    artifacts = run_dir / "cf_artifacts.npz"
    if not artifacts.exists():
        raise FileNotFoundError(f"Counterfactual artifacts not found: {artifacts}")
    python_exec = _resolve_python_executable()
    cmd = [
        python_exec,
        "analysis/counterfactual_metrics.py",
        "--artifacts",
        str(artifacts),
        "--test-scenarios",
        str(test_scenarios),
        "--out-json",
        str(run_dir / "cf_metrics.json"),
        "--out-csv",
        str(run_dir / "cf_metrics.csv"),
        "--out-patient-json",
        str(run_dir / "cf_metrics_patient.json"),
        "--out-patient-csv",
        str(run_dir / "cf_metrics_patient.csv"),
        "--horizons",
        str(horizons),
        "--n-bootstrap",
        str(int(n_bootstrap)),
        "--bootstrap-seed",
        str(int(bootstrap_seed)),
    ]
    logging.info("Launching counterfactual metrics: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run counterfactual train/test and compute counterfactual metrics.")
    p.add_argument("--train", action="store_true", help="Run training before evaluation.")
    p.add_argument("--test", action="store_true", help="Run counterfactual evaluation.")
    p.add_argument("--train-config", type=Path, required=True)
    p.add_argument("--eval-config", type=Path, required=True)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--output-root", type=Path, default=Path("outputs/results"))
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--test-scenarios", type=Path, default=None)
    p.add_argument("--horizons", type=str, default="30,60,90,120")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_cfg = _load_yaml(args.train_config)
    eval_cfg = _load_yaml(args.eval_config)
    test_scenarios = args.test_scenarios or _infer_test_scenarios(eval_cfg)

    if args.train or args.test or not (run_dir / "cf_artifacts.npz").exists():
        _run_base_counterfactual(
            train=args.train,
            test=args.test,
            train_config=args.train_config,
            eval_config=args.eval_config,
            output_root=args.output_root,
            run_id=args.run_id,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        logging.info("Reusing existing counterfactual artifacts in %s", run_dir)

    _run_metrics(
        run_dir=run_dir,
        test_scenarios=test_scenarios,
        horizons=str(args.horizons),
        n_bootstrap=int(args.n_bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    summary = {
        "run_id": args.run_id,
        "train_config": str(args.train_config),
        "eval_config": str(args.eval_config),
        "output_root": str(args.output_root),
        "test_scenarios": str(test_scenarios),
        "horizons": str(args.horizons),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (run_dir / "cf_metrics_run.json").write_text(json.dumps(summary, indent=2))
    logging.info("Counterfactual workflow completed: %s", run_dir)


if __name__ == "__main__":
    main()
