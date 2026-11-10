from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HP_ROOT = REPO_ROOT / "outputs" / "hyperparameters"
OUT_ROOT = REPO_ROOT / "configs" / "counterfactual_best"

ARCHES = ["arimax", "attengluco", "dlinear", "gru", "itransformer", "patchmlp"]
CF_DATASET = "action_cf_benchmark"
METRICS = ["rmse", "mae", "seg", "gmse", "grmse", "ceg_ratio", "peg_ratio", "seg_z_ratio"]
EVENT_THRESHOLDS = {"post_meal": 3, "post_bolus": 3, "nocturnal": 0}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _best_run_dir(arch: str, *, hp_root: Path, dataset: str) -> tuple[Path, float]:
    hp_dir = hp_root / f"{arch}_{dataset}"
    if not hp_dir.exists():
        raise FileNotFoundError(hp_dir)
    best_dir: Path | None = None
    best_val = float("inf")
    for run_dir in sorted(hp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        val_path = run_dir / "val_rmse.json"
        cfg_path = run_dir / "config.yaml"
        if not val_path.exists() or not cfg_path.exists():
            continue
        try:
            val = float(_load_json(val_path)["val_rmse"])
        except Exception:
            continue
        if val < best_val:
            best_val = val
            best_dir = run_dir
    if best_dir is None:
        raise FileNotFoundError(f"No valid hyperparameter runs found for {arch} in {hp_dir}")
    return best_dir, best_val


def _normalize_train_cfg(
    cfg: dict[str, Any],
    arch: str,
    best_run: Path,
    val_rmse: float,
    *,
    dataset: str,
) -> dict[str, Any]:
    out = dict(cfg)
    out["notes"] = (
        f"Counterfactual best config: {arch} on {dataset}; "
        f"selected from {best_run.name} with val_rmse={val_rmse:.6f}"
    )
    return out


def _build_eval_cfg(
    train_cfg: dict[str, Any],
    arch: str,
    best_run: Path,
    val_rmse: float,
    *,
    dataset: str,
) -> dict[str, Any]:
    model_data = dict(train_cfg.get("model_data") or {})
    dl_cfg = dict(model_data.get("dataloader") or {})
    dl_cfg["shuffle"] = False
    model_data["dataloader"] = dl_cfg
    return {
        "split": {
            "processed_root": "datasets_processed",
            "split_name": dataset,
            "split_part": "test",
        },
        "model_data": model_data,
        "evaluation": {
            "metrics": METRICS,
            "event_thresholds": EVENT_THRESHOLDS,
            "compile": False,
        },
        "notes": (
            f"Counterfactual eval config for {arch} on {dataset}; "
            f"selected from {best_run.name} with val_rmse={val_rmse:.6f}"
        ),
    }


def _build_glimmer_train_cfg(
    patchmlp_cfg: dict[str, Any],
    patchmlp_best_run: Path,
    patchmlp_val_rmse: float,
    *,
    dataset: str,
    glimmer_reference_config: Path,
) -> dict[str, Any]:
    out = json.loads(json.dumps(patchmlp_cfg))
    out["model_data"]["model"]["name"] = "patchmlp_glimmer"
    glimmer_ref = _load_yaml(glimmer_reference_config)
    glimmer_training = dict(glimmer_ref.get("training") or {})
    out["training"] = glimmer_training
    out["notes"] = (
        f"Counterfactual best config: patchmlp_glimmer on {dataset}; "
        f"using best PatchMLP architecture from {patchmlp_best_run.name} with val_rmse={patchmlp_val_rmse:.6f} "
        "and default GLIMMER GA settings from existing repo config"
    )
    return out


def _build_glimmer_eval_cfg(
    glimmer_train_cfg: dict[str, Any],
    patchmlp_best_run: Path,
    patchmlp_val_rmse: float,
    *,
    dataset: str,
) -> dict[str, Any]:
    model_data = dict(glimmer_train_cfg.get("model_data") or {})
    dl_cfg = dict(model_data.get("dataloader") or {})
    dl_cfg["shuffle"] = False
    model_data["dataloader"] = dl_cfg
    return {
        "split": {
            "processed_root": "datasets_processed",
            "split_name": dataset,
            "split_part": "test",
        },
        "model_data": model_data,
        "evaluation": {
            "metrics": METRICS,
            "event_thresholds": EVENT_THRESHOLDS,
            "compile": False,
        },
        "notes": (
            f"Counterfactual eval config for patchmlp_glimmer on {dataset}; "
            f"using best PatchMLP architecture from {patchmlp_best_run.name} with val_rmse={patchmlp_val_rmse:.6f}"
        ),
    }


def _write_yaml(path: Path, data: dict[str, Any], *, fail_if_exists: bool) -> None:
    if fail_if_exists and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {path}")
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best counterfactual configs from hyperparameter runs.")
    parser.add_argument("--dataset", type=str, default=CF_DATASET)
    parser.add_argument("--outputs-root", type=Path, default=HP_ROOT)
    parser.add_argument("--configs-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--glimmer-reference-config",
        type=Path,
        default=REPO_ROOT / "configs" / "policy_regret_best" / "train" / "patchmlp_glimmer_action_cf_benchmark_bolus.yaml",
        help="Existing config whose GLIMMER training block is reused.",
    )
    parser.add_argument("--fail-if-exists", action="store_true")
    args = parser.parse_args()

    train_out = args.configs_root / "train"
    eval_out = args.configs_root / "eval"
    train_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"dataset": args.dataset, "selected": {}}

    patchmlp_best_dir: Path | None = None
    patchmlp_best_val: float | None = None
    patchmlp_best_cfg: dict[str, Any] | None = None

    for arch in ARCHES:
        best_dir, best_val = _best_run_dir(arch, hp_root=args.outputs_root, dataset=args.dataset)
        train_cfg = _normalize_train_cfg(
            _load_yaml(best_dir / "config.yaml"),
            arch,
            best_dir,
            best_val,
            dataset=args.dataset,
        )
        eval_cfg = _build_eval_cfg(train_cfg, arch, best_dir, best_val, dataset=args.dataset)

        train_path = train_out / f"{arch}_{args.dataset}.yaml"
        eval_path = eval_out / f"{arch}_{args.dataset}.yaml"
        _write_yaml(train_path, train_cfg, fail_if_exists=args.fail_if_exists)
        _write_yaml(eval_path, eval_cfg, fail_if_exists=args.fail_if_exists)
        summary["selected"][arch] = {
            "best_run": _display_path(best_dir),
            "val_rmse": float(best_val),
            "train_config": _display_path(train_path),
            "eval_config": _display_path(eval_path),
        }

        if arch == "patchmlp":
            patchmlp_best_dir = best_dir
            patchmlp_best_val = best_val
            patchmlp_best_cfg = train_cfg

    if patchmlp_best_cfg is None or patchmlp_best_dir is None or patchmlp_best_val is None:
        raise RuntimeError("PatchMLP best config is required to create patchmlp_glimmer config.")

    glimmer_train = _build_glimmer_train_cfg(
        patchmlp_best_cfg,
        patchmlp_best_dir,
        patchmlp_best_val,
        dataset=args.dataset,
        glimmer_reference_config=args.glimmer_reference_config,
    )
    glimmer_eval = _build_glimmer_eval_cfg(
        glimmer_train,
        patchmlp_best_dir,
        patchmlp_best_val,
        dataset=args.dataset,
    )
    glimmer_train_path = train_out / f"patchmlp_glimmer_{args.dataset}.yaml"
    glimmer_eval_path = eval_out / f"patchmlp_glimmer_{args.dataset}.yaml"
    _write_yaml(glimmer_train_path, glimmer_train, fail_if_exists=args.fail_if_exists)
    _write_yaml(glimmer_eval_path, glimmer_eval, fail_if_exists=args.fail_if_exists)
    summary["selected"]["patchmlp_glimmer"] = {
        "based_on_patchmlp_run": _display_path(patchmlp_best_dir),
        "patchmlp_val_rmse": float(patchmlp_best_val),
        "train_config": _display_path(glimmer_train_path),
        "eval_config": _display_path(glimmer_eval_path),
    }

    summary_path = args.configs_root / "summary.json"
    if args.fail_if_exists and summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote best counterfactual configs to {args.configs_root}")


if __name__ == "__main__":
    main()
