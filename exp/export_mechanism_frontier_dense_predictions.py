from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import make_dataloader_from_cfg  # noqa: E402
from exp.run_experiment import _sanitize_state_dict, _unwrap_model, to_jsonable  # noqa: E402
from exp.run_surrogate_mse_pair_generation import DEFAULT_SPLITS, HORIZON_MINUTES, _export_split  # noqa: E402
from models import build_model_from_cfg  # noqa: E402


def _load_cfg(run_dir: Path) -> dict[str, Any]:
    for name in ("resolved_config.yaml", "config.yaml", "input_config.yaml"):
        path = run_dir / name
        if path.exists():
            cfg = yaml.safe_load(path.read_text())
            if isinstance(cfg, dict):
                return cfg
    raise FileNotFoundError(f"No config YAML found in {run_dir}")


def _load_model(run_dir: Path, cfg: dict[str, Any]) -> torch.nn.Module:
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = dict(cfg)
    dense_cfg = dict(cfg)
    dense_cfg["model_data"] = dict(cfg["model_data"])
    dense_cfg["model_data"]["dataloader"] = dict(cfg["model_data"]["dataloader"])
    dense_cfg["model_data"]["dataloader"]["stride"] = 1
    dl = make_dataloader_from_cfg(dense_cfg["split"], {**dense_cfg["model_data"]["dataloader"], "shuffle": False}, split_part="test", seed=int(dense_cfg.get("model_data", {}).get("seed", 42)))
    input_dim = len(dl.dataset.feature_cols)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("model_cfg") or cfg["model_data"]["model"]
    model = build_model_from_cfg(model_cfg, input_dim=input_dim, history_len=int(cfg["model_data"]["dataloader"]["history_len"]), pred_len=int(cfg["model_data"]["dataloader"]["pred_len"]))
    _unwrap_model(model).load_state_dict(_sanitize_state_dict(ckpt["state_dict"]))
    return model


def _infer_meta(run_dir: Path, cfg: dict[str, Any]) -> tuple[str, str, str]:
    meta = dict(cfg.get("mechanism_frontier") or cfg.get("surrogate") or {})
    dataset = str(meta.get("dataset") or cfg["split"]["split_name"].split("_patients_")[0]).upper()
    model = str(meta.get("model_architecture") or cfg["model_data"]["model"]["name"]).lower()
    loss = str(meta.get("loss_key") or cfg["training"].get("loss") or run_dir.parts[-4]).lower()
    return dataset, model, loss


def run(run_dir: Path, output_root: Path, *, run_id: str | None, overwrite: bool, splits: list[str]) -> Path:
    cfg = _load_cfg(run_dir)
    dense_cfg = json.loads(json.dumps(to_jsonable(cfg)))
    dense_cfg["model_data"]["dataloader"]["stride"] = 1
    dense_cfg["model_data"]["dataloader"]["shuffle"] = False
    dense_cfg["model_data"]["dataloader"]["num_workers"] = int(dense_cfg["model_data"]["dataloader"].get("num_workers", 0))
    dataset, model_arch, loss = _infer_meta(run_dir, dense_cfg)
    dense_id = run_id or "/".join([loss, f"{model_arch}_{dataset.lower()}", f"seed_{int(dense_cfg.get('model_data', {}).get('seed', 42))}"])
    out_dir = output_root / dense_id
    if out_dir.exists() and (out_dir / "prediction_pairs_summary.json").exists() and not overwrite:
        raise FileExistsError(f"Completed dense export exists: {out_dir}. Use --overwrite to replace it.")
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    logging.info("Dense stride-1 export from %s to %s", run_dir, out_dir)
    model = _load_model(run_dir, dense_cfg)

    with (out_dir / "resolved_config.yaml").open("w") as f:
        yaml.safe_dump(to_jsonable(dense_cfg), f, sort_keys=False)
    (out_dir / "source_run_dir.txt").write_text(str(run_dir) + "\n")

    seed = int(dense_cfg.get("model_data", {}).get("seed", 42))
    metrics = {}
    for split in splits:
        metrics[split] = _export_split(
            model=model,
            cfg=dense_cfg,
            split_part=split,
            run_dir=out_dir,
            dataset=dataset,
            model_architecture=model_arch,
            seed=seed,
        )
    pred = int(dense_cfg["model_data"]["dataloader"]["pred_len"])
    summary = {
        "dataset": dataset,
        "model_architecture": model_arch,
        "loss_key": loss,
        "source_run_dir": str(run_dir),
        "history_len": int(dense_cfg["model_data"]["dataloader"]["history_len"]),
        "pred_len": pred,
        "stride": 1,
        "dense_decision_export": True,
        "horizon_minutes": (np.arange(1, pred + 1) * HORIZON_MINUTES).tolist(),
        "value_scale": "mg/dL",
        "metrics": metrics,
    }
    (out_dir / "prediction_pairs_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stride-1 dense prediction artifacts from mechanism-frontier trained checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/dense_predictions"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--splits", default="test", help="Comma-separated splits to export; default test.")
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    invalid = set(splits) - set(DEFAULT_SPLITS)
    if invalid:
        raise ValueError(f"Invalid splits: {sorted(invalid)}")
    print(run(args.run_dir, args.output_root, run_id=args.run_id, overwrite=args.overwrite, splits=splits))


if __name__ == "__main__":
    main()
