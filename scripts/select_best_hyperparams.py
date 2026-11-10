#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


def _load_val_rmse(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("val_rmse")


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _model_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return cfg["model_data"]["model"]["params"] or {}
    except (KeyError, TypeError):
        return {}


def _parse_exclusions(items: List[str]) -> List[Tuple[str, str]]:
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"--exclude-param expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out.append((k.strip(), v.strip().lower()))
    return out


def _is_excluded(cfg: Dict[str, Any], exclusions: List[Tuple[str, str]]) -> bool:
    params = _model_params(cfg)
    for key, val in exclusions:
        if key in params and str(params[key]).strip().lower() == val:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Select best hyperparameter configs by val RMSE.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/hyperparameters"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--out-dir", type=str, default="hyperparameter_best")
    parser.add_argument(
        "--exclude-param",
        action="append",
        default=None,
        help="KEY=VALUE in model.params; configs matching are ineligible for selection "
             "but are still scored and reported as an ablation. Repeatable. "
             "Pass --exclude-param '' to disable the default.",
    )
    args = parser.parse_args()

    raw_excl = args.exclude_param
    if raw_excl is None:
        raw_excl = ["normalize_insulin_by_weight=true"]
    raw_excl = [e for e in raw_excl if e.strip()]
    exclusions = _parse_exclusions(raw_excl)
    if exclusions:
        print("selection exclusions (scored but not selectable):")
        for k, v in exclusions:
            print(f"  {k} == {v}")
    else:
        print("selection exclusions: none")

    out_root = args.outputs_root
    if not out_root.exists():
        raise FileNotFoundError(out_root)

    dest_root = args.configs_root / args.out_dir
    dest_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for pair_dir in sorted(out_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        best_rmse = None
        best_cfg = None
        best_id = None
        excl_rmse = None
        excl_id = None
        n_ok = 0
        n_excl = 0
        for run_dir in sorted(pair_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            rmse = _load_val_rmse(run_dir / "val_rmse.json")
            if rmse is None:
                continue
            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                continue
            cfg = _load_config(cfg_path)
            n_ok += 1
            if _is_excluded(cfg, exclusions):
                n_excl += 1
                if excl_rmse is None or rmse < excl_rmse:
                    excl_rmse = rmse
                    excl_id = run_dir.name
                continue
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_cfg = cfg
                best_id = run_dir.name
        if best_cfg is None:
            if n_ok:
                print(f"  {pair_dir.name}: NO SELECTABLE CONFIG "
                      f"({n_ok} scored, {n_excl} excluded) -- skipped")
            continue
        out_path = dest_root / f"{pair_dir.name}.yaml"
        out_path.write_text(yaml.safe_dump(best_cfg))
        entry = {"val_rmse": best_rmse, "config": out_path.name, "run_id": best_id,
                 "n_scored": n_ok, "n_excluded": n_excl}
        if excl_rmse is not None:
            entry["excluded_best_val_rmse"] = excl_rmse
            entry["excluded_best_run_id"] = excl_id
            entry["excluded_minus_selected"] = excl_rmse - best_rmse
        summary[pair_dir.name] = entry
        msg = (f"  {pair_dir.name}: selected {best_id} val_rmse={best_rmse:.4f} "
               f"({n_ok - n_excl} eligible / {n_ok} scored)")
        if excl_rmse is not None:
            delta = excl_rmse - best_rmse
            sign = "worse" if delta > 0 else "better"
            msg += f" | best excluded {excl_id} val_rmse={excl_rmse:.4f} ({abs(delta):.4f} {sign})"
        print(msg)

    (dest_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote best configs to {dest_root} and summary.json")


if __name__ == "__main__":
    main()
