#!/usr/bin/env python
"""
Generate a synthetic timeseries dataset using MixtureDataset and save it in the
same format as datasets_processed/<name>/timeseries.csv + metadata.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict
import types
import sys

import numpy as np
import pandas as pd


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _install_src_aliases(syn_root: Path) -> None:
    # Provide minimal src.* aliases expected by the third-party code.
    if "src" not in sys.modules:
        sys.modules["src"] = types.ModuleType("src")
    if "src.data" not in sys.modules:
        sys.modules["src.data"] = types.ModuleType("src.data")
    if "src.ergodic" not in sys.modules:
        sys.modules["src.ergodic"] = types.ModuleType("src.ergodic")

    dataset_mod = _load_module("src.data.dataset", syn_root / "data" / "dataset.py")
    base_mod = _load_module("src.ergodic.base", syn_root / "ergodic" / "base.py")
    # Ensure parent modules expose children
    sys.modules["src"].data = sys.modules["src.data"]  # type: ignore[attr-defined]
    sys.modules["src"].ergodic = sys.modules["src.ergodic"]  # type: ignore[attr-defined]
    sys.modules["src.data"].dataset = dataset_mod  # type: ignore[attr-defined]
    sys.modules["src.ergodic"].base = base_mod  # type: ignore[attr-defined]


def generate_synthetic_dataset(
    name: str,
    n_components: int = 3,
    component_model: str = "ar",
    component_kwargs: Dict[str, Any] | None = None,
    low: float = 0.0,
    high: float = 1.0,
    n_features: int = 3,
    batch_size: int = 1,
    n_samples: int = 50,
    time_length: int = 288,
    random_seed: int = 42,
    output_root: str = "datasets_processed",
    start_date: str = "2020-01-01",
    freq: str = "5min",
) -> Path:
    syn_root = Path(__file__).resolve().parents[1] / "preprocessing" / "synthetic"
    _install_src_aliases(syn_root)

    mixture_mod = _load_module("synthetic_mixture", syn_root / "data" / "mixture.py")
    ar_mod = _load_module("synthetic_ar", syn_root / "ergodic" / "ar.py")
    ou_mod = _load_module("synthetic_ou", syn_root / "ergodic" / "ornstein.py")
    arma_mod = _load_module("synthetic_arma", syn_root / "ergodic" / "arma.py")

    if component_kwargs is None:
        component_kwargs = {}
    # Defaults for AR model
    if component_model.lower() in {"ar", "autoregressive", "autoregressivemodel"}:
        component_cls = ar_mod.AutoregressiveModel
        component_kwargs = dict(component_kwargs)
        component_kwargs.setdefault("n_features", int(n_features))
        component_kwargs.setdefault("order", 2)
    elif component_model.lower() in {"ornstein", "ornstein_uhlenbeck", "ou"}:
        component_cls = ou_mod.OrnsteinUhlenbeck
        component_kwargs = dict(component_kwargs)
        component_kwargs.setdefault("n_features", int(n_features))
    elif component_model.lower() in {"arma"}:
        component_cls = arma_mod.ARMA
        component_kwargs = dict(component_kwargs)
        component_kwargs.setdefault("n_features", int(n_features))
        component_kwargs.setdefault("time_length", int(time_length))
        component_kwargs.setdefault("p", 2)
        component_kwargs.setdefault("q", 2)
    else:
        raise ValueError(f"Unsupported component_model: {component_model}")

    ds = mixture_mod.MixtureDataset(
        n_components=int(n_components),
        component_model=component_cls,
        component_kwargs=component_kwargs,
        low=float(low),
        high=float(high),
        n_features=int(n_features),
        batch_size=int(batch_size),
        n_samples=int(n_samples),
        time_length=int(time_length),
        random_seed=int(random_seed),
    )

    features = ds.data["features"]  # [N, T, F]
    targets = ds.data["targets"]    # [N, T'] or [N, T', 1]

    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets = targets[..., 0]
    if targets.ndim != 2:
        raise ValueError(f"Unexpected targets shape {targets.shape}")

    if features.shape[1] != targets.shape[1]:
        # Align features to target length (drop earliest steps)
        features = features[:, -targets.shape[1] :, :]

    n_samples, tlen = targets.shape

    dates = pd.date_range(start=start_date, periods=tlen, freq=freq)
    pid_col = np.repeat(np.arange(n_samples), tlen)
    pat_id_col = pid_col.copy()
    seq_id_col = pid_col.copy()
    date_col = np.tile(dates.to_numpy(), n_samples)
    cgm_col = targets.reshape(-1)
    feat_cols = features.reshape(-1, features.shape[-1])

    data = {
        "pid": pid_col,
        "pat_id": pat_id_col,
        "seq_id": seq_id_col,
        "date": date_col,
        "cgm": cgm_col,
    }
    for i in range(features.shape[-1]):
        data[f"c{i+1}"] = feat_cols[:, i]

    df = pd.DataFrame(data)

    out_dir = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "timeseries.csv"
    df.to_csv(csv_path, index=False)

    channels = ["cgm"] + [f"c{i+1}" for i in range(features.shape[-1])]
    meta = {
        "dataset": name,
        "source_files": [str(csv_path)],
        "rows_final": int(len(df)),
        "patients_final": int(n_samples),
        "channels": channels,
        "notes": "Synthetic mixture dataset generated via preprocessing/synthetic.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Dataset name to save under (datasets_processed/<name>)")
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--component-model", default="ar")
    p.add_argument("--component-kwargs", default="{}", help="JSON dict of kwargs for component model")
    p.add_argument("--low", type=float, default=0.0)
    p.add_argument("--high", type=float, default=1.0)
    p.add_argument("--n-features", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--time-length", type=int, default=288)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-root", default="datasets_processed")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--freq", default="5min")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        component_kwargs = json.loads(args.component_kwargs)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --component-kwargs: {e}") from e

    out_dir = generate_synthetic_dataset(
        name=args.name,
        n_components=args.n_components,
        component_model=args.component_model,
        component_kwargs=component_kwargs,
        low=args.low,
        high=args.high,
        n_features=args.n_features,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        time_length=args.time_length,
        random_seed=args.random_seed,
        output_root=args.output_root,
        start_date=args.start_date,
        freq=args.freq,
    )
    print(f"[✓] Wrote synthetic dataset to {out_dir}")


if __name__ == "__main__":
    main()
