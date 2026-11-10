from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from preprocessing.common import ensure_io


@dataclass(frozen=True)
class ActionCFConfig:
    prefix_filename: str = "prefix_factual.csv"
    suffix_prefix: str = "suffix_"
    manifest_filename: str = "manifest.csv"
    splits_filename: str = "splits.csv"


def _build_patient_dir_map(raw_dir: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for p in raw_dir.glob("patient_*"):
        if not p.is_dir():
            continue
        match = re.search(r"(\d+)$", p.name)
        if not match:
            continue
        pid = int(match.group(1))
        mapping[pid] = p
    return mapping


def _read_prefix_csv(path: Path, pat_id: int, seq_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = "time" if "time" in df.columns else "date"
    if time_col not in df.columns:
        raise KeyError(f"Expected a time/date column in {path}")

    required = {"cgm", "basal", "bolus", "meal"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")

    df = df.rename(columns={time_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cgm"] = pd.to_numeric(df["cgm"], errors="coerce")
    df["basal"] = pd.to_numeric(df["basal"], errors="coerce").fillna(0.0)
    df["bolus"] = pd.to_numeric(df["bolus"], errors="coerce").fillna(0.0)
    df["meal"] = pd.to_numeric(df["meal"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "cgm"]).copy()

    df["pat_id"] = int(pat_id)
    df["seq_id"] = int(seq_id)
    return df[["pat_id", "seq_id", "date", "cgm", "basal", "bolus", "meal"]]


def _read_suffix_csv(
    path: Path,
    pat_id: int,
    seq_id: int,
    episode_id: int,
    scenario_name: str,
    scenario_type: str,
    perturb_family: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = "time" if "time" in df.columns else "date"
    if time_col not in df.columns:
        raise KeyError(f"Expected a time/date column in {path}")

    required = {"cgm", "basal", "bolus", "meal"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")

    df = df.rename(columns={time_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cgm"] = pd.to_numeric(df["cgm"], errors="coerce")
    df["basal"] = pd.to_numeric(df["basal"], errors="coerce").fillna(0.0)
    df["bolus"] = pd.to_numeric(df["bolus"], errors="coerce").fillna(0.0)
    df["meal"] = pd.to_numeric(df["meal"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "cgm"]).copy()

    df["pat_id"] = int(pat_id)
    df["seq_id"] = int(seq_id)
    df["episode_id"] = int(episode_id)
    df["scenario_name"] = str(scenario_name)
    df["scenario_type"] = str(scenario_type)
    df["perturb_family"] = str(perturb_family)

    keep_cols = [
        "pat_id",
        "seq_id",
        "episode_id",
        "scenario_name",
        "scenario_type",
        "perturb_family",
        "date",
        "cgm",
        "basal",
        "bolus",
        "meal",
    ]
    for col in ["minutes_from_perturb_start", "in_perturb_window"]:
        if col in df.columns:
            keep_cols.append(col)
    return df[keep_cols]


def _collect_split(
    split_name: str,
    split_df: pd.DataFrame,
    patient_dirs: Dict[int, Path],
    cfg: ActionCFConfig,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    patients = split_df.loc[split_df["split"] == split_name, "pat_id"].tolist()
    if not patients:
        return pd.DataFrame(columns=["pat_id", "seq_id", "date", "cgm", "basal", "bolus", "meal"]), {
            "patients": 0,
            "sequences": 0,
            "rows": 0,
        }

    seq_id = 1
    parts: List[pd.DataFrame] = []
    missing: List[str] = []

    for pat_id in sorted(patients):
        pdir = patient_dirs.get(int(pat_id))
        if pdir is None:
            missing.append(f"patient_{pat_id}")
            continue

        episodes = sorted([d for d in pdir.glob("episode_*") if d.is_dir()])
        if not episodes:
            missing.append(str(pdir / "episode_*"))
            continue

        for ep in episodes:
            prefix = ep / cfg.prefix_filename
            if not prefix.exists():
                missing.append(str(prefix))
                continue
            parts.append(_read_prefix_csv(prefix, int(pat_id), seq_id))
            seq_id += 1

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing expected files/directories: {preview}{suffix}")

    if not parts:
        return pd.DataFrame(columns=["pat_id", "seq_id", "date", "cgm", "basal", "bolus", "meal"]), {
            "patients": len(patients),
            "sequences": 0,
            "rows": 0,
        }

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    stats = {
        "patients": len(set(int(p) for p in patients)),
        "sequences": int(df["seq_id"].nunique()),
        "rows": int(len(df)),
    }
    return df, stats


def _parse_episode_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    if not match:
        raise ValueError(f"Unexpected episode directory name: {path}")
    return int(match.group(1))


def _collect_test(
    split_df: pd.DataFrame,
    patient_dirs: Dict[int, Path],
    cfg: ActionCFConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    patients = split_df.loc[split_df["split"] == "test", "pat_id"].tolist()
    if not patients:
        empty = pd.DataFrame(columns=["pat_id", "seq_id", "date", "cgm", "basal", "bolus", "meal"])
        empty_suffix = pd.DataFrame(
            columns=[
                "pat_id",
                "seq_id",
                "episode_id",
                "scenario_name",
                "scenario_type",
                "perturb_family",
                "date",
                "cgm",
                "basal",
                "bolus",
                "meal",
            ]
        )
        empty_manifest = pd.DataFrame(
            columns=["pat_id", "seq_id", "episode_id", "scenario_name", "scenario_type", "perturb_family", "is_valid"]
        )
        return empty, empty_suffix, empty_manifest, {"patients": 0, "episodes": 0, "scenarios": 0, "rows_prefix": 0, "rows_suffix": 0}

    seq_id = 1
    prefix_parts: List[pd.DataFrame] = []
    suffix_parts: List[pd.DataFrame] = []
    scenario_rows: List[Dict[str, object]] = []
    missing: List[str] = []
    missing_counts = {
        "missing_patient_dir": 0,
        "missing_episode_dirs": 0,
        "missing_prefix": 0,
        "missing_manifest": 0,
        "missing_suffix": 0,
    }
    episodes = 0

    for pat_id in sorted(patients):
        pdir = patient_dirs.get(int(pat_id))
        if pdir is None:
            missing_counts["missing_patient_dir"] += 1
            missing.append(f"patient_{pat_id}")
            continue

        ep_dirs = sorted([d for d in pdir.glob("episode_*") if d.is_dir()])
        if not ep_dirs:
            missing_counts["missing_episode_dirs"] += 1
            missing.append(str(pdir / "episode_*"))
            continue

        for ep in ep_dirs:
            episode_id = _parse_episode_id(ep)
            prefix = ep / cfg.prefix_filename
            manifest_path = ep / cfg.manifest_filename
            if not prefix.exists():
                missing_counts["missing_prefix"] += 1
                missing.append(str(prefix))
                continue
            if not manifest_path.exists():
                missing_counts["missing_manifest"] += 1
                missing.append(str(manifest_path))
                continue

            prefix_parts.append(_read_prefix_csv(prefix, int(pat_id), seq_id))

            manifest = pd.read_csv(manifest_path)
            required_cols = {"scenario_name", "scenario_type", "perturb_family", "is_valid"}
            if required_cols - set(manifest.columns):
                raise KeyError(f"Manifest missing required columns: {manifest_path}")
            is_valid = pd.to_numeric(manifest["is_valid"], errors="coerce").fillna(0).astype(int)
            valid = manifest[is_valid == 1]

            for _, row in valid.iterrows():
                scenario_name = str(row["scenario_name"])
                scenario_type = str(row["scenario_type"])
                perturb_family = str(row["perturb_family"])
                suffix_path = ep / f"{cfg.suffix_prefix}{scenario_name}.csv"
                if not suffix_path.exists():
                    missing_counts["missing_suffix"] += 1
                    missing.append(str(suffix_path))
                    continue

                suffix_parts.append(
                    _read_suffix_csv(
                        suffix_path,
                        int(pat_id),
                        seq_id,
                        int(episode_id),
                        scenario_name,
                        scenario_type,
                        perturb_family,
                    )
                )
                scenario_rows.append(
                    {
                        "pat_id": int(pat_id),
                        "seq_id": int(seq_id),
                        "episode_id": int(episode_id),
                        "scenario_name": scenario_name,
                        "scenario_type": scenario_type,
                        "perturb_family": perturb_family,
                        "is_valid": 1,
                    }
                )

            episodes += 1
            seq_id += 1

    missing_preview = ", ".join(missing[:10]) if missing else ""

    prefix_df = pd.concat(prefix_parts, ignore_index=True) if prefix_parts else pd.DataFrame(
        columns=["pat_id", "seq_id", "date", "cgm", "basal", "bolus", "meal"]
    )
    suffix_df = pd.concat(suffix_parts, ignore_index=True) if suffix_parts else pd.DataFrame(
        columns=[
            "pat_id",
            "seq_id",
            "episode_id",
            "scenario_name",
            "scenario_type",
            "perturb_family",
            "date",
            "cgm",
            "basal",
            "bolus",
            "meal",
        ]
    )
    if scenario_rows:
        scenario_df = pd.DataFrame(scenario_rows)
    else:
        scenario_df = pd.DataFrame(
            columns=[
                "pat_id",
                "seq_id",
                "episode_id",
                "scenario_name",
                "scenario_type",
                "perturb_family",
                "is_valid",
            ]
        )

    if not prefix_df.empty:
        prefix_df = prefix_df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
    if not suffix_df.empty:
        suffix_df = suffix_df.sort_values(["pat_id", "seq_id", "scenario_name", "date"]).reset_index(drop=True)
    if not scenario_df.empty:
        scenario_df = scenario_df.sort_values(["pat_id", "seq_id", "scenario_name"]).reset_index(drop=True)

    # Normalize date formatting
    if "date" in prefix_df.columns and not prefix_df.empty:
        prefix_df["date"] = pd.to_datetime(prefix_df["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    if "date" in suffix_df.columns and not suffix_df.empty:
        suffix_df["date"] = pd.to_datetime(suffix_df["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    stats = {
        "patients": len(set(int(p) for p in patients)),
        "episodes": int(episodes),
        "scenarios": int(len(scenario_df)),
        "rows_prefix": int(len(prefix_df)),
        "rows_suffix": int(len(suffix_df)),
        **missing_counts,
    }
    if missing_preview:
        stats["missing_preview"] = missing_preview
    return prefix_df, suffix_df, scenario_df, stats


def _preprocess_action_cf_like(
    raw_dir: Path,
    output_dir: Path,
    dataset_name: str,
    notes: str,
) -> None:
    raw_dir, output_dir = ensure_io(raw_dir, output_dir)
    cfg = ActionCFConfig()

    split_path = raw_dir / cfg.splits_filename
    if not split_path.exists():
        raise FileNotFoundError(f"Expected split file at {split_path}")

    split_df = pd.read_csv(split_path)
    if {"pat_id", "split"} - set(split_df.columns):
        raise KeyError(f"Split file missing required columns: {split_path}")
    split_df = split_df[["pat_id", "split"]].copy()
    split_df["pat_id"] = pd.to_numeric(split_df["pat_id"], errors="coerce").astype(int)
    split_df["split"] = split_df["split"].astype(str).str.lower()

    patient_dirs = _build_patient_dir_map(raw_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, object] = {
        "dataset": str(dataset_name),
        "source_files": [str(split_path)],
        "notes": str(notes),
        "splits": {},
        "channels": ["cgm", "basal", "bolus", "meal"],
    }

    for split_name in ["train", "val"]:
        df, stats = _collect_split(split_name, split_df, patient_dirs, cfg)
        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        meta["splits"][split_name] = {
            **stats,
            "path": str(out_path),
        }

    prefix_df, suffix_df, scenario_df, test_stats = _collect_test(split_df, patient_dirs, cfg)
    prefix_path = output_dir / "test_prefix.csv"
    suffix_path = output_dir / "test_suffix.csv"
    scenarios_path = output_dir / "test_scenarios.csv"
    prefix_df.to_csv(prefix_path, index=False)
    suffix_df.to_csv(suffix_path, index=False)
    scenario_df.to_csv(scenarios_path, index=False)
    meta["splits"]["test"] = {
        **test_stats,
        "prefix_path": str(prefix_path),
        "suffix_path": str(suffix_path),
        "scenarios_path": str(scenarios_path),
    }

    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def preprocess_action_cf_benchmark(raw_dir: Path, output_dir: Path) -> None:
    _preprocess_action_cf_like(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="action_cf_benchmark",
        notes=(
            "Merged prefix_factual.csv sequences per patient and episode. "
            "Train/val contain factual prefixes; test also includes counterfactual suffix scenarios."
        ),
    )


def preprocess_action_cf_benchmark_bolus(raw_dir: Path, output_dir: Path) -> None:
    _preprocess_action_cf_like(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="action_cf_benchmark_bolus",
        notes=(
            "Merged prefix_factual.csv sequences per patient and episode. "
            "Test suffixes are bolus-addition perturbations at the start of the perturbation window "
            "with magnitudes 0.00x, 0.25x, ..., 2.00x of each patient's average bolus."
        ),
    )


def preprocess_action_cf_benchmark_controliq(raw_dir: Path, output_dir: Path) -> None:
    _preprocess_action_cf_like(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="action_cf_benchmark_controliq",
        notes=(
            "ControliQ version of action_cf_benchmark. Merged prefix_factual.csv sequences per patient "
            "and episode. Train/val contain factual prefixes; test also includes counterfactual suffix scenarios."
        ),
    )


def preprocess_action_cf_benchmark_bolus_controliq(raw_dir: Path, output_dir: Path) -> None:
    _preprocess_action_cf_like(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="action_cf_benchmark_bolus_controliq",
        notes=(
            "ControliQ version of action_cf_benchmark_bolus. Merged prefix_factual.csv sequences per patient "
            "and episode. Test suffixes are bolus-addition perturbations at the start of the perturbation window "
            "with magnitudes 0.00x, 0.25x, ..., 2.00x of each patient's average bolus."
        ),
    )
