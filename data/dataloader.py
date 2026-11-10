from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .adapters import load_timeseries


def _load_spec(split_name: str, split_root: Path) -> dict:
    split_root = Path(split_root)
    path = split_root / split_name
    if path.is_dir():
        candidates = list(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No split spec found in {path}")
        path = candidates[0]
    if not path.suffix:
        path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"Split spec not found: {path}")
    return json.loads(path.read_text()), path


def _default_feature_cols(df) -> List[str]:
    drop_cols = {"pat_id", "seq_id", "date"}
    return [c for c in df.columns if c not in drop_cols]


def _canon_pid(val) -> str:
    try:
        f = float(val)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(val).strip()


def _encoded_pid(val) -> int:
    """Return a stable integer suitable for tensor batching.

    Existing numeric patient ids retain their value. DIAX also permits string
    identifiers, which are mapped to stable negative integers in the exactly
    representable float64 range used by the dataset's internal array.
    """

    canonical = _canon_pid(val)
    try:
        numeric = int(canonical)
        if str(numeric) == canonical and abs(numeric) < 2**52:
            return numeric
    except ValueError:
        pass
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return -(int.from_bytes(digest[:8], "big") % (2**52 - 1) + 1)


class TimeSeriesWindowDataset(Dataset):
    """
    Sliding-window dataset built from CSV or DIAX data and a patient-level split spec.

    The window indexing keeps samples within the same sequence_id. The first column
    of the internal array is seq_id, matching the snippet:
        seq0 = data[:-ctx-pred, 0]
        seq1 = data[ctx+pred:, 0]
        indices = torch.nonzero(seq0 == seq1)[::step]
    """

    def __init__(
        self,
        split_name: str,
        split_part: str,
        split_root: Path = Path("configs/preprocessing"),
        processed_root: Path = Path("datasets_processed"),
        history_len: int = 288,
        pred_len: int = 24,
        stride: int = 12,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = "cgm",
        target_horizon: Optional[int] = None,
        univariate: bool = False,
        nocturnal_start_hour: int = 0,
        nocturnal_end_hour: int = 6,
        csv_path: Optional[Path] = None,
        subset_size: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        subset_seed: int = 42,
        source_path: Optional[Path] = None,
        source_format: str = "auto",
        source_options: Optional[Mapping[str, Any]] = None,
    ):
        if split_part not in {"train", "val", "test"}:
            raise ValueError("split_part must be one of {'train','val','test'}")

        if csv_path is not None and source_path is not None:
            raise ValueError("Use either csv_path (legacy) or source_path, not both")

        spec_path = None
        patients = None
        if csv_path is not None:
            # Preserve the legacy behavior: an explicit csv_path is not filtered
            # through a split spec. New callers should use source_path instead.
            ts_path = Path(csv_path)
            if not ts_path.exists():
                raise FileNotFoundError(ts_path)
            df, resolved_format = load_timeseries(ts_path, source_format="csv")
        else:
            spec, spec_path = _load_spec(split_name, split_root)
            dataset_name = spec.get("dataset") or split_name.split("_")[0]
            patients = spec["patients"].get(split_part)
            if patients is None:
                raise KeyError(f"Split part '{split_part}' not found in spec {spec_path}")

            if source_path is not None:
                ts_path = Path(source_path)
                ts_candidates = [ts_path]
            else:
                dataset_dirs = [
                    Path(processed_root) / dataset_name,
                    Path("dataset_processed") / dataset_name,
                    Path("datasets_preprocessed") / dataset_name,
                ]
                requested_format = source_format.lower()
                if requested_format == "csv":
                    ts_candidates = [path / "timeseries.csv" for path in dataset_dirs]
                elif requested_format == "diax":
                    ts_candidates = [
                        path for path in dataset_dirs
                        if path.is_dir() and next(path.glob("*.json"), None) is not None
                    ]
                elif requested_format == "auto":
                    ts_candidates = []
                    for path in dataset_dirs:
                        ts_candidates.append(path / "timeseries.csv")
                        if (
                            path.is_dir()
                            and next(path.glob("*.json"), None) is not None
                        ):
                            ts_candidates.append(path)
                else:
                    raise ValueError(
                        f"Unsupported source_format '{source_format}'; expected auto, csv, or diax"
                    )
                ts_path = next((path for path in ts_candidates if path.exists()), None)
            if ts_path is None:
                raise FileNotFoundError(
                    f"No {source_format} source found for {dataset_name}: tried {ts_candidates}"
                )
            df, resolved_format = load_timeseries(
                ts_path,
                source_format=source_format,
                patient_ids=patients,
                options=source_options,
            )

        import pandas as pd

        self.source_path = Path(ts_path)
        self.source_format = resolved_format
        if "pat_id" not in df.columns:
            if "pid" in df.columns:
                df = df.rename(columns={"pid": "pat_id"})
            else:
                raise KeyError(f"'pat_id' (or pid) not in dataframe {ts_path}")
        if "seq_id" not in df.columns:
            # Single sequence per patient fallback
            df = df.sort_values(["pat_id", "date"]).reset_index(drop=True)
            df["seq_id"] = df.groupby("pat_id").cumcount().mul(0)
        if patients is not None:
            pat_set = {_canon_pid(p) for p in patients}
            pat_key = df["pat_id"].apply(_canon_pid)
            df = df[pat_key.isin(pat_set)]
            if df.empty:
                raise ValueError(f"No rows found for split '{split_part}' in {ts_path}")

        df = df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
        df = df.dropna(subset=[target_col])
        if df.empty:
            raise ValueError(f"No non-NaN target rows for split '{split_part}' in {ts_path}")

        # Event masks per row (computed on full dataframe, independent of univariate slicing)
        exclude = {"cgm", "pat_id", "seq_id", "basal", "date", "pid", "patient_id", "subject_id"}
        numeric_cols = [c for c in df.columns if c.lower() not in exclude]
        num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if numeric_cols else None
        meal_cols = [c for c in (numeric_cols or []) if ("meal" in c.lower()) or ("carb" in c.lower())]
        bolus_cols = [c for c in (numeric_cols or []) if "bolus" in c.lower()]
        self.meal_mask = (num_df[meal_cols] > 0).any(axis=1).to_numpy() if meal_cols else None
        self.bolus_mask = (num_df[bolus_cols] > 0).any(axis=1).to_numpy() if bolus_cols else None
        self.nocturnal_start_hour = int(nocturnal_start_hour)
        self.nocturnal_end_hour = int(nocturnal_end_hour)

        if univariate:
            features = [target_col]
        else:
            features = list(feature_cols) if feature_cols else _default_feature_cols(df)
        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not in dataframe")
        # Always place target first to simplify downstream indexing
        features = [target_col] + [c for c in features if c != target_col]

        # Build data array: [seq_id, pat_id, features...]
        arr_seq = df["seq_id"].to_numpy(dtype=np.int64)
        raw_patient_ids = df["pat_id"].map(_canon_pid)
        arr_pat = raw_patient_ids.map(_encoded_pid).to_numpy(dtype=np.int64)
        self.patient_id_map = {
            int(encoded): raw
            for encoded, raw in zip(arr_pat, raw_patient_ids)
        }
        if len(self.patient_id_map) != raw_patient_ids.nunique():
            raise ValueError("Patient-id encoding collision; patient identifiers are not unique")
        arr_feat = df[features].to_numpy(dtype=np.float32)
        if np.isnan(arr_feat).any():
            nan_counts = np.isnan(arr_feat).sum(axis=0)
            # Target should have been dropped if NaN; enforce
            if nan_counts[0] > 0:
                spec_str = str(spec_path) if spec_path is not None else "N/A"
                raise ValueError(
                    f"NaNs found in target column '{target_col}' for split '{split_part}' "
                    f"(spec={spec_str}): {int(nan_counts[0])}"
                )
            # Fill non-target NaNs with 0.0
            arr_feat[:, 1:][np.isnan(arr_feat[:, 1:])] = 0.0
        arr_date = df["date"].to_numpy()
        self.feature_cols = features
        self.target_idx = 0  # target_col placed first
        # data: [seq_id, pat_id, date (object), features...]; keep date separately to avoid dtype mixing
        self.data = np.concatenate(
            [arr_seq[:, None], arr_pat[:, None], arr_feat], axis=1
        )  # numeric array [N, 2+F]
        self.dates = arr_date  # array of strings/timestamps aligned with rows

        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.step = int(stride)
        self.target_horizon = target_horizon

        # Compute valid start indices within same sequence
        seq0 = self.data[: -(self.history_len + self.pred_len), 0]
        seq1 = self.data[self.history_len + self.pred_len :, 0]
        self.indices = torch.nonzero(
            torch.from_numpy(seq0) == torch.from_numpy(seq1), as_tuple=False
        ).squeeze(-1)[:: self.step]
        self.num = int(self.indices.numel())
        self._subset_indices: Optional[torch.Tensor] = None

        if subset_fraction is not None:
            frac = float(subset_fraction)
            if not (0.0 < frac <= 1.0):
                raise ValueError("subset_fraction must be in (0,1]")
            subset_size = int(round(self.num * frac))
        if subset_size is not None:
            subset_size = int(subset_size)
            if subset_size <= 0:
                raise ValueError("subset_size must be > 0")
            subset_size = min(subset_size, self.num)
            rng = np.random.default_rng(int(subset_seed))
            perm = rng.permutation(self.num)[:subset_size]
            self._subset_indices = self.indices[perm]
            self.num = int(self._subset_indices.numel())

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self._subset_indices is not None:
            i = int(self._subset_indices[idx])
        else:
            i = int(self.indices[idx])
        ctx_end = i + self.history_len
        pred_end = ctx_end + self.pred_len

        seq_id = self.data[ctx_end - 1, 0]
        pat_id = self.data[ctx_end - 1, 1]
        ctx_end_date = self.dates[ctx_end - 1]

        ctx = self.data[i:ctx_end, 2:]  # drop seq_id, pat_id
        tgt = self.data[ctx_end:pred_end, 2 + self.target_idx]
        if self.target_horizon is not None:
            h = int(self.target_horizon)
            if h == -1:
                h = len(tgt) - 1
            if h < 0 or h >= len(tgt):
                raise IndexError(
                    f"target_horizon {self.target_horizon} out of range for pred_len {len(tgt)}"
                )
            tgt = np.asarray([tgt[h]], dtype=tgt.dtype)
        def _dist(mask_1d):
            if mask_1d is None:
                return -1
            win = mask_1d[i:ctx_end]
            if not win.any():
                return -1
            last_idx = win.size - 1 - np.flip(win, axis=0).argmax()
            return ctx.shape[0] - 1 - last_idx

        post_meal = _dist(self.meal_mask)
        post_bolus = _dist(self.bolus_mask)

        # Nocturnal: 12am-6am by default (configurable start/end hours)
        hour = None
        try:
            s = str(ctx_end_date)
            if len(s) >= 13 and s[10] in {" ", "T"}:
                hour = int(s[11:13])
        except Exception:
            hour = None
        if hour is None:
            nocturnal = -1
        else:
            start = self.nocturnal_start_hour
            end = self.nocturnal_end_hour
            if start < end:
                in_range = start <= hour < end
            else:
                in_range = (hour >= start) or (hour < end)
            nocturnal = 0 if in_range else -1

        evt_tuple = (post_meal, post_bolus, nocturnal)

        X = torch.as_tensor(ctx, dtype=torch.float32)
        Y = torch.as_tensor(tgt, dtype=torch.float32)
        pid = torch.as_tensor(pat_id, dtype=torch.int64)
        sid = torch.as_tensor(seq_id, dtype=torch.int64)
        meta = {"feature_cols": self.feature_cols}
        return X, Y, pid, sid, ctx_end_date, evt_tuple, meta


def make_dataloader(
    split_name: str,
    split_part: str,
    split_root: Path = Path("configs/preprocessing"),
    processed_root: Path = Path("datasets_processed"),
    history_len: int = 288,
    pred_len: int = 24,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "cgm",
    target_horizon: Optional[int] = None,
    univariate: bool = False,
    nocturnal_start_hour: int = 0,
    nocturnal_end_hour: int = 6,
    seed: Optional[int] = None,
    source_path: Optional[Path] = None,
    source_format: str = "auto",
    source_options: Optional[Mapping[str, Any]] = None,
):
    ds = TimeSeriesWindowDataset(
        split_name=split_name,
        split_part=split_part,
        split_root=split_root,
        processed_root=processed_root,
        history_len=history_len,
        pred_len=pred_len,
        stride=stride,
        feature_cols=feature_cols,
        target_col=target_col,
        target_horizon=target_horizon,
        univariate=univariate,
        nocturnal_start_hour=nocturnal_start_hour,
        nocturnal_end_hour=nocturnal_end_hour,
        source_path=source_path,
        source_format=source_format,
        source_options=source_options,
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        seed = int(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)

        def _seed_worker(worker_id: int):
            worker_seed = seed + int(worker_id)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        worker_init_fn = _seed_worker

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def make_dataloader_from_cfg(
    split_cfg: Dict[str, Any],
    dataloader_cfg: Dict[str, Any],
    *,
    split_part: str,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Build a DataLoader from config dictionaries.

    Expected keys:
      split_cfg: split_name, split_root (optional), processed_root (optional),
        source_path/data_path (optional), source_format (auto/csv/diax),
        source_options (optional DIAX adapter settings)
      dataloader_cfg: history_len, pred_len, stride, batch_size, univariate,
        num_workers, feature_cols (optional)
    """
    split_cfg = dict(split_cfg or {})
    dataloader_cfg = dict(dataloader_cfg or {})

    split_name = split_cfg.get("split_name") or split_cfg.get("name")
    if not split_name:
        raise ValueError("split_cfg must include split_name")

    split_root = Path(split_cfg.get("split_root", "configs/preprocessing"))
    processed_root = Path(split_cfg.get("processed_root", "datasets_processed"))

    if shuffle is None:
        shuffle = bool(dataloader_cfg.get("shuffle", False))
    if seed is None and "seed" in dataloader_cfg:
        seed = dataloader_cfg.get("seed")

    return make_dataloader(
        split_name=str(split_name),
        split_part=str(split_part),
        split_root=split_root,
        processed_root=processed_root,
        history_len=int(dataloader_cfg.get("history_len", 288)),
        pred_len=int(dataloader_cfg.get("pred_len", 24)),
        stride=int(dataloader_cfg.get("stride", 1)),
        batch_size=int(dataloader_cfg.get("batch_size", 64)),
        shuffle=bool(shuffle),
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
        feature_cols=dataloader_cfg.get("feature_cols", None),
        target_col=str(dataloader_cfg.get("target_col", "cgm")),
        target_horizon=dataloader_cfg.get("target_horizon", None),
        univariate=bool(dataloader_cfg.get("univariate", False)),
        nocturnal_start_hour=int(dataloader_cfg.get("nocturnal_start_hour", 0)),
        nocturnal_end_hour=int(dataloader_cfg.get("nocturnal_end_hour", 6)),
        seed=seed,
        source_path=split_cfg.get("source_path", split_cfg.get("data_path")),
        source_format=str(split_cfg.get("source_format", "auto")),
        source_options=split_cfg.get("source_options"),
    )
