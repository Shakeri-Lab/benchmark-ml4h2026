from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _default_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {"pat_id", "seq_id", "date", "scenario_name", "scenario_type", "perturb_family", "episode_id"}
    return [c for c in df.columns if c not in drop_cols]


def _canon_pid(val) -> str:
    try:
        f = float(val)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(val).strip()


def _dist_from_end(mask_1d: np.ndarray) -> int:
    if mask_1d is None or mask_1d.size == 0:
        return -1
    if not mask_1d.any():
        return -1
    last_idx = mask_1d.size - 1 - np.flip(mask_1d, axis=0).argmax()
    return int(mask_1d.size - 1 - last_idx)


def _nocturnal_flag(ctx_end_date: str, start_hour: int, end_hour: int) -> int:
    hour = None
    try:
        s = str(ctx_end_date)
        if len(s) >= 13 and s[10] in {" ", "T"}:
            hour = int(s[11:13])
    except Exception:
        hour = None
    if hour is None:
        return -1
    if start_hour < end_hour:
        in_range = start_hour <= hour < end_hour
    else:
        in_range = (hour >= start_hour) or (hour < end_hour)
    return 0 if in_range else -1


def _shift_insulin_cols(df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
    if pred_len <= 0:
        return df
    out = df.copy()
    group_cols = [c for c in ["pat_id", "seq_id"] if c in out.columns]
    if not group_cols:
        return out
    for col in ["basal", "bolus"]:
        if col in out.columns:
            out[col] = out.groupby(group_cols)[col].shift(-pred_len)
            out[col] = out[col].fillna(0.0)
    return out


class CounterfactualWindowDataset(Dataset):
    """
    Dataloader for action-conditional counterfactual dataset.

    Train/val: sliding windows over prefix data with insulin channels shifted
    left by pred_len (future insulin exposed in X).

    Test: one sample per (patient, episode, scenario), using prefix for history
    and suffix for targets (CGM) and future insulin shift.
    """

    def __init__(
        self,
        *,
        split_part: str,
        processed_root: Path = Path("datasets_processed"),
        dataset_name: str = "action_cf_benchmark",
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
        prefix_path: Optional[Path] = None,
        suffix_path: Optional[Path] = None,
        scenarios_path: Optional[Path] = None,
        subset_size: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        subset_seed: int = 42,
    ):
        if split_part not in {"train", "val", "test"}:
            raise ValueError("split_part must be one of {'train','val','test'}")

        self.split_part = split_part
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.step = int(stride)
        self.target_horizon = target_horizon
        self.nocturnal_start_hour = int(nocturnal_start_hour)
        self.nocturnal_end_hour = int(nocturnal_end_hour)
        self._subset_indices: Optional[np.ndarray] = None

        if split_part in {"train", "val"}:
            self._init_trainval(
                processed_root=processed_root,
                dataset_name=dataset_name,
                csv_path=csv_path,
                feature_cols=feature_cols,
                target_col=target_col,
                univariate=univariate,
                subset_size=subset_size,
                subset_fraction=subset_fraction,
                subset_seed=subset_seed,
            )
        else:
            self._init_test(
                processed_root=processed_root,
                dataset_name=dataset_name,
                prefix_path=prefix_path,
                suffix_path=suffix_path,
                scenarios_path=scenarios_path,
                feature_cols=feature_cols,
                target_col=target_col,
                univariate=univariate,
                subset_size=subset_size,
                subset_fraction=subset_fraction,
                subset_seed=subset_seed,
            )

    def _init_trainval(
        self,
        *,
        processed_root: Path,
        dataset_name: str,
        csv_path: Optional[Path],
        feature_cols: Optional[Sequence[str]],
        target_col: str,
        univariate: bool,
        subset_size: Optional[int],
        subset_fraction: Optional[float],
        subset_seed: int,
    ) -> None:
        if csv_path is None:
            csv_path = Path(processed_root) / dataset_name / f"{self.split_part}.csv"
        if not Path(csv_path).exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)

        if "pat_id" not in df.columns:
            if "pid" in df.columns:
                df = df.rename(columns={"pid": "pat_id"})
            else:
                raise KeyError(f"'pat_id' (or pid) not in dataframe {csv_path}")
        if "seq_id" not in df.columns:
            df = df.sort_values(["pat_id", "date"]).reset_index(drop=True)
            df["seq_id"] = df.groupby("pat_id").cumcount().mul(0)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", target_col]).copy()
        if df.empty:
            raise ValueError(f"No usable rows for split '{self.split_part}' in {csv_path}")

        # Event masks on raw (unshifted) data
        exclude = {"cgm", "pat_id", "seq_id", "basal", "date", "pid", "patient_id", "subject_id"}
        numeric_cols = [c for c in df.columns if c.lower() not in exclude]
        num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if numeric_cols else None
        meal_cols = [c for c in (numeric_cols or []) if ("meal" in c.lower()) or ("carb" in c.lower())]
        bolus_cols = [c for c in (numeric_cols or []) if "bolus" in c.lower()]
        self.meal_mask = (num_df[meal_cols] > 0).any(axis=1).to_numpy() if meal_cols else None
        self.bolus_mask = (num_df[bolus_cols] > 0).any(axis=1).to_numpy() if bolus_cols else None

        df = df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
        df = _shift_insulin_cols(df, self.pred_len)

        if univariate:
            features = [target_col]
        else:
            features = list(feature_cols) if feature_cols else _default_feature_cols(df)
        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not in dataframe")
        features = [target_col] + [c for c in features if c != target_col]

        arr_seq = df["seq_id"].to_numpy(dtype=np.int64)
        arr_pat = df["pat_id"].to_numpy()
        arr_feat = df[features].to_numpy(dtype=np.float32)
        if np.isnan(arr_feat).any():
            nan_counts = np.isnan(arr_feat).sum(axis=0)
            if nan_counts[0] > 0:
                raise ValueError(
                    f"NaNs found in target column '{target_col}' for split '{self.split_part}'"
                )
            arr_feat[:, 1:][np.isnan(arr_feat[:, 1:])] = 0.0

        self.feature_cols = features
        self.target_idx = 0
        self.data = np.concatenate([arr_seq[:, None], arr_pat[:, None], arr_feat], axis=1)
        self.dates = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

        seq0 = self.data[: -(self.history_len + self.pred_len), 0]
        seq1 = self.data[self.history_len + self.pred_len :, 0]
        indices = torch.nonzero(
            torch.from_numpy(seq0) == torch.from_numpy(seq1), as_tuple=False
        ).squeeze(-1)[:: self.step]
        self.indices = indices
        self.num = int(self.indices.numel())

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

        self._scenario_name = "factual"
        self._scenario_type = "factual"
        self._perturb_family = "none"

    def _init_test(
        self,
        *,
        processed_root: Path,
        dataset_name: str,
        prefix_path: Optional[Path],
        suffix_path: Optional[Path],
        scenarios_path: Optional[Path],
        feature_cols: Optional[Sequence[str]],
        target_col: str,
        univariate: bool,
        subset_size: Optional[int],
        subset_fraction: Optional[float],
        subset_seed: int,
    ) -> None:
        if prefix_path is None:
            prefix_path = Path(processed_root) / dataset_name / "test_prefix.csv"
        if suffix_path is None:
            suffix_path = Path(processed_root) / dataset_name / "test_suffix.csv"
        if scenarios_path is None:
            scenarios_path = Path(processed_root) / dataset_name / "test_scenarios.csv"

        if not Path(prefix_path).exists():
            raise FileNotFoundError(prefix_path)
        if not Path(suffix_path).exists():
            raise FileNotFoundError(suffix_path)

        prefix_df = pd.read_csv(prefix_path)
        suffix_df = pd.read_csv(suffix_path)

        for col in ["date", "time"]:
            if col in prefix_df.columns:
                prefix_df = prefix_df.rename(columns={col: "date"})
                break
        for col in ["date", "time"]:
            if col in suffix_df.columns:
                suffix_df = suffix_df.rename(columns={col: "date"})
                break

        prefix_df["date"] = pd.to_datetime(prefix_df["date"], errors="coerce")
        suffix_df["date"] = pd.to_datetime(suffix_df["date"], errors="coerce")
        prefix_df = prefix_df.dropna(subset=["date", target_col]).copy()
        suffix_df = suffix_df.dropna(subset=["date", target_col]).copy()

        if "scenario_type" not in suffix_df.columns or "perturb_family" not in suffix_df.columns:
            if Path(scenarios_path).exists():
                scen_df = pd.read_csv(scenarios_path)
                keep = ["pat_id", "seq_id", "scenario_name", "scenario_type", "perturb_family"]
                scen_df = scen_df[keep].drop_duplicates()
                suffix_df = suffix_df.merge(
                    scen_df,
                    on=["pat_id", "seq_id", "scenario_name"],
                    how="left",
                )

        if univariate:
            features = [target_col]
        else:
            features = list(feature_cols) if feature_cols else _default_feature_cols(prefix_df)
        if target_col not in prefix_df.columns:
            raise KeyError(f"target_col '{target_col}' not in prefix dataframe")
        features = [target_col] + [c for c in features if c != target_col]
        self.feature_cols = features
        self.target_idx = 0

        prefix_df = prefix_df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
        suffix_df = suffix_df.sort_values(["pat_id", "seq_id", "scenario_name", "date"]).reset_index(drop=True)

        prefix_groups = {
            (int(pid), int(sid)): g
            for (pid, sid), g in prefix_df.groupby(["pat_id", "seq_id"], sort=False)
        }

        samples = []
        for (pid, sid, scenario_name), suf in suffix_df.groupby(
            ["pat_id", "seq_id", "scenario_name"], sort=False
        ):
            key = (int(pid), int(sid))
            pref = prefix_groups.get(key)
            if pref is None or pref.empty:
                continue
            pref = pref.sort_values("date")
            suf = suf.sort_values("date")
            if len(pref) < self.history_len or len(suf) < self.pred_len:
                continue

            pref_ctx = pref.iloc[-self.history_len:]
            cgm_ctx = pref_ctx[target_col].to_numpy(dtype=np.float32)
            meal_ctx = (
                pref_ctx["meal"].to_numpy(dtype=np.float32)
                if "meal" in pref_ctx.columns
                else np.zeros_like(cgm_ctx)
            )
            bolus_ctx_raw = (
                pref_ctx["bolus"].to_numpy(dtype=np.float32)
                if "bolus" in pref_ctx.columns
                else np.zeros_like(cgm_ctx)
            )

            # Build shifted insulin using prefix + suffix for this scenario
            if "basal" in pref.columns and "basal" in suf.columns:
                combined_basal = pd.concat([pref["basal"], suf["basal"]], ignore_index=True)
            else:
                combined_basal = pd.Series(np.zeros(len(pref) + len(suf), dtype=float))
            if "bolus" in pref.columns and "bolus" in suf.columns:
                combined_bolus = pd.concat([pref["bolus"], suf["bolus"]], ignore_index=True)
            else:
                combined_bolus = pd.Series(np.zeros(len(pref) + len(suf), dtype=float))
            basal_shift = combined_basal.shift(-self.pred_len).to_numpy(dtype=np.float32)
            bolus_shift = combined_bolus.shift(-self.pred_len).to_numpy(dtype=np.float32)
            start = len(pref) - self.history_len
            end = len(pref)
            basal_ctx = basal_shift[start:end]
            bolus_ctx = bolus_shift[start:end]
            if np.isnan(basal_ctx).any():
                basal_ctx = np.nan_to_num(basal_ctx, nan=0.0)
            if np.isnan(bolus_ctx).any():
                bolus_ctx = np.nan_to_num(bolus_ctx, nan=0.0)

            feat_map: Dict[str, np.ndarray] = {
                target_col: cgm_ctx,
                "basal": basal_ctx,
                "bolus": bolus_ctx,
                "meal": meal_ctx,
            }
            try:
                X = np.stack([feat_map[f] for f in self.feature_cols], axis=1)
            except KeyError as e:
                raise KeyError(f"Missing feature '{e.args[0]}' in test feature map") from e

            Y = suf[target_col].iloc[: self.pred_len].to_numpy(dtype=np.float32)
            if self.target_horizon is not None:
                h = int(self.target_horizon)
                if h == -1:
                    h = len(Y) - 1
                if h < 0 or h >= len(Y):
                    raise IndexError(
                        f"target_horizon {self.target_horizon} out of range for pred_len {len(Y)}"
                    )
                Y = np.asarray([Y[h]], dtype=Y.dtype)

            ctx_end_date = pref_ctx["date"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            meal_mask = meal_ctx > 0
            bolus_mask = bolus_ctx_raw > 0
            post_meal = _dist_from_end(meal_mask)
            post_bolus = _dist_from_end(bolus_mask)
            nocturnal = _nocturnal_flag(ctx_end_date, self.nocturnal_start_hour, self.nocturnal_end_hour)
            evt_tuple = (post_meal, post_bolus, nocturnal)

            scenario_type = (
                str(suf.get("scenario_type", "counterfactual").iloc[0])
                if "scenario_type" in suf.columns
                else "counterfactual"
            )
            perturb_family = (
                str(suf.get("perturb_family", "unknown").iloc[0])
                if "perturb_family" in suf.columns
                else "unknown"
            )

            samples.append(
                {
                    "X": X.astype(np.float32),
                    "Y": Y.astype(np.float32),
                    "pat_id": int(pid),
                    "seq_id": int(sid),
                    "date": ctx_end_date,
                    "evt_tuple": evt_tuple,
                    "scenario_name": str(scenario_name),
                    "scenario_type": scenario_type,
                    "perturb_family": perturb_family,
                }
            )

        if not samples:
            raise ValueError("No valid counterfactual test samples found.")

        if subset_fraction is not None:
            frac = float(subset_fraction)
            if not (0.0 < frac <= 1.0):
                raise ValueError("subset_fraction must be in (0,1]")
            subset_size = int(round(len(samples) * frac))
        if subset_size is not None:
            subset_size = int(subset_size)
            if subset_size <= 0:
                raise ValueError("subset_size must be > 0")
            subset_size = min(subset_size, len(samples))
            rng = np.random.default_rng(int(subset_seed))
            idx = rng.permutation(len(samples))[:subset_size]
            samples = [samples[i] for i in idx]

        self._test_samples = samples
        self.num = len(samples)
        self.meal_mask = None
        self.bolus_mask = None

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.split_part in {"train", "val"}:
            if self._subset_indices is not None:
                i = int(self._subset_indices[idx])
            else:
                i = int(self.indices[idx])
            ctx_end = i + self.history_len
            pred_end = ctx_end + self.pred_len

            seq_id = self.data[ctx_end - 1, 0]
            pat_id = self.data[ctx_end - 1, 1]
            ctx_end_date = self.dates[ctx_end - 1]

            ctx = self.data[i:ctx_end, 2:]
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
                return _dist_from_end(win)

            post_meal = _dist(self.meal_mask)
            post_bolus = _dist(self.bolus_mask)
            nocturnal = _nocturnal_flag(ctx_end_date, self.nocturnal_start_hour, self.nocturnal_end_hour)
            evt_tuple = (post_meal, post_bolus, nocturnal)

            X = torch.as_tensor(ctx, dtype=torch.float32)
            Y = torch.as_tensor(tgt, dtype=torch.float32)
            pid = torch.as_tensor(pat_id)
            sid = torch.as_tensor(seq_id)
            meta = {"feature_cols": self.feature_cols, "insulin_shifted": True}
            return (
                X,
                Y,
                pid,
                sid,
                ctx_end_date,
                evt_tuple,
                meta,
                self._scenario_name,
                self._scenario_type,
                self._perturb_family,
            )

        sample = self._test_samples[idx]
        X = torch.as_tensor(sample["X"], dtype=torch.float32)
        Y = torch.as_tensor(sample["Y"], dtype=torch.float32)
        pid = torch.as_tensor(sample["pat_id"])
        sid = torch.as_tensor(sample["seq_id"])
        meta = {"feature_cols": self.feature_cols, "insulin_shifted": True}
        return (
            X,
            Y,
            pid,
            sid,
            sample["date"],
            sample["evt_tuple"],
            meta,
            sample["scenario_name"],
            sample["scenario_type"],
            sample["perturb_family"],
        )


def make_cf_dataloader(
    *,
    split_part: str,
    processed_root: Path = Path("datasets_processed"),
    dataset_name: str = "action_cf_benchmark",
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
    csv_path: Optional[Path] = None,
    prefix_path: Optional[Path] = None,
    suffix_path: Optional[Path] = None,
    scenarios_path: Optional[Path] = None,
    subset_size: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    subset_seed: int = 42,
    seed: Optional[int] = None,
) -> DataLoader:
    ds = CounterfactualWindowDataset(
        split_part=split_part,
        processed_root=processed_root,
        dataset_name=dataset_name,
        history_len=history_len,
        pred_len=pred_len,
        stride=stride,
        feature_cols=feature_cols,
        target_col=target_col,
        target_horizon=target_horizon,
        univariate=univariate,
        nocturnal_start_hour=nocturnal_start_hour,
        nocturnal_end_hour=nocturnal_end_hour,
        csv_path=csv_path,
        prefix_path=prefix_path,
        suffix_path=suffix_path,
        scenarios_path=scenarios_path,
        subset_size=subset_size,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
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
        shuffle=shuffle if split_part != "test" else False,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def make_cf_dataloader_from_cfg(
    split_cfg: Dict[str, Any],
    dataloader_cfg: Dict[str, Any],
    *,
    split_part: str,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    split_cfg = dict(split_cfg or {})
    dataloader_cfg = dict(dataloader_cfg or {})

    dataset_name = split_cfg.get("dataset") or split_cfg.get("split_name") or split_cfg.get("name")
    if not dataset_name:
        raise ValueError("split_cfg must include dataset name (split_name or dataset)")

    processed_root = Path(split_cfg.get("processed_root", "datasets_processed"))

    csv_path = split_cfg.get(f"{split_part}_csv")
    prefix_path = split_cfg.get("test_prefix_csv")
    suffix_path = split_cfg.get("test_suffix_csv")
    scenarios_path = split_cfg.get("test_scenarios_csv")

    if shuffle is None:
        shuffle = bool(dataloader_cfg.get("shuffle", False))

    return make_cf_dataloader(
        split_part=str(split_part),
        processed_root=processed_root,
        dataset_name=str(dataset_name),
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
        csv_path=Path(csv_path) if csv_path else None,
        prefix_path=Path(prefix_path) if prefix_path else None,
        suffix_path=Path(suffix_path) if suffix_path else None,
        scenarios_path=Path(scenarios_path) if scenarios_path else None,
        subset_size=dataloader_cfg.get("subset_size", None),
        subset_fraction=dataloader_cfg.get("subset_fraction", None),
        subset_seed=int(dataloader_cfg.get("subset_seed", 42)),
        seed=seed,
    )
