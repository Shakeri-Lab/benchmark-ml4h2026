from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


SLICE_LABELS: Tuple[str, ...] = ("both", "post_meal", "post_bolus", "nocturnal", "neither")


def slice_label_from_events(post_meal: int, post_bolus: int, nocturnal: int) -> str:
    has_meal = post_meal >= 0
    has_bolus = post_bolus >= 0
    is_noct = nocturnal == 0

    if has_meal and has_bolus:
        return "both"
    if has_meal and (not has_bolus):
        return "post_meal"
    if has_bolus and (not has_meal):
        return "post_bolus"
    if is_noct and (not has_meal) and (not has_bolus):
        return "nocturnal"
    return "neither"


def window_to_slice(history_len: int, a: int, b: int) -> slice:
    """
    Convert a recency window [a,b] (in steps) to a slice over the history axis.

    Semantics (LOCKED): window [a,b] means timesteps in (t-b, t-a], where t is the
    last context index. With history_len=T, t=T-1 and this corresponds to:
        start = max(0, T-b)
        end   = max(0, T-a)
        indices = [start, ..., end-1]
    """
    if history_len <= 0:
        raise ValueError("history_len must be > 0")
    if a is None or b is None:
        raise ValueError("window endpoints must be provided")
    a = int(a)
    b = int(b)
    if a < 0 or b < 0:
        raise ValueError(f"window endpoints must be nonnegative: [{a},{b}]")
    if b <= 0:
        raise ValueError(f"window must have b>0: [{a},{b}]")
    if not (a < b):
        raise ValueError(f"window must satisfy a<b: [{a},{b}]")

    start = max(0, history_len - b)
    end = max(0, history_len - a)
    if end <= start:
        return slice(0, 0)
    return slice(start, end)


@dataclass(frozen=True)
class Player:
    name: str
    modality: str
    window_steps: Tuple[int, int]  # [a,b] in steps
    time_slice: slice
    channel_idxs: Tuple[int, ...]
    channel_names: Tuple[str, ...]


def _default_windows_steps_by_modality() -> Dict[str, List[Tuple[int, int]]]:
    return {
        "cgm": [(0, 3), (3, 6), (6, 12), (12, 24)],
        "meal": [(0, 3), (3, 6), (6, 12)],
        "bolus": [(0, 3), (3, 6), (6, 12)],
        "basal": [(0, 6), (6, 12)],
        "activity": [(0, 6), (6, 12), (12, 24)],
    }


@dataclass
class DataConfig:
    split_path: Optional[str] = None  # path to split CSV (overridable via CLI)
    split_part: str = "test"
    history_len: int = 288
    pred_len: int = 24
    stride: int = 1
    batch_size: int = 64
    univariate: bool = False
    num_workers: int = 0
    processed_root: str = "datasets_processed"
    input_layout: str = "BTC"  # "BTC" or "BCT"
    nocturnal_start_hour: int = 0
    nocturnal_end_hour: int = 6
    subsample_size: Optional[int] = None
    subsample_fraction: Optional[float] = None
    subsample_seed: int = 42


@dataclass
class ShapParams:
    horizons_steps: List[int] = field(default_factory=lambda: [3, 6, 12, 18, 24])
    n_permutations: int = 8
    n_background_draws: int = 16
    microbatch_size: int = 64
    perm_batch_size: int = 1
    seed: int = 42
    resample_background_each_permutation: bool = False


@dataclass
class BackgroundPolicy:
    prefer_same_slice: bool = True
    min_pool_size_same_slice: int = 25
    allow_global_fallback: bool = False
    global_prefer_same_slice: bool = False
    exclude_self: bool = False
    max_per_pid_slice: Optional[int] = None
    max_per_pid: Optional[int] = None
    max_global: Optional[int] = None


@dataclass
class OutputConfig:
    partition_by_slice: bool = True
    file_format: str = "npz"
    max_eval_samples: Optional[int] = None
    diagnostics_subset_size: int = 128


@dataclass
class ModelConfig:
    compile: Any = False


@dataclass
class ReconstructionConfig:
    meal_bolus_k: int = 3
    meal_type: Optional[str] = None  # "binary" | "grams" | None (auto)
    meal_binary_max: float = 1.5
    meal_binary_ratio: float = 0.95


@dataclass
class ShapConfig:
    data: DataConfig = field(default_factory=DataConfig)
    shap: ShapParams = field(default_factory=ShapParams)
    background: BackgroundPolicy = field(default_factory=BackgroundPolicy)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    windows_steps_by_modality: Dict[str, List[Tuple[int, int]]] = field(
        default_factory=_default_windows_steps_by_modality
    )
    modality_channels: Optional[Dict[str, List[str]]] = None

    @staticmethod
    def _as_bool(v: Any, default: bool = False) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off", "none", "null"}:
            return False
        return default

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShapConfig":
        d = dict(d or {})

        data_d = dict(d.get("data") or {})
        shap_d = dict(d.get("shap") or {})
        bg_d = dict(d.get("background") or {})
        out_d = dict(d.get("output") or {})
        model_d = dict(d.get("model") or {})

        split_path = data_d.get("split_path", d.get("split_path", None))
        if split_path is not None:
            split_path = str(split_path).strip()
            if not split_path:
                split_path = None

        data = DataConfig(
            split_path=split_path,
            split_part=str(data_d.get("split_part", "test")),
            history_len=int(data_d.get("history_len", 288)),
            pred_len=int(data_d.get("pred_len", 24)),
            stride=int(data_d.get("stride", 1)),
            batch_size=int(data_d.get("batch_size", 64)),
            univariate=cls._as_bool(data_d.get("univariate", False)),
            num_workers=int(data_d.get("num_workers", 0)),
            processed_root=str(data_d.get("processed_root", "datasets_processed")),
            input_layout=str(data_d.get("input_layout", "BTC")).upper(),
            nocturnal_start_hour=int(data_d.get("nocturnal_start_hour", 0)),
            nocturnal_end_hour=int(data_d.get("nocturnal_end_hour", 6)),
            subsample_size=data_d.get("subsample_size", None),
            subsample_fraction=data_d.get("subsample_fraction", None),
            subsample_seed=int(data_d.get("subsample_seed", 42)),
        )
        if data.split_part not in {"train", "val", "test"}:
            raise ValueError("data.split_part must be one of {'train','val','test'}")
        if data.input_layout not in {"BTC", "BCT"}:
            raise ValueError("data.input_layout must be 'BTC' or 'BCT'")
        if data.history_len <= 0 or data.pred_len <= 0:
            raise ValueError("data.history_len and data.pred_len must be > 0")
        if data.stride <= 0 or data.batch_size <= 0:
            raise ValueError("data.stride and data.batch_size must be > 0")
        if data.subsample_size is not None:
            data.subsample_size = int(data.subsample_size)
            if data.subsample_size <= 0:
                raise ValueError("data.subsample_size must be > 0 if set")
        if data.subsample_fraction is not None:
            data.subsample_fraction = float(data.subsample_fraction)
            if not (0.0 < data.subsample_fraction <= 1.0):
                raise ValueError("data.subsample_fraction must be in (0,1] if set")

        horizons_steps = list(shap_d.get("horizons_steps", [3, 6, 12, 18, 24]))
        horizons_steps = [int(x) for x in horizons_steps]
        if any(h <= 0 for h in horizons_steps):
            raise ValueError("shap.horizons_steps must be positive integers (steps)")
        if len(set(horizons_steps)) != len(horizons_steps):
            raise ValueError("shap.horizons_steps must not contain duplicates")
        if any(h > data.pred_len for h in horizons_steps):
            raise ValueError(
                f"All horizons_steps must be within pred_len={data.pred_len}. "
                f"Got horizons_steps={horizons_steps}"
            )

        shap = ShapParams(
            horizons_steps=horizons_steps,
            n_permutations=int(shap_d.get("n_permutations", 8)),
            n_background_draws=int(shap_d.get("n_background_draws", 16)),
            microbatch_size=int(shap_d.get("microbatch_size", 64)),
            perm_batch_size=int(shap_d.get("perm_batch_size", 1)),
            seed=int(shap_d.get("seed", 42)),
            resample_background_each_permutation=cls._as_bool(
                shap_d.get("resample_background_each_permutation", False)
            ),
        )
        if (
            shap.n_permutations <= 0
            or shap.n_background_draws <= 0
            or shap.microbatch_size <= 0
            or shap.perm_batch_size <= 0
        ):
            raise ValueError("shap.n_permutations, n_background_draws, microbatch_size must be > 0")

        background = BackgroundPolicy(
            prefer_same_slice=cls._as_bool(bg_d.get("prefer_same_slice", True)),
            min_pool_size_same_slice=int(bg_d.get("min_pool_size_same_slice", 25)),
            allow_global_fallback=cls._as_bool(bg_d.get("allow_global_fallback", False)),
            global_prefer_same_slice=cls._as_bool(bg_d.get("global_prefer_same_slice", False)),
            exclude_self=cls._as_bool(bg_d.get("exclude_self", False)),
            max_per_pid_slice=bg_d.get("max_per_pid_slice", None),
            max_per_pid=bg_d.get("max_per_pid", None),
            max_global=bg_d.get("max_global", None),
        )
        if background.max_per_pid_slice is not None:
            background.max_per_pid_slice = int(background.max_per_pid_slice)
        if background.max_per_pid is not None:
            background.max_per_pid = int(background.max_per_pid)
        if background.max_global is not None:
            background.max_global = int(background.max_global)
        if background.min_pool_size_same_slice < 0:
            raise ValueError("background.min_pool_size_same_slice must be >= 0")

        output = OutputConfig(
            partition_by_slice=cls._as_bool(out_d.get("partition_by_slice", True)),
            file_format=str(out_d.get("file_format", "npz")),
            max_eval_samples=out_d.get("max_eval_samples", None),
            diagnostics_subset_size=int(out_d.get("diagnostics_subset_size", 128)),
        )
        if output.file_format.lower() != "npz":
            raise ValueError("Only file_format='npz' is supported right now")
        if output.max_eval_samples is not None:
            output.max_eval_samples = int(output.max_eval_samples)
            if output.max_eval_samples <= 0:
                raise ValueError("output.max_eval_samples must be > 0 if set")
        if output.diagnostics_subset_size < 0:
            raise ValueError("output.diagnostics_subset_size must be >= 0")

        model = ModelConfig(
            compile=model_d.get("compile", False),
        )

        recon_d = dict(d.get("reconstruction") or {})
        reconstruction = ReconstructionConfig(
            meal_bolus_k=int(recon_d.get("meal_bolus_k", 3)),
            meal_type=recon_d.get("meal_type", None),
            meal_binary_max=float(recon_d.get("meal_binary_max", 1.5)),
            meal_binary_ratio=float(recon_d.get("meal_binary_ratio", 0.95)),
        )
        if reconstruction.meal_type is not None:
            reconstruction.meal_type = str(reconstruction.meal_type).lower()
            if reconstruction.meal_type not in {"binary", "grams"}:
                raise ValueError("reconstruction.meal_type must be 'binary', 'grams', or None")

        windows_raw = d.get("windows_steps_by_modality", None)
        if windows_raw is None:
            windows = _default_windows_steps_by_modality()
        else:
            windows = {}
            for k, v in dict(windows_raw).items():
                win_list = []
                for item in v:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        raise ValueError(
                            "windows_steps_by_modality must be modality -> list of [a,b] windows"
                        )
                    a, b = int(item[0]), int(item[1])
                    _ = window_to_slice(data.history_len, a, b)  # validation + clamp logic
                    win_list.append((a, b))
                windows[str(k)] = win_list

        modality_channels = d.get("modality_channels", None)
        if modality_channels is not None:
            modality_channels = {str(k): [str(x) for x in v] for k, v in dict(modality_channels).items()}

        return cls(
            data=data,
            shap=shap,
            background=background,
            output=output,
            model=model,
            reconstruction=reconstruction,
            windows_steps_by_modality=windows,
            modality_channels=modality_channels,
        )


def load_shap_config(path: Path) -> ShapConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError("shap_config must be .yaml/.yml or .json")
    return ShapConfig.from_dict(raw or {})


def infer_modalities(
    feature_cols: Sequence[str],
    modality_channels: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, List[int]]:
    """
    Map feature columns to modality groups.

    - If modality_channels is provided, uses exact column-name matches.
    - Otherwise uses a simple heuristic based on substrings.
    """
    cols = list(feature_cols)
    if modality_channels:
        out: Dict[str, List[int]] = {m: [] for m in modality_channels.keys()}
        name_to_idx = {c: i for i, c in enumerate(cols)}
        for modality, names in modality_channels.items():
            for n in names:
                if n in name_to_idx:
                    out[str(modality)].append(int(name_to_idx[n]))
        return out

    def guess(c: str) -> str:
        s = c.lower()
        if "cgm" in s:
            return "cgm"
        if ("meal" in s) or ("carb" in s):
            return "meal"
        if "bolus" in s:
            return "bolus"
        if "basal" in s:
            return "basal"
        if ("activity" in s) or ("exercise" in s):
            return "activity"
        return "other"

    out: Dict[str, List[int]] = {}
    for i, c in enumerate(cols):
        m = guess(c)
        out.setdefault(m, []).append(int(i))
    return out


def build_players(
    feature_cols: Sequence[str],
    history_len: int,
    windows_steps_by_modality: Mapping[str, Sequence[Sequence[int]]],
    modality_channels: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[Player], np.ndarray]:
    """
    Returns:
      - players: list of Player (P)
      - fixed_mask: boolean mask [history_len, C] for indices NOT covered by any player
    """
    cols = list(feature_cols)
    modalities = infer_modalities(cols, modality_channels=modality_channels)
    players: List[Player] = []

    C = len(cols)
    covered = np.zeros((history_len, C), dtype=bool)
    for modality, win_list in windows_steps_by_modality.items():
        ch = modalities.get(modality, [])
        if not ch:
            continue
        for (a, b) in win_list:
            ts = window_to_slice(history_len, int(a), int(b))
            if ts.start == ts.stop:
                continue
            name = f"{modality}[{int(a)},{int(b)}]"
            ch_names = tuple(cols[i] for i in ch)
            p = Player(
                name=name,
                modality=str(modality),
                window_steps=(int(a), int(b)),
                time_slice=ts,
                channel_idxs=tuple(int(i) for i in ch),
                channel_names=ch_names,
            )
            players.append(p)
            covered[ts, ch] = True

    fixed_mask = ~covered
    return players, fixed_mask


def apply_fixed_mask_inplace(Xk: torch.Tensor, x_orig: torch.Tensor, fixed_mask: np.ndarray) -> None:
    """
    Xk: [K, T, C]
    x_orig: [T, C]
    fixed_mask: bool [T, C]
    """
    if fixed_mask.size == 0:
        return
    T, C = x_orig.shape
    flat_mask = torch.as_tensor(fixed_mask.reshape(-1), dtype=torch.bool, device=Xk.device)
    x_flat = x_orig.reshape(-1)
    X_flat = Xk.reshape(Xk.shape[0], -1)
    X_flat[:, flat_mask] = x_flat[flat_mask]


def apply_player_inplace(Xk: torch.Tensor, x_orig: torch.Tensor, player: Player) -> None:
    """
    Overwrite player block in Xk with original values.

    Xk: [K, T, C]
    x_orig: [T, C]
    """
    ts = player.time_slice
    ch = list(player.channel_idxs)
    if ts.start == ts.stop or not ch:
        return
    Xk[:, ts, ch] = x_orig[ts, :][:, ch]


def _forward_model(model: torch.nn.Module, x: torch.Tensor, pred_len: int) -> torch.Tensor:
    try:
        return model(x, pred_len)
    except TypeError:
        return model(x)


def predict_mean_horizons(
    model,
    Xk: torch.Tensor,
    pred_len: int,
    horizon_idxs: Sequence[int],
    device: torch.device,
    microbatch_size: int,
    input_layout: str = "BTC",
) -> np.ndarray:
    """
    Evaluate the model on a batch Xk and return the mean prediction over the batch
    for selected horizon indices.

    Returns: np.ndarray [H]
    """
    if input_layout == "BCT":
        Xk = Xk.permute(0, 2, 1).contiguous()
    elif input_layout != "BTC":
        raise ValueError("input_layout must be BTC or BCT")

    H = len(horizon_idxs)
    preds_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, Xk.shape[0], microbatch_size):
            xb = Xk[s : s + microbatch_size].to(device)
            if isinstance(model, torch.nn.Module):
                out = _forward_model(model, xb, pred_len)
                out_h = out[:, list(horizon_idxs)]
                preds_chunks.append(out_h.detach().cpu())
            else:
                # Fallback to predict() API (CPU)
                xb_np = xb.detach().cpu().numpy()
                out_np = model.predict(xb_np, pred_len)
                out_np = out_np[:, list(horizon_idxs)]
                preds_chunks.append(torch.as_tensor(out_np))
    preds = torch.cat(preds_chunks, dim=0) if preds_chunks else torch.zeros((0, H))
    mean = preds.mean(dim=0)
    return mean.cpu().numpy()
