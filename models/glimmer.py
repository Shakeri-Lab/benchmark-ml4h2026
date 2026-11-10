from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib.util
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import nn


def _add_glimmer_to_path():
    root = Path(__file__).resolve().parent.parent / "third_party" / "glimmer_upstream"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_upstream_module(module_name: str, *, subdir: str | None = "models"):
    root = _add_glimmer_to_path()
    if subdir:
        mod_path = root / subdir / f"{module_name}.py"
    else:
        mod_path = root / f"{module_name}.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"GLIMMER module not found: {mod_path}")
    spec = importlib.util.spec_from_file_location(f"glimmer_{module_name}", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GLIMMER module '{module_name}'")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_tf():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - tested via import guard
        raise ImportError(
            "TensorFlow is required for GLIMMER. Please install tensorflow-cpu (or tensorflow)."
        ) from exc
    return tf


class GlimmerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        history_len: int,
        pred_len: int,
        *,
        variant: str = "transformer",
        univariate: bool = True,
        context_len: Optional[int] = None,
        num_blocks: int = 1,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
        key_dim: int = 16,
        activation: str = "relu",
        loss_weights: Optional[list[float]] = None,
        ma_window_size: int = 200,
        epochs: int = 30,
        batch_size: int = 48,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        seed: int = 42,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.context_len = int(context_len) if context_len is not None else int(history_len)
        self.univariate = bool(univariate)
        self.variant = str(variant).lower()
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.key_dim = int(key_dim)
        self.activation = str(activation)
        self.loss_weights = loss_weights
        self.ma_window_size = int(ma_window_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.optimizer_name = str(optimizer)
        self.seed = int(seed)

        # GLIMMER is trained in prefit; no torch parameters to update
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.feature_cols: Optional[list[str]] = None
        self._feat_idx: dict[str, Any] = {}
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

        _add_glimmer_to_path()
        tf = _require_tf()
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self._tf = tf
        effective_dim = 1 if self.univariate else int(input_dim)
        self._model = self._build_tf_model(input_dim=effective_dim)

    def _build_tf_model(self, input_dim: int):
        tf = self._tf
        glimmer_cfg = _load_upstream_module("config", subdir=None)
        build_transformer = _load_upstream_module("transformer_model", subdir="models").build_model  # type: ignore[attr-defined]
        build_cnn_lstm = _load_upstream_module("cnn_lstm_model", subdir="models").build_model  # type: ignore[attr-defined]

        if self.variant not in {"transformer", "cnn_lstm"}:
            raise ValueError(f"Unknown GLIMMER variant '{self.variant}'")

        # Patch global config to match requested activation.
        glimmer_cfg.TransformerConfig.ACTIVATION = self.activation
        glimmer_cfg.GlimmerLSTMConfig.ACTIVATION = self.activation

        cfg = SimpleNamespace(
            NUM_BLOCKS=self.num_blocks,
            NUM_HEADS=self.num_heads,
            FF_DIM=self.ff_dim,
            DROPOUT=self.dropout,
            KEY_DIM=self.key_dim,
            ACTIVATION=self.activation,
        )

        input_shape = (self.context_len, int(input_dim))
        if self.variant == "transformer":
            model = build_transformer(input_shape=input_shape, output_dim=self.pred_len, config=cfg)
        else:
            model = build_cnn_lstm(input_shape=input_shape, output_dim=self.pred_len, config=glimmer_cfg.GlimmerLSTMConfig)

        return model

    def _infer_feature_indices(self, input_dim: int):
        if self._feat_idx:
            return
        cols = [c.lower() for c in (self.feature_cols or [])]
        def _idx(names):
            for name in names:
                if name in cols:
                    return cols.index(name)
            return None

        glucose_idx = _idx(["cgm", "glucose"]) or (input_dim - 1)
        basal_idx = _idx(["basal", "basal_rate"])
        carbs_idx = _idx(["carbs", "meal"])

        bolus_indices = []
        for name in ("bolus", "bolus_short", "bolus_long"):
            if name in cols:
                bolus_indices.append(cols.index(name))
        if not bolus_indices and input_dim >= 3:
            bolus_indices = [min(2, input_dim - 1)]

        self._feat_idx = {
            "glucose": glucose_idx,
            "basal": basal_idx,
            "carbs": carbs_idx,
            "bolus": bolus_indices,
        }

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.context_len:
            x = x[:, -self.context_len :, :]
        if self.univariate:
            return x[..., :1]
        return x

    def _moving_average(self, series: np.ndarray) -> np.ndarray:
        window = int(self.ma_window_size)
        if window <= 1:
            return series.copy()
        if series.shape[-1] < window:
            return np.zeros_like(series)
        kernel = np.ones(window, dtype=series.dtype) / float(window)
        ma = np.convolve(series, kernel, mode="valid")
        pad = np.zeros(window - 1, dtype=series.dtype)
        return np.concatenate((pad, ma), axis=0)

    def _build_features(self, x: torch.Tensor) -> np.ndarray:
        x = x.detach().cpu()
        if x.shape[1] < self.context_len:
            # pad on the left with the first value to reach context_len
            pad_len = self.context_len - x.shape[1]
            pad = x[:, :1, :].repeat(1, pad_len, 1)
            x = torch.cat([pad, x], dim=1)
        x_np = x.numpy().astype(np.float32)
        b, t, c = x_np.shape
        self._infer_feature_indices(c)

        idx = self._feat_idx
        glucose = x_np[:, :, idx["glucose"]]
        basal = np.zeros_like(glucose)
        carbs = np.zeros_like(glucose)
        bolus = np.zeros_like(glucose)

        if idx["basal"] is not None and idx["basal"] < c:
            basal = x_np[:, :, idx["basal"]]
        if idx["carbs"] is not None and idx["carbs"] < c:
            carbs = x_np[:, :, idx["carbs"]]
        bolus_idxs = idx.get("bolus") or []
        valid_bolus = [i for i in bolus_idxs if i is not None and i < c]
        if valid_bolus:
            bolus = np.sum(x_np[:, :, valid_bolus], axis=-1)

        ma_200 = np.stack([self._moving_average(glucose[i]) for i in range(b)], axis=0)

        glucose_class = np.zeros_like(glucose)
        glucose_class[glucose < 70.0] = 0.0
        glucose_class[(glucose >= 70.0) & (glucose <= 180.0)] = 1.0
        glucose_class[glucose > 180.0] = 2.0

        feats = np.stack([basal, carbs, bolus, glucose, ma_200, glucose_class], axis=-1)
        if feats.shape[1] > self.context_len:
            feats = feats[:, -self.context_len :, :]
        return feats

    def prefit(self, train_loader: Iterable, val_loader: Optional[Iterable] = None):
        tf = self._tf

        loss_weights = self.loss_weights or [1.0, 4.0, 2.5]

        def loss_fn(y_true, y_pred):
            mae = tf.keras.backend.abs(y_pred - y_true)
            normal_mask = tf.logical_and(y_true >= 70.0, y_true <= 180.0)
            hypo_mask = y_true < 70.0
            hyper_mask = y_true > 180.0
            normal_error = mae * tf.cast(normal_mask, tf.keras.backend.floatx()) * loss_weights[0]
            hypo_error = mae * tf.cast(hypo_mask, tf.keras.backend.floatx()) * loss_weights[1]
            hyper_error = mae * tf.cast(hyper_mask, tf.keras.backend.floatx()) * loss_weights[2]
            return tf.keras.backend.mean(normal_error + hypo_error + hyper_error)

        optimizer = tf.keras.optimizers.get({"class_name": self.optimizer_name, "config": {"learning_rate": self.learning_rate}})
        self._model.compile(optimizer=optimizer, loss=loss_fn)

        # Infer feature columns from dataset if available
        if hasattr(train_loader, "dataset"):
            self.feature_cols = getattr(train_loader.dataset, "feature_cols", None)

        # Fit standard scaler on training features
        sum_x = None
        sum_x2 = None
        count = 0
        for batch in train_loader:
            X = batch[0]
            feats = self._build_features(X)
            flat = feats.reshape(-1, feats.shape[-1])
            if sum_x is None:
                sum_x = flat.sum(axis=0)
                sum_x2 = (flat ** 2).sum(axis=0)
            else:
                sum_x += flat.sum(axis=0)
                sum_x2 += (flat ** 2).sum(axis=0)
            count += flat.shape[0]
        if count == 0:
            raise ValueError("No data available to fit GLIMMER scaler.")
        mean = sum_x / count
        var = np.maximum(sum_x2 / count - mean ** 2, 1e-12)
        std = np.sqrt(var)
        std[std == 0] = 1.0
        self._scaler_mean = mean.astype(np.float32)
        self._scaler_std = std.astype(np.float32)

        def _scale(feats: np.ndarray) -> np.ndarray:
            return (feats - self._scaler_mean) / self._scaler_std

        def gen(loader):
            for batch in loader:
                X, Y = batch[0], batch[1]
                feats = self._build_features(X)
                feats = _scale(feats)
                Y = Y.cpu().numpy().astype(np.float32)
                yield feats, Y

        steps = len(train_loader)
        val_data = None
        val_steps = None
        if val_loader is not None:
            val_data = gen(val_loader)
            val_steps = len(val_loader)

        self._model.fit(
            gen(train_loader),
            steps_per_epoch=steps,
            epochs=self.epochs,
            validation_data=val_data,
            validation_steps=val_steps,
            verbose=1,
        )

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f"pred_len mismatch: got {pred_len}, expected {self.pred_len}")

        feats = self._build_features(x)
        if self._scaler_mean is not None and self._scaler_std is not None:
            feats = (feats - self._scaler_mean) / self._scaler_std
        preds = self._model(feats, training=False).numpy().astype(np.float32)
        preds = torch.tensor(preds, dtype=torch.float32, device=x.device)
        return preds
