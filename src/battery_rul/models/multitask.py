"""Multi-task sequence model: one encoder, three heads.

Architecture
------------
::

    window of scaled features  (B, W, F)
              |
      shared temporal encoder            Transformer encoder (default)
      (pre-norm, sinusoidal positions,   or an LSTM/GRU stack
       attention pooling)
              |
        shared representation  (B, d)
         /          |          \\
    RUL head    SOH head    risk head
    (linear)    (sigmoid-   (logit)
                 free linear)

Why share
---------
The three tasks are three views of one latent process. Remaining life, present
state of health and near-term crossing probability are all functions of the same
degradation trajectory, so a shared encoder gets three supervision signals for
one set of parameters — which matters a great deal at this cohort size, where
the binding constraint is data rather than capacity. The risk head in particular
sees very few positives on its own; sharing an encoder trained partly by the two
dense regression targets is what makes it learnable at all.

Loss
----
``total = w_rul * L_rul + w_soh * L_soh + w_risk * L_risk``

with each component logged separately, because a combined loss that improves
while one component silently degrades is the standard multi-task failure and it
is invisible in the total. RUL is divided by ``multitask.rul_scale`` before its
loss so that a target measured in hundreds of cycles does not swamp an SOH target
measured in units of one. Focal loss is available for the risk head when the
positive rate is low enough that plain cross-entropy is dominated by easy
negatives.

Windowing and warm-up
---------------------
Identical to the Milestone 1 sequence models: a window ending at cycle *k* holds
cycles ``[k-w+1, k]`` of one cell, is labelled with that cell's targets *at k*,
and never crosses a cell boundary. Rows without a full window are returned as
NaN, not silently dropped, so coverage stays visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from battery_rul.config import ExperimentConfig, MultiTaskConfig
from battery_rul.models.neural import resolve_device
from battery_rul.utils.io import load_pickle, save_pickle
from battery_rul.utils.logging import get_logger
from battery_rul.utils.seed import seed_everything, torch_generator

logger = get_logger(__name__)

__all__ = ["MultiTaskDataset", "MultiTaskPrediction", "MultiTaskSequenceModel"]


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MultiTaskDataset:
    """Windowed tensors for three aligned targets."""

    X: np.ndarray  # (n, window, n_features)
    y_rul: np.ndarray
    y_soh: np.ndarray
    y_risk: np.ndarray
    battery_ids: np.ndarray
    cycle_index: np.ndarray
    feature_names: list[str]
    #: Index into the *source* frame of each window's final row.
    row_positions: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def window(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])


def make_multitask_windows(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    window: int,
    stride: int = 1,
    rul: np.ndarray | None = None,
    soh: np.ndarray | None = None,
    risk: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> MultiTaskDataset:
    """Build per-cell sliding windows aligned to the last row's targets.

    Targets may be ``None`` (inference) or contain NaN (a row with no valid label
    for that task); NaN is preserved and masked out of the loss rather than
    imputed, so a missing risk label costs that task one sample instead of
    teaching it a fabricated one.
    """
    n = len(frame)
    if len(values) != n:
        raise ValueError(f"Row-count mismatch: frame={n}, values={len(values)}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    def _column(array: np.ndarray | None) -> np.ndarray:
        return np.full(n, np.nan) if array is None else np.asarray(array, dtype=float)

    rul_v, soh_v, risk_v = _column(rul), _column(soh), _column(risk)

    xs: list[np.ndarray] = []
    r, s, k = [], [], []
    bids: list[str] = []
    cycles: list[int] = []
    positions_out: list[int] = []

    positions = np.arange(n)
    battery_values = frame["battery_id"].to_numpy()
    cycle_values = frame["cycle_index"].to_numpy()
    too_short: list[str] = []

    for battery_id in pd.unique(battery_values):
        idx = positions[battery_values == battery_id]
        idx = idx[np.argsort(cycle_values[idx], kind="stable")]
        if idx.size < window:
            too_short.append(str(battery_id))
            continue
        for end in range(window - 1, idx.size, stride):
            sl = idx[end - window + 1 : end + 1]
            last = int(idx[end])
            xs.append(values[sl])
            r.append(rul_v[last])
            s.append(soh_v[last])
            k.append(risk_v[last])
            bids.append(str(battery_id))
            cycles.append(int(cycle_values[last]))
            positions_out.append(last)

    if too_short:
        logger.warning(
            "%d cell(s) shorter than window=%d produced no multi-task windows: %s",
            len(too_short),
            window,
            too_short,
        )
    if not xs:
        raise ValueError(f"No windows produced with window={window}")

    return MultiTaskDataset(
        X=np.asarray(xs, dtype=np.float32),
        y_rul=np.asarray(r, dtype=np.float32),
        y_soh=np.asarray(s, dtype=np.float32),
        y_risk=np.asarray(k, dtype=np.float32),
        battery_ids=np.asarray(bids),
        cycle_index=np.asarray(cycles, dtype=np.int32),
        feature_names=list(feature_names or []),
        row_positions=np.asarray(positions_out, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions — no parameters to overfit at this data size."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        pe = cast(Tensor, self.pe)
        return x + pe[:, : x.size(1), :]


class MultiTaskNetwork(nn.Module):
    """Shared encoder with three task heads."""

    def __init__(self, n_features: int, cfg: MultiTaskConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_kind = cfg.encoder

        if cfg.encoder == "transformer":
            self.input_proj = nn.Linear(n_features, cfg.d_model)
            self.pos = _PositionalEncoding(cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=cfg.num_layers, enable_nested_tensor=False
            )
            self.attn_pool = nn.Linear(cfg.d_model, 1)
            representation = cfg.d_model
        else:
            rnn_cls = nn.LSTM if cfg.encoder == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            representation = cfg.hidden_size * 2  # last state ++ sequence mean

        self.shared_norm = nn.LayerNorm(representation)
        self.shared_dropout = nn.Dropout(cfg.dropout)
        self.rul_head = self._head(representation, cfg)
        self.soh_head = self._head(representation, cfg)
        self.risk_head = self._head(representation, cfg)
        #: Kept for diagnostics only — see the explainability docs on why
        #: attention weights are not treated as explanations.
        self.last_attention: Tensor | None = None

    @staticmethod
    def _head(representation: int, cfg: MultiTaskConfig) -> nn.Module:
        return nn.Sequential(
            nn.Linear(representation, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def encode(self, x: Tensor) -> Tensor:
        if self.encoder_kind == "transformer":
            h = self.pos(self.input_proj(x))
            h = self.encoder(h)
            weights = torch.softmax(self.attn_pool(h), dim=1)
            self.last_attention = weights.detach()
            pooled = (h * weights).sum(dim=1)
        else:
            out, _ = self.rnn(x)
            pooled = torch.cat([out[:, -1, :], out.mean(dim=1)], dim=-1)
        return self.shared_dropout(self.shared_norm(pooled))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        shared = self.encode(x)
        return {
            "rul": self.rul_head(shared).squeeze(-1),
            "soh": self.soh_head(shared).squeeze(-1),
            "risk_logit": self.risk_head(shared).squeeze(-1),
            "representation": shared,
        }


def _focal_bce(logits: Tensor, targets: Tensor, gamma: float, pos_weight: Tensor) -> Tensor:
    """Focal binary cross-entropy: down-weights the easy negatives."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    return (((1 - p_t) ** gamma) * bce).mean()


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MultiTaskPrediction:
    """Row-aligned outputs; NaN where no full window exists."""

    rul: np.ndarray
    soh: np.ndarray
    risk_probability: np.ndarray
    scoreable: np.ndarray

    def to_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "battery_id": frame["battery_id"].to_numpy(),
                "cycle_index": frame["cycle_index"].to_numpy(),
                "rul_pred": self.rul,
                "soh_pred": self.soh,
                "risk_probability_raw": self.risk_probability,
                "scoreable": self.scoreable,
            }
        )


@dataclass
class MultiTaskSequenceModel:
    """Trainable, serialisable multi-task model.

    Deliberately *not* a :class:`~battery_rul.models.base.BaseModel`: that
    interface returns a single array, and forcing three heads through it would
    mean either three forward passes or an out-of-band channel for the other two
    outputs. The Milestone 1 zoo and its evaluation paths are untouched.
    """

    cfg: ExperimentConfig
    network: MultiTaskNetwork | None = None
    device: torch.device | None = None
    fitted: bool = False
    feature_names: list[str] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    fit_metadata: dict[str, Any] = field(default_factory=dict)
    best_state: dict[str, Any] | None = field(default=None, repr=False)
    #: The architecture/window settings this instance was actually trained with.
    #: A loaded model must window and rescale exactly as it was trained, even if
    #: the runtime configuration has since moved on — silently using the runtime
    #: window would feed the encoder a sequence length it never saw.
    trained_config: MultiTaskConfig | None = None

    @property
    def mt(self) -> MultiTaskConfig:
        return self.trained_config or self.cfg.multitask

    # -- data ---------------------------------------------------------------
    def windows(
        self, frame: pd.DataFrame, values: np.ndarray, *, with_targets: bool = True
    ) -> MultiTaskDataset:
        rul = soh = risk = None
        if with_targets:
            rul = frame[self.cfg.target.name].to_numpy(dtype=float)
            soh = frame[self.cfg.soh.target_name].to_numpy(dtype=float)
            risk = pd.to_numeric(frame[self.cfg.risk.target_name], errors="coerce").to_numpy(
                dtype=float
            )
        return make_multitask_windows(
            frame,
            values,
            window=self.mt.window,
            stride=self.mt.stride,
            rul=rul,
            soh=soh,
            risk=risk,
            feature_names=self.feature_names,
        )

    # -- loss ---------------------------------------------------------------
    def _regression_loss(self, kind: str) -> Any:
        if kind == "mse":
            return nn.MSELoss(reduction="none")
        if kind == "mae":
            return nn.L1Loss(reduction="none")
        return nn.HuberLoss(reduction="none", delta=1.0)

    def _compute_losses(
        self, outputs: dict[str, Tensor], batch: dict[str, Tensor], pos_weight: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        mt = self.mt
        components: dict[str, float] = {}
        total = torch.zeros((), device=outputs["rul"].device)

        for name, weight, kind in (
            ("rul", mt.rul_weight, mt.rul_loss),
            ("soh", mt.soh_weight, mt.soh_loss),
        ):
            target = batch[name]
            mask = torch.isfinite(target)
            if weight <= 0 or not bool(mask.any()):
                components[f"{name}_loss"] = float("nan")
                continue
            criterion = self._regression_loss(kind)
            loss = criterion(outputs[name][mask], target[mask]).mean()
            components[f"{name}_loss"] = float(loss.item())
            total = total + weight * loss

        target = batch["risk"]
        mask = torch.isfinite(target)
        if mt.risk_weight > 0 and bool(mask.any()):
            logits = outputs["risk_logit"][mask]
            labels = target[mask]
            if mt.risk_loss == "focal":
                loss = _focal_bce(logits, labels, mt.focal_gamma, pos_weight)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
            components["risk_loss"] = float(loss.item())
            total = total + mt.risk_weight * loss
        else:
            components["risk_loss"] = float("nan")

        components["total_loss"] = float(total.item())
        return total, components

    # -- fit ----------------------------------------------------------------
    def fit(
        self,
        train_frame: pd.DataFrame,
        train_values: np.ndarray,
        *,
        val_frame: pd.DataFrame | None = None,
        val_values: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> MultiTaskSequenceModel:
        mt = self.cfg.multitask
        self.trained_config = mt
        seed_everything(mt.seed)
        self.device = resolve_device(mt.device)
        self.feature_names = list(feature_names or [])

        train_ds = self.windows(train_frame, train_values)
        val_ds = None
        if val_frame is not None and val_values is not None and len(val_frame):
            try:
                val_ds = self.windows(val_frame, val_values)
            except ValueError as exc:
                logger.warning("No validation windows (%s); early stopping disabled", exc)

        self.network = MultiTaskNetwork(train_ds.n_features, mt).to(self.device)
        n_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        # Class weighting rather than oversampling: duplicating rows of a time
        # series would put near-identical windows in the same batch and inflate
        # every metric computed afterwards.
        risk = train_ds.y_risk[np.isfinite(train_ds.y_risk)]
        n_pos = float(np.sum(risk == 1.0))
        n_neg = float(np.sum(risk == 0.0))
        weight = (n_neg / n_pos) if (mt.risk_weight > 0 and n_pos > 0) else 1.0
        pos_weight = torch.tensor(
            weight if self.cfg.risk.class_weight_balanced else 1.0, device=self.device
        )

        tensors = self._tensors(train_ds)
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=min(int(mt.batch_size), len(train_ds)),
            shuffle=True,
            generator=torch_generator(mt.seed),
        )
        optimiser = torch.optim.AdamW(
            self.network.parameters(), lr=mt.learning_rate, weight_decay=mt.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=max(mt.early_stopping_patience // 3, 2)
        )

        val_tensors = None
        if val_ds is not None:
            val_tensors = [t.to(self.device) for t in self._tensors(val_ds)]

        history: dict[str, list[float]] = {
            key: [] for key in ("total_loss", "rul_loss", "soh_loss", "risk_loss", "val_total_loss")
        }
        best_loss, best_epoch = float("inf"), -1

        for epoch in range(int(mt.epochs)):
            self.network.train()
            epoch_components: dict[str, list[float]] = {}
            for xb, rb, sb, kb in loader:
                xb = xb.to(self.device)
                batch = {
                    "rul": rb.to(self.device) / mt.rul_scale,
                    "soh": sb.to(self.device),
                    "risk": kb.to(self.device),
                }
                optimiser.zero_grad(set_to_none=True)
                outputs = self.network(xb)
                loss, components = self._compute_losses(outputs, batch, pos_weight)
                loss.backward()
                if mt.grad_clip:
                    nn.utils.clip_grad_norm_(self.network.parameters(), mt.grad_clip)
                optimiser.step()
                for key, value in components.items():
                    epoch_components.setdefault(key, []).append(value)

            for key in ("total_loss", "rul_loss", "soh_loss", "risk_loss"):
                values = [v for v in epoch_components.get(key, []) if np.isfinite(v)]
                history[key].append(float(np.mean(values)) if values else float("nan"))

            if val_tensors is not None:
                self.network.eval()
                with torch.no_grad():
                    outputs = self.network(val_tensors[0])
                    batch = {
                        "rul": val_tensors[1] / mt.rul_scale,
                        "soh": val_tensors[2],
                        "risk": val_tensors[3],
                    }
                    _, components = self._compute_losses(outputs, batch, pos_weight)
                val_loss = components["total_loss"]
            else:
                val_loss = history["total_loss"][-1]
            history["val_total_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch = val_loss, epoch
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()
                }
            elif epoch - best_epoch >= mt.early_stopping_patience:
                logger.info("multitask: early stop at epoch %d (best %d)", epoch, best_epoch)
                break

            if epoch % 10 == 0:
                logger.info(
                    "multitask epoch %3d  total=%.4f  rul=%.4f  soh=%.4f  risk=%.4f  val=%.4f",
                    epoch,
                    history["total_loss"][-1],
                    history["rul_loss"][-1],
                    history["soh_loss"][-1],
                    history["risk_loss"][-1],
                    val_loss,
                )

        if self.best_state is not None:
            self.network.load_state_dict(self.best_state)

        self.fitted = True
        self.history = history
        self.fit_metadata = {
            "encoder": mt.encoder,
            "window": mt.window,
            "n_features": train_ds.n_features,
            "n_train_windows": len(train_ds),
            "n_val_windows": 0 if val_ds is None else len(val_ds),
            "n_parameters": int(n_params),
            "best_epoch": best_epoch,
            "best_val_total_loss": round(best_loss, 6),
            "epochs_run": len(history["total_loss"]),
            "risk_pos_weight": float(pos_weight.item()),
            "loss_weights": {
                "rul": mt.rul_weight,
                "soh": mt.soh_weight,
                "risk": mt.risk_weight,
            },
            "rul_scale": mt.rul_scale,
            "device": str(self.device),
            "feature_names": list(self.feature_names),
        }
        logger.info(
            "multitask fitted: %s encoder, %d windows, %d params, best epoch %d (val=%.4f)",
            mt.encoder,
            len(train_ds),
            n_params,
            best_epoch,
            best_loss,
        )
        return self

    def _tensors(self, dataset: MultiTaskDataset) -> list[Tensor]:
        return [
            torch.from_numpy(dataset.X).float(),
            torch.from_numpy(dataset.y_rul).float(),
            torch.from_numpy(dataset.y_soh).float(),
            torch.from_numpy(dataset.y_risk).float(),
        ]

    # -- predict -------------------------------------------------------------
    def predict(self, frame: pd.DataFrame, values: np.ndarray) -> MultiTaskPrediction:
        """Row-aligned three-task prediction. NaN where the window is incomplete."""
        if not self.fitted or self.network is None:
            raise RuntimeError("MultiTaskSequenceModel is not fitted. Call fit() first.")

        n = len(frame)
        rul = np.full(n, np.nan)
        soh = np.full(n, np.nan)
        risk = np.full(n, np.nan)
        scoreable = np.zeros(n, dtype=bool)

        try:
            dataset = self.windows(frame, values, with_targets=False)
        except ValueError as exc:
            logger.warning("multitask: cannot window prediction data (%s)", exc)
            return MultiTaskPrediction(rul, soh, risk, scoreable)

        self.network.eval()
        outputs: list[dict[str, np.ndarray]] = []
        chunk = 256
        with torch.no_grad():
            for start in range(0, len(dataset), chunk):
                xb = torch.from_numpy(dataset.X[start : start + chunk]).float().to(self.device)
                out = self.network(xb)
                outputs.append(
                    {
                        "rul": out["rul"].cpu().numpy(),
                        "soh": out["soh"].cpu().numpy(),
                        "risk": torch.sigmoid(out["risk_logit"]).cpu().numpy(),
                    }
                )

        positions = dataset.row_positions
        rul[positions] = np.concatenate([o["rul"] for o in outputs]) * self.mt.rul_scale
        soh[positions] = np.concatenate([o["soh"] for o in outputs])
        risk[positions] = np.concatenate([o["risk"] for o in outputs])
        scoreable[positions] = True

        # Physical post-processing, applied here and recorded as such: the raw
        # head output is unconstrained, and a negative remaining life or an SOH
        # of 3.0 is a model artifact rather than a state of the world.
        rul = np.where(np.isfinite(rul), np.clip(rul, 0.0, None), rul)
        soh = np.where(
            np.isfinite(soh),
            np.clip(soh, self.cfg.soh.plausible_min, self.cfg.soh.plausible_max),
            soh,
        )
        risk = np.where(np.isfinite(risk), np.clip(risk, 0.0, 1.0), risk)
        return MultiTaskPrediction(rul, soh, risk, scoreable)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist weights and configuration (not the live torch module graph)."""
        if not self.fitted or self.network is None:
            raise RuntimeError("Refusing to save an unfitted model")
        payload = {
            "state_dict": {k: v.cpu() for k, v in self.network.state_dict().items()},
            "multitask_config": self.mt.model_dump(mode="json"),
            "feature_names": list(self.feature_names),
            "fit_metadata": self.fit_metadata,
            "history": self.history,
            "n_features": self.fit_metadata.get("n_features"),
        }
        return save_pickle(payload, path)

    @classmethod
    def load(cls, path: str | Path, cfg: ExperimentConfig) -> MultiTaskSequenceModel:
        payload = load_pickle(path)
        required = {"state_dict", "multitask_config", "feature_names", "n_features"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"{path} is not a multi-task bundle; missing keys: {missing}")

        model = cls(cfg=cfg)
        model.feature_names = list(payload["feature_names"])
        model.fit_metadata = dict(payload.get("fit_metadata", {}))
        model.history = dict(payload.get("history", {}))
        model.device = resolve_device(cfg.multitask.device)
        mt = MultiTaskConfig(**payload["multitask_config"])
        model.trained_config = mt
        model.network = MultiTaskNetwork(int(payload["n_features"]), mt).to(model.device)
        model.network.load_state_dict(payload["state_dict"])
        model.network.eval()
        model.fitted = True
        return model
