"""Sequence models: LSTM, GRU and a Transformer encoder.

All three share one training loop (:class:`SequenceModel`) and differ only in
their encoder, so the comparison is apples-to-apples: same windows, same scaler,
same optimiser, same early-stopping rule, same seed.

Windowing contract
------------------
A window ending at cycle *k* holds cycles ``[k-w+1, k]`` of one cell and is
labelled ``RUL(k)``. Consequently the first ``w-1`` cycles of every cell cannot
be scored. :meth:`SequenceModel.predict` returns ``NaN`` for those rows rather
than dropping them, so predictions stay row-aligned with the tabular models and
the evaluator can compare like with like on the intersection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from battery_rul.features.sequences import make_sequences
from battery_rul.models.base import BaseModel, TrainingData, register_model
from battery_rul.utils.logging import get_logger
from battery_rul.utils.seed import seed_everything, torch_generator

logger = get_logger(__name__)

__all__ = ["GRUModel", "LSTMModel", "SequenceModel", "TransformerModel", "resolve_device"]


def resolve_device(preference: str) -> torch.device:
    """Pick a compute device, honouring an explicit request when it is usable."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class _RecurrentEncoder(nn.Module):
    """LSTM or GRU stack with a mean+last pooled head.

    Pooling both the final hidden state and the sequence mean is a small but
    consistently useful choice here: the last state captures the cell's current
    condition, the mean captures the trajectory it took to get there, and RUL
    depends on both.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        cell: str,
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        pooled = hidden_size * directions * 2
        self.head = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Dropout(dropout),
            nn.Linear(pooled, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.rnn(x)
        pooled = torch.cat([out[:, -1, :], out.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)


class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions.

    Learned embeddings would be the usual choice, but with only a few hundred
    windows they overfit immediately; the sinusoidal version adds no parameters.
    """

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


class _TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder with attention pooling."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm trains far more stably on tiny datasets
            activation="gelu",
        )
        # enable_nested_tensor is incompatible with norm_first and only warns;
        # disable it explicitly rather than emit a warning on every construction.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.attn_pool = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        weights = torch.softmax(self.attn_pool(h), dim=1)
        pooled = (h * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------
@dataclass
class SequenceModel(BaseModel):
    """Windowed neural regressor with early stopping and LR scheduling."""

    is_sequence: ClassVar[bool] = True
    network: nn.Module | None = None
    device: torch.device | None = None
    best_state: dict[str, Any] | None = field(default=None, repr=False)

    def _build_network(self, n_features: int) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- loss -------------------------------------------------------------
    def _loss_fn(self) -> nn.Module:
        train_cfg = self.cfg.models.training
        if train_cfg.loss == "mse":
            return nn.MSELoss()
        if train_cfg.loss == "mae":
            return nn.L1Loss()
        # Huber is the default: RUL residuals have heavy tails near end of life,
        # and squared error there drags the whole fit toward the last few cycles.
        return nn.HuberLoss(delta=train_cfg.huber_delta)

    # -- data -------------------------------------------------------------
    def _windows(self, data: TrainingData):
        seq_cfg = self.cfg.models.sequence
        return make_sequences(
            data.frame,
            data.X,
            data.y,
            window=int(self.params.get("window", seq_cfg.window)),
            stride=int(self.params.get("stride", seq_cfg.stride)),
            feature_names=data.feature_names,
        )

    # -- fit ---------------------------------------------------------------
    def fit(self, train: TrainingData, val: TrainingData | None = None) -> SequenceModel:
        train_cfg = self.cfg.models.training
        seed_everything(train_cfg.seed)
        self.device = resolve_device(train_cfg.device)

        train_batch = self._windows(train)
        val_batch = None
        if val is not None and not val.is_empty:
            try:
                val_batch = self._windows(val)
            except ValueError as exc:
                logger.warning(
                    "%s: no validation windows (%s); early stopping disabled", self.name, exc
                )

        n_features = train_batch.n_features
        self.network = self._build_network(n_features).to(self.device)
        n_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_batch.X).float(), torch.from_numpy(train_batch.y).float()
            ),
            batch_size=min(int(train_cfg.batch_size), len(train_batch)),
            shuffle=True,
            drop_last=False,
            num_workers=train_cfg.num_workers,
            generator=torch_generator(train_cfg.seed),
        )

        optimiser = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.params.get("learning_rate", train_cfg.learning_rate)),
            weight_decay=float(self.params.get("weight_decay", train_cfg.weight_decay)),
        )
        scheduler = self._build_scheduler(optimiser, train_cfg)
        criterion = self._loss_fn()

        val_tensors = None
        if val_batch is not None:
            val_tensors = (
                torch.from_numpy(val_batch.X).float().to(self.device),
                torch.from_numpy(val_batch.y).float().to(self.device),
            )

        best_loss = float("inf")
        best_epoch = -1
        patience = int(train_cfg.early_stopping_patience)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(int(train_cfg.epochs)):
            self.network.train()
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(self.network(xb), yb)
                loss.backward()
                if train_cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.network.parameters(), train_cfg.grad_clip)
                optimiser.step()
                running += float(loss.item()) * xb.size(0)

            train_loss = running / max(len(train_batch), 1)
            history["train_loss"].append(train_loss)
            history["lr"].append(float(optimiser.param_groups[0]["lr"]))

            if val_tensors is not None:
                self.network.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.network(val_tensors[0]), val_tensors[1]).item())
            else:
                val_loss = train_loss
            history["val_loss"].append(val_loss)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()
                }
            elif epoch - best_epoch >= patience:
                logger.info(
                    "%s: early stop at epoch %d (best %d, val_loss=%.4f)",
                    self.name,
                    epoch,
                    best_epoch,
                    best_loss,
                )
                break

            if epoch % 10 == 0 or epoch == train_cfg.epochs - 1:
                logger.info(
                    "%s epoch %3d/%d  train=%.4f  val=%.4f",
                    self.name,
                    epoch,
                    train_cfg.epochs,
                    train_loss,
                    val_loss,
                )

        if self.best_state is not None:
            self.network.load_state_dict(self.best_state)

        self.fitted = True
        self.train_history = history
        self.fit_metadata = {
            "n_train_rows": len(train),
            "n_train_windows": len(train_batch),
            "n_val_windows": 0 if val_batch is None else len(val_batch),
            "n_features": n_features,
            "feature_names": list(train.feature_names),
            "n_parameters": int(n_params),
            "window": train_batch.window,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_loss, 5),
            "epochs_run": len(history["train_loss"]),
            "device": str(self.device),
        }
        logger.info(
            "%s fitted: %d windows, %d params, best epoch %d (val=%.4f)",
            self.name,
            len(train_batch),
            n_params,
            best_epoch,
            best_loss,
        )
        return self

    @staticmethod
    def _build_scheduler(optimiser: torch.optim.Optimizer, train_cfg: Any) -> Any:
        if train_cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=train_cfg.epochs)
        if train_cfg.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode="min",
                factor=0.5,
                patience=max(train_cfg.early_stopping_patience // 3, 2),
            )
        return None

    # -- predict ------------------------------------------------------------
    def predict(self, data: TrainingData) -> np.ndarray:
        """Row-aligned predictions; ``NaN`` where no full window exists yet."""
        self._check_fitted()
        assert self.network is not None

        out = np.full(len(data), np.nan, dtype=float)
        try:
            batch = self._windows(data)
        except ValueError as exc:
            logger.warning("%s: cannot window prediction data (%s)", self.name, exc)
            return out

        self.network.eval()
        preds: list[np.ndarray] = []
        chunk = 512
        with torch.no_grad():
            for start in range(0, len(batch), chunk):
                xb = torch.from_numpy(batch.X[start : start + chunk]).float().to(self.device)
                preds.append(self.network(xb).cpu().numpy())
        values = np.concatenate(preds) if preds else np.empty(0)

        # Map each window back to the row holding its final cycle.
        lookup = {
            (str(b), int(c)): i
            for i, (b, c) in enumerate(zip(data.battery_ids, data.cycle_index, strict=True))
        }
        for value, battery_id, cycle in zip(
            values, batch.battery_ids, batch.cycle_index, strict=True
        ):
            row = lookup.get((str(battery_id), int(cycle)))
            if row is not None:
                out[row] = float(value)
        return out


@register_model("lstm")
class LSTMModel(SequenceModel):
    """Stacked LSTM — the standard recurrent baseline for battery prognostics."""

    def _build_network(self, n_features: int) -> nn.Module:
        s = self.cfg.models.sequence
        return _RecurrentEncoder(
            n_features=n_features,
            hidden_size=int(self.params.get("hidden_size", s.hidden_size)),
            num_layers=int(self.params.get("num_layers", s.num_layers)),
            dropout=float(self.params.get("dropout", s.dropout)),
            bidirectional=bool(self.params.get("bidirectional", s.bidirectional)),
            cell="lstm",
        )


@register_model("gru")
class GRUModel(SequenceModel):
    """GRU — fewer gates than the LSTM, which usually helps at this sample size."""

    def _build_network(self, n_features: int) -> nn.Module:
        s = self.cfg.models.sequence
        return _RecurrentEncoder(
            n_features=n_features,
            hidden_size=int(self.params.get("hidden_size", s.hidden_size)),
            num_layers=int(self.params.get("num_layers", s.num_layers)),
            dropout=float(self.params.get("dropout", s.dropout)),
            bidirectional=bool(self.params.get("bidirectional", s.bidirectional)),
            cell="gru",
        )


@register_model("transformer")
class TransformerModel(SequenceModel):
    """Pre-norm Transformer encoder with attention pooling."""

    def _build_network(self, n_features: int) -> nn.Module:
        s = self.cfg.models.sequence
        d_model = int(self.params.get("d_model", s.d_model))
        nhead = int(self.params.get("nhead", s.nhead))
        if d_model % nhead:
            d_model = max(nhead, (d_model // nhead) * nhead)
            logger.warning("Adjusted d_model to %d so it divides nhead=%d", d_model, nhead)
        return _TransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(self.params.get("num_layers", s.num_layers)),
            dim_feedforward=int(self.params.get("dim_feedforward", s.dim_feedforward)),
            dropout=float(self.params.get("dropout", s.dropout)),
        )
