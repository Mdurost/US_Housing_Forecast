"""Model, trainer, and checkpoint helpers."""

from __future__ import annotations

import io
import logging
import random
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_optimizer import Ranger
from torchmetrics import Metric

from housing_forecast.config import Config

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def resolve_accelerator(accelerator: str) -> str:
    if accelerator == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return accelerator


def build_model(training: TimeSeriesDataSet, config: Config) -> TemporalFusionTransformer:
    m = config.model
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=m.learning_rate,
        hidden_size=m.hidden_size,
        attention_head_size=m.attention_head_size,
        dropout=m.dropout,
        hidden_continuous_size=m.hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=m.log_interval,
        optimizer=Ranger,
        reduce_on_plateau_patience=m.reduce_on_plateau_patience,
    )
    logger.info("Number of parameters in network: %.1fk", model.size() / 1e3)
    return model


def build_trainer(config: Config, extra_loggers: list | None = None) -> tuple[pl.Trainer, ModelCheckpoint]:
    t = config.trainer
    checkpoint = ModelCheckpoint(
        dirpath=config.resolve(config.paths.checkpoint_dir),
        filename="best-tft-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    # TensorBoard must stay first: pytorch-forecasting draws its figures on the first logger.
    loggers = [TensorBoardLogger(config.resolve(config.paths.log_dir)), *(extra_loggers or [])]
    trainer = pl.Trainer(
        max_epochs=t.max_epochs,
        accelerator=resolve_accelerator(t.accelerator),
        enable_model_summary=True,
        gradient_clip_val=t.gradient_clip_val,
        limit_train_batches=t.limit_train_batches,
        callbacks=[
            LearningRateMonitor(),
            EarlyStopping(
                monitor="val_loss",
                min_delta=t.early_stopping_min_delta,
                patience=t.early_stopping_patience,
                mode="min",
            ),
            checkpoint,
        ],
        logger=loggers,
    )
    return trainer, checkpoint


def _relocate_metrics(value: Any, device: torch.device) -> None:
    """Point pickled torchmetrics objects (loss, logging metrics) at ``device``.

    They remember the device they were trained on; a GPU-trained checkpoint otherwise
    fails to load on a CPU-only machine with "Torch not compiled with CUDA enabled".
    """
    if isinstance(value, torch.nn.Module):
        for module in value.modules():
            if isinstance(module, Metric):
                module._device = device
    elif isinstance(value, dict):
        for item in value.values():
            _relocate_metrics(item, device)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _relocate_metrics(item, device)


def load_model(path: Path | str, map_location: str = "cpu") -> TemporalFusionTransformer:
    # Our own checkpoints pickle dataset parameters and metric objects, so full unpickling is required.
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    _relocate_metrics(checkpoint.get("hyper_parameters", {}), torch.device(map_location))
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    model = TemporalFusionTransformer.load_from_checkpoint(buffer, map_location=map_location, weights_only=False)
    model.eval()
    return model
