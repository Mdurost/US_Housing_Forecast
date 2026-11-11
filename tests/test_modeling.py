import lightning.pytorch as pl
import torch
from torchmetrics import Metric

from housing_forecast.dataset import build_training_dataset, train_dataloader
from housing_forecast.features import fit_features
from housing_forecast.modeling import build_model, load_model


def test_load_model_ignores_training_device_of_metrics(raw_panel, config, tmp_path):
    """A GPU-trained checkpoint must load on CPU-only machines (Docker, CI)."""
    frame, spec = fit_features(raw_panel, config)
    training = build_training_dataset(frame, spec, config)
    model = build_model(training, config)
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader(training, config))
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)

    # Simulate a checkpoint saved from a CUDA run: metric objects remember the cuda device.
    checkpoint = torch.load(path, weights_only=False)
    for value in checkpoint["hyper_parameters"].values():
        if isinstance(value, torch.nn.Module):
            for module in value.modules():
                if isinstance(module, Metric):
                    module._device = torch.device("cuda")
    torch.save(checkpoint, path)

    loaded = load_model(path)
    assert loaded.loss.device == torch.device("cpu")
    assert not loaded.training
    assert loaded.dataset_parameters["time_varying_unknown_reals"] == spec.time_varying_unknown_reals
