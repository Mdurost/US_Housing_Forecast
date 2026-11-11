from pathlib import Path

import pytest

from housing_forecast.config import load_config


def test_default_config_matches_notebook_hyperparameters():
    config = load_config()
    assert config.dataset.max_encoder_length == 36
    assert config.dataset.max_prediction_length == 12
    assert config.model.hidden_size == 70
    assert config.features.corr_threshold == 0.90
    assert (config.project_root / "pyproject.toml").is_file()


def test_overrides_are_parsed_as_yaml(tmp_path: Path):
    config = load_config(None, ["trainer.max_epochs=3", "trainer.accelerator=cpu", f"project_root={tmp_path}"])
    assert config.trainer.max_epochs == 3
    assert config.trainer.accelerator == "cpu"
    assert config.resolve(config.paths.model_checkpoint) == tmp_path / "models" / "tft_housing.ckpt"


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown TrainerConfig keys"):
        load_config(None, ["trainer.max_epoch=3"])
