"""Model artifact layout (champion vs candidate) and provenance metadata (model cards)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from housing_forecast.config import Config


@dataclass(frozen=True)
class ModelArtifacts:
    """The files that make up one deployable model."""

    checkpoint: Path
    feature_spec: Path
    model_card: Path

    @classmethod
    def champion(cls, config: Config) -> ModelArtifacts:
        p = config.paths
        return cls(config.resolve(p.model_checkpoint), config.resolve(p.feature_spec), config.resolve(p.model_card))

    @classmethod
    def candidate(cls, config: Config) -> ModelArtifacts:
        directory = config.resolve(config.paths.candidate_dir)
        p = config.paths
        return cls(directory / p.model_checkpoint.name, directory / p.feature_spec.name, directory / p.model_card.name)

    def files(self) -> list[Path]:
        return [self.checkpoint, self.feature_spec, self.model_card]

    def exists(self) -> bool:
        return self.checkpoint.is_file()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_card(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_card(card: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")


def git_info(root: Path) -> dict[str, Any]:
    """Current commit and whether tracked files have uncommitted changes."""

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def data_summary(path: Path | None, raw: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(raw["date"])
    summary: dict[str, Any] = {
        "rows": len(raw),
        "start_date": dates.min().strftime("%Y-%m-%d"),
        "end_date": dates.max().strftime("%Y-%m-%d"),
        "n_states": int(raw["state"].nunique()),
    }
    if path is not None and path.is_file():
        summary["path"] = path.as_posix()
        summary["sha256"] = file_sha256(path)
    return summary
