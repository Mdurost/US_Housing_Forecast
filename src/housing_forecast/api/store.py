"""Read-only access to pipeline outputs (forecasts, promotion reports, champion model card)."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from housing_forecast.config import Config
from housing_forecast.provenance import ModelArtifacts, load_card

TAG_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_STATE_FILE = re.compile(r"^state_forecast_(\d{4}-\d{2})\.csv$")


class NotFound(LookupError):
    pass


class ForecastStore:
    def __init__(self, config: Config):
        self.config = config
        self.forecast_dir = config.resolve(config.paths.output_dir) / "forecasts"
        self.evaluation_dir = config.resolve(config.paths.output_dir) / "evaluation"
        self._cache: dict[Path, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def tags(self) -> list[str]:
        if not self.forecast_dir.is_dir():
            return []
        found = (_STATE_FILE.match(p.name) for p in self.forecast_dir.iterdir())
        return sorted(m.group(1) for m in found if m)

    def resolve_tag(self, tag: str) -> str:
        if tag == "latest":
            tags = self.tags()
            if not tags:
                raise NotFound("No forecasts have been generated yet. Run the pipeline first.")
            return tags[-1]
        if not TAG_PATTERN.match(tag):
            raise NotFound(f"Invalid forecast month {tag!r}; expected YYYY-MM or 'latest'.")
        return tag

    def _cached(self, path: Path, loader) -> Any:
        if not path.is_file():
            raise NotFound(f"{path.name} not found")
        mtime = path.stat().st_mtime
        with self._lock:
            hit = self._cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]
        value = loader(path)
        with self._lock:
            self._cache[path] = (mtime, value)
        return value

    def _frame(self, path: Path) -> pd.DataFrame:
        return self._cached(path, lambda p: pd.read_csv(p, parse_dates=["date"]))

    def national(self, tag: str) -> pd.DataFrame:
        return self._frame(self.forecast_dir / f"national_forecast_{tag}.csv")

    def states(self, tag: str) -> pd.DataFrame:
        return self._frame(self.forecast_dir / f"state_forecast_{tag}.csv")

    def forecast_metadata(self, tag: str) -> dict[str, Any]:
        path = self.forecast_dir / f"forecast_metadata_{tag}.json"
        try:
            return self._cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))
        except NotFound:
            return {}

    def promotion_report(self, tag: str) -> dict[str, Any]:
        path = self.evaluation_dir / f"promotion_{tag}.json"
        return self._cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))

    def champion_card(self) -> dict[str, Any]:
        return load_card(ModelArtifacts.champion(self.config).model_card)

    def champion_exists(self) -> bool:
        return ModelArtifacts.champion(self.config).exists()
