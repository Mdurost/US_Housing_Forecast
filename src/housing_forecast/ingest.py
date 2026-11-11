"""Download raw sources (with retries) and snapshot them for reproducible rebuilds."""

from __future__ import annotations

import datetime as dt
import json
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from housing_forecast.config import Config

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
_EXCEL_SUFFIXES = (".xlsx", ".xls")


def _session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _suffix(url: str) -> str:
    path = url.lower().split("?")[0]
    return ".xlsx" if path.endswith(_EXCEL_SUFFIXES) else ".csv"


def parse_bytes(raw_bytes: bytes, is_excel: bool, encodings=("utf-8", "latin-1", "cp1252")) -> pd.DataFrame:
    if is_excel:
        return pd.read_excel(BytesIO(raw_bytes))
    for encoding in encodings:
        try:
            return pd.read_csv(BytesIO(raw_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(raw_bytes), encoding="latin-1")


def download_sources(config: Config, snapshot: bool = True) -> tuple[dict[str, pd.DataFrame], Path | None]:
    """Download every configured source. Returns frames and the snapshot directory (if any)."""
    session = _session(config.ingest.max_retries)
    snapshot_dir = None
    if snapshot:
        snapshot_dir = config.resolve(config.paths.raw_dir) / dt.date.today().isoformat()
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    manifest: dict[str, dict[str, str]] = {}
    for name, url in config.sources.items():
        logger.info("Downloading %s", name)
        response = session.get(url, timeout=config.ingest.request_timeout_seconds)
        response.raise_for_status()
        suffix = _suffix(url)
        frames[name] = parse_bytes(response.content, is_excel=suffix == ".xlsx")
        if snapshot_dir is not None:
            filename = f"{name}{suffix}"
            (snapshot_dir / filename).write_bytes(response.content)
            manifest[name] = {"url": url, "file": filename}

    if snapshot_dir is not None:
        manifest_path = snapshot_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Raw snapshot written to %s", snapshot_dir)
    return frames, snapshot_dir


def load_snapshot(snapshot_dir: Path | str, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load sources from a snapshot directory created by :func:`download_sources`."""
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name, entry in manifest.items():
        if names is not None and name not in names:
            continue
        path = snapshot_dir / entry["file"]
        frames[name] = parse_bytes(path.read_bytes(), is_excel=path.suffix == ".xlsx")
    return frames
