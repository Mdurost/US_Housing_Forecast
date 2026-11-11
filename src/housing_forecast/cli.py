"""Command-line entry point: ``housing-forecast <command>`` or ``python -m housing_forecast``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from housing_forecast.config import Config, load_config

logger = logging.getLogger("housing_forecast")


def _data_tag(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m")


def cmd_ingest(config: Config, args: argparse.Namespace) -> Path:
    from housing_forecast.clean import build_panel
    from housing_forecast.ingest import download_sources, load_snapshot

    if args.from_snapshot:
        sources = load_snapshot(args.from_snapshot)
    else:
        sources, _ = download_sources(config, snapshot=not args.no_snapshot)
    panel = build_panel(sources, config)

    output = config.resolve(config.paths.processed_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    logger.info("Processed data written to %s", output)
    return output


def _load_raw(config: Config) -> pd.DataFrame:
    from housing_forecast.features import load_processed_data

    return load_processed_data(config)


def cmd_train(config: Config, args: argparse.Namespace) -> None:
    from housing_forecast.train import train

    train(_load_raw(config), config, data_path=config.resolve(config.paths.processed_data))


def _artifacts(config: Config, candidate: bool):
    from housing_forecast.provenance import ModelArtifacts

    return ModelArtifacts.candidate(config) if candidate else ModelArtifacts.champion(config)


def cmd_evaluate(config: Config, args: argparse.Namespace) -> Path:
    from housing_forecast.artifacts import load_model_and_frame
    from housing_forecast.evaluate import evaluate

    candidate = getattr(args, "candidate", False)
    model, spec, frame = load_model_and_frame(_load_raw(config), config, _artifacts(config, candidate))
    metrics = evaluate(model, frame, spec.target, config)
    suffix = "_candidate" if candidate else ""
    output = (
        config.resolve(config.paths.output_dir)
        / "evaluation"
        / f"metrics_{_data_tag(frame['date'].max())}{suffix}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    logger.info("Evaluation metrics written to %s\n%s", output, metrics.to_string(index=False))
    return output


def cmd_promote(config: Config, args: argparse.Namespace) -> bool:
    from housing_forecast.promote import promote

    result = promote(_load_raw(config), config, force=getattr(args, "force", False))
    logger.info("Promotion report written to %s", result.report_path)
    return result.decision.promote


def cmd_predict(config: Config, args: argparse.Namespace) -> Path:
    from housing_forecast.artifacts import load_model_and_frame
    from housing_forecast.predict import forecast
    from housing_forecast.provenance import ModelArtifacts, load_card, save_card, utc_now
    from housing_forecast.tracking import Tracker

    champion = ModelArtifacts.champion(config)
    model, spec, frame = load_model_and_frame(_load_raw(config), config, champion)
    result = forecast(model, frame, spec, config)
    tag = _data_tag(result.last_observed_date)
    output_dir = config.resolve(config.paths.output_dir) / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"state_forecast_{tag}.csv"
    national_path = output_dir / f"national_forecast_{tag}.csv"
    result.state.to_csv(state_path, index=False)
    result.national.to_csv(national_path, index=False)

    # Attach forecasts to the champion's MLflow run so each model's predictions are traceable.
    card = load_card(champion.model_card)
    run_id = card.get("mlflow_run_id")
    metadata_path = output_dir / f"forecast_metadata_{tag}.json"
    save_card(
        {
            "forecast_month": tag,
            "generated_at": utc_now(),
            "model": {k: card.get(k) for k in ("mlflow_run_id", "registered_version", "created_at", "promoted_at")},
        },
        metadata_path,
    )
    tracker = Tracker(config)
    tracker.attach_artifacts(run_id, [state_path, national_path, metadata_path], f"forecasts/{tag}")

    national = result.national.set_index("date").round(0).reset_index()
    logger.info("Forecasts written to %s\n%s", output_dir, national.to_string(index=False))
    return output_dir


def cmd_tune(config: Config, args: argparse.Namespace) -> None:
    from housing_forecast.tune import tune

    tune(_load_raw(config), config, learning_rate=args.learning_rate)


def cmd_serve(config: Config, args: argparse.Namespace) -> None:
    import uvicorn

    from housing_forecast.api import create_app

    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level=args.log_level.lower())


def cmd_run(config: Config, args: argparse.Namespace) -> None:
    if not args.skip_ingest:
        cmd_ingest(config, args)
    cmd_train(config, args)
    if not cmd_promote(config, args):
        logger.warning("Candidate was not promoted; forecasting with the existing champion.")
    cmd_predict(config, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="housing-forecast", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set trainer.max_epochs=5 (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_ingest_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--from-snapshot", type=Path, help="Rebuild from a saved data/raw/<date> snapshot")
        p.add_argument("--no-snapshot", action="store_true", help="Do not save raw downloads")

    ingest = sub.add_parser("ingest", help="Download sources and build the processed dataset")
    add_ingest_args(ingest)
    ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("train", help="Train a candidate model -> models/candidate/ (tracked in MLflow)").set_defaults(
        func=cmd_train
    )
    evaluate = sub.add_parser("evaluate", help="Score a model vs naive baselines on the hold-out window")
    evaluate.add_argument("--candidate", action="store_true", help="Evaluate the candidate instead of the champion")
    evaluate.set_defaults(func=cmd_evaluate)
    promote = sub.add_parser("promote", help="Promote the candidate if it beats the baseline and the champion")
    promote.add_argument("--force", action="store_true", help="Promote even if the gate rejects the candidate")
    promote.set_defaults(func=cmd_promote)
    sub.add_parser("predict", help="Write 12-month forecasts from the champion model").set_defaults(func=cmd_predict)

    tune = sub.add_parser("tune", help="LR finder + Optuna hyperparameter search (slow)")
    tune.add_argument("--learning-rate", type=float, help="Skip the LR finder and center the search here")
    tune.set_defaults(func=cmd_tune)

    run = sub.add_parser("run", help="ingest -> train -> promote -> predict")
    add_ingest_args(run)
    run.add_argument("--skip-ingest", action="store_true", help="Use the existing processed dataset")
    run.add_argument("--force", action="store_true", help="Promote the new model even if the gate rejects it")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="Start the FastAPI service (forecasts + Airflow pipeline control)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    # pytorch-forecasting renders validation plots; use a non-GUI backend for headless runs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config, args.overrides)
    args.func(config, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
