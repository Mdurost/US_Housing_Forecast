"""Champion/challenger gate: promote the candidate only if it beats the baseline and champion."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from housing_forecast.artifacts import load_model_and_frame
from housing_forecast.config import Config
from housing_forecast.evaluate import evaluate
from housing_forecast.provenance import ModelArtifacts, load_card, save_card, utc_now
from housing_forecast.tracking import MODEL_ARTIFACT_PATH, Tracker, metric_key

logger = logging.getLogger(__name__)

MODEL_ROW = "TFT"


@dataclass
class PromotionDecision:
    promote: bool
    reason: str


@dataclass
class PromotionResult:
    decision: PromotionDecision
    candidate_metrics: pd.DataFrame
    champion_metrics: pd.DataFrame | None
    report_path: Path
    registered_version: str | None


def decide(
    candidate: float,
    baseline: float | None,
    champion: float | None,
    min_improvement: float = 0.0,
    metric: str = "MASE",
) -> PromotionDecision:
    """Lower is better. Candidate must beat the baseline (if given) and improve on the champion."""
    if baseline is not None and not candidate < baseline:
        return PromotionDecision(False, f"candidate {metric} {candidate:.4f} does not beat baseline {baseline:.4f}")
    if champion is None:
        return PromotionDecision(True, f"no current champion; candidate {metric} {candidate:.4f}")
    threshold = champion * (1 - min_improvement)
    if not candidate < threshold:
        return PromotionDecision(
            False,
            f"candidate {metric} {candidate:.4f} does not improve on champion {champion:.4f}"
            + (f" by {min_improvement:.1%}" if min_improvement else ""),
        )
    return PromotionDecision(True, f"candidate {metric} {candidate:.4f} beats champion {champion:.4f}")


def model_metric(metrics: pd.DataFrame, model: str, metric: str) -> float:
    row = metrics.loc[metrics["model"] == model, metric]
    if row.empty:
        raise KeyError(f"No {metric!r} for model {model!r} in evaluation results")
    return float(row.iloc[0])


def _evaluate(raw: pd.DataFrame, config: Config, artifacts: ModelArtifacts) -> tuple[pd.DataFrame, pd.Timestamp]:
    model, spec, frame = load_model_and_frame(raw, config, artifacts)
    return evaluate(model, frame, spec.target, config), frame["date"].max()


def _archive_champion(champion: ModelArtifacts, config: Config) -> Path | None:
    if not champion.exists():
        return None
    destination = config.resolve(config.paths.archive_dir) / utc_now().replace(":", "")
    destination.mkdir(parents=True, exist_ok=True)
    for path in champion.files():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    logger.info("Previous champion archived to %s", destination)
    return destination


def promote(raw: pd.DataFrame, config: Config, force: bool = False) -> PromotionResult:
    p = config.promotion
    candidate, champion = ModelArtifacts.candidate(config), ModelArtifacts.champion(config)
    if not candidate.exists():
        raise FileNotFoundError(f"No candidate model at {candidate.checkpoint}. Run `housing-forecast train` first.")

    candidate_metrics, last_date = _evaluate(raw, config, candidate)
    champion_metrics = _evaluate(raw, config, champion)[0] if champion.exists() else None

    candidate_score = model_metric(candidate_metrics, MODEL_ROW, p.metric)
    baseline_score = model_metric(candidate_metrics, p.baseline_model, p.metric) if p.baseline_model else None
    champion_score = model_metric(champion_metrics, MODEL_ROW, p.metric) if champion_metrics is not None else None
    decision = decide(candidate_score, baseline_score, champion_score, p.min_improvement, p.metric)
    if force and not decision.promote:
        decision = PromotionDecision(True, f"forced: {decision.reason}")
    logger.info("Promotion %s: %s", "APPROVED" if decision.promote else "REJECTED", decision.reason)

    card = load_card(candidate.model_card)
    run_id = card.get("mlflow_run_id")
    holdout = {
        "data_end_date": last_date.strftime("%Y-%m-%d"),
        "metric": p.metric,
        "candidate": candidate_metrics.to_dict("records"),
        "champion": champion_metrics.to_dict("records") if champion_metrics is not None else None,
        "decision": {"promote": decision.promote, "reason": decision.reason, "forced": force},
        "candidate_run_id": run_id,
        "champion_run_id": load_card(champion.model_card).get("mlflow_run_id"),
        "evaluated_at": utc_now(),
    }
    report_path = (
        config.resolve(config.paths.output_dir) / "evaluation" / f"promotion_{last_date.strftime('%Y-%m')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(holdout, indent=2), encoding="utf-8")

    tracker = Tracker(config)
    if run_id and not tracker.run_exists(run_id):
        logger.warning("Candidate MLflow run %s not found in the current tracking store; not logging.", run_id)
        run_id = None
    metrics = {
        f"holdout_{metric_key(row['model'])}_{name}": row[name]
        for row in candidate_metrics.to_dict("records")
        for name in ("MASE", "MAE", "RMSE", "MAPE")
    }
    metrics["holdout_champion_" + p.metric] = champion_score
    tracker.log_metrics(run_id, metrics)
    tracker.set_tags(run_id, {"promoted": decision.promote, "promotion_reason": decision.reason})
    tracker.log_artifact(run_id, report_path, "evaluation")

    registered_version = None
    if decision.promote:
        card.update({"holdout": holdout, "promoted_at": utc_now()})
        registered_version = tracker.register_champion(run_id, description=decision.reason)
        card["registered_version"] = registered_version
        save_card(card, candidate.model_card)
        tracker.log_artifact(run_id, candidate.model_card, MODEL_ARTIFACT_PATH)

        _archive_champion(champion, config)
        for source, target in zip(candidate.files(), champion.files()):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        logger.info("Candidate promoted to %s", champion.checkpoint)

    return PromotionResult(decision, candidate_metrics, champion_metrics, report_path, registered_version)
