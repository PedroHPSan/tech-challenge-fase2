"""Treina, compara, rastreia e registra classificadores de propensão."""

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED_DIR, METRICS_DIR, MODELS_DIR, SEED, settings
from src.evaluation import evaluate_classifier, save_evaluation_artifacts
from src.features import build_pipeline, get_numeric_columns
from src.models import (
    ModelCandidate,
    build_model,
    cross_validation_metrics,
    load_candidates,
)
from src.params import get_stage_params
from src.preprocess import TARGET_COLUMN
from src.tracking import (
    configure_tracking,
    log_candidate_run,
    register_champion,
    registry_is_available,
)

MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = METRICS_DIR / "metrics.json"


@dataclass
class CandidateResult:
    """Resultado necessário para comparar e persistir um candidato."""

    candidate: ModelCandidate
    pipeline: Pipeline
    metrics: dict[str, float]
    run_id: str


def load_split(
    name: str, directory: Path = DATA_PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega um split processado e separa features do alvo."""
    dataframe = pd.read_csv(directory / f"{name}.csv")
    return dataframe.drop(columns=[TARGET_COLUMN]), dataframe[TARGET_COLUMN].astype(int)


def train_candidate(
    candidate: ModelCandidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    folds: int,
) -> CandidateResult:
    """Treina, avalia e rastreia um candidato de modelo."""
    estimator = build_model(candidate, SEED)
    pipeline = build_pipeline(estimator, get_numeric_columns(X_train))
    cv_metrics = cross_validation_metrics(pipeline, X_train, y_train, folds, SEED)
    pipeline.fit(X_train, y_train)
    metrics = {**cv_metrics, **evaluate_classifier(pipeline, X_test, y_test)}
    run_id = _log_run(candidate, pipeline, metrics, X_test, y_test)
    return CandidateResult(candidate, pipeline, metrics, run_id)


def _log_run(
    candidate: ModelCandidate,
    pipeline: Pipeline,
    metrics: dict[str, float],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Gera e registra os artefatos de avaliação de uma run."""
    with TemporaryDirectory(prefix="purchase-evaluation-") as directory:
        artifact_dir = Path(directory)
        save_evaluation_artifacts(pipeline, X_test, y_test, artifact_dir)
        return log_candidate_run(candidate, pipeline, metrics, X_test, artifact_dir, SEED)


def select_champion(
    results: list[CandidateResult], selection_metric: str
) -> CandidateResult:
    """Seleciona o maior resultado da métrica de validação configurada."""
    if not results:
        raise ValueError("Nenhum candidato foi configurado para treinamento.")
    return max(results, key=lambda result: result.metrics[selection_metric])


def save_champion(result: CandidateResult) -> None:
    """Persiste o pipeline e as métricas do campeão para o DVC."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")


def maybe_register_champion(
    result: CandidateResult, config: dict[str, Any]
) -> str | None:
    """Registra o campeão quando um backend compatível está configurado."""
    should_register = bool(config.get("register_champion", True))
    if not should_register or not registry_is_available(settings.mlflow_tracking_uri):
        return None
    return register_champion(result.run_id, str(config["registered_model_name"]))


def run() -> None:
    """Executa comparação, seleção, persistência e registro do campeão."""
    config = get_stage_params("train")
    configure_tracking(settings.mlflow_tracking_uri, str(config["experiment_name"]))
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    candidates = load_candidates(config)
    folds = int(config.get("cv_folds", 5))
    results = [
        train_candidate(candidate, X_train, y_train, X_test, y_test, folds)
        for candidate in candidates
    ]
    champion = select_champion(
        results, str(config.get("selection_metric", "cv_f1_mean"))
    )
    save_champion(champion)
    version = maybe_register_champion(champion, config)
    print(_summary(champion, version))


def _summary(champion: CandidateResult, version: str | None) -> str:
    """Formata o resumo final apresentado no terminal."""
    registry = f", registry_version={version}" if version else ", registry=ignorado"
    return f"Campeão: {champion.candidate.name}, métricas={champion.metrics}{registry}"


if __name__ == "__main__":
    run()
