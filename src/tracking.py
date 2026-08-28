"""Integração de experimentos e Model Registry com o MLflow."""

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.models import ModelCandidate


def configure_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Configura o backend e seleciona o experimento do projeto."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_candidate_run(
    candidate: ModelCandidate,
    pipeline: Pipeline,
    metrics: dict[str, float],
    X_sample: pd.DataFrame,
    artifact_dir: Path,
    seed: int,
) -> str:
    """Registra um candidato treinado e devolve o identificador da run."""
    with mlflow.start_run(run_name=candidate.name) as active_run:
        mlflow.log_params(_tracking_params(candidate, seed))
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        _log_model(pipeline, X_sample)
        return active_run.info.run_id


def _tracking_params(candidate: ModelCandidate, seed: int) -> dict[str, Any]:
    """Normaliza parâmetros do candidato para o MLflow."""
    params = {f"model_{key}": value for key, value in candidate.params.items()}
    return {
        "candidate": candidate.name,
        "estimator": candidate.estimator,
        "seed": seed,
        **params,
    }


def _log_model(pipeline: Pipeline, X_sample: pd.DataFrame) -> None:
    """Registra o pipeline com exemplo de entrada e assinatura."""
    example = X_sample.head(5)
    signature = infer_signature(example, pipeline.predict(example))
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=example,
        signature=signature,
    )


def registry_is_available(tracking_uri: str) -> bool:
    """Indica se o backend configurado suporta o Model Registry."""
    return bool(tracking_uri and not tracking_uri.startswith(("file:", "./")))


def register_champion(run_id: str, model_name: str) -> str:
    """Cria uma versão registrada e atribui o alias champion."""
    model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", model_version.version)
    client.set_model_version_tag(
        model_name,
        model_version.version,
        "validation_status",
        "passed",
    )
    return str(model_version.version)
