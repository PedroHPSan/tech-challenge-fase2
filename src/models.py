"""Fábrica de modelos e validação cruzada dos candidatos."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

SCORING = ("f1", "precision", "recall", "roc_auc")


@dataclass(frozen=True)
class ModelCandidate:
    """Configuração de um modelo candidato ao registro."""

    name: str
    estimator: str
    params: dict[str, Any]


def load_candidates(config: dict[str, Any]) -> list[ModelCandidate]:
    """Converte as configurações YAML em candidatos tipados."""
    return [
        ModelCandidate(
            name=item["name"],
            estimator=item["estimator"],
            params=item.get("params", {}),
        )
        for item in config.get("candidates", [])
    ]


def _build_logistic(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Cria uma Regressão Logística reproduzível."""
    return LogisticRegression(**params, random_state=seed)


def _build_random_forest(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Cria uma Random Forest reproduzível."""
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)


MODEL_BUILDERS: dict[str, Callable[[dict[str, Any], int], BaseEstimator]] = {
    "logistic_regression": _build_logistic,
    "random_forest": _build_random_forest,
}


def build_model(candidate: ModelCandidate, seed: int) -> BaseEstimator:
    """Instancia o estimador solicitado usando uma fábrica de modelos."""
    try:
        builder = MODEL_BUILDERS[candidate.estimator]
    except KeyError as error:
        raise ValueError(f"Modelo não suportado: {candidate.estimator}") from error
    return builder(candidate.params, seed)


def cross_validation_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
) -> dict[str, float]:
    """Calcula métricas médias e desvios em validação cruzada estratificada."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(pipeline, X, y, cv=splitter, scoring=SCORING, n_jobs=1)
    return _summarize_scores(scores)


def _summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Resume os resultados retornados pelo Scikit-Learn."""
    metrics: dict[str, float] = {}
    for name in SCORING:
        values = scores[f"test_{name}"]
        metrics[f"cv_{name}_mean"] = float(np.mean(values))
        metrics[f"cv_{name}_std"] = float(np.std(values))
    return metrics
