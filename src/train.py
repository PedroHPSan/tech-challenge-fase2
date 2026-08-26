"""Estágio de treinamento do pipeline DVC.

Baseline intencionalmente simples (Regressão Logística) para fechar o pipeline
de ponta a ponta. O Membro 5 substitui a construção do modelo e acrescenta o
tracking do MLflow, mantendo o mesmo contrato de entradas e saídas:

    entradas : data/processed/train.csv, data/processed/test.csv
    saídas   : models/model.pkl, metrics/metrics.json
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED_DIR, METRICS_DIR, MODELS_DIR, SEED
from src.features import build_pipeline, get_numeric_columns
from src.params import get_stage_params
from src.preprocess import TARGET_COLUMN

MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = METRICS_DIR / "metrics.json"


def load_split(name: str, directory: Path = DATA_PROCESSED_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega um split processado e separa features do alvo."""
    df = pd.read_csv(directory / f"{name}.csv")
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN].astype(int)


def build_model(params: dict[str, Any], seed: int = SEED) -> BaseEstimator:
    """Instancia o classificador de baseline a partir dos parâmetros."""
    return LogisticRegression(
        max_iter=int(params.get("max_iter", 1000)),
        C=float(params.get("C", 1.0)),
        class_weight=params.get("class_weight", "balanced"),
        random_state=seed,
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calcula as métricas preditivas exigidas pelo desafio."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "f1": float(f1_score(y, predictions)),
        "precision": float(precision_score(y, predictions)),
        "recall": float(recall_score(y, predictions)),
        "auc": float(roc_auc_score(y, probabilities)),
    }


def save_artifacts(model: Pipeline, metrics: dict[str, float]) -> None:
    """Persiste o modelo treinado e o arquivo de métricas."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run() -> None:
    """Executa o estágio de treinamento do pipeline."""
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    model = build_model(get_stage_params("train"))
    pipeline = build_pipeline(model, get_numeric_columns(X_train))
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    save_artifacts(pipeline, metrics)
    print(f"Treinamento concluído: {metrics}")


if __name__ == "__main__":
    run()
