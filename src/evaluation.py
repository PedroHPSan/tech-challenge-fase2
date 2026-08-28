"""Métricas e artefatos visuais para classificação binária."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_classifier(
    model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Calcula as métricas obrigatórias no conjunto de teste."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "f1": float(f1_score(y, predictions)),
        "precision": float(precision_score(y, predictions)),
        "recall": float(recall_score(y, predictions)),
        "auc": float(roc_auc_score(y, probabilities)),
    }


def save_evaluation_artifacts(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    """Salva curvas, matriz de confusão e relatório do modelo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    _save_confusion_matrix(y, predictions, output_dir)
    _save_roc_curve(y, probabilities, output_dir)
    _save_precision_recall_curve(y, probabilities, output_dir)
    _save_classification_report(y, predictions, output_dir)


def _save_confusion_matrix(
    y: pd.Series, predictions: Any, output_dir: Path
) -> None:
    """Salva a matriz de confusão em PNG."""
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, predictions, ax=axis, cmap="Blues")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def _save_roc_curve(y: pd.Series, probabilities: Any, output_dir: Path) -> None:
    """Salva a curva ROC em PNG."""
    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, probabilities, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(figure)


def _save_precision_recall_curve(
    y: pd.Series, probabilities: Any, output_dir: Path
) -> None:
    """Salva a curva de precisão e recall em PNG."""
    figure, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(figure)


def _save_classification_report(
    y: pd.Series, predictions: Any, output_dir: Path
) -> None:
    """Salva o relatório completo de classificação em JSON."""
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    content = json.dumps(report, indent=2)
    (output_dir / "classification_report.json").write_text(content, encoding="utf-8")
