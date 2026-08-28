"""Testes de métricas e artefatos de avaliação."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_classifier, save_evaluation_artifacts


def fitted_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Cria um classificador pequeno para os testes."""
    X = pd.DataFrame({"feature": [0.0, 0.2, 0.8, 1.0, 1.2, 1.4]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = Pipeline([("model", LogisticRegression(random_state=42))])
    pipeline.fit(X, y)
    return pipeline, X, y


def test_evaluate_classifier_retorna_metricas_obrigatorias() -> None:
    pipeline, X, y = fitted_pipeline()

    metrics = evaluate_classifier(pipeline, X, y)

    assert set(metrics) == {"f1", "precision", "recall", "auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_save_evaluation_artifacts_gera_arquivos(tmp_path) -> None:
    pipeline, X, y = fitted_pipeline()

    save_evaluation_artifacts(pipeline, X, y, tmp_path)

    expected = {
        "classification_report.json",
        "confusion_matrix.png",
        "precision_recall_curve.png",
        "roc_curve.png",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
