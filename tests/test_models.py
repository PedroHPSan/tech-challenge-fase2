"""Testes da fábrica de modelos."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models import ModelCandidate, build_model, load_candidates


def test_load_candidates_preserva_configuracao() -> None:
    config = {
        "candidates": [
            {
                "name": "baseline",
                "estimator": "logistic_regression",
                "params": {"C": 0.5},
            }
        ]
    }

    candidates = load_candidates(config)

    assert candidates == [
        ModelCandidate("baseline", "logistic_regression", {"C": 0.5})
    ]


@pytest.mark.parametrize(
    ("estimator", "expected_type"),
    [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
    ],
)
def test_build_model_usa_factory(estimator: str, expected_type: type) -> None:
    candidate = ModelCandidate("candidate", estimator, {})

    model = build_model(candidate, seed=42)

    assert isinstance(model, expected_type)
    assert model.random_state == 42


def test_build_model_rejeita_estimador_desconhecido() -> None:
    candidate = ModelCandidate("invalid", "unsupported", {})

    with pytest.raises(ValueError, match="Modelo não suportado"):
        build_model(candidate, seed=42)
