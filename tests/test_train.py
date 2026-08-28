"""Testes da seleção do modelo campeão."""

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.models import ModelCandidate
from src.train import CandidateResult, select_champion


def make_result(name: str, score: float) -> CandidateResult:
    """Cria um resultado mínimo para testar a seleção."""
    pipeline = Pipeline([("model", DummyClassifier())])
    candidate = ModelCandidate(name, "logistic_regression", {})
    return CandidateResult(candidate, pipeline, {"cv_f1_mean": score}, f"run-{name}")


def test_select_champion_escolhe_maior_metrica() -> None:
    results = [make_result("baseline", 0.6), make_result("forest", 0.7)]

    champion = select_champion(results, "cv_f1_mean")

    assert champion.candidate.name == "forest"


def test_select_champion_rejeita_lista_vazia() -> None:
    with pytest.raises(ValueError, match="Nenhum candidato"):
        select_champion([], "cv_f1_mean")
