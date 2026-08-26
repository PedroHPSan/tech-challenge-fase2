"""Leitura dos parâmetros do pipeline definidos em params.yaml."""

from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("params.yaml")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Carrega params.yaml; retorna dicionário vazio se o arquivo não existir."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_stage_params(stage: str, path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Retorna o bloco de parâmetros de um estágio específico do pipeline."""
    return load_params(path).get(stage, {})
