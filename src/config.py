"""Configurações centrais do projeto, carregadas de variáveis de ambiente/.env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.params import load_params


class Settings(BaseSettings):
    """Configurações de infraestrutura da aplicação."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    mlflow_tracking_uri: str = ""
    dvc_remote_path: str = "~/dvc-storage/techchallenge-fase2"
    seed: int = 42


settings = Settings()

DATA_RAW_PATH = Path("data/raw/online_shoppers_intention.csv")
DATA_PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

# params.yaml é a fonte de verdade para o DVC (que rastreia o arquivo e
# invalida o cache quando ele muda); o .env permanece como fallback.
SEED = int(load_params().get("seed", settings.seed))
