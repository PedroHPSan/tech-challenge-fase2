"""Download reprodutível do dataset bruto de propensão de compra (UCI)."""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_RAW_PATH

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
CSV_NAME_IN_ZIP = "online_shoppers_intention.csv"


def fetch_zip(url: str = DATASET_URL) -> bytes:
    """Baixa o arquivo zip do dataset a partir do repositório da UCI."""
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def extract_csv(archive: bytes, member: str = CSV_NAME_IN_ZIP) -> bytes:
    """Extrai o CSV do dataset de dentro do zip baixado."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        return zf.read(member)


def save_csv(content: bytes, destination: Path = DATA_RAW_PATH) -> Path:
    """Grava o CSV bruto em disco, criando o diretório de destino."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def run() -> None:
    """Baixa e materializa o dataset bruto em data/raw/."""
    if DATA_RAW_PATH.exists():
        print(f"Dataset já presente em {DATA_RAW_PATH}; nada a fazer.")
        return
    path = save_csv(extract_csv(fetch_zip()))
    print(f"Dataset salvo em {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(run())
