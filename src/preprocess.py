"""Pré-processamento do dataset de intenção de compra."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED_DIR, DATA_RAW_PATH, SEED
from src.params import get_stage_params

TARGET_COLUMN = "Revenue"
TEST_SIZE = float(get_stage_params("preprocess").get("test_size", 0.2))


def load_raw_data(path: Path = DATA_RAW_PATH) -> pd.DataFrame:
    """Carrega o CSV bruto do dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e linhas com valores ausentes."""
    return df.drop_duplicates().dropna().reset_index(drop=True)


def split_data(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide o dataset em treino e teste com estratificação pelo alvo."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = DATA_PROCESSED_DIR,
) -> None:
    """Salva os conjuntos de treino e teste em disco."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        df = X.copy()
        df[TARGET_COLUMN] = y.to_numpy()
        df.to_csv(output_dir / f"{name}.csv", index=False)


def run() -> None:
    """Executa o estágio de pré-processamento do pipeline."""
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed(X_train, X_test, y_train, y_test)
    print(f"Pré-processamento concluído: treino={len(X_train)}, teste={len(X_test)}")


if __name__ == "__main__":
    run()
