"""Testes unitários do módulo de engenharia de features."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features import (
    CATEGORICAL_COLUMNS,
    build_pipeline,
    build_preprocessor,
    get_numeric_columns,
)
from src.preprocess import TARGET_COLUMN, clean_data, load_raw_data


def test_get_numeric_columns_exclui_categoricas() -> None:
    df = clean_data(load_raw_data()).drop(columns=[TARGET_COLUMN])
    numericas = get_numeric_columns(df)
    for coluna in CATEGORICAL_COLUMNS:
        assert coluna not in numericas
    assert len(numericas) == len(df.columns) - len(CATEGORICAL_COLUMNS)


def test_preprocessor_transforma_sem_nulos() -> None:
    df = clean_data(load_raw_data()).drop(columns=[TARGET_COLUMN])
    transformado = build_preprocessor(get_numeric_columns(df)).fit_transform(df)
    assert transformado.shape[0] == len(df)
    assert transformado.shape[1] > df.shape[1]
    assert not np.isnan(transformado).any()


def test_pipeline_completo_treina_sem_erro() -> None:
    df = clean_data(load_raw_data()).sample(n=500, random_state=42)
    X, y = df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN].astype(int)
    pipeline = build_pipeline(LogisticRegression(max_iter=500), get_numeric_columns(X))
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(y)
