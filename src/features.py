"""Engenharia de features e transformadores do pipeline de modelagem."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["Month", "VisitorType", "Weekend"]


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Retorna as colunas numéricas do dataframe de features."""
    return [c for c in df.columns if c not in CATEGORICAL_COLUMNS]


def build_preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    """Monta o transformador: OneHot para categóricas e scaling para numéricas."""
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), numeric_columns),
        ]
    )


def build_pipeline(model: BaseEstimator, numeric_columns: list[str]) -> Pipeline:
    """Monta o pipeline completo com pré-processamento e modelo."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_columns)),
            ("model", model),
        ]
    )
