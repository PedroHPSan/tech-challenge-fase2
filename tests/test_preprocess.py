"""Testes unitários do módulo de pré-processamento."""

from src.preprocess import clean_data, load_raw_data, split_data


def test_clean_data_remove_nulos_e_duplicatas() -> None:
    df = clean_data(load_raw_data())
    assert not df.isnull().any().any()
    assert not df.duplicated().any()


def test_split_respeita_test_size() -> None:
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert abs(len(X_test) / len(df) - 0.2) < 0.01


def test_split_estratificado_preserva_proporcao_do_alvo() -> None:
    df = clean_data(load_raw_data())
    _, _, y_train, y_test = split_data(df, test_size=0.2)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
