# Tech Challenge Fase 2 — Propensão de Compra em E-commerce

Sistema preditivo para identificar a **propensão de compra** de usuários de um
e-commerce com base no comportamento de navegação.

## Dataset

[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(UCI) — 12.330 sessões de navegação, 17 features (numéricas e categóricas) e
alvo binário `Revenue` (a sessão terminou em compra ou não). O arquivo CSV fica
em `data/raw/` e não é versionado no Git.

## Estrutura do projeto

```
├── data/
│   ├── raw/            # dataset original (não versionado no Git)
│   └── processed/      # saída do preprocessamento
├── models/             # artefatos de modelo
├── src/
│   ├── config.py       # caminhos e constantes do projeto
│   ├── preprocess.py   # limpeza e split treino/teste
│   └── features.py     # transformações (ColumnTransformer)
└── tests/              # testes unitários (pytest)
```

## Convenções de código

- Funções curtas, com type hints e docstrings
- Nomes descritivos, módulos com responsabilidade única
- Seed fixado (`SEED = 42` em `src/config.py`) para reprodutibilidade

## Testes

Com as dependências instaladas (`pandas`, `scikit-learn`, `pytest`):

```bash
python -m pytest
```
