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
│   ├── config.py       # caminhos e configurações (carregadas do .env)
│   ├── preprocess.py   # limpeza e split treino/teste
│   └── features.py     # transformações (ColumnTransformer)
└── tests/              # testes unitários (pytest)
```

## Instalação do zero (Poetry)

Pré-requisito: ter o Python 3.11+ e o [Poetry](https://python-poetry.org/) instalados.

```bash
# 1. Instale as dependências (garante as versões exatas do poetry.lock)
poetry install

# 2. Copie o arquivo de configuração de exemplo e ajuste se necessário
cp .env.example .env

# 3. (Opcional) Ative o ambiente virtual do Poetry em seu terminal
poetry shell
```

> **Reprodutibilidade garantida:** o arquivo `poetry.lock` está commitado no
> repositório. Isso significa que `poetry install` sempre instalará as mesmas
> versões de todas as bibliotecas, independente da máquina.

## Configurações de ambiente (.env)

Todas as configurações sensíveis ou que variam por ambiente ficam externalizadas
no arquivo `.env` (ver `.env.example` como modelo):

| Variável | Descrição |
|---|---|
| `MLFLOW_TRACKING_URI` | URI do servidor MLflow (vazio = tracking local) |
| `DVC_REMOTE_PATH` | Caminho do remote local para armazenamento do DVC |
| `SEED` | Seed global para reprodutibilidade de todos os passos |

## Convenções de código

- Funções curtas, com type hints e docstrings
- Nomes descritivos, módulos com responsabilidade única
- Seed fixado via variável de ambiente `SEED` (default: 42) para reprodutibilidade
- Qualidade de código verificada com `ruff`

## Qualidade e testes

```bash
# Lint do código
poetry run ruff check src/

# Testes unitários
poetry run pytest
```
