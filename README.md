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
│   ├── raw/            # dataset original (versionado por DVC, não por Git)
│   └── processed/      # saída do estágio preprocess (gerado pelo DVC)
├── models/             # artefatos de modelo (gerado pelo DVC)
├── metrics/            # métricas do estágio train (metrics.json)
├── scripts/
│   └── download_dataset.py  # download reprodutível do dataset da UCI
├── src/
│   ├── config.py       # caminhos e configurações (.env + params.yaml)
│   ├── params.py       # leitura dos parâmetros do pipeline
│   ├── preprocess.py   # limpeza e split treino/teste
│   ├── features.py     # transformações (ColumnTransformer)
│   └── train.py        # treinamento e avaliação do classificador
├── tests/              # testes unitários (pytest)
├── params.yaml         # parâmetros rastreados pelo DVC
└── dvc.yaml            # definição do pipeline (preprocess → train)
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

## Versionamento de dados e pipeline (DVC)

O dataset bruto e os artefatos gerados **não ficam no Git**. O Git versiona
apenas os ponteiros (`*.dvc`, `dvc.lock`), e o conteúdo real vai para o remote
do DVC. Isso mantém o repositório leve e o histórico limpo.

### Obter o dataset

```bash
poetry run python -m scripts.download_dataset
```

O script baixa o CSV direto do repositório da UCI para
`data/raw/online_shoppers_intention.csv`. Se o arquivo já existir, não faz nada.
Quem já tem acesso ao remote pode simplesmente usar `dvc pull`.

### Rodar o pipeline

```bash
poetry run dvc repro
```

O pipeline tem dois estágios encadeados, definidos em `dvc.yaml`:

| Estágio | Comando | Entradas | Saídas |
|---|---|---|---|
| `preprocess` | `python -m src.preprocess` | `data/raw/*.csv` | `data/processed/train.csv`, `test.csv` |
| `train` | `python -m src.train` | `data/processed/*` | `models/model.pkl`, `metrics/metrics.json` |

O DVC só reexecuta um estágio quando alguma dependência muda — código, dados ou
parâmetros. Rodar `dvc repro` duas vezes seguidas não refaz trabalho nenhum.

### Parâmetros

Todos os parâmetros do pipeline ficam em `params.yaml` (seed, `test_size`,
hiperparâmetros do modelo). Alterar um valor lá invalida o cache do estágio
correspondente e faz o `dvc repro` reexecutá-lo.

```bash
poetry run dvc params diff     # o que mudou nos parâmetros
poetry run dvc metrics show    # métricas da última execução
poetry run dvc dag             # visualiza o grafo do pipeline
```

### Remote: Google Drive

O remote padrão é uma pasta no Google Drive, configurada em `.dvc/config`
(versionado). As credenciais **nunca** vão para o Git — ficam em
`.dvc/config.local`, que está no `.gitignore` do próprio DVC.

Configuração inicial, uma vez por pessoa do grupo:

A pasta do projeto já está apontada no `.dvc/config`:
<https://drive.google.com/drive/folders/1MIoZ5pr_6IlQyg_pUoK003O6TFJveu1m>

```bash
# 1. Peça ao responsável acesso de Editor à pasta acima (necessário para push;
#    acesso de Leitor basta para quem só vai rodar `dvc pull`).

# 2. Na primeira sincronização o DVC abre o navegador para você autorizar
#    o acesso com sua conta Google. O token fica só na sua máquina.
poetry run dvc pull
```

> As credenciais OAuth são gravadas localmente pelo DVC e não são commitadas.
> Se o grupo preferir uma app OAuth própria (recomendado para evitar limites de
> cota do cliente padrão), grave o ID e o segredo em `.dvc/config.local`:
> `dvc remote modify --local gdrive gdrive_client_id <ID>` e
> `dvc remote modify --local gdrive gdrive_client_secret <SECRET>`.

### Remote alternativo: pasta local

Para trabalhar offline, em CI ou dentro do container, há um remote `localstore`
apontando para uma pasta fora do repositório (configurado em `.dvc/config.local`,
por máquina, a partir de `DVC_REMOTE_PATH` no `.env`):

```bash
poetry run dvc remote add --local localstore "$DVC_REMOTE_PATH"
poetry run dvc push -r localstore
```

### Comandos do dia a dia

```bash
poetry run dvc status    # o que está fora de sincronia
poetry run dvc push      # envia os dados para o remote padrão (Google Drive)
poetry run dvc pull      # traz os dados do remote
```

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
