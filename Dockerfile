# syntax=docker/dockerfile:1

FROM python:3.12.10-slim-bookworm AS builder

ARG POETRY_VERSION=2.4.1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN python -m pip install "poetry==${POETRY_VERSION}" \
    && python -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root --no-interaction --no-ansi


FROM python:3.12.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p data/raw data/processed models metrics \
    && chown -R appuser:appgroup /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup . .

USER appuser

CMD ["sh", "-c", "python -m scripts.download_dataset && exec dvc repro"]
