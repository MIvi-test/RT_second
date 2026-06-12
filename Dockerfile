# syntax=docker/dockerfile:1
FROM python:3.12-slim

ARG TORCH_VARIANT=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем uv
RUN pip install --upgrade pip uv --quiet

# Отключаем кэш uv
ENV UV_NO_CACHE=1

# Установка torch через uv с флагом --system
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
        uv pip install --system \
            --index-url https://download.pytorch.org/whl/cu124 \
            torch==2.5.1 --quiet; \
    else \
        uv pip install --system \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.5.1 --quiet; \
    fi

COPY pyproject.toml ./
# Установка зависимостей проекта через uv
RUN uv pip install --system . --quiet

COPY app.py llm.py search.py settings.py index.py query.py entrypoint.sh ./

RUN mkdir -p /storage && chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV USE_RERANKER=false
ENV USE_GPU=false
ENV USE_OLLAMA=true
ENV STORAGE_DIR=/storage

EXPOSE 8501
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]