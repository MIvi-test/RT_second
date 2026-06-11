# syntax=docker/dockerfile:1
FROM python:3.12-slim

ARG TORCH_VARIANT=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cpu  — ~250 MB torch wheel (default, smaller image)
# cuda — GPU torch; requires nvidia-container-toolkit at runtime
RUN pip install --upgrade pip --quiet && \
    if [ "$TORCH_VARIANT" = "cuda" ]; then \
        pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu124 \
            torch==2.5.1 --quiet; \
    else \
        pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.5.1 --quiet; \
    fi

COPY pyproject.toml ./
RUN pip install --no-cache-dir . --quiet

COPY app.py llm.py search.py settings.py index.py query.py entrypoint.sh ./

RUN mkdir -p /storage && chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV USE_RERANKER=true
ENV USE_GPU=false
ENV USE_OLLAMA=true
ENV STORAGE_DIR=/storage

EXPOSE 8501
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
