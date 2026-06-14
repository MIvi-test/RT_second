ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Настройки окружения uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Копируем конфигурацию проекта. 
# Запись uv.lock* (со звездочкой) означает: "скопируй, если он есть, но не падай, если его нет"
COPY pyproject.toml uv.lock* /app/

# Принимаем аргумент сборки (cpu или cuda)
ARG TORCH_VARIANT=cpu
ENV TORCH_VARIANT=${TORCH_VARIANT}

# Заменяем shell-скрипт на встроенную фичу uv. 
# Мы передаем индекс PyTorch напрямую через переменную окружения uv, если выбрана cuda
RUN if [ "${TORCH_VARIANT}" = "cuda" ]; then \
        export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"; \
    fi && \
    uv sync --no-install-project --no-dev

# Прописываем путь к виртуальному окружению в PATH
ENV PATH="/app/.venv/bin:$PATH"

# Копируем исходный код и entrypoint
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]