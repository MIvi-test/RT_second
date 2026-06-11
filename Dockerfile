FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Установка системных зависимостей (build-essential для компиляции, 
# curl для healthcheck, а также дополнительные библиотеки для torch/chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Копируем только pyproject.toml для кеширования слоя зависимостей
COPY pyproject.toml ./

# Обновляем pip и устанавливаем зависимости из pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Копируем исходный код приложения
COPY app.py llm.py search.py index.py query.py ./

# Создаём необходимые директории для монтирования томов и кеширования
RUN mkdir -p /app/storage /app/cache && \
    # Создаём пустые файлы-заглушки (будут перезаписаны томами или при инициализации)
    touch bm25_index.pkl bm25_meta.json results.json

# Указываем HuggingFace кеш внутри образа (будет переопределён томом в compose)
ENV HF_HOME=/app/cache/huggingface

# Создаём непривилегированного пользователя для безопасности
RUN addgroup --system appgroup && \
    adduser --system --no-create-home --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

# Переключаемся на непривилегированного пользователя
USER appuser

# Открываем порт для Streamlit
EXPOSE 8501

# Команда по умолчанию (может быть переопределена в docker-compose)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]