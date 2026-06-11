# второй этап Ростеком кейса

Семантический поиск по коду с гибридным BM25 + векторным поиском, опциональным reranker и RAG через Ollama.

## Локальный запуск (без Docker)

### 1. Зависимости

```bash
uv sync
```

Для GPU (опционально, нужен NVIDIA + CUDA):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Индексация (один раз)

```bash
# путь к датасету — при необходимости через SOURCE_PATH
uv run index.py
```

### 3. Ollama (опционально, для RAG-ответов)

```bash
ollama serve          # отдельный терминал
ollama pull mistral:7b
```

### 4. UI

```bash
uv run streamlit run app.py
```

Откройте http://localhost:8501

### Быстрый старт — 3 команды

```bash
ollama serve &
uv run index.py
USE_GPU=true USE_RERANKER=true uv run streamlit run app.py
```

Без LLM:

```bash
USE_OLLAMA=false uv run index.py
USE_OLLAMA=false uv run streamlit run app.py
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `SOURCE_PATH` | `./dataset_case3_v1.0_fix/gymhero` | Корень Python-кода для индексации |
| `STORAGE_DIR` | `./storage` | Каталог индексов (ChromaDB, BM25) |
| `USE_RERANKER` | `true` | CrossEncoder `BAAI/bge-reranker-v2-m3`; `false` — быстрее, без rerank |
| `USE_GPU` | `false` | `true` — cuda для embedder и reranker (если CUDA доступна) |
| `USE_OLLAMA` | `true` | RAG-генерация через Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Адрес Ollama |

Модели эмбеддингов и reranker **не меняются**:

- эмбеддинги: `intfloat/multilingual-e5-large`
- reranker: `BAAI/bge-reranker-v2-m3`
- LLM: `mistral:7b` (Ollama)

## Docker

CPU (по умолчанию):

```bash
docker compose build
docker compose up
```

GPU (нужен [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

После старта: http://localhost:8501

### Оценка размера на диске

| Компонент | CPU | GPU |
|-----------|-----|-----|
| Образы Docker (`app` + `ollama`) | ~4–6 GB | ~10–14 GB |
| Модели HF (e5-large + reranker) | ~2.2 GB | ~2.2 GB |
| Ollama `mistral:7b` | ~4 GB | ~4 GB |
| Индекс (`storage`) | ~50–500 MB | ~50–500 MB |

Итого после первого полного запуска: **~12–18 GB** (CPU) или **~18–28 GB** (GPU).

## Кэш моделей

При старте Streamlit модели загружаются **один раз** через `@st.cache_resource` и остаются в RAM до перезапуска процесса. Веса Hugging Face кэшируются на диске (`~/.cache/huggingface`); в Docker для этого смонтирован volume `hf_cache`.
