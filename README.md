# CodeLens — умный поиск по кодовой базе

> Чемпионат ГК «Ростелеком», 2-й этап, направление «Искусственный интеллект»

## Описание

> RAG-система для интеллектуального поиска по Python-кодовой базе

Система принимает вопрос на русском или английском языке и находит топ-5 наиболее релевантных фрагментов кода из репозитория — функций и классов. При включённом LLM дополнительно генерирует связный человекочитаемый ответ на основе найденного кода.

Реализовано:
- Под капотом **гибридный поиск**: векторный (ChromaDB + `multilingual-e5-large`) и полнотекстовый (BM25) с настраиваемыми весами. Запросы на русском автоматически переводятся на английский перед поиском.

- Опциональный CrossEncoder-реранкер (`BAAI/bge-reranker-v2-m3`), который переранжирует кандидатов, анализируя пару (вопрос, код) совместно.

- LLM-ответы, локально генерирующиеся через Ollama (`mistral:7b` или другую модель в `LLM_MODEL`) без внешних API.

**Стек:** Python 3.12 · ChromaDB · sentence-transformers · BM25 · Ollama · Streamlit · uv · Docker

### Структура проекта

``` ASCII
RT_second/
├── src/
│   ├── app.py               # Streamlit UI
│   ├── index.py             # AST-парсинг, эмбеддинги, ChromaDB + BM25
│   ├── search.py            # semantic_search, hybrid_search
│   ├── llm.py               # RAG через Ollama (generate_rag_answer)
│   └── settings.py          # USE_GPU, USE_RERANKER, resolve_device()
├── storage/                 # Генерируется при index.py (в .gitignore)
│   ├── chroma_db/
│   │   └── chroma.sqlite3
│   ├── bm25_index.pkl
│   └── bm25_meta.json
├── .gitignore
├── docker-compose.yml
├── docker-compose.gpu.yml
├── Dockerfile
├── entrypoint.sh
├── pyproject.toml
├── uv.lock
└── README.md
```

---
### Клонирование репозитория

```bash
git clone https://github.com/MIvi-test/RT_second.git
cd RT_second
```
---
## Быстрый запуск

Перед запуском необходимо настроить пути для директорий исходного кода, кешa HuggingFace и хранения базы данных ChromaDB (SOURCE, HF_HOME и BD_PATH в соответсвенно в разделе volumes файлов docker-compose).

| Переменная | По умолчанию                         | Описание                                |
| ---------- | ------------------------------------ | --------------------------------------- |
| `BD_PATH`  | `.\storage`                          | Каталог с индексами (ChromaDB, BM25)    |
| `HF_HOME`  | `${USERPROFILE}\.cache\huggingface}` | Путь для сохранения моделей HuggingFace |
| `SOURCE`   | `${USERPROFILE}\Downloads\dataset}`  | Путь к исходному коду для индексации    |

`HF_HOME` также используется для глобального обращения к моделям, которые **уже** установлены в системе.

Более детально информация о значении переменных и их настройке находится в [разделе "Переменные окружения"](#variables).

---
### Linux

``` bash
# пример настройки
export SOURCE_PATH=/app/dataset_case3_v1.0_fix
```

**CPU** (по умолчанию):

```bash
docker compose \  
-f docker-compose.yml \  
up
```

**С поддержкой GPU** (нужен **nvidia-container-toolkit**):

```bash
docker compose \  
-f docker-compose.gpu.linux.yml \  
up
```

---
### Windows

``` PowerShell
# пример настройки
$env:SOURCE="/app/dataset_case3_v1.0_fix"
```

> Для запуска docker-compose с поддержкой LLM в ОС Windows обязательно должен быть запущен docker desktop

**CPU** (по умолчанию):

```bash
docker compose \  
-f docker-compose.yml \  
up
```

**С поддержкой GPU**:

```bash
docker compose \  
-f docker-compose.gpu.win.yml \  
up
```

---
После старта: http://localhost:8501
Для завершения работы выполняются аналогичные команды, но вместо `up --build` используется `down`:

```bash
docker compose down
```

---
## Локальный запуск

### 1. Установка uv

**Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

После установки `uv sync` создаст виртуальное окружение и установит все зависимости из `pyproject.toml` автоматически.

### 2. Установка зависимостей

```bash
uv sync
```

Для GPU (опционально, требуется NVIDIA + CUDA):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Индексация (один раз)

```bash
# путь к датасету — при необходимости через SOURCE_PATH
uv run index.py
```

### 4. Ollama (опционально, для RAG-ответов)

```bash
# Отдельный терминал
ollama serve
# Скачать модель (один раз)
ollama pull qwen3.5:9b 
```

Для выбора другой модели необходимо изменить переменную `LLM_MODEL` (см. [далее](#variables))
### 5. UI

```bash
uv run streamlit run app.py
```

Откройте http://localhost:8501

---
<h2 id="variables">Переменные окружения</h2>

| Переменная     | По умолчанию                       | Описание                                        |
| -------------- | ---------------------------------- | ----------------------------------------------- |
| `SOURCE_PATH`  | `./dataset_case3_v1.0_fix/gymhero` | Путь к исходному коду для индексации            |
| `STORAGE_DIR`  | `./storage`                        | Каталог с индексами (ChromaDB, BM25)            |
| `USE_RERANKER` | `true`                             | Включить CrossEncoder `BAAI/bge-reranker-v2-m3` |
| `USE_GPU`      | `false`                            | Использовать CUDA для эмбеддингов и реранкера   |
| `USE_OLLAMA`   | `true`                             | Включить RAG-генерацию через Ollama             |
| `OLLAMA_HOST`  | `http://localhost:11434`           | Адрес Ollama-сервера                            |
| `LLM_MODEL`    | `qwen3.5:9b`                       | Модель для RAG-генерации                        |
### Настройка запуска через Docker

Для этого необходимо изменить следущие строки в раздели `enviroment` в файлах `docker-compose `и `docker-compose.gpu`
```yml
environment:
- OLLAMA_HOST=http://ollama:11434
- LLM_MODEL=qwen3.5:9b
- PYTHONUNBUFFERED=1
- USE_RERANKER=false
- USE_GPU=false
- USE_OLLAMA=false
- STORAGE_DIR=/storage
- SOURCE_PATH=/dataset_case3_v1.0_fix/gymhero
```
### Настройка локального запуска

#### Linux

``` bash
export SOURCE_PATH=./dataset_case3_v1.0_fix/gymhero
export STORAGE_DIR=./storage
export USE_RERANKER=true
export USE_GPU=false
export USE_OLLAMA=true
export OLLAMA_HOST=http://localhost:11434
export LLM_MODEL=qwen3.5:9b
```

#### Windows

``` PowerShell
$env:SOURCE_PATH="./dataset_case3_v1.0_fix/gymhero"
$env:STORAGE_DIR="./storage"
$env:USE_RERANKER="true"
$env:USE_GPU="false"
$env:USE_OLLAMA="true"
$env:OLLAMA_HOST="http://localhost:11434"
$env:LLM_MODEL="qwen3.5:9b"
```

---

## Архитектура решения

### Индексация (`index.py`)

- **AST-парсинг** — код делится по логическим границам (классы, методы внутри классов, функции верхнего уровня), а не по количеству символов. Дефектные файлы с `SyntaxError` пропускаются.
- **Metadata Injection** — перед векторизацией в текст чанка добавляются путь к файлу, тип и имя объекта: `File path: ...\nObject type: ...\nCode: ...`. Это позволяет модели учитывать контекст расположения кода.
- **Токенизатор `tokenize_code`** — разбивает `camelCase` и `snake_case` на отдельные токены (`create_user` → `create user`), что критически повышает точность BM25 при поиске по именам функций.
- **ChromaDB** — коллекция `code_chunks` с метрикой косинусного расстояния (`hnsw:space: cosine`), запись батчами по 200 элементов. При повторном запуске коллекция пересоздаётся во избежание дубликатов.
- **BM25-индекс** — строится параллельно с ChromaDB и сохраняется на диск (`bm25_index.pkl` + `bm25_meta.json`) в `STORAGE_DIR`.

### Поиск (`search.py`)

| Метод             | Описание                                                                         |
| ----------------- | -------------------------------------------------------------------------------- |
| `semantic_search` | ChromaDB → топ-75 кандидатов → (опционально) реранкер → топ-5                    |
| `hybrid_search`   | ChromaDB (50%) + BM25 (50%) → топ-30 кандидатов → (опционально) реранкер → топ-5 |

Оба метода нормализуют оценки в диапазон 0–1 перед слиянием. Запросы на русском автоматически переводятся на английский через `deep-translator` перед поиском, что повышает релевантность, так как кодовая база написана на английском.

Реранкер `BAAI/bge-reranker-v2-m3` анализирует совместимость пары (вопрос, код) совместно через механизм Attention, в отличие от bi-encoder, который кодирует их раздельно. Это даёт значительный прирост точности на сложных запросах. Отключается через `USE_RERANKER=false`.

Все ресурсы (модель, ChromaDB, BM25) загружаются один раз при первом обращении (`_load()`) и остаются в памяти до перезапуска процесса.

### RAG (`llm.py` + `app.py`)

- При старте UI проверяется доступность Ollama и наличие выбранной в переменной окружения `LLM_MODEL` модели. Если сервер не запущен — переключатель LLM отключается с понятным сообщением.
- В промпт передаются исходный вопрос и топ-5 найденных фрагментов с указанием пути, имени, кода и оценки релевантности.
- Промпт адаптируется под язык вопроса, т.е. ответ генерируется на том же языке, что и запрос.
- Streamlit кэширует модели через `@st.cache_resource`, т.е. повторные запросы не перезагружают модель.

### Docker

Проект полностью контейнеризирован. Доступно два профиля:

| Профиль | Файл                                                           | Требования                        |
| ------- | -------------------------------------------------------------- | --------------------------------- |
| CPU     | `docker-compose.yml`                                           | Docker                            |
| GPU     | `docker-compose.gpu.win.yml` или`docker-compose.gpu.linux.yml` | Docker + nvidia-container-toolkit |

`entrypoint.sh` запускает `index.py` и `streamlit` последовательно внутри контейнера. Веса HuggingFace кэшируются в volume `hf_cache` (`~/.cache/huggingface`) — повторный старт контейнера не перескачивает модели.

## Формат chunk_id

Соответствует спецификации датасета: `{relative_path}:{name}:{start_line}`

Примеры:
- `gymhero/security.py:create_access_token:12`
- `gymhero/crud/base.py:CRUDRepository.get_many:53`
- `gymhero/config.py:Settings:11`

| Поле | Описание |
|---|---|
| `relative_path` | Путь от `SOURCE_PATH`, прямые слэши (`/`) |
| `name` | Имя функции или класса; для метода — `ClassName.method_name` |
| `start_line` | Номер первой строки определения (`def` или `class`), возвращаемый `ast.parse` |

При сравнении `score.py` допускает отклонение ±2 строки по `start_line`.

## Кэш моделей

При старте Streamlit модели загружаются **один раз** через `@st.cache_resource` и остаются в RAM до перезапуска процесса. Веса Hugging Face кэшируются на диске (`~/.cache/huggingface`); в Docker для этого смонтирован volume `hf_cache`.

## Выбор модели  эмбедингов

## Выбор LLM
