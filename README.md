# CodeLens — умный поиск по кодовой базе

> Чемпионат ГК «Ростелеком», 2-й этап, направление «Искусственный интеллект»

## Описание

> RAG-система для интеллектуального поиска по Python-кодовой базе

Система принимает вопрос на русском или английском языке и находит топ-5 наиболее релевантных фрагментов кода из репозитория — функций и классов. При включённом LLM дополнительно генерирует связный человекочитаемый ответ на основе найденного кода.

Реализовано:
- Под капотом **гибридный поиск**: векторный (ChromaDB + `BAAI/bge-m3`) и полнотекстовый (BM25) с настраиваемыми весами. Запросы на русском автоматически переводятся на английский перед поиском.

- Опциональный CrossEncoder-реранкер (`BAAI/bge-reranker-v2-m3`), который переранжирует кандидатов, анализируя пару (вопрос, код) совместно.

- LLM-ответы, локально генерирующиеся через Ollama (`llama3.2:3b` или другую модель в `LLM_MODEL`) без внешних API.

**Стек:** Python 3.12 · ChromaDB · sentence-transformers · BM25 · Ollama · Streamlit · uv · Docker

### Структура проекта

``` ASCII
RT_second/
├── src/
│   ├── app.py               # Streamlit UI
│   ├── index.py             # AST-парсинг, эмбеддинги, ChromaDB + BM25
│   ├── index_java.py
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
├── docker-compose.gpu.win.yml
├── docker-compose.gpu.linux.yml
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

Установить uv как переменную окружения path, если это не произошло автоматически:

```PowerShell
#временно (на текущую сессию в powershell)
$env:Path += ";C:\путь\к\папке\с\uv"
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
uv run index_java.py
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

| Переменная               | По умолчанию                          | Описание                                        |
| :----------------------- | :------------------------------------ | :---------------------------------------------- |
| **SCRIPT_DIR**           | `Path(__file__).resolve().parent`     | Корневая директория текущего скрипта            |
| **STORAGE_DIR**          | `SCRIPT_DIR / "storage"`              | Каталог для хранения индексов и метаданных      |
| **CHROMA_PATH**          | `STORAGE_DIR / "chroma_db"`           | Путь к векторной базе данных ChromaDB           |
| **COLLECTION_NAME**      | `"code_chunks"`                       | Название коллекции в ChromaDB                   |
| **SOURCE_PATH**          | `SCRIPT_DIR`                          | Путь к исходному коду для индексации            |
| **BM25_INDEX**           | `STORAGE_DIR / "bm25_index.pkl"`      | Файл основного индекса BM25                     |
| **BM25_META**            | `STORAGE_DIR / "bm25_meta.json"`      | Файл метаданных BM25                            |
| **BM25_INDEX_JAVA**      | `STORAGE_DIR / "bm25_index_java.pkl"` | Индекс BM25 для Java-файлов                     |
| **BM25_META_JAVA**       | `STORAGE_DIR / "bm25_meta_java.json"` | Метаданные BM25 для Java-файлов                 |
| **DEFAULT_PREDICTIONS**  | `SOURCE_PATH / "results.json"`        | Путь для сохранения результатов предсказаний    |
| **DEFAULT_QUESTIONS**    | `SOURCE_PATH / "eval_questions.json"` | Файл с тестовыми вопросами для валидации        |
| **SCORE_SCRIPT**         | `SOURCE_PATH / "score.py"`            | Скрипт для расчета финальных метрик (Precision) |
| **EVAL_SCRIPT**          | `SOURCE_PATH / "eval_questions.json"` | Путь к скрипту или данным оценки качества       |
| **EMBEDDING_MODEL_NAME** | `"BAAI/bge-m3"`                       | Активная модель эмбеддингов                     |
| **LLM_MODEL_NAME**       | `"llama3.2:3b"`                       | Модель для RAG-генерации через Ollama           |
| **RERANKER_MODEL_NAME**  | `"BAAI/bge-reranker-v2-m3"`           | Модель для Cross-Encoder переранжирования       |
| **USE_GPU**              | `false`                               | Флаг использования CUDA                         |
| **USE_RERANKER**         | `false`                               | Флаг включения переранжирования результатов     |
| **USE_OLLAMA**           | `false`                               | Флаг включения генерации ответов через LLM      |
| **LOG_LEVEL**            | `"INFO"`                              | Уровень логирования системы                     |

### Настройка запуска через Docker

Для этого необходимо изменить следущие строки в раздели `enviroment` в файлах `docker-compose `и `docker-compose.gpu`
```yml
# Пример
environment:
- OLLAMA_HOST=http://ollama:11434
- LLM_MODEL=qwen3.5:9b
- USE_RERANKER=false
- USE_GPU=false
- USE_OLLAMA=false
- STORAGE_DIR=/storage
- SOURCE_PATH=/dataset_case3_v1.0_fix/gymhero
```
### Настройка локального запуска

#### Linux

``` bash
# Пример
export SOURCE_PATH=./dataset_case3_v1.0_fix/gymhero
export STORAGE_DIR=./storage
export USE_RERANKER=true
export USE_GPU=false
export USE_OLLAMA=true
export OLLAMA_HOST=http://localhost:11434
export LLM_MODEL_NAME=qwen3.5:9b
```

#### Windows

``` PowerShell
# Пример
$env:SOURCE_PATH="./dataset_case3_v1.0_fix/gymhero"
$env:STORAGE_DIR="./storage"
$env:USE_RERANKER="true"
$env:USE_GPU="false"
$env:USE_OLLAMA="true"
$env:OLLAMA_HOST="http://localhost:11434"
$env:LLM_MODEL_NAME="qwen3.5:9b"
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

---
## Выбор модели эмбедингов

Для построения векторного индекса системы RAG было проведено детальное сравнительное тестирование 5 моделей на фиксированном тестовом наборе данных (`eval_questions.json`) при помощи предоставленного `score.py`.

Каждая модель оценивалась по метрике качества **Precision@5** в 4 конфигурациях:

1. **Семантический поиск**
2. **Семантический поиск с автопереводом запроса на английский язык**
3. **Гибридный поиск**
4. **Гибридный поиск с автопереводом запроса на английский язык**

### Итоговые результаты

Модель **BAAI/bge-m3** была выбрана как основная (Production-режим), так как она демонстрирует лучший результат по Precision@5, а также предоставляет наиболее полные возможности для поиска.

| Модель                                | Лучший P@5 | Режим    | Перевод |
| ------------------------------------- | ---------- | -------- | ------- |
| **BAAI/bge-m3**                       | **0.800**  | semantic | -       |
| 44WXNRFEELS.../CODE_VERONICA          | 0.767      | semantic | -       |
| intfloat/multilingual-e5-large        | 0.767      | semantic | +       |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.700      | semantic | +       |
| sentence-transformers/LaBSE           | *0.589*    | semantic | -       |
> Курсивом отмечены результаты, которые не прошли порог равный 0.6.

---
### Дополнительные результаты

| Модель                                    | Средний P@5 | Худший P@5 | Размер     | Объем/ ед. метрики (GB) |
| ----------------------------------------- | ----------- | ---------- | ---------- | ----------------------- |
| **44WXNR.../CODE_VERONICA**               | **0.722**   | **0.667**  | 2.3 GB     | 3                       |
| BAAI/bge-m3                               | 0.714       | 0.611      | 4.6 GB     | 5.75                    |
| intfloat/multilingual-e5-large            | 0.692       | *0.589*    | 2.3 GB     | 3                       |
| **paraphrase-multilingual-MiniLM-L12-v2** | 0.667       | 0.633      | **0.5 GB** | **0.714**               |
| sentence-transformers/LaBSE               | *0.570*     | *0.544*    | 1.9 GB     | 3.226                   |
Модели, которые имеют своё применение:
- Модель **paraphrase-multilingual-MiniLM-L12-v2** является наиболее эффективной, если учитывать размер, что может быть полезно при ограниченных ресурсах или быстром развертывании
- Модель **44WXNRFEELSLIKEPINSANDNEEDLESINMYHEART/CODE_VERONICA** показывает самый стабильный результат, который не зависит от конкретного решения индексации и архитектуры системы в целом.
---
## Выбор LLM

В качестве базового текстового генератора (генеративной части RAG-конвейера) была выбрана и развернута локальная модель **llama3.2:3b** через экосистему Ollama.

**Почему именно она:**
- **Самые развернутые ответы:** Модель не просто выплевывает кусок кода, а выдает максимально подробный и структурированный разбор: объясняет логику, пишет комментарии и пошагово расписывает, как этот код внедрить.
- **Высокая скорость:** Модель очень быстрая, начинает генерировать ответ уже через **0.6 сек** после запроса и выдает около **40 токенов/сек**. Стриминг текста в интерфейсе идет плавно и без зависаний.
- **Легковесность:** Занимает всего **~2.2 ГБ**. Это критично, так как всё запускается локально, модель оставляет достаточно оперативной памяти для работы векторного индекса FAISS и веб-интерфейса Streamlit, исключая падения по нехватке памяти.