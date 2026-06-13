"""Streamlit UI: поиск по коду + опциональный LLM-ответ (RAG) + оценка Precision@5."""
import sys
import streamlit as st
import json
from pathlib import Path

# ------------------- Функция поиска файла -------------------
def find_file(filename: str) -> Path | None:
    """Ищет файл в текущей директории и всех поддиректориях."""
    root = Path.cwd()
    # 1. Проверяем прямо в корне
    direct_path = root / filename
    if direct_path.is_file():
        return direct_path
    # 2. Рекурсивный поиск
    for path in root.rglob(filename):
        if path.is_file():
            return path
    return None

# ------------------- Подключение score.py -------------------
score_path = find_file("score.py")
if score_path is None:
    st.error("Файл score.py не найден. Оценка Precision@5 недоступна.")
    def score_question(top5, correct):
        return 0.0
else:
    # Добавляем директорию score.py в sys.path, если её там нет
    score_dir = score_path.parent
    if str(score_dir) not in sys.path:
        sys.path.insert(0, str(score_dir))
    try:
        from score import score_question
    except ImportError:
        st.error("Не удалось импортировать score_question из score.py")
        def score_question(top5, correct):
            return 0.0

# ------------------- Остальные импорты -------------------
from llm import USE_OLLAMA, check_ollama, fetch_documents_for_chunks, generate_rag_answer
from search import hybrid_search, initialize_search
from settings import USE_GPU, USE_RERANKER

# Пытаемся импортировать psutil для проверки памяти
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def _load_search_engine():
    """Загрузить модели и индексы."""
    return initialize_search()

def _cached_check_ollama():
    return check_ollama()

# Функция для получения топ-5 chunk_id по запросу (используется при оценке)
def get_top5_chunk_ids(query: str) -> list[str]:
    """Возвращает список chunk_id (топ-5) для заданного запроса."""
    raw_results = hybrid_search(query)
    top5 = [r["chunk_id"] for r in raw_results[:5]]
    return top5

# 1. Настройка страницы
st.set_page_config(
    page_title="Advanced Code Search",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

search_runtime = _load_search_engine()
st.title("Семантический поиск по коду")
st.markdown("---")

# 2. Инициализация переменных в st.session_state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_documents" not in st.session_state:
    st.session_state.search_documents = None
if "llm_answer" not in st.session_state:
    st.session_state.llm_answer = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False
if "eval_predictions" not in st.session_state:
    st.session_state.eval_predictions = None

# 3. БОКОВАЯ ПАНЕЛЬ НАСТРОЕК (SIDEBAR)
with st.sidebar:
    st.header("Настройки")
    ollama_ok, ollama_err = _cached_check_ollama()
    if ollama_ok:
        st.success("Локальная LLM доступна")
    else:
        st.error("LLM недоступна")
        if ollama_err:
            st.caption(f"Ошибка: {ollama_err}")

    st.markdown("---")
    st.subheader("Режим поиска")
    st.caption(f"Устройство: `{search_runtime.get('device', 'Неизвестно')}`")
    # Добавлен вывод модели эмбеддингов в боковую панель (угол)
    st.caption(f"Модель эмбеддингов: `{search_runtime.get('embedding_model', 'Неизвестно')}`")
    st.caption(f"Реранкер: `{'вкл' if USE_RERANKER else 'выкл'}` (USE_RERANKER)")
    st.caption(f"GPU: `{'запрошен' if USE_GPU else 'выкл'}` (USE_GPU)")
    st.caption(f"LLM: `{'вкл' if USE_OLLAMA else 'выкл'}` (USE_OLLAMA)")

    st.markdown("---")
    st.subheader("Параметры поиска")

    enable_llm = st.checkbox(
        "Включить генерацию RAG-ответа",
        value=ollama_ok,
        disabled=not ollama_ok,
    )

    filter_type = st.multiselect(
        "Фильтр по типу объектов из ТОП-5",
        options=["function", "class", "method"],
        default=["function", "class", "method"],
        help="Уберите типы, которые не хотите видеть в текущей выдаче",
    )

# 4. ОСНОВНАЯ ЗОНА ИНТЕРФЕЙСА
with st.form(key="search_form"):
    query = st.text_input(
        "Введите поисковый запрос:",
        value=st.session_state.last_query,
        placeholder="Например: как устроена авторизация пользователя?",
    )
    submitted = st.form_submit_button("Запустить поиск", type="primary", use_container_width=True)

# Если форма отправлена и запрос не пустой
if submitted and query.strip():
    q_cleaned = query.strip()
    with st.spinner("Ищем совпадения в репозитории..."):
        raw_results = hybrid_search(q_cleaned)

        LOW_THRESHOLD = 30.0
        filtered_results = [r for r in raw_results if r.get("score", 0) >= LOW_THRESHOLD]

        st.session_state.search_results = filtered_results
        st.session_state.last_query = q_cleaned
        st.session_state.llm_answer = None

    if raw_results:
        chunk_ids = [r["chunk_id"] for r in raw_results]
        st.session_state.search_documents = fetch_documents_for_chunks(chunk_ids)
    else:
        st.session_state.search_documents = None

# Три вкладки: результаты, LLM-пояснение, оценка
tab_results, tab_llm, tab_eval = st.tabs(["Найденные фрагменты кода", "Пояснение от ИИ", "Оценка Precision@5"])

search_clicked = st.button("Запустить поиск", type="primary", use_container_width=True)
search_triggered = search_clicked or st.session_state.get("trigger_search", False)

if (search_triggered or (query.strip() and query.strip() != st.session_state.last_query)) and query.strip():
    if st.session_state.get("trigger_search"):
        st.session_state.trigger_search = False
    q_cleaned = query.strip()

    with st.spinner("Ищем совпадения в репозитории..."):
        raw_results = hybrid_search(q_cleaned)

        # Мягкий выходной фильтр (опускаем до 45.0, чтобы не резать базу)
        LOW_THRESHOLD = 45.0
        filtered_results = [r for r in raw_results if r.get("score", 0) >= LOW_THRESHOLD]

        st.session_state.search_results = filtered_results
        st.session_state.last_query = q_cleaned
        st.session_state.llm_answer = None

    if raw_results:
        chunk_ids = [r["chunk_id"] for r in raw_results]
        st.session_state.search_documents = fetch_documents_for_chunks(chunk_ids)
    else:
        st.session_state.search_documents = None

if st.session_state.search_results:
    results = [r for r in st.session_state.search_results if r["type"] in filter_type]
    
    with tab_results:
        if not results:
            st.warning("В текущем ТОП-5 нет объектов выбранного типа. Измените фильтр в боковой панели.")
        else:
            st.subheader(f"Отображено фрагментов: {len(results)} из 5")

            for idx, hit in enumerate(results, 1):
                with st.container(border=True):
                    col_info, col_metric = st.columns([4, 1])

                    with col_info:
                        st.markdown(f"### {idx}. `{hit['name']}`")
                        st.markdown(
                            f"**Путь к файлу:** `{hit['file_path']}` | **Тип:** `{hit['type']}`"
                        )

                    with col_metric:
                        st.metric(label="Релевантность", value=f"{hit['score']}%")

                    with st.expander("Посмотреть исходный код фрагмента", expanded=(idx == 1)):
                        code_content = st.session_state.search_documents.get(
                            hit["chunk_id"], "# Код отсутствует"
                        )
                        st.code(code_content, language="python")

    with tab_llm:
        if enable_llm:
            if not results:
                st.info("Невозможно сгенерировать ответ: список фрагментов пуст из-за фильтров.")
            else:
                # Добавлен вывод названия LLM-модели во вкладку с LLM
                st.caption(f"Модель LLM: `{search_runtime.get('llm_model', 'Неизвестно')}`")
                
                if st.session_state.llm_answer is None:
                    with st.spinner("Нейросеть анализирует контекст и пишет ответ..."):
                        st.session_state.llm_answer = generate_rag_answer(
                            st.session_state.last_query,
                            results,
                            st.session_state.search_documents,
                        )
                st.subheader("Сгенерированный ответ архитектора")
                st.markdown(st.session_state.llm_answer)
        else:
            st.info("Генерация ответов отключена. Включите чекбокс в боковой панели (требуется Ollama).")

elif st.session_state.search_results is not None:
    with tab_results:
        st.warning("Ничего не найдено по данному запросу.")

# 5. ВКЛАДКА ОЦЕНКИ PRECISION@5
with tab_eval:
    st.header("Оценка точности поиска (Precision@5)")
    st.markdown("Метрика вычисляется по тестовому набору `eval_questions.json` с использованием логики `score.py` (допуск +-2 строки).")
    
    # Поиск eval_questions.json через find_file
    eval_file_path = find_file("eval_questions.json")
    if eval_file_path is None:
        st.error("Файл eval_questions.json не найден. Поместите его в одну из директорий проекта.")
    else:
        if st.button("Запустить оценку", type="primary"):
            with st.spinner("Загрузка вопросов и выполнение поиска..."):
                # 1. Загружаем эталонные вопросы
                with open(eval_file_path, encoding="utf-8") as f:
                    questions = json.load(f)

                # 2. Для каждого вопроса получаем предсказания (топ-5 chunk_id)
                predictions = []
                progress_bar = st.progress(0)
                for i, q in enumerate(questions):
                    qid = q["question_id"]
                    query_text = q["query"]
                    top5_ids = get_top5_chunk_ids(query_text)
                    predictions.append({
                        "question_id": qid,
                        "top_5_chunks": top5_ids
                    })
                    progress_bar.progress((i + 1) / len(questions))

                # Сохраняем в session_state для возможности сохранения
                st.session_state.eval_predictions = predictions

                # 3. Вычисляем Precision@5, используя функцию score_question из score.py
                per_question = []
                for q, pred in zip(questions, predictions):
                    correct = q.get("correct_chunk_ids", [])
                    score = score_question(pred["top_5_chunks"], correct)
                    per_question.append({
                        "question_id": q["question_id"],
                        "difficulty": q.get("difficulty", "unknown"),
                        "language": q.get("language", "unknown"),
                        "n_correct": len(correct),
                        "score": score,
                    })

                # 4. Агрегация
                total = len(per_question)
                mean_score = sum(r["score"] for r in per_question) / total
                by_difficulty = {}
                for r in per_question:
                    d = r["difficulty"]
                    by_difficulty.setdefault(d, []).append(r["score"])
                by_language = {}
                for r in per_question:
                    l = r["language"]
                    by_language.setdefault(l, []).append(r["score"])

                # 5. Вывод результатов
                st.success(f"Оценка завершена. Средний Precision@5 = {mean_score:.3f}")
                st.metric("Итоговый Score", f"{mean_score:.3f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("По сложности")
                    for diff in ["easy", "medium", "hard"]:
                        scores = by_difficulty.get(diff, [])
                        if scores:
                            avg = sum(scores)/len(scores)
                            st.metric(diff.capitalize(), f"{avg:.3f}", f"{len(scores)} вопросов")
                with col2:
                    st.subheader("По языку")
                    for lang in ["ru", "en"]:
                        scores = by_language.get(lang, [])
                        if scores:
                            avg = sum(scores)/len(scores)
                            lang_name = "Русский" if lang == "ru" else "Английский"
                            st.metric(lang_name, f"{avg:.3f}", f"{len(scores)} вопросов")

                st.subheader("Детализация по вопросам")
                data = []
                for r in per_question:
                    matched = round(r["score"] * min(5, r["n_correct"]))
                    data.append({
                        "Вопрос": r["question_id"],
                        "Сложность": r["difficulty"],
                        "Язык": r["language"],
                        "Precision@5": f"{r['score']:.2f}",
                        "Найдено/Ожидалось": f"{matched}/{r['n_correct']}"
                    })
                st.dataframe(data, use_container_width=True)

        # Кнопка сохранения results.json (появляется после оценки)
        if st.session_state.get("eval_predictions"):
            if st.button("Сохранить results.json для отчёта"):
                output_path = Path("results.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(st.session_state["eval_predictions"], f, ensure_ascii=False, indent=2)
                st.success(f"Файл сохранён: {output_path.absolute()}")
                st.info("Вы можете проверить его командой: `python score.py --predictions results.json --questions eval_questions.json`")