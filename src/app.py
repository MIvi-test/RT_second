"""Streamlit UI: поиск по коду + опциональный LLM-ответ (RAG) + оценка Precision@5."""

from search import hybrid_search, semantic_search
from llm import fetch_documents_for_chunks, generate_rag_answer
import sys
import streamlit as st
import json
import hashlib
from pathlib import Path

from config import (
    SOURCE_PATH,
    LLM_MODEL_NAME,
    USE_GPU,
    USE_RERANKER,
    USE_OLLAMA,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ---------------------- КЭШИРОВАНИЕ ----------------------


@st.cache_resource
def _load_search_engine():
    """Загрузить модели и индексы (кэшируется)."""
    from search import initialize_search

    return initialize_search()


@st.cache_data(ttl=60)
def _cached_check_ollama():
    from llm import check_ollama

    return check_ollama()


# ---------------------- ОСТАЛЬНЫЕ ФУНКЦИИ ----------------------
__all__ = [
    "find_file",
    "get_top5_chunk_ids",
    "_load_search_engine",
    "_cached_check_ollama",
]


def find_file(filename: str) -> Path | None:
    """Ищет файл в корневой директории и всех поддиректориях."""
    root = Path(__file__).resolve().parent.parent
    direct_path = root / filename
    if direct_path.is_file():
        return direct_path
    for path in root.rglob(filename):
        if path.is_file():
            return path
    return None


# Подключение score.py
score_path = find_file("score.py")
if score_path is None:
    st.error("Файл score.py не найден. Оценка Precision@5 недоступна.")

    def score_question(top5, correct):
        return 0.0
else:
    score_dir = score_path.parent
    if str(score_dir) not in sys.path:
        sys.path.insert(0, str(score_dir))
    try:
        with open(score_path, "rb") as f:
            digest = hashlib.file_digest(f, "sha512")
        # Хеш-сумма для проверки целостности (опционально)
        if digest.hexdigest() == "08f8c2eb03086eebba4998569d55b227610d791f9f66ac3d7d741ce89915d88a887592a5966c2e7e429ae120cc2250de4fe1a878e6d06dc612e2eae9951a1c71":
            from score import score_question
    except ImportError:
        st.error("Не удалось импортировать score_question из score.py")

        def score_question(top5, correct):
            return 0.0


def get_top5_chunk_ids(query: str) -> list[str]:
    """Возвращает список chunk_id (топ-5) для заданного запроса с учётом выбранного языка и перевода."""
    mode = st.session_state.get("search_mode", "hybrid")
    lang = st.session_state.get("search_lang", "python")
    use_translation = st.session_state.get("use_translation", True)

    if mode == "semantic":
        raw_results = semantic_search(
            query, lang=lang, use_translation=use_translation)
    else:
        raw_results = hybrid_search(
            query, lang=lang, use_translation=use_translation)

    top5 = [r["chunk_id"] for r in raw_results[:5]]
    return top5


# ---------------------- НАСТРОЙКА СТРАНИЦЫ ----------------------
st.set_page_config(
    page_title="Advanced Code Search",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

search_runtime = _load_search_engine()
st.title("Поиск по коду")
st.markdown("---")

# Инициализация session_state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_documents" not in st.session_state:
    st.session_state.search_documents = None
if "llm_answer" not in st.session_state:
    st.session_state.llm_answer = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "eval_predictions" not in st.session_state:
    st.session_state.eval_predictions = None
if "save_triggered" not in st.session_state:
    st.session_state["save_triggered"] = False
if "search_input" not in st.session_state:
    st.session_state.search_input = ""
if "search_lang" not in st.session_state:
    st.session_state.search_lang = "python"
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "hybrid"
if "use_translation" not in st.session_state:
    st.session_state.use_translation = True
if "enable_llm" not in st.session_state:
    st.session_state.enable_llm = False

# ---------------------- БОКОВАЯ ПАНЕЛЬ (информация и фильтр) ----------------------
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
    st.subheader("Информация о системе")
    st.caption(f"Устройство: `{search_runtime.get('device', 'Неизвестно')}`")
    st.caption(
        f"Модель эмбеддингов: `{search_runtime.get('embedding_model', 'Неизвестно')}`")
    st.caption(f"Реранкер: `{'вкл' if USE_RERANKER else 'выкл'}`")
    st.caption(f"GPU: `{'запрошен' if USE_GPU else 'выкл'}`")
    st.caption(f"LLM: `{'вкл' if USE_OLLAMA else 'выкл'}`")
    if USE_OLLAMA:
        st.caption(f"Модель: `{LLM_MODEL_NAME}`")

    st.markdown("---")
    st.subheader("Параметры поиска")

    lang_choice = st.selectbox(
        "Язык кода:",
        options=["Python", "Java"],
        index=0 if st.session_state.search_lang == "python" else 1,
        help="Выберите язык, по которому будет выполняться поиск (BM25 и фильтрация коллекции).",
    )
    st.session_state["search_lang"] = "python" if lang_choice == "Python" else "java"

    search_mode = st.radio(
        "Режим поиска",
        options=["semantic", "hybrid"],
        index=1 if st.session_state.search_mode == "hybrid" else 0,
        help="semantic — только эмбеддинги; hybrid — BM25 + эмбеддинги",
    )
    st.session_state["search_mode"] = search_mode

    use_translation = st.checkbox(
        "Переводить запросы на английский",
        value=st.session_state.use_translation,
        help="Если включено, русские запросы будут переводиться перед поиском",
    )
    st.session_state["use_translation"] = use_translation

    enable_llm = st.checkbox(
        "Включить генерацию RAG-ответа",
        value=st.session_state.enable_llm or ollama_ok,
        disabled=not ollama_ok,
    )
    st.session_state["enable_llm"] = enable_llm

    st.markdown("---")
    st.subheader("Фильтр типов (мгновенный)")
    filter_type = st.multiselect(
        "Показывать только",
        options=["function", "class", "method"],
        default=["function", "class", "method"],
        help="Изменение фильтра не вызывает перезапуск поиска",
    )

# ---------------------- ОСНОВНАЯ ФОРМА (поиск + параметры) ----------------------
with st.form(key="search_form"):
    query = st.text_input(
        "Введите поисковый запрос:",
        value=st.session_state.last_query,
        placeholder="Например: как устроена авторизация пользователя?",
        max_chars=200,
    )

    submitted = st.form_submit_button(
        "Запустить поиск", type="primary", width="stretch")

# ---------------------- ОБРАБОТКА ПОИСКА ----------------------
if submitted and query.strip():
    q_cleaned = query.strip()
    with st.spinner("Ищем совпадения в репозитории..."):
        mode = st.session_state["search_mode"]
        lang = st.session_state["search_lang"]
        use_translation = st.session_state["use_translation"]

        try:
            if mode == "semantic":
                raw_results = semantic_search(
                    q_cleaned, lang=lang, use_translation=use_translation)
            else:
                raw_results = hybrid_search(
                    q_cleaned, lang=lang, use_translation=use_translation)
        except Exception as e:
            st.error(f"Ошибка при выполнении поиска: {e}")
            raw_results = []

        # Фильтр по минимальному скору (порог 0, т.е. все)
        filtered_results = [r for r in raw_results if r.get("score", 0) >= 0.0]
        st.session_state.search_results = filtered_results
        st.session_state.last_query = q_cleaned
        st.session_state.llm_answer = None
        st.session_state.search_documents = {}

        if raw_results:
            try:
                chunk_ids = [r["chunk_id"] for r in raw_results]
                st.session_state.search_documents = fetch_documents_for_chunks(
                    chunk_ids)
            except Exception as e:
                st.error(f"Не удалось загрузить тексты фрагментов: {e}")
                st.session_state.search_documents = {}

# ---------------------- ВКЛАДКИ ----------------------
tab_results, tab_llm, tab_eval = st.tabs(
    ["Найденные фрагменты кода", "Пояснение от ИИ", "Оценка Precision@5"])

if st.session_state.search_results:
    results = [
        r for r in st.session_state.search_results if r["type"] in filter_type]
    documents = st.session_state.search_documents or {}

    with tab_results:
        if not results:
            st.warning(
                "В текущем ТОП-5 нет объектов выбранного типа. Измените фильтр в боковой панели.")
        else:
            st.subheader(
                f"Отображено фрагментов: {len(results)} из {len(st.session_state.search_results)}")
            for idx, hit in enumerate(results, 1):
                with st.container(border=True):
                    col_info, col_metric = st.columns([4, 1])
                    with col_info:
                        st.markdown(f"### {idx}. `{hit['name']}`")
                        st.markdown(
                            f"**Путь к файлу:** `{hit['file_path']}` | **Тип:** `{hit['type']}`")
                    with col_metric:
                        st.metric(label="Релевантность",
                                  value=f"{hit['score']}%")
                    with st.expander("Посмотреть исходный код фрагмента", expanded=(idx == 1)):
                        code_content = documents.get(
                            hit["chunk_id"], "# Код отсутствует")
                        # Определяем язык для подсветки по расширению файла или по выбранному языку
                        file_path = hit.get("file_path", "")
                        if file_path.endswith(".py"):
                            code_lang = "python"
                        elif file_path.endswith(".java"):
                            code_lang = "java"
                        else:
                            code_lang = st.session_state.get(
                                "search_lang", "python")
                        st.code(code_content, language=code_lang)

    with tab_llm:
        if st.session_state.get("enable_llm", False):
            if not results:
                st.info(
                    "Невозможно сгенерировать ответ: список фрагментов пуст из-за фильтров.")
            else:
                st.caption(f"Модель LLM: `{LLM_MODEL_NAME}`")
                if st.session_state.llm_answer is None:
                    with st.spinner("Нейросеть анализирует контекст и пишет ответ..."):
                        try:
                            st.session_state.llm_answer = generate_rag_answer(
                                st.session_state.last_query,
                                results,
                                documents,
                            )
                        except Exception as e:
                            st.session_state.llm_answer = f"Ошибка генерации ответа: {e}"
                st.subheader("Сгенерированный ответ архитектора")
                st.markdown(st.session_state.llm_answer)
        else:
            st.info(
                "Генерация ответов отключена. Включите чекбокс в боковой панели (требуется Ollama).")

elif st.session_state.search_results is not None:
    with tab_results:
        st.warning("Ничего не найдено по данному запросу.")

# ---------------------- ОЦЕНКА ----------------------
with tab_eval:
    st.header("Оценка точности поиска (Precision@5)")
    st.markdown(
        "Метрика вычисляется по тестовому набору `eval_questions.json` с использованием логики `score.py` (допуск +-2 строки)."
    )
    eval_file_path = find_file("eval_questions.json")
    if eval_file_path is None:
        st.error(
            "Файл eval_questions.json не найден. Поместите его в одну из директорий проекта.")
    else:
        if "eval_running" not in st.session_state:
            st.session_state.eval_running = False

        if st.button("Запустить оценку", type="primary", disabled=st.session_state.eval_running):
            st.session_state.eval_running = True
            try:
                with st.spinner("Загрузка вопросов и выполнение поиска..."):
                    with open(eval_file_path, encoding="utf-8") as f:
                        questions = json.load(f)
                    predictions = []
                    progress_bar = st.progress(0)
                    for i, q in enumerate(questions):
                        try:
                            top5_ids = get_top5_chunk_ids(q["query"])
                        except Exception as e:
                            st.error(
                                f"Ошибка поиска для вопроса {q.get('question_id')}: {e}")
                            top5_ids = []
                        predictions.append(
                            {"question_id": q["question_id"], "top_5_chunks": top5_ids})
                        progress_bar.progress((i + 1) / len(questions))
                    st.session_state.eval_predictions = predictions

                    per_question = []
                    for q, pred in zip(questions, predictions):
                        correct = q.get("correct_chunk_ids", [])
                        score = score_question(pred["top_5_chunks"], correct)
                        per_question.append(
                            {
                                "question_id": q["question_id"],
                                "difficulty": q.get("difficulty", "unknown"),
                                "language": q.get("language", "unknown"),
                                "n_correct": len(correct),
                                "score": score,
                            }
                        )
                    total = len(per_question)
                    mean_score = sum(r["score"] for r in per_question) / total
                    by_difficulty = {}
                    by_language = {}
                    for r in per_question:
                        by_difficulty.setdefault(
                            r["difficulty"], []).append(r["score"])
                        by_language.setdefault(
                            r["language"], []).append(r["score"])

                    st.success(
                        f"Оценка завершена. Средний Precision@5 = {mean_score:.3f}")
                    st.metric("Итоговый Score", f"{mean_score:.3f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("По сложности")
                        for diff in ["easy", "medium", "hard"]:
                            scores = by_difficulty.get(diff, [])
                            if scores:
                                avg = sum(scores) / len(scores)
                                st.metric(
                                    diff.capitalize(), f"{avg:.3f}", f"{len(scores)} вопросов")
                    with col2:
                        st.subheader("По языку")
                        for lang in ["ru", "en"]:
                            scores = by_language.get(lang, [])
                            if scores:
                                avg = sum(scores) / len(scores)
                                lang_name = "Русский" if lang == "ru" else "Английский"
                                st.metric(
                                    lang_name, f"{avg:.3f}", f"{len(scores)} вопросов")

                    st.subheader("Детализация по вопросам")
                    data = []
                    for r in per_question:
                        matched = round(r["score"] * min(5, r["n_correct"]))
                        data.append(
                            {
                                "Вопрос": r["question_id"],
                                "Сложность": r["difficulty"],
                                "Язык": r["language"],
                                "Precision@5": f"{r['score']:.2f}",
                                "Найдено/Ожидалось": f"{matched}/{r['n_correct']}",
                            }
                        )
                    st.dataframe(data, width="stretch")
            finally:
                st.session_state.eval_running = False

        if st.session_state.get("eval_predictions"):
            if st.button("Сохранить results.json для отчёта"):
                st.session_state["save_triggered"] = True
            if st.session_state.get("save_triggered"):
                output_path = Path("results.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        st.session_state["eval_predictions"], f, ensure_ascii=False, indent=2)
                st.success(f"Файл сохранён: {output_path.absolute()}")
                st.info(
                    "Вы можете проверить его командой: `python score.py --predictions results.json --questions eval_questions.json`"
                )
                st.session_state["save_triggered"] = False
