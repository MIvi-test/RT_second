"""Streamlit UI: поиск по коду + опциональный LLM-ответ (RAG)."""

import time
import streamlit as st

from llm import USE_OLLAMA, check_ollama, fetch_documents_for_chunks, generate_rag_answer
from search import hybrid_search, initialize_search
from settings import USE_GPU, USE_RERANKER

# Пытаемся импортировать psutil для проверки памяти
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@st.cache_resource
def _load_search_engine():
    """Загрузить модели и индексы один раз на процесс Streamlit."""
    return initialize_search()


@st.cache_data(ttl=30)
def _cached_check_ollama():
    return check_ollama()


# 1. Настройка страницы
st.set_page_config(
    page_title="Advanced Code Search",
    page_icon="🦄",
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
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False

if PSUTIL_AVAILABLE:
    mem = psutil.virtual_memory()
    available_mb = mem.available / (1024 * 1024)
    HISTORY_ENABLED = available_mb >= 100
else:
    HISTORY_ENABLED = True

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
    st.caption(f"Устройство: `{search_runtime['device']}`")
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

    st.markdown("---")
    st.subheader("История запросов")
    if not HISTORY_ENABLED:
        st.warning("Недостаточно свободной памяти. История отключена.")
    else:
        if st.session_state.search_history:
            for i, q in enumerate(st.session_state.search_history):
                if st.button(q, key=f"hist_{i}"):
                    st.session_state.last_query = q
                    st.session_state.search_results = None
                    st.session_state.llm_answer = None
                    st.session_state.trigger_search = True
                    st.rerun()
        else:
            st.caption("История пуста")
        if st.button("Очистить историю", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()

# 4. ОСНОВНАЯ ЗОНА ИНТЕРФЕЙСА
query = st.text_input(
    "Введите поисковый запрос:",
    value=st.session_state.last_query,
    placeholder="Например: как устроена авторизация пользователя?",
)

tab_results, tab_llm = st.tabs(["Найденные фрагменты кода", "Пояснение от ИИ"])

search_clicked = st.button("Запустить поиск", type="primary", use_container_width=True)

search_triggered = search_clicked or st.session_state.get("trigger_search", False)

if (search_triggered or (query.strip() and query.strip() != st.session_state.last_query)) and query.strip():
    if st.session_state.get("trigger_search"):
        st.session_state.trigger_search = False

    if query.strip() == st.session_state.last_query and st.session_state.search_results is not None:
        with st.spinner("Повторный поиск (загрузка из кэша)..."):
            time.sleep(0.4)
    else:
        with st.spinner("Ищем совпадения в репозитории..."):
            raw_results = hybrid_search(query.strip())

            st.session_state.search_results = raw_results
            st.session_state.last_query = query.strip()
            st.session_state.llm_answer = None

            if HISTORY_ENABLED and raw_results:
                q = query.strip()
                if q in st.session_state.search_history:
                    st.session_state.search_history.remove(q)
                st.session_state.search_history.append(q)
                if len(st.session_state.search_history) > 5:
                    st.session_state.search_history = st.session_state.search_history[-5:]

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