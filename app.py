"""Streamlit UI: поиск по коду + опциональный LLM-ответ (RAG)."""

import streamlit as st

from llm import check_ollama, fetch_documents_for_chunks, generate_rag_answer
from search import hybrid_search

st.set_page_config(page_title="Code Search", layout="wide")
st.title("Семантический поиск по коду")

ollama_ok, ollama_err = check_ollama()

query = st.text_input("Запрос", placeholder="Например: как создать пользователя?")
enable_llm = st.checkbox("Включить LLM-ответ", disabled=not ollama_ok)

if not ollama_ok and ollama_err:
    st.info(ollama_err)

if st.button("Поиск", type="primary") and query.strip():
    with st.spinner("Поиск..."):
        results = hybrid_search(query.strip(), top_k=5)
        chunk_ids = [r["chunk_id"] for r in results]
        documents = fetch_documents_for_chunks(chunk_ids)

    if not results:
        st.warning("Ничего не найдено.")
    else:
        for hit in results:
            with st.container(border=True):
                st.markdown(f"**{hit['file_path']}** — `{hit['name']}` ({hit['type']})")
                st.caption(f"Релевантность: {hit['score']}%")
                st.code(documents.get(hit["chunk_id"], ""), language="python")

        if enable_llm:
            with st.spinner("Генерация LLM-ответа..."):
                answer = generate_rag_answer(query.strip(), results, documents)
            st.subheader("Ответ LLM")
            st.markdown(answer)
