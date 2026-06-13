"""RAG-ответ через Ollama (mistral:7b) на основе найденных фрагментов кода."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import chromadb

SCRIPT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", SCRIPT_DIR / "storage"))
CHROMA_PATH = STORAGE_DIR / "chroma_db"
COLLECTION_NAME = "code_chunks"
DEFAULT_MODEL = "mistral:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
USE_OLLAMA = os.environ.get("USE_OLLAMA", "false").lower() not in {"0", "false", "no", "off"}

# Ensure ollama Python client picks up correct host
os.environ.setdefault("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

logger = logging.getLogger(__name__)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def check_ollama(model: str = DEFAULT_MODEL) -> tuple[bool, str | None]:
    """Проверить доступность Ollama и наличие модели. Возвращает (ok, сообщение об ошибке)."""
    if not USE_OLLAMA:
        return False, "LLM disabled"

    try:
        import ollama
    except ImportError:
        return False, "Пакет ollama не установлен"

    try:
        response = ollama.list()
        names = [m.model for m in response.models]
        if not any(n == model or n.startswith(f"{model}:") or model in n for n in names):
            return False, f"Модель {model} не найдена. Выполните: ollama pull {model}"
        return True, None
    except Exception:
        logger.exception("Ollama unavailable")
        return False, "Ollama недоступна. Запустите: ollama serve"


def fetch_documents_for_chunks(chunk_ids: list[str]) -> dict[str, str]:
    """Загрузить тексты чанков из ChromaDB по их id."""
    if not chunk_ids:
        return {}

    data = _get_collection().get(ids=chunk_ids, include=["documents"])
    return {cid: doc for cid, doc in zip(data["ids"], data["documents"])}


def _build_prompt(question: str, results: list[dict], documents: dict[str, str]) -> str:
    parts = [
        "Ты — ассистент по анализу кодовой базы. "
        "На основе найденных фрагментов кода дай связный человекочитаемый ответ на вопрос пользователя. "
        "Отвечай на том же языке, что и вопрос. "
        "Если в фрагментах нет достаточной информации — скажи об этом честно.",
        f"\nВопрос: {question}\n",
        "Найденные фрагменты:",
    ]

    for i, hit in enumerate(results, start=1):
        code = documents.get(hit["chunk_id"], "")
        parts.append(
            f"\n--- Фрагмент {i}: {hit['file_path']} :: {hit['name']} "
            f"(релевантность {hit.get('score', 0)}%) ---\n{code}"
        )

    parts.append("\nСвязный ответ:")
    return "\n".join(parts)


def generate_rag_answer(
    question: str,
    results: list[dict],
    documents: dict[str, str] | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Сгенерировать RAG-ответ: вопрос + топ-N фрагментов → связный текст."""
    if not results:
        return "По запросу не найдено фрагментов для формирования ответа."

    if not USE_OLLAMA:
        return "LLM generation disabled. Запустите Ollama и включите USE_OLLAMA=true, если хотите использовать эту функцию."

    try:
        import ollama
    except ImportError:
        return "Пакет ollama не установлен. Установите зависимость ollama, чтобы использовать LLM."

    if documents is None:
        chunk_ids = [r["chunk_id"] for r in results]
        documents = fetch_documents_for_chunks(chunk_ids)

    prompt = _build_prompt(question, results, documents)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content.strip()
