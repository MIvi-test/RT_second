import json
import os
import pickle
import re
import logging
from functools import lru_cache
import numpy as np
from pathlib import Path
import chromadb
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from config import *
from typing import Any, Dict, List, Optional

# Logger
logger = logging.getLogger(__name__)
try:
    logging.basicConfig()
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
except Exception:
    # fallback if LOG_LEVEL invalid or basicConfig already called
    logger.setLevel(logging.INFO)

__all__ = [
    "initialize_search",
    "semantic_search",
    "hybrid_search",
    "tokenize_code",
]

# Module-level singletons: resources are loaded lazily on first search call
_model: Optional[Any] = None
_collection: Optional[Any] = None
_bm25: Optional[Any] = None
_bm25_meta: Optional[dict] = None
_reranker: Optional[Any] = None
_device: Optional[str] = None

# Method name constants to avoid magic strings
METHOD_SEMANTIC = "semantic"
METHOD_HYBRID = "hybrid"
METHOD_SUFFIX_RERANK = " + reranker"
METHOD_SEMANTIC_WITH_RERANK = METHOD_SEMANTIC + METHOD_SUFFIX_RERANK
METHOD_HYBRID_WITH_RERANK = METHOD_HYBRID + METHOD_SUFFIX_RERANK


def _translate_to_english(text: str) -> str:
    """Translate query to English if it contains Cyrillic characters.

    This function always attempts translation for Cyrillic text. Control of
    whether translation should occur per-call is handled by the caller
    (search functions) via their `use_translation` parameter.
    """
    # Простая детекция кириллицы — переводим только русскоязычные запросы
    if re.search(r"[а-яА-ЯёЁ]", text):
        try:
            logger.debug("Translating query to English via GoogleTranslator")
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            logger.exception("Translation failed; using original text")
            return text  # если перевод упал — используем оригинал
    return text


def tokenize_code(text: str) -> list[str]:
    # split CamelCase and non-alphanumeric chars, return lowercase tokens
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]", " ", text)
    return text.lower().split()


def _load():
    """Lazy load model, optional reranker, Chroma collection and BM25 index.

    Note: this function mutates module-level singletons; callers should use
    `_ensure_loaded()` which is cached to avoid repeated heavy work.
    """
    global _model, _collection, _bm25, _bm25_meta, _reranker, _device

    if _device is None:
        _device = resolve_device()
        logger.info("[search] device=%s, reranker=%s", _device, USE_RERANKER)

    if _model is None:
        logger.info("Loading SentenceTransformer model on %s...", _device)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)

    if USE_RERANKER and _reranker is None:
        logger.info("Loading CrossEncoder reranker on %s...", _device)
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=_device)

    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)

    if _bm25 is None:
        bm25_path = STORAGE_DIR / "bm25_index.pkl"
        meta_path = STORAGE_DIR / "bm25_meta.json"

        if not bm25_path.exists():
            raise FileNotFoundError("bm25_index.pkl not found — run index.py")

        with open(bm25_path, "rb") as f:
            _bm25 = pickle.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            _bm25_meta = json.load(f)


# Cache the heavy load so repeated Streamlit reruns don't retrigger it.
@lru_cache(maxsize=1)
def _ensure_loaded() -> bool:
    _load()
    return True


def initialize_search() -> dict:
    """Load search resources once and return runtime info for UI/logging."""
    _ensure_loaded()
    return {
        "device": _device or "cpu",
        "use_reranker": USE_RERANKER,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": "BAAI/bge-reranker-v2-m3" if USE_RERANKER else None,
    }


def _apply_reranker(
    query: str,
    candidates: list[dict],
    *,
    score_key: str,
    method_with: str,
    method_without: str,
) -> list[dict]:
    """Score candidates with CrossEncoder or fall back to base scores."""
    if USE_RERANKER and _reranker is not None:
        pairs = [[query, c["document"]] for c in candidates if c.get("document")]
        if pairs:
            rerank_scores = _reranker.predict(pairs)
            idx = 0
            for c in candidates:
                if c.get("document"):
                    c["score"] = round(float(rerank_scores[idx]) * 100, 2)
                    c["method"] = method_with
                    idx += 1
                else:
                    c["score"] = round(c[score_key] * 100, 2)
                    c["method"] = method_without
        else:
            for c in candidates:
                c["score"] = round(c[score_key] * 100, 2)
                c["method"] = method_without
    else:
        for c in candidates:
            c["score"] = round(c[score_key] * 100, 2)
            c["method"] = method_without

    for c in candidates:
        c.pop("document", None)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def semantic_search(query: str, top_k: int = 5, use_translation: Optional[bool] = True) -> list[dict]:
    """Semantic search with optional CrossEncoder reranking."""
    _ensure_loaded()
    if use_translation:
        query = _translate_to_english(query)

    fetch_k = 75 if USE_RERANKER else top_k
    include_docs = USE_RERANKER

    query_vec = _model.encode([query], convert_to_numpy=True)

    hits = _collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=fetch_k,
        include=["metadatas", "distances"] + (["documents"] if include_docs else []),
    )

    candidates = []
    documents = hits["documents"][0] if include_docs else [None] * len(hits["ids"][0])
    for _id, meta, dist, doc in zip(
        hits["ids"][0], hits["metadatas"][0], hits["distances"][0], documents
    ):
        item = {
            "chunk_id": _id,
            "file_path": meta.get("file_path", "?"),
            "type": meta.get("type", "?"),
            "name": meta.get("name", "?"),
            "semantic_score": 1 - dist,
        }
        if include_docs:
            item["document"] = doc
        candidates.append(item)

    scored = _apply_reranker(
        query,
        candidates,
        score_key="semantic_score",
        method_with=METHOD_SEMANTIC_WITH_RERANK,
        method_without=METHOD_SEMANTIC,
    )
    return scored[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.5,
    use_translation: Optional[bool] = True,
) -> list[dict]:
    """Combine semantic and BM25 scores, optionally rerank with CrossEncoder."""
    _ensure_loaded()
    if use_translation:
        query = _translate_to_english(query)

    fetch_k = top_k * 3
    include_docs = USE_RERANKER

    query_vec = _model.encode([query], convert_to_numpy=True)
    hits = _collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=fetch_k,
        include=["metadatas", "distances"] + (["documents"] if include_docs else []),
    )

    sem_scores: dict[str, float] = {}
    meta_by_id: dict[str, dict] = {}
    doc_by_id: dict[str, str] = {}
    documents = hits["documents"][0] if include_docs else [None] * len(hits["ids"][0])
    for _id, meta, dist, doc in zip(
        hits["ids"][0], hits["metadatas"][0], hits["distances"][0], documents
    ):
        cid = _id
        sem_scores[cid] = 1 - dist
        meta_by_id[cid] = meta
        if include_docs and doc:
            doc_by_id[cid] = doc

    tokens = tokenize_code(query)
    raw_scores = _bm25.get_scores(tokens)

    max_score = raw_scores.max()
    bm25_scores: dict[str, float] = {}

    if max_score > 0:
        norm = raw_scores / max_score
        top_indices = np.argsort(norm)[::-1][:fetch_k]

        for idx in top_indices:
            if norm[idx] > 0:
                bm25_meta_item = _bm25_meta[idx]
                rel_path = bm25_meta_item.get("file_path", "?")
                name = bm25_meta_item.get("name", "?")
                start_line = bm25_meta_item.get("start_line", "0")
                cid = f"{rel_path}:{name}:{start_line}"
                bm25_scores[cid] = float(norm[idx])
                if cid not in meta_by_id:
                    meta_by_id[cid] = bm25_meta_item

    all_ids = set(sem_scores) | set(bm25_scores)

    combined = []
    for cid in all_ids:
        sem = sem_scores.get(cid, 0.0)
        bm25 = bm25_scores.get(cid, 0.0)
        final = semantic_weight * sem + bm25_weight * bm25

        meta = meta_by_id[cid]
        item = {
            "chunk_id": cid,
            "file_path": meta.get("file_path", "?"),
            "type": meta.get("type", "?"),
            "name": meta.get("name", "?"),
            "hybrid_score": final,
        }
        if include_docs:
            item["document"] = doc_by_id.get(cid, "")
        combined.append(item)

    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_candidates = combined[:30] if USE_RERANKER else combined[:top_k]

    scored = _apply_reranker(
        query,
        top_candidates,
        score_key="hybrid_score",
        method_with=METHOD_HYBRID_WITH_RERANK,
        method_without=METHOD_HYBRID,
    )
    return scored[:top_k]
