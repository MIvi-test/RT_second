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
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import *
from typing import Any, Dict, List, Optional

# Logger
logger = logging.getLogger(__name__)
try:
    logging.basicConfig()
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
except Exception:
    logger.setLevel(logging.INFO)

__all__ = [
    "initialize_search",
    "semantic_search",
    "hybrid_search",
    "tokenize_code",
]

# Module-level singletons
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None
_reranker: Optional[CrossEncoder] = None
_device: Optional[str] = None

# BM25 indices (language‑specific)
_bm25_python: Optional[Any] = None
_bm25_meta_python: Optional[dict] = None
_bm25_java: Optional[Any] = None
_bm25_meta_java: Optional[dict] = None

# Method name constants
METHOD_SEMANTIC = "semantic"
METHOD_HYBRID = "hybrid"
METHOD_SUFFIX_RERANK = " + reranker"
METHOD_SEMANTIC_WITH_RERANK = METHOD_SEMANTIC + METHOD_SUFFIX_RERANK
METHOD_HYBRID_WITH_RERANK = METHOD_HYBRID + METHOD_SUFFIX_RERANK


def _translate_to_english(text: str) -> str:
    """Translate query to English if it contains Cyrillic characters."""
    if re.search(r"[а-яА-ЯёЁ]", text):
        try:
            logger.debug("Translating query to English via GoogleTranslator")
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            logger.exception("Translation failed; using original text")
            return text
    return text


def tokenize_code(text: str) -> List[str]:
    """Tokenize code: split CamelCase and keep only alphanumeric."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]", " ", text)
    return text.lower().split()


def _matches_lang(file_path: str, lang: str) -> bool:
    """Filter chunks by file extension (Chroma $contains is unreliable for paths)."""
    if lang == "java":
        return file_path.endswith(".java")
    if lang == "python":
        return file_path.endswith(".py")
    return True


def _init_core() -> None:
    """Load model, Chroma collection and reranker (once)."""
    global _model, _collection, _reranker, _device

    if _device is None:
        _device = resolve_device()
        logger.info("[search] device=%s, reranker=%s", _device, USE_RERANKER)

    if _model is None:
        logger.info("Loading SentenceTransformer model on %s...", _device)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)

    if USE_RERANKER and _reranker is None:
        logger.info("Loading CrossEncoder reranker on %s...", _device)
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, device=_device)

    if _collection is None:
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Chroma collection '{COLLECTION_NAME}': {e}. "
                "Make sure indexing has been run."
            )


def _load_bm25(lang: str):
    """Load and return (bm25_index, metadata) for the requested language."""
    global _bm25_python, _bm25_meta_python, _bm25_java, _bm25_meta_java

    if lang == "java":
        if _bm25_java is None:
            bm25_path = BM25_INDEX_JAVA
            meta_path = BM25_META_JAVA
            if not bm25_path.exists():
                raise FileNotFoundError(
                    f"Java BM25 index not found: {bm25_path}")
            with open(bm25_path, "rb") as f:
                _bm25_java = pickle.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                _bm25_meta_java = json.load(f)
        return _bm25_java, _bm25_meta_java
    else:  # python
        if _bm25_python is None:
            if not BM25_INDEX.exists():
                raise FileNotFoundError(
                    "bm25_index.pkl not found — run index.py")
            with open(BM25_INDEX, "rb") as f:
                _bm25_python = pickle.load(f)
            with open(BM25_META, "r", encoding="utf-8") as f:
                _bm25_meta_python = json.load(f)
        return _bm25_python, _bm25_meta_python


@lru_cache(maxsize=1)
def _ensure_loaded() -> bool:
    """Legacy function: loads core components and Python BM25."""
    _init_core()
    _load_bm25("python")  # preload Python BM25 for compatibility
    return True


def initialize_search() -> dict:
    """Load search resources once and return runtime info for UI/logging."""
    _ensure_loaded()
    return {
        "device": _device or "cpu",
        "use_reranker": USE_RERANKER,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME if USE_RERANKER else None,
    }


def _apply_reranker(
    query: str,
    candidates: List[dict],
    *,
    score_key: str,
    method_with: str,
    method_without: str,
) -> List[dict]:
    """Score candidates with CrossEncoder or fall back to base scores."""
    if USE_RERANKER and _reranker is not None:
        pairs = [[query, c["document"]]
                 for c in candidates if c.get("document")]
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


def semantic_search(
    query: str,
    top_k: int = 5,
    lang: str = "python",
    use_translation: bool = False
) -> List[dict]:
    """Semantic search with optional CrossEncoder reranking, filtered by language."""
    _init_core()

    if use_translation:
        query = _translate_to_english(query)

    fetch_k = (25 if USE_RERANKER else top_k) * 4
    include_docs = USE_RERANKER

    query_vec = _model.encode([query], convert_to_numpy=True)

    hits = _collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=fetch_k,
        include=["metadatas", "distances"] +
        (["documents"] if include_docs else []),
    )

    candidates = []
    documents = hits["documents"][0] if include_docs else [
        None] * len(hits["ids"][0])
    for _id, meta, dist, doc in zip(
        hits["ids"][0], hits["metadatas"][0], hits["distances"][0], documents
    ):
        file_path = meta.get("file_path", "?")
        if not _matches_lang(file_path, lang):
            continue
        item = {
            "chunk_id": _id,
            "file_path": file_path,
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
    lang: str = "python",
    use_translation: bool = False
) -> List[dict]:
    """Combine semantic and BM25 scores, optionally rerank with CrossEncoder."""
    _init_core()
    bm25, bm25_meta = _load_bm25(lang)

    if use_translation:
        query = _translate_to_english(query)

    fetch_k = top_k * 12
    include_docs = USE_RERANKER

    query_vec = _model.encode([query], convert_to_numpy=True)

    hits = _collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=fetch_k,
        include=["metadatas", "distances"] +
        (["documents"] if include_docs else []),
    )

    sem_scores: Dict[str, float] = {}
    meta_by_id: Dict[str, dict] = {}
    doc_by_id: Dict[str, str] = {}
    documents = hits["documents"][0] if include_docs else [
        None] * len(hits["ids"][0])
    for _id, meta, dist, doc in zip(
        hits["ids"][0], hits["metadatas"][0], hits["distances"][0], documents
    ):
        file_path = meta.get("file_path", "?")
        if not _matches_lang(file_path, lang):
            continue
        cid = _id
        sem_scores[cid] = 1 - dist
        meta_by_id[cid] = meta
        if include_docs and doc:
            doc_by_id[cid] = doc

    tokens = tokenize_code(query)
    raw_scores = bm25.get_scores(tokens)

    max_score = raw_scores.max()
    bm25_scores: Dict[str, float] = {}

    if max_score > 0:
        norm = raw_scores / max_score
        top_indices = np.argsort(norm)[::-1][:fetch_k]

        for idx in top_indices:
            if norm[idx] > 0:
                bm25_meta_item = bm25_meta[idx]
                rel_path = bm25_meta_item.get("file_path", "?")
                if not _matches_lang(rel_path, lang):
                    continue
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
        bm25_val = bm25_scores.get(cid, 0.0)
        final = semantic_weight * sem + bm25_weight * bm25_val

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
