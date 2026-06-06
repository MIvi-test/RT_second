import json
import pickle
import re
import numpy as np
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "code_chunks"
MODEL_NAME = "intfloat/multilingual-e5-large"

# Module-level singletons: resources are loaded lazily on first search call
_model = None
_collection = None
_bm25 = None
_bm25_meta = None
_reranker = None


def tokenize_code(text: str) -> list[str]:
    # split CamelCase and non-alphanumeric chars, return lowercase tokens
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9]', ' ', text)
    return text.lower().split()


def _load():
    """Lazy load model, reranker, Chroma collection and BM25 index."""
    global _model, _collection, _bm25, _bm25_meta, _reranker

    if _reranker is None:
        print("Loading CrossEncoder reranker...")
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    if _model is None:
        print("Loading SentenceTransformer model...")
        _model = SentenceTransformer(MODEL_NAME)

    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)

    if _bm25 is None:
        bm25_path = SCRIPT_DIR / "bm25_index.pkl"
        meta_path = SCRIPT_DIR / "bm25_meta.json"

        if not bm25_path.exists():
            raise FileNotFoundError("bm25_index.pkl not found — run index.py")

        with open(bm25_path, "rb") as f:
            _bm25 = pickle.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            _bm25_meta = json.load(f)


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Pure semantic search with CrossEncoder reranking."""
    _load()

    fetch_k = 75  # wide candidate funnel

    # encode query vector
    query_vec = _model.encode([query], convert_to_numpy=True)

    # retrieve candidates (include documents for reranking)
    hits = _collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=fetch_k,
        include=["metadatas", "distances", "documents"],
    )

    # collect candidates with base semantic score
    candidates = []
    for meta, dist, doc in zip(hits["metadatas"][0], hits["distances"][0], hits["documents"][0]):
        candidates.append({
            "chunk_id": meta["chunk_id"],
            "file_path": meta["file_path"],
            "type": meta["type"],
            "name": meta["name"],
            "document": doc,              # keep code for reranker
            "semantic_score": 1 - dist    # base Chroma score
        })

    # rerank with CrossEncoder
    pairs = [[query, c["document"]] for c in candidates]
    rerank_scores = _reranker.predict(pairs)

    # write final scores and remove large document text
    for c, r_score in zip(candidates, rerank_scores):
        c["score"] = round(float(r_score) * 100, 2)
        c["method"] = "semantic + reranker"
        del c["document"]

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates[:top_k]

def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """Combine semantic and BM25 scores, then rerank with CrossEncoder."""
    _load()

    fetch_k = top_k * 3

    # semantic candidates
    query_vec = _model.encode([query], convert_to_numpy=True)
    hits = _collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=fetch_k,
        include=["metadatas", "distances"],
    )

    sem_scores: dict[str, float] = {}
    meta_by_id: dict[str, dict] = {}
    for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
        cid = meta["chunk_id"]
        sem_scores[cid] = 1 - dist
        meta_by_id[cid] = meta

    # BM25 candidates
    tokens = tokenize_code(query)
    raw_scores = _bm25.get_scores(tokens)

    max_score = raw_scores.max()
    bm25_scores: dict[str, float] = {}

    if max_score > 0:
        norm = raw_scores / max_score
        top_indices = np.argsort(norm)[::-1][:fetch_k]

        for idx in top_indices:
            if norm[idx] > 0:
                cid = _bm25_meta[idx]["chunk_id"]
                bm25_scores[cid] = float(norm[idx])
                if cid not in meta_by_id:
                    meta_by_id[cid] = _bm25_meta[idx]

    all_ids = set(sem_scores) | set(bm25_scores)

    combined = []
    for cid in all_ids:
        sem = sem_scores.get(cid, 0.0)
        bm25 = bm25_scores.get(cid, 0.0)
        final = semantic_weight * sem + bm25_weight * bm25

        meta = meta_by_id[cid]
        combined.append({
            "chunk_id": cid,
            "file_path": meta["file_path"],
            "type": meta["type"],
            "name": meta["name"],
            "hybrid_score": final,
            "method": "hybrid + reranker",
        })

    # keep top candidates for reranking
    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_candidates = combined[:30]

    # get documents for candidates and prepare pairs for reranker
    candidate_ids = [c["chunk_id"] for c in top_candidates]
    db_data = _collection.get(ids=candidate_ids, include=["documents"])
    doc_map = {cid: doc for cid, doc in zip(db_data["ids"], db_data["documents"])}

    pairs = []
    valid_candidates = []
    for c in top_candidates:
        cid = c["chunk_id"]
        if cid in doc_map:
            pairs.append([query, doc_map[cid]])
            valid_candidates.append(c)

    # rerank and write scores
    rerank_scores = _reranker.predict(pairs)
    for c, r_score in zip(valid_candidates, rerank_scores):
        c["score"] = round(float(r_score) * 100, 2)

    valid_candidates.sort(key=lambda x: x["score"], reverse=True)

    return valid_candidates[:top_k]