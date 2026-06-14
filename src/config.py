from pathlib import Path
import os
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", SCRIPT_DIR / "storage"))
CHROMA_PATH = STORAGE_DIR / "chroma_db"
COLLECTION_NAME = "code_chunks"
SOURCE_PATH = Path(os.environ.get("SOURCE_PATH", SCRIPT_DIR))
REPO_ROOT = SOURCE_PATH
BM25_INDEX = STORAGE_DIR / "bm25_index.pkl"
BM25_META = STORAGE_DIR / "bm25_meta.json"
BM25_INDEX_JAVA = STORAGE_DIR / "bm25_index_java.pkl"
BM25_META_JAVA = STORAGE_DIR / "bm25_meta_java.json"
DEFAULT_PREDICTIONS = SOURCE_PATH / "results.json"
DEFAULT_QUESTIONS = SOURCE_PATH / "eval_questions.json"
SCORE_SCRIPT = SOURCE_PATH / "score.py"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", os.environ.get("OLLAMA_MODEL", "llama3.2:3b"))

# Flags from environment
USE_GPU = os.environ.get("USE_GPU", "false").lower() not in {"0", "false", "no", "off"}
USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() not in {"0", "false", "no", "off"}
USE_OLLAMA = os.environ.get("USE_OLLAMA", "false").lower() not in {"0", "false", "no", "off"}
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


def resolve_device() -> str:
    """Return 'cuda' if GPU requested and available, else 'cpu'."""
    if USE_GPU:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"
