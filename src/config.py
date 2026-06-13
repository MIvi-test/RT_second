from pathlib import Path
import os 


SCRIPT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", SCRIPT_DIR / "storage"))
CHROMA_PATH = STORAGE_DIR / "chroma_db"
COLLECTION_NAME = "code_chunks"
MODEL_NAME = "intfloat/multilingual-e5-large"
SOURCE_PATH = Path(os.environ.get("SOURCE_PATH", SCRIPT_DIR))
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", SCRIPT_DIR / "storage"))
REPO_ROOT = SOURCE_PATH
BM25_INDEX = STORAGE_DIR / "bm25_index.pkl"
BM25_META = STORAGE_DIR / "bm25_meta.json"
DEFAULT_PREDICTIONS = SOURCE_PATH / "results.json"
DEFAULT_QUESTIONS = SOURCE_PATH / "eval_questions.json"
SCORE_SCRIPT = SOURCE_PATH / "score.py"
# EMDENDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMDENDING_MODEL_NAME = "BAИI/bge-m3"
