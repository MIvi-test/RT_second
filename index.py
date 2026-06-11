import pickle
from rank_bm25 import BM25Okapi
import traceback
import ast
import subprocess
import sys
import re
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Build an embeddings index and BM25 index from repository Python files.
# - extract functions/classes into chunks
# - compute embeddings and store them in ChromaDB
# - build and save a BM25 index for hybrid search

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "gymhero"
CHROMA_PATH = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "code_chunks"
MODEL_NAME = "intfloat/multilingual-e5-large" #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PREDICTIONS = SCRIPT_DIR / "results.json"
DEFAULT_QUESTIONS = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "eval_questions.json"
SCORE_SCRIPT = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "score.py"


def get_node_source(lines: list[str], node: ast.AST) -> str:
    start_line = node.lineno - 1
    end_line = getattr(node, "end_lineno", len(lines))
    return "\n".join(lines[start_line:end_line])

# Split code-like text into tokens for BM25

def tokenize_code(text: str) -> list[str]:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9]', ' ', text)
    return text.lower().split()

def extract_chunks_from_file(py_file: Path, repo_root: Path) -> list[dict]:
    pure_rel_path = py_file.relative_to(repo_root).as_posix()
    rel_path = f"{pure_rel_path}"

    # parse file into AST and extract top-level functions and classes
    try:
        src = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, Exception) as e:
        traceback.print_exc()
        return []

    lines = src.splitlines()
    chunks = []

    # Обходим только верхнеуровневые узлы, чтобы жестко контролировать вложенность
    for node in tree.body:
        # 1. Если это изолированная функция верхнего уровня
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            source_code = get_node_source(lines, node)
            chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": class_id,
                    "document": f"File path: {rel_path}\nObject type: {node.__class__.__name__}\nObject name: {node.name}\nCode:\n{get_node_source(lines, node)}",
                    "metadata": {
                        "file_path": rel_path,
                        "name": node.name,
                        "type": "function",
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    },
                }
            )

        # 2. Если это класс, извлекаем ТОЛЬКО его методы, исключая дублирование всего класса
        elif isinstance(node, ast.ClassDef):
            for sub_node in node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    source_code = get_node_source(lines, sub_node)
                    chunk_id = (
                        f"{rel_path}:{node.name}.{sub_node.name}:{sub_node.lineno}"
                    )
                    chunks.append(
                        {
                            "id": chunk_id,
                            "document": source_code,
                            "metadata": {
                                "file_path": rel_path,
                                "name": f"{node.name}.{sub_node.name}",
                                "type": "method",
                                "start_line": sub_node.lineno,
                                "end_line": getattr(
                                    node, "end_lineno", sub_node.lineno
                                ),
                            },
                        }
                    )
    return chunks


def main() -> int:
    if not REPO_ROOT.exists():
        print(f"Ошибка: Путь к репозиторию {REPO_ROOT} не найден.")
        return 1

    all_chunks = []
    for py_file in REPO_ROOT.rglob("*.py"):
        file_chunks = extract_chunks_from_file(py_file, REPO_ROOT)
        all_chunks.extend(file_chunks)

    # nothing to index
    if not all_chunks:
        print("Внимание: Не найдено подходящих чанков кода для индексации.")
        return 0

    # load sentence transformer model for embeddings
    try:
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        

        traceback.print_exc()
        return 1

    ids = [item["id"] for item in all_chunks]
    documents = [item["document"] for item in all_chunks]
    metadatas = [item["metadata"] for item in all_chunks]

    # build and save BM25 index and metadata
    try:
        

        tokenized_corpus = [tokenize_code(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(SCRIPT_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

        with open(SCRIPT_DIR / "bm25_meta.json", "w", encoding="utf-8") as f:
            
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        traceback.print_exc()
        return 1
    
    # compute embeddings for all chunks
    try:
        chunk_embeddings = embed_model.encode(
            documents, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
    except Exception as e:

        traceback.print_exc()
        return 1

    # store embeddings in ChromaDB (persistent)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        # Тюнинг HNSW-индекса под маленький датасет для максимального Precision@5
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 32,
            },
        )

        batch_size = 200
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            collection.add(
                ids=ids[i:end],
                embeddings=chunk_embeddings[i:end].tolist(),
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
    except Exception as e:
        traceback.print_exc()
        return 1

    # === АВТОМАТИЧЕСКИЙ ВЫЗОВ СКРИПТА ОЦЕНКИ ===
    if (
        SCORE_SCRIPT.exists()
        and DEFAULT_PREDICTIONS.exists()
        and DEFAULT_QUESTIONS.exists()
    ):
        print("\n--- Запуск утилиты оценки score.py ---")
        subprocess.run(
            [
                sys.executable,
                str(SCORE_SCRIPT),
                "--predictions",
                str(DEFAULT_PREDICTIONS),
                "--questions",
                str(DEFAULT_QUESTIONS),
            ],
            check=False,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
