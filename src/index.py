import os
import pickle
import traceback
import ast
import subprocess
import sys
import re
import json
import warnings
from pathlib import Path
from config import *
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

__all__ = [
    "get_node_source",
    "tokenize_code",
    "extract_chunks_from_file",
    "main",
]

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def get_node_source(lines: list[str], node: ast.AST) -> str:
    """Return the source lines that belong to an AST node."""
    return "\n".join(lines[node.lineno - 1 : getattr(node, "end_lineno", len(lines))])


def tokenize_code(text: str) -> list[str]:
    """Normalize and split code into BM25-friendly tokens."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # split camelCase
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]", " ", text)  # drop punctuation
    return text.lower().split()


def extract_chunks_from_file(py_file: Path, repo_root: Path) -> list[dict]:
    """
    Parse one file and return a chunk per top-level function,
    per class (as a whole), and per method inside classes.
    """
    rel_path = py_file.relative_to(repo_root).as_posix()
    try:
        src = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        print(f"[WARN] Could not parse {py_file}")
        traceback.print_exc()
        return []

    lines = src.splitlines()
    chunks = []

    for node in tree.body:
        # ---- Top-level functions ----
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": chunk_id,
                    "document": (
                        f"File path: {rel_path}\n"
                        f"Object type: {node.__class__.__name__}\n"
                        f"Object name: {node.name}\n"
                        f"Code:\n{get_node_source(lines, node)}"
                    ),
                    "metadata": {
                        "file_path": rel_path,
                        "name": node.name,
                        "type": "function",
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    },
                }
            )

        # ---- Classes ----
        elif isinstance(node, ast.ClassDef):
            # 1. Index the whole class as a chunk
            class_chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": class_chunk_id,
                    "document": (
                        f"File path: {rel_path}\n"
                        f"Object type: {node.__class__.__name__}\n"
                        f"Object name: {node.name}\n"
                        f"Code:\n{get_node_source(lines, node)}"
                    ),
                    "metadata": {
                        "file_path": rel_path,
                        "name": node.name,
                        "type": "class",
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    },
                }
            )

            # 2. Index each method inside the class separately
            for sub_node in node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = f"{node.name}.{sub_node.name}"
                    chunk_id = f"{rel_path}:{method_name}:{sub_node.lineno}"
                    chunks.append(
                        {
                            "id": chunk_id,
                            "document": (
                                f"File path: {rel_path}\n"
                                f"Object type: {sub_node.__class__.__name__}\n"
                                f"Object name: {method_name}\n"
                                f"Code:\n{get_node_source(lines, sub_node)}"
                            ),
                            "metadata": {
                                "file_path": rel_path,
                                "name": method_name,
                                "type": "method",
                                "start_line": sub_node.lineno,
                                "end_line": getattr(
                                    sub_node, "end_lineno", sub_node.lineno
                                ),
                            },
                        }
                    )

    return chunks


def main() -> int:
    print(f"[index] SOURCE_PATH : {SOURCE_PATH}")
    print(f"[index] STORAGE_DIR : {STORAGE_DIR}")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Сбор всех .py файлов и извлечение чанков ---
    py_files = set(SOURCE_PATH.rglob("*.py"))
    all_chunks = []
    print(f"[index] Scanning {len(py_files)} files …")
        
    for py_file in py_files:
        all_chunks.extend(extract_chunks_from_file(py_file, SOURCE_PATH))

    if not all_chunks:
        print("[WARN] No chunks found – nothing to index.")
        return 0

    print(f"[index] {len(all_chunks)} chunks extracted")

    ids = [c["id"] for c in all_chunks]
    documents = [c["document"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # --- 2. BM25 индекс ---
    print("[index] Building BM25 index …")
    try:
        bm25 = BM25Okapi([tokenize_code(doc) for doc in documents])

        with open(BM25_INDEX, "wb") as f:
            pickle.dump(bm25, f)

        with open(BM25_META, "w", encoding="utf-8") as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)

        print(f"[index] BM25 saved → {BM25_INDEX}")
    except Exception:
        print("[ERROR] BM25 build failed")
        traceback.print_exc()
        return 1

    # --- 3. Вычисление эмбеддингов ---
    device = resolve_device()
    print(f"[index] Loading {EMBEDDING_MODEL_NAME} on {device} …")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception:
        print(f"[ERROR] Could not load model '{EMBEDDING_MODEL_NAME}'")
        traceback.print_exc()
        return 1

    print(f"[index] Encoding {len(documents)} chunks …")
    try:
        chunk_embeddings = embed_model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception:
        print("[ERROR] Encoding failed")
        traceback.print_exc()
        return 1

    # --- 4. Запись в ChromaDB ---
    print(f"[index] Writing ChromaDB → {CHROMA_PATH} …")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        
        # Получаем или создаем общую коллекцию
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 32,
            },
        )

        # Выборочно удаляем старые записи только для Python-файлов, чтобы не задеть Java
        try:
            collection.delete(where={"file_path": {"$like": "%.py"}})
        except Exception:
            pass

        batch_size = 200
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            collection.add(
                ids=ids[i:end],
                embeddings=chunk_embeddings[i:end].tolist(),
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            print(f"[index]   batch {i}–{end}")

        print(
            f"[index] ChromaDB updated – Python vectors merged into '{COLLECTION_NAME}'"
        )
    except Exception:
        print("[ERROR] ChromaDB write failed")
        traceback.print_exc()
        return 1

    print("[index] Done ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())