import pickle
import ast
import subprocess
import sys
import re
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
        import traceback

        traceback.print_exc()
        return []

    lines = src.splitlines()
    chunks = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": class_id,
                    "document": f"File path: {rel_path}\nObject type: {node.__class__.__name__}\nObject name: {node.name}\nCode:\n{get_node_source(lines, node)}",
                    "metadata": {
                        "chunk_id": class_id,
                        "file_path": rel_path,
                        "type": "class",
                        "name": node.name,
                        "start_line": node.lineno,
                    },
                }
            )

            for sub_node in node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = f"{node.name}.{sub_node.name}"
                    method_id = f"{rel_path}:{method_name}:{sub_node.lineno}"
                    chunks.append(
                        {
                            "id": method_id,
                            "document": get_node_source(lines, sub_node),
                            "metadata": {
                                "chunk_id": method_id,
                                "file_path": rel_path,
                                "type": "method",
                                "name": method_name,
                                "start_line": sub_node.lineno,
                            },
                        }
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": func_id,
                    "document": get_node_source(lines, node),
                    "metadata": {
                        "chunk_id": func_id,
                        "file_path": rel_path,
                        "type": "function",
                        "name": node.name,
                        "start_line": node.lineno,
                    },
                }
            )

    return chunks


def main() -> int:
    if not REPO_ROOT.exists():
        return 1

    all_chunks = []
    for py_file in REPO_ROOT.rglob("*.py"):
        file_chunks = extract_chunks_from_file(py_file, REPO_ROOT)
        all_chunks.extend(file_chunks)

    # nothing to index
    if not all_chunks:
        return 0

    # load sentence transformer model for embeddings
    try:
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return 1

    ids = [item["id"] for item in all_chunks]
    documents = [item["document"] for item in all_chunks]
    metadatas = [item["metadata"] for item in all_chunks]

    # build and save BM25 index and metadata
    try:
        import pickle
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [tokenize_code(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(SCRIPT_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

        with open(SCRIPT_DIR / "bm25_meta.json", "w", encoding="utf-8") as f:
            import json
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    
    # compute embeddings for all chunks
    try:
        chunk_embeddings = embed_model.encode(
            documents, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return 1

    # store embeddings in ChromaDB (persistent)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
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
        import traceback

        traceback.print_exc()
        return 1

    if (
        SCORE_SCRIPT.exists()
        and DEFAULT_PREDICTIONS.exists()
        and DEFAULT_QUESTIONS.exists()
    ):
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
