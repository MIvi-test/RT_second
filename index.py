import ast
import subprocess
import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "gymhero"
CHROMA_PATH = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "code_chunks"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PREDICTIONS = SCRIPT_DIR / "results.json"
DEFAULT_QUESTIONS = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "eval_questions.json"
SCORE_SCRIPT = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "score.py"

def get_node_source(lines: list[str], node: ast.AST) -> str:
    start_line = node.lineno - 1
    end_line = getattr(node, "end_lineno", len(lines))
    return "\n".join(lines[start_line:end_line])


def extract_chunks_from_file(py_file: Path, repo_root: Path) -> list[dict]:
    pure_rel_path = py_file.relative_to(repo_root).as_posix()
    rel_path = f"{pure_rel_path}"

    try:
        src = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, Exception):
        return []

    lines = src.splitlines()
    chunks = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": class_id,
                    "document": get_node_source(lines, node),
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

    if not all_chunks:
        return 0

    try:
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception:
        return 1

    ids = [item["id"] for item in all_chunks]
    documents = [item["document"] for item in all_chunks]
    metadatas = [item["metadata"] for item in all_chunks]

    try:
        chunk_embeddings = embed_model.encode(
            documents, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
    except Exception:
        return 1

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
    except Exception:
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
