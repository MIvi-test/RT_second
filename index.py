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

    # Обходим только верхнеуровневые узлы, чтобы жестко контролировать вложенность
    for node in tree.body:
        # 1. Если это изолированная функция верхнего уровня
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            source_code = get_node_source(lines, node)
            chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append(
                {
                    "id": chunk_id,
                    "document": source_code,
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

    if not all_chunks:
        print("Внимание: Не найдено подходящих чанков кода для индексации.")
        return 0

    try:
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Ошибка при загрузке модели {MODEL_NAME}: {e}")
        return 1

    ids = [item["id"] for item in all_chunks]
    documents = [item["document"] for item in all_chunks]
    metadatas = [item["metadata"] for item in all_chunks]

    try:
        chunk_embeddings = embed_model.encode(
            documents, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
    except Exception as e:
        print(f"Ошибка при расчете эмбеддингов: {e}")
        return 1

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
        print(f"Успешно проиндексировано {len(documents)} атомарных чанков кода.")
    except Exception as e:
        print(f"Ошибка при работе с ChromaDB: {e}")
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
