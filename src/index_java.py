import json
import os
import pickle
import re
import sys
import traceback
import warnings
from pathlib import Path

import chromadb
import javalang
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    REPO_ROOT,
    STORAGE_DIR,
    BM25_META_JAVA,
    BM25_INDEX_JAVA,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    resolve_device,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def tokenize_code(text: str) -> list[str]:
    """Нормализация текста для BM25 (разбивка camelCase, удаление пунктуации)."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text.lower().split()


def extract_method_source(lines: list[str], start_line: int) -> str:
    """
    Извлекает исходный код метода/конструктора, начиная со start_line.
    Ищет баланс фигурных скобок, чтобы определить конец тела метода.
    """
    if start_line < 1 or start_line > len(lines):
        return ""

    # Ищем строку, содержащую открывающую фигурную скобку
    brace_start_line = None
    for i in range(start_line - 1, len(lines)):
        if "{" in lines[i]:
            brace_start_line = i
            break
    if brace_start_line is None:
        return ""

    balance = 0
    end_line = None
    for i in range(brace_start_line, len(lines)):
        line = lines[i]
        balance += line.count("{") - line.count("}")
        if balance == 0:
            end_line = i
            break

    if end_line is None:
        end_line = len(lines) - 1

    return "\n".join(lines[start_line - 1 : end_line + 1])


def process_type_node(
    type_node,
    rel_path: str,
    lines: list[str],
    outer_name: str,
    chunks: list,
):
    """
    Рекурсивно обходит классы/интерфейсы/enum и добавляет чанки для методов и конструкторов.
    """
    # Полное имя типа
    if outer_name:
        full_type_name = f"{outer_name}.{type_node.name}"
    else:
        full_type_name = type_node.name

    for member in type_node.body:
        if isinstance(member, javalang.tree.MethodDeclaration):
            start_line = member.position.line
            code = extract_method_source(lines, start_line)
            if not code:
                continue

            method_name = member.name
            full_name = f"{full_type_name}.{method_name}"
            chunk_id = f"{rel_path}:{full_name}:{start_line}"

            doc_text = (
                f"File path: {rel_path}\n"
                f"Object type: method\n"
                f"Object name: {full_name}\n"
                f"Code:\n{code}"
            )

            chunks.append({
                "id": chunk_id,
                "document": doc_text,
                "metadata": {
                    "file_path": rel_path,
                    "name": full_name,
                    "type": "method",
                    "start_line": start_line,
                    "end_line": start_line + code.count('\n'),
                },
            })

        elif isinstance(member, javalang.tree.ConstructorDeclaration):
            start_line = member.position.line
            code = extract_method_source(lines, start_line)
            if not code:
                continue

            # Имя конструктора — имя класса
            constructor_name = full_type_name.split('.')[-1]
            full_name = f"{full_type_name}.{constructor_name}"
            chunk_id = f"{rel_path}:{full_name}:{start_line}"

            doc_text = (
                f"File path: {rel_path}\n"
                f"Object type: constructor\n"
                f"Object name: {full_name}\n"
                f"Code:\n{code}"
            )

            chunks.append({
                "id": chunk_id,
                "document": doc_text,
                "metadata": {
                    "file_path": rel_path,
                    "name": full_name,
                    "type": "constructor",
                    "start_line": start_line,
                    "end_line": start_line + code.count('\n'),
                },
            })

        # Вложенные типы
        elif isinstance(member, (javalang.tree.ClassDeclaration,
                                 javalang.tree.InterfaceDeclaration,
                                 javalang.tree.EnumDeclaration)):
            process_type_node(member, rel_path, lines, full_type_name, chunks)


def extract_chunks_from_java_file(java_file: Path, repo_root: Path) -> list[dict]:
    """
    Парсит один .java файл и возвращает список чанков (методы и конструкторы).
    """
    rel_path = java_file.relative_to(repo_root).as_posix()
    try:
        src = java_file.read_text(encoding="utf-8", errors="replace")
        lines = src.splitlines()
        tree = javalang.parse.parse(src)
    except Exception:
        print(f"[WARN] Could not parse {java_file}")
        traceback.print_exc()
        return []

    chunks = []

    # Обрабатываем все верхнеуровневые типы
    for path_node in tree.types:
        if isinstance(path_node, (javalang.tree.ClassDeclaration,
                                  javalang.tree.InterfaceDeclaration,
                                  javalang.tree.EnumDeclaration)):
            process_type_node(path_node, rel_path, lines, "", chunks)

    return chunks


def main() -> int:
    print(f"[index_java] SOURCE_PATH : {REPO_ROOT}")
    print(f"[index_java] STORAGE_DIR : {STORAGE_DIR}")

    if not REPO_ROOT.exists():
        print(f"[ERROR] Repo root not found: {REPO_ROOT}")
        return 1

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Сбор всех .java файлов и извлечение чанков ---
    java_files = list(REPO_ROOT.rglob("*.java"))
    all_chunks = []
    print(f"[index_java] Scanning {len(java_files)} Java files …")

    for java_file in java_files:
        all_chunks.extend(extract_chunks_from_java_file(java_file, REPO_ROOT))

    if not all_chunks:
        print("[WARN] No chunks found – nothing to index.")
        return 0

    print(f"[index_java] {len(all_chunks)} chunks extracted")

    ids = [c["id"] for c in all_chunks]
    documents = [c["document"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # --- 2. BM25 индекс ---
    print("[index_java] Building BM25 index …")
    try:
        bm25 = BM25Okapi([tokenize_code(doc) for doc in documents])

        with open(BM25_INDEX_JAVA, "wb") as f:
            pickle.dump(bm25, f)

        with open(BM25_META_JAVA, "w", encoding="utf-8") as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)

        print(f"[index_java] BM25 saved → {BM25_INDEX_JAVA}")
    except Exception:
        print("[ERROR] BM25 build failed")
        traceback.print_exc()
        return 1

    # --- 3. Вычисление эмбеддингов ---
    device = resolve_device()
    print(f"[index_java] Loading {EMBEDDING_MODEL_NAME} on {device} …")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception:
        print(f"[ERROR] Could not load model '{EMBEDDING_MODEL_NAME}'")
        traceback.print_exc()
        return 1

    print(f"[index_java] Encoding {len(documents)} chunks …")
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
    print(f"[index_java] Writing ChromaDB → {CHROMA_PATH} …")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

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
            print(f"[index_java]   batch {i}–{end}")

        print(f"[index_java] ChromaDB ready – {len(documents)} vectors in '{COLLECTION_NAME}'")
    except Exception:
        print("[ERROR] ChromaDB write failed")
        traceback.print_exc()
        return 1

    print("[index_java] Done ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())