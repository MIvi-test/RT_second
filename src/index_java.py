import json
import os
import pickle
import re
import sys
import traceback
import warnings
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    SOURCE_PATH,
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

# tree-sitter setup (requires tree-sitter>=0.25, tree-sitter-java)
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language(tsjava.language())
_parser = Parser(JAVA_LANGUAGE)


# Типы узлов, которые считаются «типами» (класс / интерфейс / enum)
_TYPE_NODE_KINDS = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "annotation_type_declaration",
    "record_declaration",  # Java 16+
}

# Типы узлов, которые считаются «членами» (методы / конструкторы)
_MEMBER_NODE_KINDS = {
    "method_declaration",
    "constructor_declaration",
}

# Поле, в котором tree-sitter хранит имя узла
_NAME_FIELD = "name"


def _node_text(node, src_bytes: bytes) -> str:
    """Возвращает исходный текст узла."""
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node, src_bytes: bytes) -> str:
    """Возвращает имя узла (поле 'name'), либо пустую строку."""
    name_node = node.child_by_field_name(_NAME_FIELD)
    if name_node is None:
        return ""
    return src_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")


def _obj_type_label(kind: str) -> str:
    """Человекочитаемый тип объекта для поля 'Object type'."""
    return {
        "class_declaration": "ClassDeclaration",
        "interface_declaration": "InterfaceDeclaration",
        "enum_declaration": "EnumDeclaration",
        "annotation_type_declaration": "AnnotationDeclaration",
        "record_declaration": "RecordDeclaration",
        "method_declaration": "method",
        "constructor_declaration": "constructor",
    }.get(kind, kind)


def _walk_type_node(
    node,
    src_bytes: bytes,
    rel_path: str,
    outer_name: str,
    chunks: list,
) -> None:
    """
    Рекурсивно обходит узел типа (class/interface/enum/record) и добавляет чанки:
      1. Сам тип целиком — аналог ClassDef-чанка в index.py
      2. Каждый метод / конструктор внутри — аналог method-чанка в index.py
      3. Вложенные типы — рекурсия
    """
    type_name = _node_name(node, src_bytes)
    full_type_name = f"{outer_name}.{type_name}" if outer_name else type_name

    # 1. Чанк для самого типа целиком
    start_line = node.start_point[0] + 1  # tree-sitter: 0-based → 1-based
    end_line = node.end_point[0] + 1
    type_code = _node_text(node, src_bytes)

    chunks.append({
        "id": f"{rel_path}:{full_type_name}:{start_line}",
        "document": (
            f"File path: {rel_path}\n"
            f"Object type: {_obj_type_label(node.type)}\n"
            f"Object name: {full_type_name}\n"
            f"Code:\n{type_code}"
        ),
        "metadata": {
            "file_path": rel_path,
            "name": full_type_name,
            "type": "class",
            "start_line": start_line,
            "end_line": end_line,
        },
    })

    # 2. Проходим по телу типа
    body_node = node.child_by_field_name("body")
    if body_node is None:
        return

    for child in body_node.children:
        if child.type in _MEMBER_NODE_KINDS:
            # --- Метод или конструктор ---
            member_name = _node_name(child, src_bytes)
            full_member_name = f"{full_type_name}.{member_name}"
            m_start = child.start_point[0] + 1
            m_end = child.end_point[0] + 1
            member_code = _node_text(child, src_bytes)

            chunks.append({
                "id": f"{rel_path}:{full_member_name}:{m_start}",
                "document": (
                    f"File path: {rel_path}\n"
                    f"Object type: {_obj_type_label(child.type)}\n"
                    f"Object name: {full_member_name}\n"
                    f"Code:\n{member_code}"
                ),
                "metadata": {
                    "file_path": rel_path,
                    "name": full_member_name,
                    "type": "method" if child.type == "method_declaration" else "constructor",
                    "start_line": m_start,
                    "end_line": m_end,
                },
            })

        elif child.type in _TYPE_NODE_KINDS:
            # --- Вложенный тип — рекурсия ---
            _walk_type_node(child, src_bytes, rel_path, full_type_name, chunks)


def extract_chunks_from_java_file(java_file: Path, repo_root: Path) -> list[dict]:
    """
    Парсит один .java файл через tree-sitter и возвращает список чанков.
    """
    rel_path = java_file.relative_to(repo_root).as_posix()
    try:
        src = java_file.read_text(encoding="utf-8", errors="replace")
        src_bytes = src.encode("utf-8", errors="replace")
        tree = _parser.parse(src_bytes)
    except Exception:
        print(f"[WARN] Could not parse {java_file}")
        traceback.print_exc()
        return []

    chunks: list[dict] = []

    # Корень дерева — compilation_unit; ищем верхнеуровневые типы
    for child in tree.root_node.children:
        if child.type in _TYPE_NODE_KINDS:
            _walk_type_node(child, src_bytes, rel_path, "", chunks)

    return chunks


def tokenize_code(text: str) -> list[str]:
    """Нормализация текста для BM25 (разбивка camelCase, удаление пунктуации)."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]", " ", text)
    return text.lower().split()


def main() -> int:
    print(f"[index_java] SOURCE_PATH : {SOURCE_PATH}")
    print(f"[index_java] STORAGE_DIR : {STORAGE_DIR}")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Сбор всех .java файлов и извлечение чанков ---
    java_files = list(SOURCE_PATH.rglob("*.java"))
    all_chunks: list[dict] = []
    print(f"[index_java] Scanning {len(java_files)} Java files …")

    for java_file in java_files:
        all_chunks.extend(extract_chunks_from_java_file(java_file, SOURCE_PATH))

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

        # Получаем существующую или создаем новую коллекцию
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 32,
            },
        )

        # Выборочно удаляем только старые Java-векторы по маске пути
        try:
            collection.delete(where={"file_path": {"$like": "%.java"}})
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
            print(f"[index_java]   batch {i}–{end}")

        print(f"[index_java] ChromaDB ready – Java vectors merged into '{COLLECTION_NAME}'")
    except Exception:
        print("[ERROR] ChromaDB write failed")
        traceback.print_exc()
        return 1

    print("[index_java] Done ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())