#!/bin/bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"
MODEL="${LLM_MODEL:-llama3.2:3b}"
STORAGE_DIR="${STORAGE_DIR:-/storage}"
SOURCE_PATH="${SOURCE_PATH:-/app/dataset_case3_v1.0_fix/}"

echo "========================================================================="
echo " entrypoint.sh — code search"
echo " OLLAMA_HOST  = ${OLLAMA_HOST}"
echo " MODEL        = ${MODEL}"
echo " STORAGE_DIR  = ${STORAGE_DIR}"
echo " SOURCE_PATH  = ${SOURCE_PATH}"
echo " USE_RERANKER = ${USE_RERANKER:-false}"
echo " USE_GPU      = ${USE_GPU:-false}"
echo " USE_OLLAMA   = ${USE_OLLAMA:-false}"
echo "========================================================================="

# 1. Проверить что модель доступна (если LLM включена)
if [ "${USE_OLLAMA:-false}" != "false" ]; then
    echo "[1] Waiting for Ollama at ${OLLAMA_HOST}..."
    for i in {1..60}; do
        if curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
            echo "[1] Ollama is ready."
            break
        fi
        echo "  ... attempt $i/60"
        sleep 3
    done

    echo "[2] Checking model ${MODEL}..."
    if curl -sf "${OLLAMA_HOST}/api/show" -d "{\"model\": \"${MODEL}\"}" > /dev/null 2>&1; then
        echo "[2] Model ${MODEL} is available."
    else
        echo "[2] WARNING: Model ${MODEL} not found. Make sure it's pulled locally: ollama pull ${MODEL}"
    fi
fi

# 3. Index Python code (if not already indexed)
mkdir -p "${STORAGE_DIR}"
export SOURCE_PATH="${SOURCE_PATH}"
export STORAGE_DIR="${STORAGE_DIR}"

if [ -f "${STORAGE_DIR}/bm25_index.pkl" ]; then
    echo "[3] Python index already exists, skipping."
else
    echo "[3] Building Python index..."
    python src/index.py
    echo "[3] Python indexing complete."
fi

# 4. Index Java code (if not already indexed)
if [ -f "${STORAGE_DIR}/bm25_index_java.pkl" ]; then
    echo "[4] Java index already exists, skipping."
else
    echo "[4] Building Java index..."
    python src/index_java.py
    echo "[4] Java indexing complete."
fi

# 5. Start Streamlit UI
echo "[5] Starting Streamlit UI on port 8501..."
exec streamlit run src/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=info