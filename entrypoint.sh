#!/bin/bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODEL="${LLM_MODEL:-qwen3.5:9b}"
STORAGE_DIR="${STORAGE_DIR:-/storage}"
SOURCE_PATH="${SOURCE_PATH:-/app/dataset_case3_v1.0_fix/}"

echo "========================================================================="
echo " entrypoint.sh — code search with Ollama"
echo " OLLAMA_HOST  = ${OLLAMA_HOST}"
echo " MODEL        = ${MODEL}"
echo " STORAGE_DIR  = ${STORAGE_DIR}"
echo " SOURCE_PATH  = ${SOURCE_PATH}"
echo " USE_RERANKER = ${USE_RERANKER:-false}"
echo " USE_GPU      = ${USE_GPU:-false}"
echo " USE_OLLAMA   = ${USE_OLLAMA:-true}"
echo "========================================================================="

# 1. Wait for Ollama
echo "[1] Waiting for Ollama..."
for i in {1..60}; do
    if curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "[1] Ollama is ready."
        break
    fi
    echo "  ... attempt $i/60"
    sleep 3
done

# 2. Pull model if not already present (model comes from host via ~/.ollama volume)
echo "[2] Checking model ${MODEL}..."
if curl -sf "${OLLAMA_HOST}/api/show" -d "{\"model\": \"${MODEL}\"}" > /dev/null 2>&1; then
    echo "[2] Model ${MODEL} already available (from host volume)."
else
    echo "[2] Model ${MODEL} not found in volume — pulling..."
    curl -sf -X POST "${OLLAMA_HOST}/api/pull" \
        -d "{\"model\": \"${MODEL}\", \"stream\": false}" > /dev/null 2>&1 || true
fi

# 3. Index code
if [ -f "${STORAGE_DIR}/bm25_index.pkl" ]; then
    echo "[3] Indexes already exist, skipping indexing."
else
    echo "[3] Building indexes..."
    mkdir -p "${STORAGE_DIR}"
    export SOURCE_PATH="${SOURCE_PATH}"
    export STORAGE_DIR="${STORAGE_DIR}"
    python src/index.py
    echo "[3] Indexing complete."
fi

# 4. Start Streamlit UI
echo "[4] Starting Streamlit UI on port 8501..."
exec streamlit run src/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=info
