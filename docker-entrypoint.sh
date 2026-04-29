#!/usr/bin/env bash
# autotune + Ollama entrypoint
# Starts Ollama in the background, waits for it to be ready,
# optionally pulls a model, then starts autotune serve in the foreground.
set -euo pipefail

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
AUTOTUNE_PORT="${AUTOTUNE_PORT:-8765}"
# OLLAMA_HOST controls where Ollama listens (0.0.0.0 = all interfaces so the
# port can be forwarded out of the container if the user maps it).
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0}"

echo "[entrypoint] Starting Ollama on ${OLLAMA_HOST}:${OLLAMA_PORT}…"
/bin/ollama serve &
OLLAMA_PID=$!

# Wait up to 30 s for Ollama to become reachable
RETRIES=60
until curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        echo "[entrypoint] ERROR: Ollama did not start within 30 s"
        exit 1
    fi
    sleep 0.5
done
echo "[entrypoint] Ollama is ready."

# Optional: pull a model on first start
if [ -n "${OLLAMA_MODEL:-}" ]; then
    echo "[entrypoint] Pulling model: ${OLLAMA_MODEL}"
    /bin/ollama pull "${OLLAMA_MODEL}"
    echo "[entrypoint] Model ready."
fi

# autotune reads AUTOTUNE_OLLAMA_URL to locate Ollama (defaults to localhost).
# In single-container mode this is already correct; docker-compose users can
# override it with AUTOTUNE_OLLAMA_URL=http://ollama:11434.
echo "[entrypoint] Starting autotune on 0.0.0.0:${AUTOTUNE_PORT}…"
exec autotune serve --host 0.0.0.0 --port "${AUTOTUNE_PORT}"
