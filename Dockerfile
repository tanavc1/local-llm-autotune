# autotune + Ollama — single container
#
# Build:  docker build -t autotune .
# Run:    docker run -p 8765:8765 -v ollama_models:/root/.ollama autotune
#
# The image bundles Ollama (the LLM runtime) and autotune (the OpenAI-compatible
# optimisation middleware).  The entrypoint starts Ollama in the background,
# waits until it is ready, then starts autotune serve on port 8765.
#
# Environment variables
# ---------------------
#   OLLAMA_MODEL         Pull this model on first start (e.g. "llama3.2:3b")
#   AUTOTUNE_PORT        autotune listen port (default 8765)
#   OLLAMA_HOST          Ollama bind address passed to ollama serve (default 0.0.0.0)
#   AUTOTUNE_OLLAMA_URL  URL autotune uses to reach Ollama (default http://localhost:11434)
#                        Set to http://ollama:11434 when using docker-compose multi-container mode

FROM ollama/ollama:latest

# Install Python 3 and pip
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages llm-autotune

# Expose autotune's API port (Ollama's 11434 is already exposed by the base image)
EXPOSE 8765

# Persist downloaded models across container restarts
VOLUME ["/root/.ollama"]

# Copy and set the entrypoint
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
