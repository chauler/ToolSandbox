#!/bin/sh
# Pull the required model(s) on startup so they are ready when the
# benchmark container starts making requests.
set -e

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"

echo "Starting Ollama server …"
ollama serve &
SERVER_PID=$!

# Wait for the server socket
echo "Waiting for Ollama to listen …"
for i in $(seq 1 120); do
    if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    sleep 1
done

# Pre-pull the model so it's cached in the volume
echo "Ensuring model '${OLLAMA_MODEL}' is available …"
ollama pull "${OLLAMA_MODEL}"
echo "Model ready."

# Keep the server in the foreground
wait $SERVER_PID
