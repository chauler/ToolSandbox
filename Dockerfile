# ToolSandbox container — runs benchmarks against an Ollama server.
# Python 3.9 matches the project's primary target version.
FROM python:3.9-slim

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install git (needed for git SHA in result_summary) and curl (health checks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy full project source
COPY . .

# Install the project and pin httpx to avoid the
# openai 1.17 + httpx 0.28 incompatibility (proxies kwarg removed).
RUN pip install --no-cache-dir -e '.[dev]' && \
    pip install --no-cache-dir 'httpx==0.27.2'

# Default output directory inside the container
ENV TOOL_SANDBOX_OUTPUT_DIR=/app/data

# The entrypoint script handles waiting for Ollama and running benchmarks
ENTRYPOINT ["python", "-u", "/app/docker/entrypoint.py"]
