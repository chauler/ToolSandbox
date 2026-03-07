#!/usr/bin/env python3
"""Entrypoint for the ToolSandbox Docker container.

Waits for the Ollama server to become healthy, then runs the requested
benchmark scenarios via the ``tool_sandbox`` CLI.

Environment variables
---------------------
OLLAMA_HOST          Ollama server URL (default: http://ollama:11434)
OLLAMA_MODEL         Model tag to use (default: qwen2.5:3b)
SCENARIOS            Space-separated scenario names, or "all" / "test"
                     (default: test)
PARALLEL             Number of parallel workers (default: 1)
OUTPUT_DIR           Where to write results (default: /app/data)
"""

import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
SCENARIOS = os.environ.get("SCENARIOS", "test")
PARALLEL = os.environ.get("PARALLEL", "1")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/data")

# Expose the Ollama endpoint to the openai library used by OllamaAgent/User
os.environ["OLLAMA_BASE_URL"] = f"{OLLAMA_HOST}/v1"
# A dummy key so the openai library doesn't complain
os.environ.setdefault("OPENAI_API_KEY", "unused")


def wait_for_ollama(timeout: int = 300) -> None:
    """Block until Ollama's API responds, up to *timeout* seconds."""
    deadline = time.time() + timeout
    url = f"{OLLAMA_HOST}/api/version"
    print(f"Waiting for Ollama at {url} …", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                print(f"  Ollama is up — version {data.get('version', '?')}", flush=True)
                return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    print("ERROR: Ollama did not become reachable within timeout.", flush=True)
    sys.exit(1)


def ensure_model_ready() -> None:
    """Make sure the desired model is pulled on the Ollama server."""
    # Check if model is already available
    try:
        url = f"{OLLAMA_HOST}/api/tags"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if OLLAMA_MODEL in models:
                print(f"  Model '{OLLAMA_MODEL}' is already available.", flush=True)
                return
    except (urllib.error.URLError, OSError, KeyError):
        pass

    # Pull the model via the Ollama API
    print(f"  Pulling model '{OLLAMA_MODEL}' …", flush=True)
    req_data = json.dumps({"name": OLLAMA_MODEL, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/pull",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            body = resp.read().decode()
            print(f"  Pull complete: {body[:200]}", flush=True)
    except Exception as exc:
        print(f"ERROR pulling model: {exc}", flush=True)
        sys.exit(1)


def run_benchmark() -> None:
    """Invoke the ``tool_sandbox`` CLI."""
    cmd = [
        "tool_sandbox",
        "--user", "Ollama",
        "--agent", "Ollama",
        "-p", PARALLEL,
        "-o", OUTPUT_DIR,
    ]

    if SCENARIOS == "all":
        pass  # no --scenario flag means all
    elif SCENARIOS == "test":
        cmd.append("-t")
    else:
        cmd.extend(["-s"] + SCENARIOS.split())

    print(f"\nRunning: {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main() -> None:
    wait_for_ollama()
    ensure_model_ready()
    run_benchmark()


if __name__ == "__main__":
    main()
