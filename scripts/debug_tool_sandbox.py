"""Debug entrypoint for ToolSandbox in VS Code.

This launcher lets VS Code users pass all CLI args through one environment
variable so they can quickly tweak scenarios, parallelism, and agent config
without editing Python code.
"""

from __future__ import annotations

import os
import shlex
import sys

from tool_sandbox.cli import main as cli_main

DEFAULT_ARGS = "--agent Ollama --user Ollama -t -p 1 -o data/debug"


def build_argv_from_env() -> list[str]:
    """Build argv from TOOL_SANDBOX_ARGS with a safe default."""
    raw = os.environ.get("TOOL_SANDBOX_ARGS", "").strip() or DEFAULT_ARGS
    try:
        parsed_args = shlex.split(raw)
    except ValueError as exc:
        raise ValueError(
            "Invalid TOOL_SANDBOX_ARGS. Check quotes/escaping in the launch profile."
        ) from exc
    return ["tool_sandbox", *parsed_args]


def main() -> None:
    sys.argv = build_argv_from_env()
    print(f"[debug-launch] Executing: {' '.join(sys.argv)}", flush=True)
    print(
        "[debug-launch] OLLAMA_BASE_URL="
        f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}",
        flush=True,
    )
    cli_main()


if __name__ == "__main__":
    main()
