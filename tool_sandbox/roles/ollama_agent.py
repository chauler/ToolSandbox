# Agent role for models served through Ollama's OpenAI-compatible API.
"""Agent role for any model served through Ollama.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1,
so we can reuse the OpenAIAPIAgent with a different base_url and api_key.
"""

import os

from openai import OpenAI

from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent


class OllamaAgent(OpenAIAPIAgent):
    """Agent using a model served by a local Ollama instance.

    Expects the OLLAMA_BASE_URL environment variable to be set
    (defaults to http://localhost:11434/v1 if not set).
    """

    def __init__(self, model_name: str = "qwen2.5:3b") -> None:
        # Skip OpenAIAPIAgent.__init__ which hardcodes the OpenAI base_url.
        self.model_name = model_name
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.openai_client: OpenAI = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require a real key
        )
