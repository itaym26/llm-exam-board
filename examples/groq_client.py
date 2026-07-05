"""
A real LLM client factory backed by the Groq API (free tier).

This module bridges the pipeline's generic `Callable[[str], str]` client
contract to the official `groq` Python SDK, so the pipeline can be run
against real, fast, open-weight models (e.g., Llama) using a free API key
from the Groq console.
"""

import os
from typing import Callable

from groq import Groq


def make_groq_client(model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024) -> Callable[[str], str]:
    """Builds a pipeline-compatible LLM client backed by the Groq API.

    Each call creates a brand-new, single-turn chat completion with no
    prior history, which is what StudentResponder relies on to guarantee
    Context Isolation.

    Args:
        model: The Groq-hosted model identifier to use. Defaults to
            "llama-3.3-70b-versatile", a strong, free-tier-eligible model.
        max_tokens: The maximum number of tokens to request in the response.

    Returns:
        Callable[[str], str]: A function that sends a prompt to the given
        model and returns its raw text response.

    Raises:
        RuntimeError: If the GROQ_API_KEY environment variable is not set.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it in your shell before running: "
            '$env:GROQ_API_KEY = "..."'
        )

    # A single shared client instance reuses the underlying HTTP connection
    # across calls; it holds no conversational state itself.
    client = Groq()

    def _client(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return _client
