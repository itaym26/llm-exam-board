"""
A real LLM client factory backed by Google's Gemini API (free tier).

This module bridges the pipeline's generic `Callable[[str], str]` client
contract to the official `google-genai` Python SDK, so the pipeline can be
run against a real Gemini model using a free API key from Google AI Studio.
"""

import os
from typing import Callable

from google import genai


def make_gemini_client(model: str = "gemini-2.0-flash") -> Callable[[str], str]:
    """Builds a pipeline-compatible LLM client backed by the Gemini API.

    Each call issues a single, standalone content-generation request with no
    prior conversation history, which is what StudentResponder relies on to
    guarantee Context Isolation.

    Args:
        model: The Gemini model identifier to use. Defaults to
            "gemini-2.0-flash", a fast model available on the free tier.

    Returns:
        Callable[[str], str]: A function that sends a prompt to the given
        model and returns its raw text response.

    Raises:
        RuntimeError: If the GEMINI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it in your shell before running: "
            '$env:GEMINI_API_KEY = "..."'
        )

    # A single shared client instance reuses the underlying HTTP connection
    # across calls; it holds no conversational state itself.
    client = genai.Client(api_key=api_key)

    def _client(prompt: str) -> str:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text or ""

    return _client
