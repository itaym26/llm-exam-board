"""
A real LLM client factory backed by Google's Gemini API (free tier).

This module bridges the pipeline's generic `Callable[[str], str]` client
contract to the official `google-genai` Python SDK, so the pipeline can be
run against a real Gemini model using a free API key from Google AI Studio.
"""

import os
from typing import Callable

from google import genai
from google.genai import types


def make_gemini_client(model: str = "gemini-2.5-flash", max_output_tokens: int = 2048) -> Callable[[str], str]:
    """Builds a pipeline-compatible LLM client backed by the Gemini API.

    Each call issues a single, standalone content-generation request with no
    prior conversation history, which is what StudentResponder relies on to
    guarantee Context Isolation.

    Args:
        model: The Gemini model identifier to use. Defaults to
            "gemini-2.5-flash". Earlier "gemini-2.0-*" models were found (see
            runs/002-.../ISSUES.md) to return 429 RESOURCE_EXHAUSTED
            immediately on a fresh free-tier key, and "gemini-1.5-flash" is
            fully retired (404); "gemini-2.5-flash" is the model confirmed
            to work on the free tier as of this writing.
        max_output_tokens: The maximum number of tokens to request in the
            response, bounding both latency and free-tier token usage.
            Defaults to 2048 rather than a smaller value because
            gemini-2.5-flash's internal "thinking" tokens are counted
            against this same budget (see runs/003-.../ISSUES.md): a
            moderately complex prompt was observed spending ~980 of a 1024
            budget on thinking alone, truncating the actual visible answer
            to a few dozen tokens. Thinking is disabled below instead, but
            the larger budget remains as a safety margin for longer
            responses (e.g. multi-item rubrics with embedded code).

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
    # across calls; it holds no conversational state itself. An explicit
    # timeout prevents a stalled connection from hanging the run forever,
    # and a bounded retry count avoids retrying indefinitely on transient
    # rate-limit (429) or server (5xx) errors.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60_000,  # milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=3,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )

    def _client(prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                # This pipeline only needs a final structured JSON answer, not
                # visible reasoning, so thinking is disabled entirely -- both
                # to avoid it silently consuming most of max_output_tokens
                # (see the truncation issue this fixed) and to reduce latency.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text or ""

    return _client
