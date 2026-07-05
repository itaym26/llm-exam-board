"""
A real LLM client factory backed by the Anthropic API.

This module bridges the pipeline's generic `Callable[[str], str]` client
contract (defined independently in generators.py, responders.py, and
judge_engine.py) to the official `anthropic` Python SDK, so the pipeline
can be run against a real Claude model instead of a mock function.
"""

import os
from typing import Callable

import anthropic


def make_anthropic_client(model: str, max_tokens: int = 1024) -> Callable[[str], str]:
    """Builds a pipeline-compatible LLM client backed by the Anthropic API.

    Each call creates a brand-new, single-turn message with no prior
    history, which is what StudentResponder relies on to guarantee
    Context Isolation.

    Args:
        model: The Anthropic model identifier to use (e.g., "claude-sonnet-5").
        max_tokens: The maximum number of tokens to request in the response.

    Returns:
        Callable[[str], str]: A function that sends a prompt to the given
        model and returns its raw text response.

    Raises:
        RuntimeError: If the ANTHROPIC_API_KEY environment variable is not set.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell before running: "
            '$env:ANTHROPIC_API_KEY = "sk-ant-..."'
        )

    # A single shared client instance reuses the underlying HTTP connection
    # across calls; it holds no conversational state itself.
    client = anthropic.Anthropic()

    def _client(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # A text response may be split across multiple content blocks;
        # concatenating them yields the complete raw text.
        return "".join(block.text for block in response.content if block.type == "text")

    return _client
