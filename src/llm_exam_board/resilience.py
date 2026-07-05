"""
Micro-Resilience utilities for the V14 LLM Evaluation Pipeline.

This module exists to shield the rest of the pipeline from the unpredictable
text formatting produced by LLM APIs (markdown code fences, trailing prose,
trailing commas, truncated output). Every module that needs to interpret a
raw LLM response as JSON must route through ResilienceUtils rather than
calling json.loads directly.
"""

import json
import re
from typing import Any, Dict, Optional


class ResilienceUtils:
    """A stateless collection of static helpers for safely parsing noisy LLM output.

    This class intentionally holds no instance state; all attributes are
    class-level, private, immutable regex patterns used purely as
    implementation detail. It is never instantiated.
    """

    # Matches a fenced code block (```json ... ``` or ``` ... ```) and
    # captures its inner content, since LLMs frequently wrap JSON in markdown.
    __CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    # Matches a comma immediately followed by a closing brace/bracket, which
    # is a common (invalid-JSON) mistake made by LLMs when listing items.
    __TRAILING_COMMA_PATTERN = re.compile(r",(\s*[\}\]])")

    # Matches the first opening brace or bracket in a string, used as the
    # entry point for the balanced-bracket scan below.
    __JSON_START_PATTERN = re.compile(r"[\{\[]")

    @classmethod
    def clean_json_response(cls, raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
        """Safely extracts a JSON object from noisy, unpredictable LLM output.

        This method never raises on malformed input. It progressively
        attempts several recovery strategies and returns None if none of
        them succeed, allowing callers to treat parse failure as ordinary
        data rather than a crash.

        Args:
            raw_text: The raw text returned by an LLM API call. May contain
                markdown fences, leading/trailing prose, or minor JSON
                syntax mistakes.

        Returns:
            Optional[Dict[str, Any]]: The parsed JSON object, or None if no
            valid JSON object could be recovered from the input.
        """
        if not raw_text or not raw_text.strip():
            return None

        candidate = cls.__strip_code_fences(raw_text)

        # Strategy 1: the response is already clean, directly parseable JSON.
        parsed = cls.__try_parse(candidate)
        if parsed is not None:
            return parsed

        # Strategy 2: locate a balanced JSON object/array embedded in noise
        # (e.g., "Sure, here is the result: { ... } Let me know if...").
        balanced_candidate = cls.__extract_balanced_json(candidate)
        if balanced_candidate is not None:
            parsed = cls.__try_parse(balanced_candidate)
            if parsed is not None:
                return parsed

            # Strategy 3: escape raw control characters (newlines, carriage
            # returns, tabs) embedded inside string literals. This is a very
            # common mistake when an LLM embeds multi-line code or text as a
            # JSON string value without escaping it, which otherwise makes
            # the string literal itself invalid JSON.
            control_char_escaped = cls.__escape_control_chars_in_strings(balanced_candidate)
            parsed = cls.__try_parse(control_char_escaped)
            if parsed is not None:
                return parsed

            # Strategy 4: repair the common "trailing comma" mistake on top
            # of the control-character-escaped candidate, in case both
            # mistakes occur in the same response.
            repaired_candidate = cls.__TRAILING_COMMA_PATTERN.sub(r"\1", control_char_escaped)
            parsed = cls.__try_parse(repaired_candidate)
            if parsed is not None:
                return parsed

        # All recovery strategies failed; the caller must decide how to
        # handle a genuine absence of structured data.
        return None

    @classmethod
    def __strip_code_fences(cls, text: str) -> str:
        """Removes markdown code fences and returns the inner content if present.

        Args:
            text: Raw text that may or may not contain a fenced code block.

        Returns:
            str: The content of the first fenced code block if found,
            otherwise the original text, in both cases stripped of
            leading/trailing whitespace.
        """
        fence_match = cls.__CODE_FENCE_PATTERN.search(text)
        if fence_match:
            return fence_match.group(1).strip()
        return text.strip()

    @classmethod
    def __extract_balanced_json(cls, text: str) -> Optional[str]:
        """Scans text for the first structurally balanced JSON object or array.

        A regex locates the starting brace/bracket; from there, a manual
        character scan tracks nesting depth while correctly ignoring braces
        that appear inside string literals (including escaped quotes), since
        naive regex alone cannot reliably match balanced, nested structures.

        Args:
            text: Text expected to contain a JSON object or array, possibly
                surrounded by other content.

        Returns:
            Optional[str]: The substring spanning a structurally balanced
            JSON object/array, or None if no complete balanced structure
            could be found.
        """
        start_match = cls.__JSON_START_PATTERN.search(text)
        if not start_match:
            return None

        start_index = start_match.start()
        opening_char = text[start_index]
        closing_char = "}" if opening_char == "{" else "]"

        depth = 0
        in_string = False
        escape_next = False

        for index in range(start_index, len(text)):
            char = text[index]

            if escape_next:
                # This character was escaped by a preceding backslash; skip
                # any special interpretation of it.
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                # Braces/brackets inside string literals must not affect depth.
                continue

            if char == opening_char:
                depth += 1
            elif char == closing_char:
                depth -= 1
                if depth == 0:
                    return text[start_index : index + 1]

        # Reached end of text without the depth returning to zero: the
        # JSON structure is truncated or otherwise unbalanced.
        return None

    @staticmethod
    def __escape_control_chars_in_strings(text: str) -> str:
        """Escapes raw newline/carriage-return/tab characters found inside string literals.

        LLMs frequently embed multi-line code or text as a JSON string value
        using a literal line break rather than the escaped `\\n` sequence
        strict JSON requires, which otherwise makes `json.loads` reject an
        otherwise well-structured response. Only characters found between
        unescaped double quotes are rewritten; everything outside string
        literals (including insignificant whitespace between tokens) is
        left untouched.

        Args:
            text: A candidate JSON string, potentially containing raw
                control characters inside its string literals.

        Returns:
            str: The text with control characters inside string literals
            replaced by their escaped equivalents.
        """
        result_chars = []
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                # This character is already part of an escape sequence
                # (e.g., the "n" in "\n"); pass it through untouched.
                result_chars.append(char)
                escape_next = False
                continue
            if char == "\\":
                result_chars.append(char)
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                result_chars.append(char)
                continue

            if in_string and char == "\n":
                result_chars.append("\\n")
            elif in_string and char == "\r":
                result_chars.append("\\r")
            elif in_string and char == "\t":
                result_chars.append("\\t")
            else:
                result_chars.append(char)

        return "".join(result_chars)

    @staticmethod
    def __try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        """Attempts to parse a string as a JSON object, suppressing all errors.

        Args:
            candidate: The string to attempt to parse as JSON.

        Returns:
            Optional[Dict[str, Any]]: The parsed dictionary if `candidate`
            is valid JSON representing an object, otherwise None. Top-level
            JSON arrays or scalars are deliberately rejected, since every
            consumer in this pipeline expects a JSON object.
        """
        try:
            result = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        return result if isinstance(result, dict) else None
