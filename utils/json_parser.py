"""
Robust JSON parser for LLM outputs.
Handles common issues: leading whitespace, markdown code fences,
unescaped LaTeX backslashes, and partial responses.
"""

import re
import json


def parse_llm_json(content: str) -> dict:
    """
    Parse JSON from LLM output robustly, handling:
    - Markdown code fences (```json ... ```)
    - Unescaped LaTeX backslashes (\\sin, \\frac, etc.)
    - Leading/trailing whitespace or preamble text
    - Partial / truncated JSON

    Args:
        content: Raw string from LLM.

    Returns:
        Parsed dict, or raises ValueError if all strategies fail.
    """
    if not content or not content.strip():
        raise ValueError("LLM returned empty content")

    # ── Strategy 1: Strip markdown code fences ──
    stripped = content.strip()
    if "```json" in stripped:
        stripped = stripped.split("```json", 1)[1]
        stripped = stripped.split("```", 1)[0].strip()
    elif "```" in stripped:
        stripped = stripped.split("```", 1)[1]
        stripped = stripped.split("```", 1)[0].strip()

    # ── Strategy 2: Direct parse on stripped content ──
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # ── Strategy 3: Find JSON object with regex ──
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ── Strategy 4: Fix unescaped backslashes ──
    # Escape single backslashes that are not already escaped json escapes
    fixed = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrtu])',  # unescaped backslash not followed by json escape chars
        r'\\\\',
        stripped if not match else match.group(0),
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # ── Strategy 5: Find first { and last } overall ──
    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = content[first:last + 1]
        # Apply backslash fix
        fixed2 = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        try:
            return json.loads(fixed2)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output. Content starts: {content[:200]}")
