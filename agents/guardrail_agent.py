"""
Guardrail Agent.
Validates inputs before processing — blocks unsafe, off-topic, or malicious prompts.
Ensures the system only processes genuine math questions.
"""

from utils.logger import get_logger

logger = get_logger("agents.guardrail")

# Keywords that indicate non-math or unsafe content
_BLOCKED_KEYWORDS = [
    "hack", "exploit", "injection", "password", "credit card",
    "social security", "bomb", "weapon", "drug", "kill",
    "ignore previous", "ignore instructions", "forget your rules",
    "system prompt", "jailbreak",
]

# Allowed math-related keywords (if none present, likely not a math question)
_MATH_INDICATORS = [
    "solve", "find", "calculate", "compute", "evaluate", "simplify",
    "differentiate", "integrate", "derive", "prove", "show that",
    "equation", "expression", "function", "limit", "sum", "product",
    "matrix", "determinant", "eigenvalue", "probability", "permutation",
    "combination", "area", "volume", "angle", "triangle", "circle",
    "quadratic", "polynomial", "logarithm", "exponent", "factorial",
    "sin", "cos", "tan", "sqrt", "root", "fraction", "ratio",
    "x", "y", "z", "=", "+", "-", "*", "/", "^", "²", "³",
    "∫", "∑", "π", "θ", "∞", "derivative", "integral",
    "algebra", "calculus", "geometry", "trigonometry", "statistics",
    "mean", "median", "variance", "standard deviation",
    "maximum", "minimum", "optimization", "converge", "diverge",
]


def guardrail_check(text: str) -> dict:
    """
    Validate that the input is a safe, math-related question.

    Args:
        text: The user's input text (raw or after OCR/ASR).

    Returns:
        dict with keys:
            - is_safe (bool): Whether the input passes all checks.
            - is_math (bool): Whether the input appears to be math-related.
            - blocked_reason (str|None): Reason for blocking, if any.
            - warnings (list[str]): Non-blocking warnings.
    """
    text_lower = text.lower().strip()
    warnings = []

    # ── Check 1: Empty input ──
    if not text_lower or len(text_lower) < 3:
        logger.warning("Guardrail: Empty or too-short input")
        return {
            "is_safe": False,
            "is_math": False,
            "blocked_reason": "Input is empty or too short. Please provide a math question.",
            "warnings": [],
        }

    # ── Check 2: Blocked keywords (unsafe / prompt injection) ──
    for keyword in _BLOCKED_KEYWORDS:
        if keyword in text_lower:
            logger.warning("Guardrail: Blocked keyword detected — '%s'", keyword)
            return {
                "is_safe": False,
                "is_math": False,
                "blocked_reason": f"Input contains restricted content. This system only handles math problems.",
                "warnings": [],
            }

    # ── Check 3: Math relevance ──
    has_math = any(indicator in text_lower for indicator in _MATH_INDICATORS)
    # Also check for numbers as a fallback
    has_numbers = any(c.isdigit() for c in text)

    if not has_math and not has_numbers:
        logger.warning("Guardrail: Input does not appear to be math-related")
        return {
            "is_safe": True,
            "is_math": False,
            "blocked_reason": "This doesn't appear to be a math question. Please provide a math problem.",
            "warnings": ["Input may not be math-related. System works best with math questions."],
        }

    # ── Check 4: Excessive length ──
    if len(text) > 5000:
        warnings.append("Input is very long. Consider breaking it into smaller problems.")

    # ── All checks passed ──
    logger.info("Guardrail: Input passed all checks (is_math=%s)", has_math)
    return {
        "is_safe": True,
        "is_math": True,
        "blocked_reason": None,
        "warnings": warnings,
    }
