"""
Parser Agent — JEE Algebra Solver
===================================
Transforms raw user input (typed, OCR'd, or voice-transcribed) into a clean,
structured problem object matching the JEE Algebra Solver spec.

Notation normalisation table:
  x², x^2, x**2  → x**2
  √x, √(x+1)     → sqrt(x), sqrt(x+1)
  ∞               → oo
  ≤, ≥            → <=, >=
  log x, ln x     → log(x, 10), log(x)
"""

import json
import re
from utils.logger import get_logger
from utils.config import LLM_MODEL_NAME, get_llm_client

logger = get_logger("agents.parser")

# ── Supported topics from the spec ──────────────────────────────────────────
SUPPORTED_TOPICS = [
    "number_system",
    "inequalities",
    "quadratic_equations",
    "polynomial_equations",
    "complex_numbers",
    "progressions",
    "permutation_combination",
    "binomial_theorem",
    "matrices_determinants",
    "probability",
    "calculus",
]

# ── Notation normalisation ───────────────────────────────────────────────────
_NOTATION_FIXES = [
    (r"x\^2|x²",           "x**2"),
    (r"x\^3|x³",           "x**3"),
    (r"\^(\d+)",           r"**\1"),
    (r"√\(([^)]+)\)",      r"sqrt(\1)"),
    (r"√([a-zA-Z0-9]+)",   r"sqrt(\1)"),
    (r"∞",                 "oo"),
    (r"≤",                 "<="),
    (r"≥",                 ">="),
    (r"≠",                 "!="),
    (r"×",                 "*"),
    (r"÷",                 "/"),
    (r"\blog\s+([a-z])",   r"log(\1, 10)"),
    (r"\bln\s+([a-z])",    r"log(\1)"),
]


def _normalise_notation(text: str) -> str:
    """Apply notation normalisation rules to raw input."""
    for pattern, repl in _NOTATION_FIXES:
        text = re.sub(pattern, repl, text)
    return text


_PARSER_SYSTEM_PROMPT = """You are a strict math problem parser for JEE Advanced/Mains-level algebra.

Given raw text (possibly from OCR or voice), produce a structured JSON object.

NOTATION NORMALISATION RULES (apply these before generating sympy_expression):
- x², x^2, x**2  → x**2
- √x              → sqrt(x)
- ∞               → oo
- ≤, ≥            → <=, >=
- log x           → log(x, 10)   (base-10)
- ln x            → log(x)       (natural log)

SUPPORTED TOPICS (pick the best match):
  number_system | inequalities | quadratic_equations | polynomial_equations |
  complex_numbers | progressions | permutation_combination |
  binomial_theorem | matrices_determinants | probability | calculus

PROBLEM TYPES:
  equation_solving | inequality_solving | range_finding | simplification |
  factorization | matrix_operations | probability_classical | sequences_series |
  binomial_expansion | combinatorics | number_theory | derivative_problem |
  limit_problem | optimization_problem | matrix_problem | determinant_problem

Return ONLY this JSON — no extra text:
{
  "problem_text": "<cleaned, well-formatted problem statement>",
  "topic": "<one of the 10 supported topics>",
  "problem_type": "<one of the problem types above>",
  "goal": "<solve | simplify | find_range | evaluate | expand | verify>",
  "variables": ["x"],
  "constraints": ["x > 0"],
  "sympy_expression": "<SymPy-ready string or empty string>",
  "parse_confidence": 0.85
}

Rules:
- parse_confidence reflects how confidently you understood the problem (0.0–1.0)
- If the input is incomplete or ambiguous, set parse_confidence < 0.6
- sympy_expression should contain the core algebraic expression (not the full sentence)
- Always return valid JSON with all fields present"""


def parse_problem(raw_text: str, input_type: str = "text") -> dict:
    """
    Parse and structure a raw JEE algebra problem.

    Applies notation normalisation, then calls the LLM to produce a fully
    structured problem object.  Falls back gracefully if the LLM is unavailable.

    Args:
        raw_text   : Raw text from user, OCR, or ASR.
        input_type : One of 'text', 'ocr', 'audio'.

    Returns:
        Parsed problem dict matching the JEE Algebra Solver schema:
        {problem_text, topic, problem_type, goal, variables, constraints,
         sympy_expression, parse_confidence, needs_clarification,
         clarification_reason}

    Failure mode: parse_confidence < 0.6 → caller should ask for clarification.
    """
    _EMPTY = {
        "problem_text": "",
        "topic": "unknown",
        "problem_type": "unknown",
        "goal": "unknown",
        "variables": [],
        "constraints": [],
        "sympy_expression": "",
        "parse_confidence": 0.0,
        "needs_clarification": True,
        "clarification_reason": "Empty input received.",
    }

    if not raw_text or not raw_text.strip():
        return _EMPTY

    # Pre-process: normalise notation before sending to LLM
    normalised = _normalise_notation(raw_text.strip())

    try:
        client = get_llm_client()
        user_msg = (
            f"Input type: {input_type}\n\n"
            f"Raw text:\n{raw_text}\n\n"
            f"Notation-normalised text (for reference):\n{normalised}"
        )

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": _PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        content = content.strip()

        # Strip code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)

        # Validate topic
        topic = parsed.get("topic", "quadratic_equations")
        if topic not in SUPPORTED_TOPICS:
            topic = "quadratic_equations"

        confidence = float(parsed.get("parse_confidence", 0.8))
        needs_clarification = confidence < 0.6

        result = {
            "problem_text":     parsed.get("problem_text", raw_text),
            "topic":            topic,
            "problem_type":     parsed.get("problem_type", "equation_solving"),
            "goal":             parsed.get("goal", "solve"),
            "variables":        parsed.get("variables", ["x"]),
            "constraints":      parsed.get("constraints", []),
            "sympy_expression": parsed.get("sympy_expression", normalised),
            "parse_confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_reason": (
                parsed.get("clarification_reason", "")
                if needs_clarification else ""
            ),
        }

        logger.info(
            "Parser: topic=%s, type=%s, confidence=%.2f",
            result["topic"], result["problem_type"], confidence
        )
        return result

    except json.JSONDecodeError as e:
        logger.error("Parser LLM returned invalid JSON: %s", e)
        return {
            "problem_text":     raw_text,
            "topic":            "quadratic_equations",
            "problem_type":     "equation_solving",
            "goal":             "solve",
            "variables":        ["x"],
            "constraints":      [],
            "sympy_expression": normalised,
            "parse_confidence": 0.5,
            "needs_clarification": True,
            "clarification_reason": f"Could not parse problem structure: {e}",
        }
    except Exception as e:
        logger.error("Parser agent error: %s", e)
        return {
            "problem_text":     raw_text,
            "topic":            "quadratic_equations",
            "problem_type":     "equation_solving",
            "goal":             "solve",
            "variables":        ["x"],
            "constraints":      [],
            "sympy_expression": normalised,
            "parse_confidence": 0.5,
            "needs_clarification": True,
            "clarification_reason": f"Parser error: {e}",
        }
