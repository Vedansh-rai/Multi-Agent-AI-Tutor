"""
Explainer Agent.
Generates student-friendly explanations with:
  - Step-by-step solution walkthrough
  - Intuitive explanation
  - Real-world analogy
  - Common mistake warnings
"""

import json
from utils.logger import get_logger
from utils.config import LLM_MODEL_NAME, get_llm_client
from utils.json_parser import parse_llm_json

logger = get_logger("agents.explainer")

_EXPLAINER_SYSTEM_PROMPT = """You are a friendly, expert JEE Advanced/Mains mathematics tutor.

Given a solved problem, create a student-friendly explanation using this MANDATORY structure:

{
  "concept_identified": "Name of the mathematical concept (e.g., Quadratic Equations — Discriminant method)",
  "relevant_formula": "The exact formula or theorem used, in LaTeX (e.g., $$x = \\\\frac{-b \\\\pm \\\\sqrt{b^2-4ac}}{2a}$$)",
  "step_by_step": [
    "Step 1: Setup — rewrite problem in standard form",
    "Step 2: Apply formula or method",
    "Step 3: Execute computation (reference tool result)",
    "Step 4: Interpret result"
  ],
  "verification": "Show substitution or check that confirms the answer (with actual numbers)",
  "key_insight": "One-sentence conceptual takeaway for the student",
  "common_pitfall": "One common mistake to avoid with this type of problem",
  "difficulty_level": "easy|medium|hard",
  "related_topics": ["topic1", "topic2"]
}

TONE: Clear, encouraging, tutor-like. Avoid jargon without explanation. Use 'we' framing (e.g., 'we set up ...', 'we substitute ...').
FORMATTING: Use $ for inline math ($x^2$) and $$ for block equations. DO NOT use \\( or \\[ delimiters.

CRITICAL: RETURN ONLY VALID JSON. NO PREAMBLE. JUST THE JSON OBJECT."""


def generate_explanation(
    problem_text: str,
    topic: str,
    solution_steps: list,
    final_answer: str,
    verification_result: dict,
    problem_type: str = "",
) -> dict:
    """
    Generate a comprehensive, student-friendly JEE explanation.

    Args:
        problem_text       : The original problem.
        topic              : Math topic (e.g., "quadratic_equations").
        solution_steps     : Solver's step list with tool results.
        final_answer       : The verified final answer.
        verification_result: Verifier's output dict.
        problem_type       : Problem type from router (e.g., "equation_solving").

    Returns:
        dict with {concept_identified, relevant_formula, step_by_step,
                   verification, key_insight, common_pitfall,
                   difficulty_level, related_topics, formatted_explanation}
    """
    try:
        client = get_llm_client()

        # Format solver steps for context
        steps_text = ""
        for s in solution_steps:
            step_num = s.get("step", "?")
            desc = s.get("description", "")
            work = s.get("work", "")
            tool_result = s.get("tool_result", "")
            steps_text += f"Step {step_num}: {desc}\n  Working: {work}\n"
            if tool_result:
                steps_text += f"  Tool output: {tool_result}\n"

        confidence = verification_result.get("confidence_score", verification_result.get("confidence", 0))
        issues = verification_result.get("issues", [])
        issues_text = "; ".join(issues) if issues else "None"

        user_msg = f"""Problem: {problem_text}
Topic: {topic}{f" | Type: {problem_type}" if problem_type else ""}

Solver Steps (with tool results):
{steps_text}

Final Answer: {final_answer}

Verification: confidence={confidence:.2f}, issues={issues_text}

Please generate the full student-friendly JEE explanation following the mandatory structure."""

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": _EXPLAINER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        explanation = parse_llm_json(content)

        # Build the canonical formatted_explanation markdown block
        steps_list = explanation.get("step_by_step", [])
        if isinstance(steps_list, str):
            steps_md = steps_list
        else:
            steps_md = "\n".join(
                f"   {s}" if s.startswith("Step") else f"   {s}"
                for s in steps_list
            )

        formatted_explanation = (
            f"\U0001f4cc **Concept Identified:** {explanation.get('concept_identified', '')}\n\n"
            f"\U0001f4da **Relevant Formula/Theorem:** {explanation.get('relevant_formula', '')}\n\n"
            f"\U0001f522 **Step-by-Step Solution:**\n{steps_md}\n\n"
            f"\u2705 **Verification:** {explanation.get('verification', '')}\n\n"
            f"\U0001f4a1 **Key Insight:** {explanation.get('key_insight', '')}"
        )

        result = {
            "success": True,
            "concept_identified": explanation.get("concept_identified", ""),
            "relevant_formula": explanation.get("relevant_formula", ""),
            "step_by_step": steps_list,
            "verification": explanation.get("verification", ""),
            "key_insight": explanation.get("key_insight", ""),
            "common_pitfall": explanation.get("common_pitfall", ""),
            "difficulty_level": explanation.get("difficulty_level", "medium"),
            "related_topics": explanation.get("related_topics", []),
            "formatted_explanation": formatted_explanation,
            "error": None,
        }

        logger.info(
            "Explanation generated (concept=%s, difficulty=%s)",
            result["concept_identified"],
            result["difficulty_level"],
        )
        return result

    except Exception as e:
        logger.error("Explainer agent failed: %s", e)
        return {
            "success": False,
            "concept_identified": "",
            "relevant_formula": "",
            "step_by_step": [],
            "verification": "",
            "key_insight": "",
            "common_pitfall": "",
            "difficulty_level": "unknown",
            "related_topics": [],
            "formatted_explanation": "",
            "error": str(e),
        }
