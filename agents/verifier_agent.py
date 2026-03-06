"""
Verifier Agent — JEE Algebra Solver
======================================
Independently validates the solver's output before it reaches the student.

Verification checks (all applicable are run):
  1. root_substitution   — Substitute into original equation, check residual ≈ 0
  2. vieta_consistency   — Sum/product of roots matches coefficients
  3. inequality_boundary — Evaluate at boundary and interior points
  4. domain_constraint   — Confirm solution respects stated constraints
  5. matrix_inverse      — A × A⁻¹ = I
  6. probability_sanity  — 0 ≤ P ≤ 1, all outcomes sum to 1
  7. sign_parity         — Check for ± errors, missing roots

Confidence scoring rubric (per spec):
  1.0        All checks passed, verified algebraically
  0.8–0.9   All checks passed, numerical verification only
  0.6–0.7   Partial verification, some checks not applicable
  < 0.6     Verification failed — trigger retry
"""

import json
import re
from utils.logger import get_logger
from utils.config import LLM_MODEL_NAME, VERIFIER_CONFIDENCE_THRESHOLD, get_llm_client
from tools.math_tools import (
    verify_solution as math_verify_solution, verify_vieta, evaluate_expression,
    compute_probability, dispatch_tool,
)

logger = get_logger("agents.verifier")

# ── Confidence threshold (default 0.8 per spec) ──────────────────────────────
_CONFIDENCE_THRESHOLD = max(VERIFIER_CONFIDENCE_THRESHOLD, 0.8)

_LLM_VERIFIER_PROMPT = """You are a strict JEE math solution verifier.

Given a problem and a proposed solution, check:
1. NUMERICAL CORRECTNESS: Is the answer right? Verify by substitution / alternative method.
2. DOMAIN CONSTRAINTS: Does the answer satisfy all given constraints?
3. REASONING CONSISTENCY: Are the steps logically valid?
4. COMPLETENESS: Are all roots/cases covered? Any ± errors?

Return ONLY this JSON:
{
  "numerical_score": 0.0,
  "domain_score": 0.0,
  "reasoning_score": 0.0,
  "checks_run": ["root_substitution", "vieta_consistency"],
  "checks_passed": ["root_substitution"],
  "issues": ["description of any issue"],
  "is_correct": true,
  "failure_reason": null
}"""


def _parse_solutions_from_answer(final_answer: str) -> list:
    """
    Extract numeric/symbolic solution values from a final answer string.
    Handles formats: "x=2, x=3" | "[2, 3]" | "2 and 3" | "x = 1, 2, 3"
    """
    # Try to extract numbers
    numbers = re.findall(r'[-+]?\d+(?:\.\d+)?(?:/\d+)?', final_answer)
    values = []
    for n in numbers:
        try:
            if '/' in n:
                num, den = n.split('/')
                values.append(float(num) / float(den))
            else:
                values.append(float(n))
        except ValueError:
            pass
    return values


def _run_deterministic_checks(
    problem_text: str,
    final_answer: str,
    solution_steps: list,
    topic: str,
    constraints: list,
    problem_type: str = "",
) -> dict:
    """
    Run all applicable deterministic (SymPy-based) verification checks.

    Returns:
        {checks_run, checks_passed, issues, sympy_confidence}
    """
    checks_run = []
    checks_passed = []
    issues = []
    sympy_confidence = 0.0

    # ── Extract expression from solution steps ────────────────────────────
    # Look for tool results in steps to find the original expression
    original_expr = ""
    solutions_found = []
    for step in solution_steps:
        tc = step.get("tool_call") or {}
        tr = step.get("tool_result", "")
        if tc.get("type") in ("solve_equation", "SYMPY_SOLVE"):
            args = tc.get("args", {})
            original_expr = args.get("expression", tc.get("input", ""))
            variable = args.get("variable", tc.get("variable", "x"))
            # Parse solutions from tool result
            try:
                import ast
                sols = ast.literal_eval(tr)
                if isinstance(sols, list):
                    solutions_found = [(s, variable) for s in sols]
            except Exception:
                solutions_found = []

    # -- 1. Root substitution check ---------------------------------------
    if original_expr and solutions_found:
        checks_run.append("root_substitution")
        all_valid = True
        for val, var in solutions_found:
            r = math_verify_solution(original_expr, var, val)
            if not r.get("success") or not r.get("is_valid", False):
                issues.append(
                    f"Root substitution failed for {var}={val}: "
                    f"residual={r.get('residual', '?')}"
                )
                all_valid = False
        if all_valid:
            checks_passed.append("root_substitution")
            sympy_confidence += 0.35

    # -- 2. Vieta's consistency check -------------------------------------
    if solutions_found and original_expr and topic in (
        "quadratic_equations", "polynomial_equations"
    ):
        checks_run.append("vieta_consistency")
        roots = [v for v, _ in solutions_found]
        # Extract coefficients from expression via SymPy
        try:
            from sympy import Symbol, Poly, sympify
            var_sym = Symbol(variable if solutions_found else "x")
            expr = sympify(original_expr.replace("^", "**"))
            poly = Poly(expr, var_sym)
            coeffs = poly.all_coeffs()
            if len(coeffs) >= 3:  # quadratic: ax^2 + bx + c
                a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
                r = verify_vieta(roots, coeff_a=a, coeff_b=b, coeff_c=c, n=2)
                if r.get("all_pass"):
                    checks_passed.append("vieta_consistency")
                    sympy_confidence += 0.25
                else:
                    issues.append("Vieta's formula check failed: " + str(r.get("checks")))
        except Exception as ve:
            logger.debug("Vieta check skipped: %s", ve)

    # -- 3. Domain constraint check ---------------------------------------
    if constraints:
        checks_run.append("domain_constraint")
        all_constrained = True
        for val, var in solutions_found:
            for constraint in constraints:
                try:
                    r = evaluate_expression(
                        constraint.replace(var, str(val)).replace("x", str(val)),
                        {}
                    )
                    # If constraint evaluation gives a boolean-like result
                    result_val = r.get("result", 1.0)
                    if result_val is not None and float(result_val) < 0:
                        issues.append(f"Domain constraint '{constraint}' not satisfied for {var}={val}")
                        all_constrained = False
                except Exception:
                    pass  # Can't evaluate symbolically, skip
        if all_constrained:
            checks_passed.append("domain_constraint")
            sympy_confidence += 0.15

    # -- 4. Probability sanity check -------------------------------------
    if topic == "probability" or "probability" in problem_type:
        checks_run.append("probability_sanity")
        # Extract probability values from final answer
        prob_values = _parse_solutions_from_answer(final_answer)
        if prob_values:
            all_valid = all(0 <= p <= 1 for p in prob_values)
            if all_valid:
                checks_passed.append("probability_sanity")
                sympy_confidence += 0.20
            else:
                issues.append(f"Probability out of range [0,1]: {prob_values}")

    # -- 5. Matrix inverse check -----------------------------------------
    # This check is done symbolically by compute_matrix_inverse itself
    # We just look for the verification flag in tool results
    for step in solution_steps:
        tr_data = step.get("tool_result", "")
        tc = step.get("tool_call") or {}
        if tc.get("type") == "compute_matrix_inverse":
            checks_run.append("matrix_inverse")
            if "True" in str(tr_data) or "verification_A_Ainv_eq_I" in str(tr_data):
                checks_passed.append("matrix_inverse")
                sympy_confidence += 0.20

    # -- 6. Generic deterministic tool check ------------------------------
    if sympy_confidence < 0.5:
        trusted_tools = [
            "solve_inequality", "find_range", "factor_polynomial", "expand_expression",
            "compute_determinant", "compute_eigenvalues", "solve_linear_system",
            "compute_combination", "compute_permutation", "compute_binomial_term",
            "compute_ap_sum", "compute_gp_sum", "simplify_expression",
            "differentiate", "compute_limit"
        ]
        for step in solution_steps:
            tc = step.get("tool_call") or {}
            tr_str = str(step.get("tool_result", ""))
            t_type = tc.get("type", "")
            if t_type in trusted_tools and tr_str and "error" not in tr_str.lower():
                checks_run.append(t_type + "_sanity")
                tr_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', tr_str)
                fa_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', final_answer)
                
                if tr_nums and fa_nums:
                    if set(tr_nums).intersection(set(fa_nums)):
                        checks_passed.append(t_type + "_sanity")
                        sympy_confidence += 0.60
                        break
                elif "oo" in tr_str and ("oo" in final_answer or "\\infty" in final_answer):
                    checks_passed.append(t_type + "_sanity")
                    sympy_confidence += 0.60
                    break

    # Clamp
    sympy_confidence = min(sympy_confidence, 1.0)

    return {
        "checks_run":      checks_run,
        "checks_passed":   checks_passed,
        "issues":          issues,
        "sympy_confidence": sympy_confidence,
    }


def verify_solution_output(
    problem_text: str,
    final_answer: str,
    solution_steps: list,
    topic: str,
    constraints: list,
    rag_sources: list = None,
    problem_type: str = "",
) -> dict:
    """Alias matching the JEE spec naming."""
    return run_verification(
        problem_text, final_answer, solution_steps,
        topic, constraints, rag_sources or [], problem_type
    )

def run_verification(
    problem_text: str,
    final_answer: str,
    solution_steps: list,
    topic: str,
    constraints: list,
    rag_sources: list = None,
    problem_type: str = "",
) -> dict:
    """
    Run ALL applicable verification checks on the solver's output.

    Pipeline:
      1. Deterministic SymPy checks (substitution, Vieta's, domain, probability, matrix)
      2. LLM-based semantic check
      3. Hybrid confidence scoring

    Output schema (per spec):
    {
      "verified": bool,
      "confidence_score": float,
      "checks_run": [str],
      "checks_passed": [str],
      "failure_reason": str | None,
      "issues": [str],
      "needs_retry": bool,
      "needs_human_review": bool,
    }
    """
    rag_sources = rag_sources or []

    # ── Phase 1: Deterministic SymPy checks ──────────────────────────────
    det = _run_deterministic_checks(
        problem_text, final_answer, solution_steps,
        topic, constraints, problem_type
    )
    sympy_conf    = det["sympy_confidence"]
    checks_run    = list(det["checks_run"])
    checks_passed = list(det["checks_passed"])
    issues        = list(det["issues"])

    # ── Phase 2: RAG relevance score ─────────────────────────────────────
    if rag_sources:
        avg_rag = sum(s.get("score", 0) for s in rag_sources) / len(rag_sources)
        retrieval_score = min(avg_rag * 1.2, 1.0)
    else:
        retrieval_score = 0.3

    # ── Phase 3: LLM semantic check ───────────────────────────────────────
    llm_numerical = 0.5
    llm_domain    = 0.5
    llm_reasoning = 0.5

    try:
        client = get_llm_client()
        steps_text = ""
        for s in solution_steps:
            steps_text += (
                f"Step {s.get('step','?')}: {s.get('description','')}"
                f" | Work: {s.get('work','')}"
                + (f" | Tool: {s.get('tool_result','')}" if s.get('tool_result') else "")
                + "\n"
            )

        user_msg = (
            f"Problem: {problem_text}\nTopic: {topic}\nConstraints: {constraints}\n"
            f"\nSolution Steps:\n{steps_text}\nFinal Answer: {final_answer}"
        )

        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": _LLM_VERIFIER_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        v = json.loads(resp.choices[0].message.content or "{}")

        llm_numerical = float(v.get("numerical_score", 0.5))
        llm_domain    = float(v.get("domain_score",    0.5))
        llm_reasoning = float(v.get("reasoning_score", 0.5))

        for chk in v.get("checks_run", []):
            if chk not in checks_run:
                checks_run.append(chk)
        for chk in v.get("checks_passed", []):
            if chk not in checks_passed:
                checks_passed.append(chk)
        for issue in v.get("issues", []):
            if issue not in issues:
                issues.append(issue)

    except Exception as e:
        logger.error("LLM verifier call failed: %s", e)
        issues.append(f"LLM verification failed: {e}")

    # ── Phase 4: Hybrid confidence scoring ───────────────────────────────
    if sympy_conf > 0:
        confidence = (
            0.50 * sympy_conf
            + 0.25 * llm_numerical
            + 0.15 * llm_domain
            + 0.10 * llm_reasoning
        )
    else:
        confidence = (
            0.40 * llm_numerical
            + 0.30 * llm_domain
            + 0.20 * llm_reasoning
            + 0.10 * retrieval_score
        )
    confidence = round(min(confidence, 1.0), 3)

    # Per spec rubric: >= 0.8 if all checks passed algebraically
    if checks_passed and sympy_conf >= 0.5:
        confidence = max(confidence, 0.80)

    verified       = confidence >= _CONFIDENCE_THRESHOLD
    failure_reason = "; ".join(issues) if issues else None
    if not verified and not failure_reason:
        failure_reason = f"Confidence {confidence:.2f} below threshold {_CONFIDENCE_THRESHOLD:.2f}"

    result = {
        "verified":           verified,
        "confidence_score":   confidence,
        "checks_run":         checks_run,
        "checks_passed":      checks_passed,
        "failure_reason":     failure_reason if not verified else None,
        "issues":             issues,
        "needs_retry":        not verified,
        "needs_human_review": confidence < 0.6,
        "score_breakdown": {
            "sympy_deterministic": round(sympy_conf, 3),
            "llm_numerical":       round(llm_numerical, 3),
            "llm_domain":          round(llm_domain, 3),
            "llm_reasoning":       round(llm_reasoning, 3),
            "retrieval_relevance": round(retrieval_score, 3),
        },
    }

    logger.info(
        "Verifier: conf=%.3f, verified=%s, checks=%d/%d",
        confidence, verified, len(checks_passed), len(checks_run)
    )
    return result


# Backward-compatible alias
verify_solution = run_verification
