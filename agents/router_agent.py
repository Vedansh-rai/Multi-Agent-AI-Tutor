"""
Intent Router Agent — JEE Algebra Solver
==========================================
Classifies the parsed problem and determines which solver pipeline
and tool chain to activate.

Problem types (per spec):
  equation_solving | inequality_solving | range_finding | probability |
  polynomial_factorization | simplification | matrix_operations |
  sequences_series | binomial | combinatorics
"""

import re
from utils.logger import get_logger

logger = get_logger("agents.router")

# ── Full routing table ───────────────────────────────────────────────────────

_ROUTING_TABLE = {
    "equation_solving": {
        "triggers": [r"\bsolve\b", r"\bfind\s+x\b", r"\broots?\s+of\b",
                     r"\bequation\b", r"=\s*0", r"\bquadratic\b", r"\bcubic\b"],
        "tool_chain": ["solve_equation"],
        "fallback": "factor_polynomial then solve_equation",
        "rag_hints": "quadratic equation roots discriminant Vieta formula",
    },
    "inequality_solving": {
        "triggers": [r"[<>](?!=)", r"\binequality\b", r"\brange\s+where\b",
                     r"\bless\s+than\b", r"\bgreater\s+than\b"],
        "tool_chain": ["solve_inequality"],
        "fallback": "evaluate at boundary points",
        "rag_hints": "inequality AM-GM Cauchy-Schwarz sign analysis intervals",
    },
    "range_finding": {
        "triggers": [r"\brange\s+of\b", r"\bpossible\s+values?\b",
                     r"\bvalues?\s+taken\b", r"\bset\s+of\s+values\b"],
        "tool_chain": ["find_range"],
        "fallback": "critical point analysis via differentiation",
        "rag_hints": "range of function domain codomain critical points",
    },
    "probability": {
        "triggers": [r"\bprobability\b", r"\bP\s*\(", r"\bchance\b",
                     r"\bbayes\b", r"\bconditional\b"],
        "tool_chain": ["compute_probability", "compute_combination"],
        "fallback": "enumerate sample space manually",
        "rag_hints": "probability classical definition conditional Bayes theorem",
    },
    "polynomial_factorization": {
        "triggers": [r"\bfactoriz", r"\bfactor\b", r"\bfactorise\b"],
        "tool_chain": ["factor_polynomial"],
        "fallback": "solve_equation then reconstruct factors",
        "rag_hints": "polynomial factorization remainder theorem rational roots",
    },
    "simplification": {
        "triggers": [r"\bsimplif", r"\breduce\b"],
        "tool_chain": ["simplify_expression", "expand_expression"],
        "fallback": "expand then collect like terms",
        "rag_hints": "algebraic simplification algebraic identities",
    },
    "matrix_operations": {
        "triggers": [r"\bdeterminant\b", r"\bdet\b", r"\binverse\b",
                     r"\beigenvalue\b", r"\bmatri[cx]", r"\brank\b"],
        "tool_chain": ["compute_determinant", "compute_matrix_inverse",
                       "compute_eigenvalues", "solve_linear_system"],
        "fallback": "cofactor expansion manually",
        "rag_hints": "matrix determinant inverse eigenvalue Cramer cofactor",
    },
    "sequences_series": {
        "triggers": [r"\bAP\b", r"\bGP\b", r"\bHP\b",
                     r"\barithmetic\s+progression\b", r"\bgeometric\s+progression\b",
                     r"\bnth\s+term\b", r"\bsum\s+(?:of|to)\s+\d*\s*terms?\b"],
        "tool_chain": ["compute_ap_sum", "compute_gp_sum"],
        "fallback": "formula lookup + evaluate_expression",
        "rag_hints": "AP GP HP nth term sum formulas infinite series convergence",
    },
    "binomial": {
        "triggers": [r"\bbinomial\b", r"\bcoefficient\s+of\b",
                     r"\bmiddle\s+term\b", r"\bgeneral\s+term\b", r"\bT_?\d+\b"],
        "tool_chain": ["compute_binomial_term", "expand_expression"],
        "fallback": "expand_expression and extract coefficient",
        "rag_hints": "binomial theorem general term middle term coefficient expansion",
    },
    "combinatorics": {
        "triggers": [r"\barrangements?\b", r"\bselections?\b",
                     r"\bnCr\b", r"\bnPr\b", r"\bcombinations?\b",
                     r"\bpermutations?\b", r"\bderangements?\b"],
        "tool_chain": ["compute_combination", "compute_permutation"],
        "fallback": "factorial counting manually",
        "rag_hints": "permutation combination nCr nPr derangement factorial counting",
    },
    "derivative_problem": {
        "triggers": [r"\bderivative\b", r"\bdifferentiate\b", r"\brate\s+of\s+change\b", r"f'\s*\("],
        "tool_chain": ["differentiate"],
        "fallback": "use limit definition of derivative",
        "rag_hints": "derivative differentiation rate of change chain rule product rule quotient rule",
    },
    "limit_problem": {
        "triggers": [r"\blimit\b", r"\blim\b", r"x\s*(?:->|\\rightarrow|to)\s*"],
        "tool_chain": ["compute_limit"],
        "fallback": "L'Hopital's rule or algebraic manipulation",
        "rag_hints": "limit L'Hopital rule continuity evaluation",
    },
    "optimization_problem": {
        "triggers": [r"\bmaximi[zs]e\b", r"\bminimi[zs]e\b", r"\bmaximum\b", r"\bminimum\b", r"\bextremum\b"],
        "tool_chain": ["differentiate", "solve_equation"],
        "fallback": "vertex form for quadratics or inequalities",
        "rag_hints": "optimization maximum minimum derivative critical points second derivative test",
    },
}

_TOPIC_TO_TYPE = {
    "quadratic_equations":     "equation_solving",
    "polynomial_equations":    "equation_solving",
    "inequalities":            "inequality_solving",
    "matrices_determinants":   "matrix_operations",
    "probability":             "probability",
    "permutation_combination": "combinatorics",
    "binomial_theorem":        "binomial",
    "progressions":            "sequences_series",
    "complex_numbers":         "equation_solving",
    "number_system":           "simplification",
    "calculus":                "derivative_problem",
}

_STRATEGIES = {
    "equation_solving":         "Apply SymPy solve(). Verify via Vieta's formulas and root substitution.",
    "inequality_solving":       "Use solve_univariate_inequality(). Check boundary and interior points.",
    "range_finding":            "Use function_range() from SymPy calculus. Validate at endpoints and critical points.",
    "probability":              "Count favorable/total outcomes using combinations. Verify 0 <= P <= 1.",
    "polynomial_factorization": "Apply sympy.factor(). Verify by expansion.",
    "simplification":           "Apply sympy.simplify() or expand(). Verify by numeric evaluation.",
    "matrix_operations":        "Build SymPy Matrix. Apply det/inv/eigenvals. Verify A*A^-1 = I.",
    "sequences_series":         "Identify a, d/r, n. Apply AP/GP formula. Verify nth term and partial sum.",
    "binomial":                 "T_{r+1} = C(n,r)*a^(n-r)*b^r. Expand and extract coefficient.",
    "combinatorics":            "Apply nCr = n!/(r!(n-r)!) or nPr = n!/(n-r)!",
    "derivative_problem":       "Apply sympy.diff(). Evaluate at point if required.",
    "limit_problem":            "Apply sympy.limit(). Apply L'Hopital's rule if indeterminate.",
    "optimization_problem":     "Find f'(x). Solve f'(x) = 0 for critical points. Check endpoints and f''(x).",
}


def classify_and_route(problem_text: str, parsed: dict) -> dict:
    """
    Classify a parsed problem and return a full routing decision.

    Args:
        problem_text : Cleaned problem text from Parser Agent.
        parsed       : Full parser output dict.

    Returns:
        {problem_type, tool_chain, rag_query, solver_strategy,
         fallback_strategy, topic, router_confidence, all_scores}
    """
    text_lower = problem_text.lower()
    scores = {}
    for ptype, config in _ROUTING_TABLE.items():
        scores[ptype] = sum(1 for t in config["triggers"] if re.search(t, text_lower))

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    parser_topic = parsed.get("topic", "")
    parser_type  = parsed.get("problem_type", "")

    if parser_type in _ROUTING_TABLE:
        final_type = parser_type
        confidence = 0.9 if scores.get(parser_type, 0) > 0 else 0.75
    elif best_score > 0:
        final_type = best_type
        confidence = min(0.5 + best_score * 0.15, 0.97)
    else:
        final_type = _TOPIC_TO_TYPE.get(parser_topic, "equation_solving")
        confidence = 0.55

    cfg = _ROUTING_TABLE[final_type]
    result = {
        "problem_type":      final_type,
        "tool_chain":        cfg["tool_chain"],
        "rag_query":         cfg["rag_hints"],
        "solver_strategy":   _STRATEGIES.get(final_type, _STRATEGIES["equation_solving"]),
        "fallback_strategy": cfg["fallback"],
        "topic":             parser_topic,
        "router_confidence": round(confidence, 2),
        "all_scores":        scores,
    }
    logger.info("Router: type=%s (conf=%.2f), tools=%s", final_type, confidence, cfg["tool_chain"])
    return result


def classify_topic(problem_text: str, parser_topic: str = "other") -> dict:
    """Backward-compatible wrapper returning topic/confidence/strategy."""
    r = classify_and_route(problem_text, {"topic": parser_topic, "problem_type": ""})
    return {
        "topic":        r["topic"],
        "confidence":   r["router_confidence"],
        "strategy":     r["solver_strategy"],
        "problem_type": r["problem_type"],
        "all_scores":   r["all_scores"],
    }
