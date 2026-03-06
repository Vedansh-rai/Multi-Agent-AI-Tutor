"""
Math Tool Executor — JEE Algebra Solver
========================================
Production-grade deterministic math computation via SymPy.
All results are computed symbolically or numerically — NEVER via LLM estimation.

Tools exposed:
  - solve_equation          : symbolic equation solving
  - solve_inequality        : inequality → interval set
  - find_range              : range of f(x) over a domain
  - simplify_expression     : algebraic simplification
  - factor_polynomial       : polynomial factoring over rationals
  - expand_expression       : polynomial expansion
  - compute_determinant     : n×n matrix determinant
  - compute_matrix_inverse  : matrix inverse
  - compute_eigenvalues     : eigenvalues of a matrix
  - solve_linear_system     : system of linear equations
  - compute_probability     : classical probability
  - compute_combination     : nCr
  - compute_permutation     : nPr
  - compute_binomial_term   : rth term in (ax+by)^n expansion
  - compute_ap_sum          : sum of n terms of AP
  - compute_gp_sum          : sum of n terms / infinite GP
  - verify_solution         : substitute and check ≈ 0
  - verify_vieta            : Vieta's formulas consistency
  - evaluate_expression     : numeric evaluation at a point
  - compute_hcf_lcm         : HCF and LCM of integers
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import sympy
from sympy import (
    Symbol, symbols, Rational, oo, I,
    solve, simplify, factor, expand, latex,
    Eq, Abs, sqrt, sympify, N,
    Matrix, det, eye,
    binomial, factorial,
    re, im, arg, Mod,
    Interval, Union, FiniteSet, EmptySet,
    S, pi, E,
    gcd, lcm,
    diff, limit
)
from sympy.calculus.util import function_range, continuous_domain
from sympy.solvers.inequalities import solve_univariate_inequality, reduce_rational_inequalities
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from utils.logger import get_logger

logger = get_logger("tools.math_tools")

# ── Parsing helpers ──────────────────────────────────────────────────────────

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_NOTATION_MAP = {
    "∞": "oo",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "√": "sqrt",
    "×": "*",
    "÷": "/",
    "²": "**2",
    "³": "**3",
    "ln": "log",   # sympy log = natural log by default
}


def _normalise(expr_str: str) -> str:
    """Replace unicode math notation with SymPy-safe ASCII equivalents."""
    for uni, ascii_ in _NOTATION_MAP.items():
        expr_str = expr_str.replace(uni, ascii_)
    return expr_str


def _parse(expr_str: str, local_dict: dict | None = None) -> sympy.Expr:
    """Flexible string → SymPy expression parser with fallback."""
    expr_str = _normalise(expr_str.strip())
    try:
        return parse_expr(expr_str, local_dict=local_dict, transformations=_TRANSFORMS)
    except Exception:
        return sympify(expr_str, locals=local_dict)


def _tool_error(kind: str, message: str, fallback: str = "") -> dict:
    """Standardised error response matching the spec."""
    return {"success": False, "error": kind, "message": message, "fallback": fallback}


# ── 1. Solve Equation ────────────────────────────────────────────────────────

def solve_equation(expression: str, variable: str = "x") -> dict:
    """
    Solve algebraic equation = 0 symbolically using SymPy.

    Accepts:
        - "x**2 - 5*x + 6"         → solves LHS = 0
        - "x**2 - 5*x + 6 = 0"     → same
        - "2*x + 3 = 7"             → solves 2x+3=7

    Returns:
        {success, solutions: [str], solutions_latex: [str],
         solutions_numeric: [float], variable}
    """
    try:
        var = Symbol(variable, real=True)
        expr_str = _normalise(expression)

        if "=" in expr_str:
            lhs_s, rhs_s = expr_str.split("=", 1)
            lhs = _parse(lhs_s, {variable: var})
            rhs = _parse(rhs_s, {variable: var})
            eq = Eq(lhs, rhs)
            sols = solve(eq, var)
        else:
            expr = _parse(expr_str, {variable: var})
            sols = solve(expr, var)

        # Also attempt complex solutions if no real solutions found
        if not sols:
            var_c = Symbol(variable)
            expr_str_c = _normalise(expression)
            if "=" in expr_str_c:
                lhs_s, rhs_s = expr_str_c.split("=", 1)
                lhs = _parse(lhs_s, {variable: var_c})
                rhs = _parse(rhs_s, {variable: var_c})
                sols = solve(Eq(lhs, rhs), var_c)
            else:
                sols = solve(_parse(expr_str_c, {variable: var_c}), var_c)

        solutions_str = [str(s) for s in sols]
        solutions_latex_ = [latex(s) for s in sols]
        solutions_num = []
        for s in sols:
            try:
                solutions_num.append(float(N(s)))
            except Exception:
                solutions_num.append(None)

        logger.info("solve_equation('%s', %s) → %s", expression, variable, solutions_str)
        return {
            "success": True,
            "solutions": solutions_str,
            "solutions_latex": solutions_latex_,
            "solutions_numeric": solutions_num,
            "num_solutions": len(sols),
            "variable": variable,
            "error": None,
        }
    except Exception as e:
        logger.error("solve_equation failed: %s", e)
        return _tool_error(
            "solve_error", str(e),
            "Try numerical solving via nsolve or check expression syntax"
        )


# ── 2. Solve Inequality ──────────────────────────────────────────────────────

def solve_inequality(expression: str, variable: str = "x") -> dict:
    """
    Solve a polynomial/rational inequality and return the solution set.

    Example: "x**2 - 4*x + 3 < 0" → "(1, 3)"

    Returns:
        {success, solution_set: str, solution_set_latex: str, interval: str}
    """
    try:
        var = Symbol(variable, real=True)
        expr_str = _normalise(expression)

        # Parse the inequality into (expr, rel) form
        for op in ["<=", ">=", "<", ">"]:
            if op in expr_str:
                lhs_s, rhs_s = expr_str.split(op, 1)
                lhs = _parse(lhs_s, {variable: var})
                rhs = _parse(rhs_s, {variable: var})
                diff_expr = lhs - rhs

                rel_map = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge"}
                rel = rel_map[op]
                from sympy import Lt, Gt, Le, Ge, And
                rel_fn = {"lt": Lt, "gt": Gt, "le": Le, "ge": Ge}[rel]

                result = solve_univariate_inequality(
                    rel_fn(diff_expr, 0), var, relational=False
                )
                result_str = str(result)
                result_latex = latex(result)

                logger.info("solve_inequality('%s') → %s", expression, result_str)
                return {
                    "success": True,
                    "solution_set": result_str,
                    "solution_set_latex": result_latex,
                    "variable": variable,
                    "error": None,
                }

        return _tool_error("parse_error", "No inequality operator found in expression")
    except Exception as e:
        logger.error("solve_inequality failed: %s", e)
        return _tool_error("solve_error", str(e), "Check inequality syntax")


# ── 3. Find Range ────────────────────────────────────────────────────────────

def _parse_domain(domain_str: str, var: sympy.Symbol) -> sympy.Set:
    """
    Parse a domain string into a SymPy Set.

    Handles:
        "(-oo, oo)"  → S.Reals
        "(-1, oo)"   → Interval.open(-1, oo)
        "[0, oo)"    → Interval.Lopen(0, oo)
        "x > -1"     → SymPy inequalities
    """
    if not domain_str or domain_str in ("(-oo, oo)", "S.Reals"):
        return S.Reals

    # Try inequality form: "x > -1", "x >= 0", etc.
    import re as _re
    ineq_pat = _re.match(r"\s*[a-zA-Z]\s*([><=!]+)\s*(.+)", domain_str)
    if ineq_pat:
        op, rhs_str = ineq_pat.group(1), ineq_pat.group(2).strip()
        rhs_val = sympify(rhs_str)
        op_map = {
            ">":  S.Reals & sympy.Interval.open(rhs_val, sympy.oo),
            ">=": S.Reals & sympy.Interval(rhs_val, sympy.oo),
            "<":  S.Reals & sympy.Interval.open(-sympy.oo, rhs_val),
            "<=": S.Reals & sympy.Interval(-sympy.oo, rhs_val),
        }
        return op_map.get(op, S.Reals)

    # Try interval notation: "(a, b)", "[a, b)", etc.
    inter_pat = _re.match(r"\s*([\[\(])\s*(.+?)\s*,\s*(.+?)\s*([\]\)])\s*", domain_str)
    if inter_pat:
        left_bracket, lo_str, hi_str, right_bracket = inter_pat.groups()
        lo = sympify(lo_str.replace("-oo", "-oo").replace("oo", "oo"))
        hi = sympify(hi_str.replace("-oo", "-oo").replace("oo", "oo"))
        left_open  = (left_bracket == "(")
        right_open = (right_bracket == ")")
        return sympy.Interval(lo, hi, left_open, right_open)

    return S.Reals


def find_range(expression: str, variable: str = "x",
               domain_str: str = "(-oo, oo)") -> dict:
    """
    Find the range of f(x) over the given domain using SymPy calculus utilities.

    Args:
        expression : e.g. "x**2"
        variable   : default "x"
        domain_str : interval string "(-1, oo)" or inequality "x > -1"

    Returns:
        {success, range: str, range_latex: str}
    """
    try:
        var = Symbol(variable, real=True)
        expr = _parse(expression, {variable: var})

        domain = _parse_domain(domain_str, var)

        rng = function_range(expr, var, domain)
        rng_str = str(rng)
        rng_latex = latex(rng)

        logger.info("find_range('%s', domain=%s) → %s", expression, domain_str, rng_str)
        return {
            "success": True,
            "range": rng_str,
            "range_latex": rng_latex,
            "domain": domain_str,
            "error": None,
        }
    except Exception as e:
        logger.error("find_range failed: %s", e)
        return _tool_error("range_error", str(e), "Try manual critical point analysis")


# ── 4. Simplify Expression ───────────────────────────────────────────────────

def simplify_expression(expression: str) -> dict:
    """
    Algebraically simplify an expression.

    Returns:
        {success, result: str, result_latex: str}
    """
    try:
        expr = _parse(expression)
        simplified = simplify(expr)
        logger.info("simplify('%s') → %s", expression, simplified)
        return {
            "success": True,
            "result": str(simplified),
            "result_latex": latex(simplified),
            "original": expression,
            "error": None,
        }
    except Exception as e:
        logger.error("simplify_expression failed: %s", e)
        return _tool_error("simplify_error", str(e))


# ── 5. Factor Polynomial ─────────────────────────────────────────────────────

def factor_polynomial(expression: str) -> dict:
    """
    Factor a polynomial over the rationals.

    Example: "x**3 - 6*x**2 + 11*x - 6" → "(x - 1)*(x - 2)*(x - 3)"

    Returns:
        {success, factored: str, factored_latex: str}
    """
    try:
        expr = _parse(expression)
        factored = factor(expr)
        logger.info("factor('%s') → %s", expression, factored)
        return {
            "success": True,
            "factored": str(factored),
            "factored_latex": latex(factored),
            "original": expression,
            "error": None,
        }
    except Exception as e:
        logger.error("factor_polynomial failed: %s", e)
        return _tool_error("factor_error", str(e))


# ── 6. Expand Expression ─────────────────────────────────────────────────────

def expand_expression(expression: str) -> dict:
    """
    Expand a polynomial expression.

    Example: "(x+1)**4" → "x**4 + 4*x**3 + 6*x**2 + 4*x + 1"

    Returns:
        {success, expanded: str, expanded_latex: str, terms: [str]}
    """
    try:
        expr = _parse(expression)
        expanded = expand(expr)

        # Extract terms as list
        from sympy import Add
        if isinstance(expanded, Add):
            terms = [str(t) for t in expanded.as_ordered_terms()]
        else:
            terms = [str(expanded)]

        logger.info("expand('%s') → %s", expression, expanded)
        return {
            "success": True,
            "expanded": str(expanded),
            "expanded_latex": latex(expanded),
            "terms": terms,
            "original": expression,
            "error": None,
        }
    except Exception as e:
        logger.error("expand_expression failed: %s", e)
        return _tool_error("expand_error", str(e))


# ── 7. Compute Determinant ───────────────────────────────────────────────────

def compute_determinant(matrix: list[list]) -> dict:
    """
    Compute the determinant of an n×n matrix.

    Args:
        matrix : 2D list, e.g. [[2,3],[1,4]]

    Returns:
        {success, determinant: str, determinant_numeric: float}
    """
    try:
        M = Matrix(matrix)
        if M.rows != M.cols:
            return _tool_error("dimension_error", "Matrix must be square for determinant")
        d = M.det()
        logger.info("det(%s) → %s", matrix, d)
        return {
            "success": True,
            "determinant": str(d),
            "determinant_numeric": float(N(d)),
            "determinant_latex": latex(d),
            "matrix_size": f"{M.rows}×{M.cols}",
            "error": None,
        }
    except Exception as e:
        logger.error("compute_determinant failed: %s", e)
        return _tool_error("matrix_error", str(e))


# ── 8. Compute Matrix Inverse ────────────────────────────────────────────────

def compute_matrix_inverse(matrix: list[list]) -> dict:
    """
    Return the inverse of a square matrix if it exists.

    Returns:
        {success, inverse: [[...]], inverse_latex: str}
    """
    try:
        M = Matrix(matrix)
        if M.rows != M.cols:
            return _tool_error("dimension_error", "Matrix must be square for inverse")
        d = M.det()
        if d == 0:
            return _tool_error(
                "singular_matrix",
                "Matrix is singular (det=0), inverse does not exist",
                "Use pseudoinverse or check the problem statement"
            )
        inv = M.inv()
        # Verify A·A⁻¹ = I
        product = M * inv
        is_identity = product == eye(M.rows)
        logger.info("inverse computed, A·A⁻¹=I: %s", is_identity)
        return {
            "success": True,
            "inverse": inv.tolist(),
            "inverse_latex": latex(inv),
            "verification_A_Ainv_eq_I": is_identity,
            "error": None,
        }
    except Exception as e:
        logger.error("compute_matrix_inverse failed: %s", e)
        return _tool_error("matrix_error", str(e))


# ── 9. Compute Eigenvalues ───────────────────────────────────────────────────

def compute_eigenvalues(matrix: list[list]) -> dict:
    """
    Compute the eigenvalues of a square matrix.

    Returns:
        {success, eigenvalues: {value: multiplicity}, characteristic_poly: str}
    """
    try:
        M = Matrix(matrix)
        evals = M.eigenvals()
        evals_str = {str(k): v for k, v in evals.items()}
        char_poly = M.charpoly()
        logger.info("eigenvalues(%s) → %s", matrix, evals_str)
        return {
            "success": True,
            "eigenvalues": evals_str,
            "eigenvalues_list": [str(k) for k in evals.keys()],
            "characteristic_polynomial": str(char_poly),
            "error": None,
        }
    except Exception as e:
        logger.error("compute_eigenvalues failed: %s", e)
        return _tool_error("matrix_error", str(e))


# ── 10. Solve Linear System ──────────────────────────────────────────────────

def solve_linear_system(equations: list[str], variables: list[str]) -> dict:
    """
    Solve a system of linear equations.

    Args:
        equations : list of equation strings, e.g. ["2*x + y = 5", "x - y = 1"]
        variables : list of variable names, e.g. ["x", "y"]

    Returns:
        {success, solution: {var: value}, solution_latex: str}
    """
    try:
        syms = [Symbol(v) for v in variables]
        sym_map = dict(zip(variables, syms))
        eq_list = []
        for eq_str in equations:
            eq_str = _normalise(eq_str)
            if "=" in eq_str:
                lhs_s, rhs_s = eq_str.split("=", 1)
                lhs = _parse(lhs_s, sym_map)
                rhs = _parse(rhs_s, sym_map)
                eq_list.append(Eq(lhs, rhs))
            else:
                eq_list.append(Eq(_parse(eq_str, sym_map), 0))

        sol = solve(eq_list, syms)
        if isinstance(sol, dict):
            sol_str = {str(k): str(v) for k, v in sol.items()}
        elif isinstance(sol, list) and sol:
            if isinstance(sol[0], tuple):
                sol_str = {str(syms[i]): str(sol[0][i]) for i in range(len(syms))}
            else:
                sol_str = {str(syms[0]): str(sol[0])}
        else:
            sol_str = {}

        logger.info("solve_linear_system → %s", sol_str)
        return {
            "success": True,
            "solution": sol_str,
            "num_solutions": len(sol_str),
            "error": None,
        }
    except Exception as e:
        logger.error("solve_linear_system failed: %s", e)
        return _tool_error("system_error", str(e))


# ── 11. Compute Probability ──────────────────────────────────────────────────

def compute_probability(favorable: int, total: int) -> dict:
    """
    Classical probability: P = favorable / total.

    Returns:
        {success, probability: str (fraction), probability_decimal: float}
    """
    try:
        favorable_val = int(favorable)
        total_val = int(total)
        if total_val <= 0:
            return _tool_error("domain_error", "Total outcomes must be positive")
        if favorable_val < 0 or favorable_val > total_val:
            return _tool_error(
                "domain_error",
                f"Favorable ({favorable_val}) must be between 0 and total ({total_val})"
            )
        prob = Rational(favorable_val, total_val)
        prob_float = float(prob)
        logger.info("P(%d/%d) = %s", favorable_val, total_val, prob)
        return {
            "success": True,
            "probability": str(prob),
            "probability_latex": latex(prob),
            "probability_decimal": prob_float,
            "favorable": favorable_val,
            "total": total_val,
            "sanity_check": 0 <= prob_float <= 1,
            "error": None,
        }
    except Exception as e:
        logger.error("compute_probability failed: %s", e)
        return _tool_error("probability_error", str(e))


# ── 12. Compute Combination ──────────────────────────────────────────────────

def compute_combination(n: int, r: int) -> dict:
    """
    Compute nCr (combinations).

    Returns:
        {success, result: int}
    """
    try:
        n_val = int(n)
        r_val = int(r)
        if r_val < 0 or r_val > n_val:
            return _tool_error(
                "domain_error", f"r={r_val} must satisfy 0 ≤ r ≤ n={n_val}"
            )
        result = int(binomial(n_val, r_val))
        logger.info("C(%d,%d) = %d", n_val, r_val, result)
        return {
            "success": True,
            "result": result,
            "formula": f"C({n_val},{r_val}) = {n_val}! / ({r_val}! × {n_val-r_val}!)",
            "error": None,
        }
    except Exception as e:
        logger.error("compute_combination failed: %s", e)
        return _tool_error("combinatorics_error", str(e))


# ── 13. Compute Permutation ──────────────────────────────────────────────────

def compute_permutation(n: int, r: int) -> dict:
    """
    Compute nPr (permutations).

    Returns:
        {success, result: int}
    """
    try:
        n_val = int(n)
        r_val = int(r)
        if r_val < 0 or r_val > n_val:
            return _tool_error(
                "domain_error", f"r={r_val} must satisfy 0 ≤ r ≤ n={n_val}"
            )
        result = int(factorial(n_val) / factorial(n_val - r_val))
        logger.info("P(%d,%d) = %d", n_val, r_val, result)
        return {
            "success": True,
            "result": result,
            "formula": f"P({n_val},{r_val}) = {n_val}! / {n_val-r_val}!",
            "error": None,
        }
    except Exception as e:
        logger.error("compute_permutation failed: %s", e)
        return _tool_error("combinatorics_error", str(e))


# ── 14. Compute Binomial Term ────────────────────────────────────────────────

def compute_binomial_term(n: int, r: int,
                          x_coeff: float = 1.0, y_coeff: float = 1.0,
                          expression: str | None = None) -> dict:
    """
    Return the (r+1)-th (0-indexed r) term in the binomial expansion of (ax+by)^n,
    i.e. T_{r+1} = C(n,r) * (ax)^(n-r) * (by)^r.

    If `expression` is provided (e.g. "(x+1)**4"), it expands that directly.

    Returns:
        {success, term: str, term_latex: str, coefficient: int/float}
    """
    try:
        if expression:
            expr = _parse(expression)
            expanded = expand(expr)
            # Extract coefficient of x^(n-r)
            x = Symbol("x")
            power = n - r
            coeff_val = expanded.coeff(x, power)
            term_str = f"C({n},{r}) * x^{power} ... see expanded form"
            return {
                "success": True,
                "term": str(coeff_val) + f"*x**{power}",
                "term_latex": latex(coeff_val) + f"x^{{{power}}}",
                "coefficient": int(coeff_val) if coeff_val.is_integer else float(coeff_val),
                "expanded": str(expanded),
                "expanded_latex": latex(expanded),
                "error": None,
            }

        x, y = symbols("x y")
        c = int(binomial(n, r))
        ax = sympy.Rational(x_coeff) * x if x_coeff != 1 else x
        by = sympy.Rational(y_coeff) * y if y_coeff != 1 else y
        term = c * ax**(n - r) * by**r
        term_simplified = simplify(term)

        logger.info("binomial_term(n=%d,r=%d) → %s", n, r, term_simplified)
        return {
            "success": True,
            "term": str(term_simplified),
            "term_latex": latex(term_simplified),
            "binomial_coeff": c,
            "r_value": r,
            "formula": f"T_{r+1} = C({n},{r}) * ({x_coeff}x)^{n-r} * ({y_coeff}y)^{r}",
            "error": None,
        }
    except Exception as e:
        logger.error("compute_binomial_term failed: %s", e)
        return _tool_error("binomial_error", str(e))


# ── 15. Compute AP Sum ───────────────────────────────────────────────────────

def compute_ap_sum(first_term: float, common_diff: float, n: int) -> dict:
    """
    Sum of first n terms of an Arithmetic Progression.

    Formula: S_n = n/2 * (2a + (n-1)d)

    Returns:
        {success, sum: float, nth_term: float}
    """
    try:
        a = Rational(first_term).limit_denominator(1000)
        d = Rational(common_diff).limit_denominator(1000)
        n_sym = sympy.Integer(n)
        s_n = n_sym * (2 * a + (n_sym - 1) * d) / 2
        t_n = a + (n_sym - 1) * d
        logger.info("AP sum(a=%s,d=%s,n=%d) = %s", a, d, n, s_n)
        return {
            "success": True,
            "sum": float(s_n),
            "sum_exact": str(s_n),
            "nth_term": float(t_n),
            "nth_term_exact": str(t_n),
            "formula": f"S_{n} = {n}/2 × (2×{a} + ({n}-1)×{d})",
            "error": None,
        }
    except Exception as e:
        logger.error("compute_ap_sum failed: %s", e)
        return _tool_error("sequence_error", str(e))


# ── 16. Compute GP Sum ───────────────────────────────────────────────────────

def compute_gp_sum(first_term: float, common_ratio: float,
                   n: int | None = None, infinite: bool = False) -> dict:
    """
    Sum of GP terms.

    Finite:   S_n = a(r^n - 1)/(r - 1)  if r ≠ 1
    Infinite: S_∞ = a/(1-r)             if |r| < 1

    Returns:
        {success, sum: float | str}
    """
    try:
        a = Rational(first_term).limit_denominator(1000)
        r = Rational(common_ratio).limit_denominator(1000)

        if infinite:
            if abs(float(r)) >= 1:
                return _tool_error(
                    "domain_error",
                    f"|r|={abs(float(r))} ≥ 1; infinite GP sum diverges",
                    "Check |r| < 1 condition for convergence"
                )
            s_inf = a / (1 - r)
            logger.info("GP infinite sum(a=%s,r=%s) = %s", a, r, s_inf)
            return {
                "success": True,
                "sum": float(s_inf),
                "sum_exact": str(s_inf),
                "sum_latex": latex(s_inf),
                "type": "infinite",
                "converges": True,
                "formula": f"S_∞ = {a}/(1-{r})",
                "error": None,
            }

        if n is None:
            return _tool_error("missing_param", "Provide n for finite GP sum")

        n_sym = sympy.Integer(n)
        if r == 1:
            s_n = a * n_sym
        else:
            s_n = a * (r**n_sym - 1) / (r - 1)
        t_n = a * r**(n_sym - 1)

        logger.info("GP finite sum(a=%s,r=%s,n=%d) = %s", a, r, n, s_n)
        return {
            "success": True,
            "sum": float(s_n),
            "sum_exact": str(s_n),
            "nth_term": float(t_n),
            "type": "finite",
            "formula": f"S_{n} = {a}×({r}^{n}-1)/({r}-1)",
            "error": None,
        }
    except Exception as e:
        logger.error("compute_gp_sum failed: %s", e)
        return _tool_error("sequence_error", str(e))


# ── 17. Verify Solution ──────────────────────────────────────────────────────

def verify_solution(expression: str, variable: str, value: Any) -> dict:
    """
    Substitute value into expression and check if result ≈ 0.

    Returns:
        {success, is_valid: bool, residual: float, residual_exact: str}
    """
    try:
        var = Symbol(variable)
        expr = _parse(expression, {variable: var})

        if "=" in expression:
            lhs_s, rhs_s = expression.split("=", 1)
            lhs = _parse(lhs_s, {variable: var})
            rhs = _parse(rhs_s, {variable: var})
            expr = lhs - rhs

        val = sympify(str(value))
        residual = expr.subs(var, val)
        residual_simplified = simplify(residual)
        residual_float = float(N(residual_simplified))
        is_valid = abs(residual_float) < 1e-9

        logger.info(
            "verify_solution('%s', %s=%s) → residual=%s, valid=%s",
            expression, variable, value, residual_float, is_valid
        )
        return {
            "success": True,
            "is_valid": is_valid,
            "residual": residual_float,
            "residual_exact": str(residual_simplified),
            "variable": variable,
            "value_tested": str(value),
            "error": None,
        }
    except Exception as e:
        logger.error("verify_solution failed: %s", e)
        return _tool_error("verify_error", str(e))


# ── 18. Verify Vieta's Formulas ──────────────────────────────────────────────

def verify_vieta(roots: list, coeff_a: float = 1.0,
                 coeff_b: float = None, coeff_c: float = None,
                 n: int = 2) -> dict:
    """
    Check Vieta's formulas for a polynomial with given roots.

    For quadratic ax²+bx+c:
        sum_of_roots   = −b/a
        product_of_roots = c/a

    Returns:
        {success, sum_check: bool, product_check: bool, details: dict}
    """
    try:
        root_syms = [sympify(str(r)) for r in roots]
        actual_sum = sum(root_syms)
        actual_product = sympy.prod(root_syms)

        result = {
            "success": True,
            "actual_sum": str(actual_sum),
            "actual_product": str(actual_product),
            "checks": {},
            "all_pass": True,
            "error": None,
        }

        if n == 2 and coeff_b is not None and coeff_c is not None:
            a = Rational(coeff_a).limit_denominator(1000)
            b = Rational(coeff_b).limit_denominator(1000)
            c = Rational(coeff_c).limit_denominator(1000)

            expected_sum = -b / a
            expected_product = c / a

            sum_residual = float(N(actual_sum - expected_sum))
            prod_residual = float(N(actual_product - expected_product))

            sum_ok = abs(sum_residual) < 1e-9
            prod_ok = abs(prod_residual) < 1e-9

            result["checks"]["sum"] = {
                "expected": str(expected_sum),
                "actual": str(actual_sum),
                "pass": sum_ok,
            }
            result["checks"]["product"] = {
                "expected": str(expected_product),
                "actual": str(actual_product),
                "pass": prod_ok,
            }
            result["all_pass"] = sum_ok and prod_ok

        logger.info("verify_vieta(roots=%s) → all_pass=%s", roots, result["all_pass"])
        return result
    except Exception as e:
        logger.error("verify_vieta failed: %s", e)
        return _tool_error("vieta_error", str(e))


# ── 19. Evaluate Expression ──────────────────────────────────────────────────

def evaluate_expression(expression: str, substitutions: dict) -> dict:
    """
    Numerically evaluate an expression at given point(s).

    Args:
        expression    : e.g. "x**2 + 2*x + 1"
        substitutions : e.g. {"x": 3}

    Returns:
        {success, result: float, result_exact: str}
    """
    try:
        sym_map = {k: Symbol(k) for k in substitutions}
        expr = _parse(expression, sym_map)
        for var_name, val in substitutions.items():
            expr = expr.subs(Symbol(var_name), sympify(str(val)))
        result_exact = simplify(expr)
        result_float = float(N(result_exact))
        logger.info("evaluate('%s', %s) → %s", expression, substitutions, result_float)
        return {
            "success": True,
            "result": result_float,
            "result_exact": str(result_exact),
            "result_latex": latex(result_exact),
            "error": None,
        }
    except Exception as e:
        logger.error("evaluate_expression failed: %s", e)
        return _tool_error("eval_error", str(e))


# ── 20. Compute HCF / LCM ───────────────────────────────────────────────────

def compute_hcf_lcm(numbers: list[int]) -> dict:
    """
    Compute HCF (GCD) and LCM of a list of integers.

    Returns:
        {success, hcf: int, lcm: int}
    """
    try:
        if len(numbers) < 2:
            return _tool_error("domain_error", "Provide at least 2 integers")
        hcf_val = numbers[0]
        lcm_val = numbers[0]
        for n in numbers[1:]:
            hcf_val = math.gcd(hcf_val, n)
            lcm_val = lcm_val * n // math.gcd(lcm_val, n)
        logger.info("HCF/LCM(%s) → HCF=%d, LCM=%d", numbers, hcf_val, lcm_val)
        return {
            "success": True,
            "hcf": hcf_val,
            "lcm": lcm_val,
            "numbers": numbers,
            "error": None,
        }
    except Exception as e:
        logger.error("compute_hcf_lcm failed: %s", e)
        return _tool_error("arithmetic_error", str(e))


# ── Calculus Tools ───────────────────────────────────────────────────────────

def differentiate(expression: str, variable: str = "x") -> dict:
    """
    Compute the derivative of an expression with respect to `variable`.
    """
    try:
        var = Symbol(variable, real=True)
        expr_str = _normalise(expression)
        expr = _parse(expr_str, {variable: var})
        
        derivative = diff(expr, var)
        
        logger.info("differentiate('%s', %s) → %s", expression, variable, derivative)
        return {
            "success": True,
            "derivative": str(derivative),
            "derivative_latex": latex(derivative),
            "variable": variable,
            "error": None,
        }
    except Exception as e:
        logger.error("differentiate failed: %s", e)
        return _tool_error("calculus_error", str(e))

def compute_limit(expression: str, variable: str = "x", point: str = "0", direction: str = "+-") -> dict:
    """
    Compute the limit of an expression as `variable` approaches `point`.
    """
    try:
        var = Symbol(variable, real=True)
        expr_str = _normalise(expression)
        pt_str = _normalise(point)
        
        expr = _parse(expr_str, {variable: var})
        pt = _parse(pt_str)
        
        dir_sym = direction if direction in ("+", "-") else "+-"
        
        res = limit(expr, var, pt, dir=dir_sym)
        
        logger.info("compute_limit('%s', %s, %s, %s) → %s", expression, variable, point, direction, res)
        return {
            "success": True,
            "limit": str(res),
            "limit_latex": latex(res),
            "variable": variable,
            "point": str(pt),
            "error": None,
        }
    except Exception as e:
        logger.error("compute_limit failed: %s", e)
        return _tool_error("calculus_error", str(e))

# ── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "solve_equation": solve_equation,
    "solve_inequality": solve_inequality,
    "find_range": find_range,
    "simplify_expression": simplify_expression,
    "factor_polynomial": factor_polynomial,
    "expand_expression": expand_expression,
    "compute_determinant": compute_determinant,
    "compute_matrix_inverse": compute_matrix_inverse,
    "compute_eigenvalues": compute_eigenvalues,
    "solve_linear_system": solve_linear_system,
    "compute_probability": compute_probability,
    "compute_combination": compute_combination,
    "compute_permutation": compute_permutation,
    "compute_binomial_term": compute_binomial_term,
    "compute_ap_sum": compute_ap_sum,
    "compute_gp_sum": compute_gp_sum,
    "verify_solution": verify_solution,
    "verify_vieta": verify_vieta,
    "evaluate_expression": evaluate_expression,
    "compute_hcf_lcm": compute_hcf_lcm,
    "differentiate": differentiate,
    "compute_limit": compute_limit,
}


def dispatch_tool(tool_name: str, **kwargs) -> dict:
    """
    Dispatch a tool call by name. Returns a standardised result dict.

    Args:
        tool_name : one of the keys in TOOL_REGISTRY
        **kwargs  : tool-specific arguments

    Returns:
        Tool result dict (always has 'success' key).
    """
    if tool_name not in TOOL_REGISTRY:
        return _tool_error(
            "unknown_tool",
            f"Tool '{tool_name}' not found. Available: {list(TOOL_REGISTRY.keys())}"
        )
    try:
        return TOOL_REGISTRY[tool_name](**kwargs)
    except TypeError as e:
        return _tool_error("arg_error", f"Wrong arguments for '{tool_name}': {e}")
