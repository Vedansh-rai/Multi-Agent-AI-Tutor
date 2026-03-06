"""
SymPy Solver Tool.
Provides symbolic math capabilities: equation solving, simplification,
differentiation, integration, and expression parsing.
"""

import sympy
from sympy import (
    symbols,
    solve,
    simplify,
    diff,
    integrate,
    sympify,
    latex,
    Eq,
    Symbol,
    oo,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from utils.logger import get_logger

logger = get_logger("tools.sympy_solver")

# Standard parsing transformations for natural math input
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _parse(expr_str: str):
    """Parse a string into a SymPy expression with flexible transformations."""
    try:
        return parse_expr(expr_str, transformations=_TRANSFORMATIONS)
    except Exception:
        # Fallback to basic sympify
        return sympify(expr_str)


def solve_equation(equation_str: str, variable: str = "x") -> dict:
    """
    Solve an equation symbolically.

    Accepts formats:
        - "x**2 - 4" → solves x² - 4 = 0
        - "x**2 - 4 = 0" → same
        - "2*x + 3 = 7" → solves 2x + 3 = 7

    Args:
        equation_str: The equation string.
        variable: Variable to solve for (default "x").

    Returns:
        dict with solutions, latex representation, and step info.
    """
    try:
        var = Symbol(variable)

        if "=" in equation_str:
            lhs_str, rhs_str = equation_str.split("=", 1)
            lhs = _parse(lhs_str.strip())
            rhs = _parse(rhs_str.strip())
            equation = Eq(lhs, rhs)
            solutions = solve(equation, var)
        else:
            expr = _parse(equation_str)
            solutions = solve(expr, var)

        solutions_str = [str(s) for s in solutions]
        solutions_latex = [latex(s) for s in solutions]

        logger.info(
            "solve_equation('%s', var=%s) → %s", equation_str, variable, solutions_str
        )

        return {
            "success": True,
            "solutions": solutions_str,
            "solutions_latex": solutions_latex,
            "num_solutions": len(solutions),
            "variable": variable,
            "error": None,
        }
    except Exception as e:
        logger.error("solve_equation failed: %s", e)
        return {
            "success": False,
            "solutions": [],
            "solutions_latex": [],
            "num_solutions": 0,
            "variable": variable,
            "error": str(e),
        }


def simplify_expression(expr_str: str) -> dict:
    """
    Simplify a mathematical expression.

    Args:
        expr_str: Expression string (e.g., "(x+1)**2 - x**2").

    Returns:
        dict with simplified result and latex form.
    """
    try:
        expr = _parse(expr_str)
        simplified = simplify(expr)
        logger.info("simplify('%s') → %s", expr_str, simplified)
        return {
            "success": True,
            "result": str(simplified),
            "result_latex": latex(simplified),
            "original": expr_str,
            "error": None,
        }
    except Exception as e:
        logger.error("simplify failed: %s", e)
        return {
            "success": False,
            "result": None,
            "result_latex": None,
            "original": expr_str,
            "error": str(e),
        }


def differentiate(expr_str: str, variable: str = "x", order: int = 1) -> dict:
    """
    Differentiate an expression.

    Args:
        expr_str: Expression to differentiate.
        variable: Variable to differentiate with respect to.
        order: Order of derivative (default 1).

    Returns:
        dict with derivative and latex.
    """
    try:
        expr = _parse(expr_str)
        var = Symbol(variable)
        result = diff(expr, var, order)
        logger.info("diff('%s', %s, %d) → %s", expr_str, variable, order, result)
        return {
            "success": True,
            "result": str(result),
            "result_latex": latex(result),
            "error": None,
        }
    except Exception as e:
        logger.error("differentiate failed: %s", e)
        return {
            "success": False,
            "result": None,
            "result_latex": None,
            "error": str(e),
        }


def integrate_expression(
    expr_str: str, variable: str = "x", lower: str = None, upper: str = None
) -> dict:
    """
    Integrate an expression (definite or indefinite).

    Args:
        expr_str: Expression to integrate.
        variable: Variable of integration.
        lower: Lower bound (None for indefinite).
        upper: Upper bound (None for indefinite).

    Returns:
        dict with integral result and latex.
    """
    try:
        expr = _parse(expr_str)
        var = Symbol(variable)

        if lower is not None and upper is not None:
            lo = _parse(lower)
            hi = _parse(upper)
            result = integrate(expr, (var, lo, hi))
            integral_type = "definite"
        else:
            result = integrate(expr, var)
            integral_type = "indefinite"

        logger.info("integrate('%s', %s) → %s", expr_str, integral_type, result)
        return {
            "success": True,
            "result": str(result),
            "result_latex": latex(result),
            "integral_type": integral_type,
            "error": None,
        }
    except Exception as e:
        logger.error("integrate failed: %s", e)
        return {
            "success": False,
            "result": None,
            "result_latex": None,
            "integral_type": None,
            "error": str(e),
        }
