"""
Python Calculator Tool.
Provides safe mathematical expression evaluation and numerical checking.
"""

import math
import ast
import operator
from utils.logger import get_logger

logger = get_logger("tools.calculator")

# Safe operators for AST-based evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions accessible in expressions
_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "factorial": math.factorial,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node):
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Tuple):
        # Handle comma-separated expressions like "(-1)**2, 2**2"
        return [_safe_eval_node(elt) for elt in node.elts]
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_func(operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            func = _SAFE_FUNCTIONS[node.func.id]
            args = [_safe_eval_node(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.

    Supports: +, -, *, /, //, %, **, sqrt, sin, cos, tan, log, exp, pi, e, etc.
    Also supports comma-separated expressions like "1**2, 2**2" → [1, 4].

    Args:
        expression: Math expression as a string (e.g., "sqrt(2) + 3**2").

    Returns:
        dict with keys: result (float|int|list), expression (str), success (bool), error (str|None).
    """
    try:
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree)
        logger.info("calculate('%s') = %s", expression, result)
        return {
            "result": result,
            "expression": expression,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.error("calculate('%s') failed: %s", expression, e)
        return {
            "result": None,
            "expression": expression,
            "success": False,
            "error": str(e),
        }


def numerical_check(
    expected: float, actual: float, tolerance: float = 1e-6
) -> dict:
    """
    Check if two numerical values match within a tolerance.

    Args:
        expected: The expected value.
        actual: The computed value.
        tolerance: Acceptable absolute difference.

    Returns:
        dict with keys: match (bool), difference (float), within_tolerance (bool).
    """
    diff = abs(expected - actual)
    match = diff <= tolerance
    logger.info(
        "numerical_check(expected=%.6f, actual=%.6f, tol=%.6f) → %s",
        expected,
        actual,
        tolerance,
        "PASS" if match else "FAIL",
    )
    return {
        "match": match,
        "expected": expected,
        "actual": actual,
        "difference": round(diff, 10),
        "within_tolerance": match,
    }
