"""
Golden Test Suite — JEE Algebra Solver
========================================
Validates the 8 canonical JEE test cases end-to-end using the Math Tool
Executor (SymPy). Tests are deterministic — they never call the LLM.

Run:
    pytest tests/test_golden.py -v

Each test:
  1. Calls the relevant tool(s) directly via ``dispatch_tool``.
  2. Asserts the tool succeeds (``result["success"] == True``).
  3. Checks the numerical answer matches the expected value.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.math_tools import dispatch_tool


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _roots_match(result: dict, expected: list[float], tol: float = 1e-6) -> bool:
    """Check that the tool's solutions list matches expected (order-independent)."""
    assert result["success"], f"Tool failed: {result}"
    sols = result.get("solutions") or result.get("roots") or []
    if not sols:
        return False
    try:
        import sympy as sp
        found = sorted([float(sp.sympify(str(r))) for r in sols])
        exp   = sorted([float(e) for e in expected])
        return len(found) == len(exp) and all(
            abs(f - e) < tol for f, e in zip(found, exp)
        )
    except Exception:
        return False


# ─────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────

class TestGolden:
    """Golden test cases for the JEE Algebra Solver."""

    # ── Case 1: Simple quadratic ─────────────────────────────────────
    def test_01_quadratic_roots(self):
        """x² - 5x + 6 = 0  →  roots = [2, 3]"""
        result = dispatch_tool("solve_equation", expression="x**2 - 5*x + 6", variable="x")
        assert result["success"], f"solve_equation failed: {result}"
        assert _roots_match(result, [2.0, 3.0]), (
            f"Expected roots [2, 3], got {result.get('solutions')}"
        )

    # ── Case 2: Quadratic inequality ─────────────────────────────────
    def test_02_quadratic_inequality(self):
        """x² - 4x + 3 < 0  →  1 < x < 3"""
        result = dispatch_tool("solve_inequality", expression="x**2 - 4*x + 3 < 0", variable="x")
        assert result["success"], f"solve_inequality failed: {result}"
        solution_str = str(result.get("solution_set", "")).replace(" ", "")
        assert "1" in solution_str and "3" in solution_str, (
            f"Expected interval containing 1 and 3, got: {result.get('solution_set')}"
        )

    # ── Case 3: Range of x² for x > -1 ──────────────────────────────
    def test_03_range_of_expression(self):
        """Range of x² for x > -1  →  [0, ∞)"""
        result = dispatch_tool(
            "find_range",
            expression="x**2",
            variable="x",
            domain_str="(-1, oo)",
        )
        assert result["success"], f"find_range failed: {result}"
        range_str = str(result.get("range", "")).replace(" ", "")
        # Range must start at 0 and extend to infinity
        assert "0" in range_str and ("oo" in range_str.lower() or "inf" in range_str.lower()), (
            f"Expected range [0, oo), got: {result.get('range')}"
        )

    # ── Case 4: Determinant of 2×2 matrix ────────────────────────────
    def test_04_determinant_2x2(self):
        """det([[2, 3], [1, 4]])  →  5"""
        result = dispatch_tool("compute_determinant", matrix=[[2, 3], [1, 4]])
        assert result["success"], f"compute_determinant failed: {result}"
        det_val = float(result.get("determinant_numeric", float("nan")))
        assert abs(det_val - 5.0) < 1e-9, f"Expected det=5, got {det_val}"

    # ── Case 5: Hypergeometric probability ───────────────────────────
    def test_05_probability_no_replacement(self):
        """P(2 red | 2 red, 3 blue, draw 2 without replacement) = C(2,2)*C(3,0)/C(5,2) = 1/10"""
        result = dispatch_tool(
            "compute_probability",
            favorable=1,   # C(2,2)*C(3,0) = 1
            total=10,      # C(5,2) = 10
        )
        assert result["success"], f"compute_probability failed: {result}"
        prob = float(result.get("probability_decimal", float("nan")))
        assert abs(prob - 0.1) < 1e-9, f"Expected P=0.1, got {prob}"

    # ── Case 6: AP sum ───────────────────────────────────────────────
    def test_06_ap_sum(self):
        """Sum of AP: 2, 5, 8, … n=10  →  155"""
        result = dispatch_tool(
            "compute_ap_sum",
            first_term=2,
            common_diff=3,
            n=10,
        )
        assert result["success"], f"compute_ap_sum failed: {result}"
        s = float(result.get("sum", float("nan")))
        assert abs(s - 155.0) < 1e-9, f"Expected sum=155, got {s}"

    # ── Case 7: Binomial coefficient ─────────────────────────────────
    def test_07_binomial_coefficient_x_squared(self):
        """Coefficient of x² in (x + 1)⁴  →  6
        T_{r+1} = C(4,r) x^(4-r) · 1^r; power=2 → r=2 → C(4,2)=6
        """
        result = dispatch_tool(
            "compute_binomial_term",
            n=4,
            r=2,
            expression="(x+1)**4",
        )
        assert result["success"], f"compute_binomial_term failed: {result}"
        coeff = float(result.get("coefficient", float("nan")))
        assert abs(coeff - 6.0) < 1e-9, f"Expected coeff=6, got {coeff}"

    # ── Case 8: Cubic roots ──────────────────────────────────────────
    def test_08_cubic_roots(self):
        """x³ - 6x² + 11x - 6 = 0  →  roots = [1, 2, 3]"""
        result = dispatch_tool(
            "solve_equation",
            expression="x**3 - 6*x**2 + 11*x - 6",
            variable="x",
        )
        assert result["success"], f"solve_equation failed: {result}"
        assert _roots_match(result, [1.0, 2.0, 3.0]), (
            f"Expected roots [1, 2, 3], got {result.get('solutions')}"
        )


# ─────────────────────────────────────────────
# Additional SymPy-level sanity checks
# ─────────────────────────────────────────────

class TestSymPySanity:
    """Sanity checks on the Math Tool Executor tools."""

    def test_factor_polynomial(self):
        """x² - 5x + 6 should factor to (x - 2)(x - 3)."""
        result = dispatch_tool("factor_polynomial", expression="x**2 - 5*x + 6")
        assert result["success"]
        factored = str(result.get("factored", ""))
        assert "x - 2" in factored or "(x-2)" in factored.replace(" ", ""), (
            f"Expected (x-2)(x-3) factor, got: {factored}"
        )

    def test_vieta_check(self):
        """Vieta's check for x² - 5x + 6 = 0 with roots [2, 3]."""
        result = dispatch_tool(
            "verify_vieta",
            roots=["2", "3"],
            coeff_a=1.0,
            coeff_b=-5.0,
            coeff_c=6.0,
            n=2,
        )
        assert result["success"], f"verify_vieta failed: {result}"
        assert result.get("checks", {}).get("sum", {}).get("pass") is True, (
            f"Vieta sum check failed: {result}"
        )
        assert result.get("checks", {}).get("product", {}).get("pass") is True, (
            f"Vieta product check failed: {result}"
        )

    def test_verify_root_substitution(self):
        """Verify that x=2 satisfies x² - 5x + 6 = 0."""
        result = dispatch_tool(
            "verify_solution",
            expression="x**2 - 5*x + 6",
            variable="x",
            value=2,
        )
        assert result["success"]
        assert result.get("is_valid") is True, f"Root substitution check failed: {result}"

    def test_determinant_3x3(self):
        """det([[1,2,3],[4,5,6],[7,8,10]]) = -3."""
        result = dispatch_tool(
            "compute_determinant",
            matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 10]],
        )
        assert result["success"]
        det_val = float(result.get("determinant_numeric", float("nan")))
        assert abs(det_val - (-3.0)) < 1e-9, f"Expected det=-3, got {det_val}"

    def test_gp_sum(self):
        """Sum of GP: 1, 2, 4, … n=5  →  31."""
        result = dispatch_tool(
            "compute_gp_sum",
            first_term=1,
            common_ratio=2,
            n=5,
        )
        assert result["success"]
        s = float(result.get("sum", float("nan")))
        assert abs(s - 31.0) < 1e-9, f"Expected sum=31, got {s}"

    def test_ncr(self):
        """C(10, 3) = 120."""
        result = dispatch_tool("compute_combination", n=10, r=3)
        assert result["success"]
        val = float(result.get("result", float("nan")))
        assert abs(val - 120.0) < 1e-9, f"Expected 120, got {val}"

    def test_hcf_lcm(self):
        """HCF(12,18)=6, LCM(12,18)=36."""
        result = dispatch_tool("compute_hcf_lcm", numbers=[12, 18])
        assert result["success"], f"compute_hcf_lcm failed: {result}"
        assert result.get("hcf") == 6, f"Expected HCF=6, got {result.get('hcf')}"
        assert result.get("lcm") == 36, f"Expected LCM=36, got {result.get('lcm')}"

