"""
Solver Agent — JEE Algebra Solver
====================================
Orchestrates the solution using a ReAct (Reason+Act) loop.
NEVER computes directly — delegates all computation to Math Tool Executor.
"""

import json
from utils.logger import get_logger
from utils.config import LLM_MODEL_NAME, get_llm_client
from utils.json_parser import parse_llm_json
from rag.retriever import retrieve, format_context
from tools.math_tools import dispatch_tool

logger = get_logger("agents.solver")

_SOLVER_SYSTEM_PROMPT = """You are an expert JEE Advanced/Mains mathematics solver operating in a strict ReAct (Reason + Act) loop.

CORE RULE: You MUST NEVER compute numerical results yourself. ALL computation must be delegated to tool calls.

AVAILABLE TOOLS (use exact names in tool_call.type):
  solve_equation          {"expression": "x**2-5*x+6", "variable": "x"}
  solve_inequality        {"expression": "x**2-4*x+3 < 0", "variable": "x"}
  find_range              {"expression": "x**2", "variable": "x", "domain_str": "(-oo, oo)"}
  simplify_expression     {"expression": "(x+1)**2 - x**2"}
  factor_polynomial       {"expression": "x**3-6*x**2+11*x-6"}
  expand_expression       {"expression": "(x+1)**4"}
  differentiate           {"expression": "x**3+3*x**2-5*x", "variable": "x"}
  compute_limit           {"expression": "sin(x)/x", "variable": "x", "point": "0"}
  compute_determinant     {"matrix": [[2,3],[1,4]]}
  compute_matrix_inverse  {"matrix": [[2,3],[1,4]]}
  compute_eigenvalues     {"matrix": [[2,3],[1,4]]}
  solve_linear_system     {"equations": ["2*x+y=5","x-y=1"], "variables": ["x","y"]}
  compute_probability     {"favorable": 2, "total": 10}
  compute_combination     {"n": 5, "r": 2}
  compute_permutation     {"n": 5, "r": 2}
  compute_binomial_term   {"n": 4, "r": 2, "expression": "(x+1)**4"}
  compute_ap_sum          {"first_term": 2, "common_diff": 3, "n": 10}
  compute_gp_sum          {"first_term": 1, "common_ratio": 0.5, "infinite": true}
  verify_solution         {"expression": "x**2-5*x+6", "variable": "x", "value": 2}
  verify_vieta            {"roots": [2,3], "coeff_a": 1, "coeff_b": -5, "coeff_c": 6}
  evaluate_expression     {"expression": "x**2+1", "substitutions": {"x": 3}}
  compute_hcf_lcm         {"numbers": [12, 18]}

REASON-ACT FORMAT — your JSON MUST follow this exactly:
{
  "approach": "<brief description of method>",
  "react_trace": [
    {"thought": "<reasoning>", "action": "<tool name>", "action_input": {<tool args>}},
    {"thought": "<interpretation of result>", "observation": "<tool output — filled by system>"}
  ],
  "steps": [
    {"step": 1, "description": "<what>", "work": "<how>",
     "tool_call": {"type": "solve_equation", "args": {"expression": "x**2-5*x+6", "variable": "x"}}}
  ],
  "final_answer": "<answer — must be consistent with tool results>",
  "tools_used": ["solve_equation", "verify_solution"]
}

SPEC RULES:
- parse_confidence < 0.6 → return error, do not guess
- Always verify at least one root by verify_solution after solve_equation
- For quadratics always run verify_vieta
- For matrices: verify A*A^-1 = I after inverse
- For probability: verify 0 <= P <= 1
- FORMATTING: use $ for inline math, $$ for block math

CRITICAL: RETURN ONLY VALID JSON. NO PREAMBLE. JUST THE JSON OBJECT."""

_RETRY_SYSTEM_PROMPT = """You are re-solving a JEE math problem after a failed verification.

The previous attempt failed. The failure reason is provided below.
Correct ONLY the identified issue and return a fresh solution JSON.

CRITICAL: RETURN ONLY VALID JSON. NO PREAMBLE."""

_CONSOLIDATION_SYSTEM_PROMPT = """You are reviewing a JEE math solution after tool results have been populated.
Correct any numerical errors in steps or final_answer to match tool results.
Return corrected solution JSON with EXACTLY the same structure. NO PREAMBLE."""


def _execute_tool_call(tool_call: dict) -> str:
    """
    Execute a single tool call from the solver's step plan.

    Accepts two formats:
      Legacy  : {"type": "SYMPY_SOLVE", "input": "x**2-4"}
      New spec: {"type": "solve_equation", "args": {"expression": ..., "variable": ...}}
    """
    if not tool_call:
        return ""

    tool_name = tool_call.get("type", "")
    # Normalise legacy uppercase names
    _legacy_map = {
        "SYMPY_SOLVE": "solve_equation",
        "SYMPY_SIMPLIFY": "simplify_expression",
        "SYMPY_DIFF": "simplify_expression",
        "SYMPY_INTEGRATE": "simplify_expression",
        "CALC": "evaluate_expression",
    }
    tool_name = _legacy_map.get(tool_name, tool_name)

    # Build kwargs from 'args' dict or legacy 'input' string
    if "args" in tool_call:
        kwargs = tool_call["args"]
    else:
        # Legacy format: single input string
        inp = tool_call.get("input", "")
        var = tool_call.get("variable", "x")
        kwargs = {"expression": inp, "variable": var}

    try:
        result = dispatch_tool(tool_name, **kwargs)
        if not result.get("success", False):
            return f"Tool error: {result.get('message', 'unknown error')}"
        # Return the most useful field
        for key in ("solutions", "solution_set", "range", "result", "factored",
                    "expanded", "determinant", "inverse", "sum", "probability",
                    "term", "eigenvalues_list", "derivative", "limit"):
            if key in result:
                return str(result[key])
        return str(result)
    except Exception as e:
        logger.error("Tool execution failed (%s): %s", tool_name, e)
        return f"Tool error: {e}"


def solve_problem(
    problem_text: str,
    topic: str,
    variables: list,
    constraints: list,
    strategy: str,
    memory_results: list | None = None,
    failure_reason: str = "",
    retry_attempt: int = 0,
) -> dict:
    """
    Solve a JEE algebra problem using ReAct loop + Math Tool Executor.

    Args:
        problem_text   : Cleaned problem statement from Parser Agent.
        topic          : Classified math topic from Router Agent.
        variables      : Extracted variables.
        constraints    : Problem constraints.
        strategy       : Solver strategy from Router Agent.
        memory_results : Similar past solutions from memory store.
        failure_reason : Verifier failure reason (non-empty on retry).
        retry_attempt  : Current retry number (0 = first attempt).

    Returns:
        dict with {success, approach, steps, final_answer, tools_used,
                   tool_results, rag_context, rag_sources, error}
    """
    rag_context = ""
    rag_results = []

    try:
        # ── Step 1: RAG Retrieval ──────────────────────────────────────────
        rag_results = retrieve(problem_text)
        rag_context = format_context(rag_results)

        # ── Step 2: Format memory context ─────────────────────────────────
        memory_context = ""
        if memory_results:
            parts = []
            for m in memory_results:
                entry = f"Similar problem: {m.get('problem', '')}\n"
                entry += (
                    f"Corrected answer: {m['corrected_answer']}\n"
                    if m.get("corrected_answer")
                    else f"Previous answer: {m.get('answer', '')}\n"
                )
                entry += f"Feedback: {m.get('feedback', 'none')}"
                parts.append(entry)
            memory_context = "\n---\n".join(parts)

        # ── Step 3: Build prompt ───────────────────────────────────────────
        retry_note = ""
        if failure_reason:
            retry_note = (
                f"\n\nRETRY ATTEMPT {retry_attempt}/3 — Previous verification FAILED:\n"
                f"Failure reason: {failure_reason}\n"
                "Please correct ONLY the identified issue.\n"
            )

        system_prompt = _RETRY_SYSTEM_PROMPT if failure_reason else _SOLVER_SYSTEM_PROMPT

        user_msg = (
            f"Problem: {problem_text}\n"
            f"Topic: {topic}\n"
            f"Variables: {variables}\n"
            f"Constraints: {constraints}\n"
            f"Strategy: {strategy}\n"
            f"\nKnowledge Base Context:\n{rag_context}\n"
            + (f"\nPast Similar Solutions:\n{memory_context}" if memory_context
               else "\nNo similar past solutions found.")
            + retry_note
            + "\n\nSolve this problem step by step using tool calls."
        )

        # ── Step 4: LLM Planning ───────────────────────────────────────────
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1 if failure_reason else 0.2,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        solution = parse_llm_json(content)

        # ── Step 5: Execute tool calls ─────────────────────────────────────
        tool_results = []
        steps = solution.get("steps", [])
        for step in steps:
            tc = step.get("tool_call")
            if tc:
                obs = _execute_tool_call(tc)
                tool_results.append({
                    "step":   step.get("step"),
                    "tool":   tc.get("type"),
                    "result": obs,
                })
                step["tool_result"] = obs

        # ── Step 6: Consolidation pass ─────────────────────────────────────
        if tool_results:
            try:
                tool_summary = "\n".join(
                    f"Step {tr['step']} [{tr['tool']}]: {tr['result']}"
                    for tr in tool_results
                )
                cons_msg = (
                    f"Original problem: {problem_text}\n\n"
                    f"Solution plan:\n{content}\n\n"
                    f"Actual tool results:\n{tool_summary}\n\n"
                    "Return the corrected JSON."
                )
                cons_resp = client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
                        {"role": "user",   "content": cons_msg},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
                corrected = parse_llm_json(cons_resp.choices[0].message.content or "")
                for orig, corr in zip(steps, corrected.get("steps", [])):
                    orig["description"] = corr.get("description", orig.get("description", ""))
                    orig["work"]        = corr.get("work",        orig.get("work", ""))
                new_answer = corrected.get("final_answer", "")
                if new_answer:
                    solution["final_answer"] = new_answer
                logger.info("Consolidation updated answer: %s",
                            solution["final_answer"][:80])
            except Exception as ce:
                logger.warning("Consolidation pass failed (keeping original): %s", ce)

        logger.info(
            "Solver done: steps=%d, tool_calls=%d, answer=%s",
            len(steps), len(tool_results),
            str(solution.get("final_answer", ""))[:80]
        )

        return {
            "success":      True,
            "approach":     solution.get("approach", ""),
            "steps":        steps,
            "final_answer": solution.get("final_answer", ""),
            "tools_used":   solution.get("tools_used", []),
            "tool_results": tool_results,
            "rag_context":  rag_context,
            "rag_sources":  [{"source": r["source"], "score": r["score"]} for r in rag_results],
            "memory_used":  bool(memory_results),
            "error":        None,
        }

    except Exception as e:
        logger.error("Solver agent failed: %s", e)
        return {
            "success":      False,
            "approach":     "",
            "steps":        [],
            "final_answer": "",
            "tools_used":   [],
            "tool_results": [],
            "rag_context":  rag_context,
            "rag_sources":  [],
            "memory_used":  False,
            "error":        str(e),
        }
