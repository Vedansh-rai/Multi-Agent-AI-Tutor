"""
Agent Orchestrator — JEE Algebra Solver
=========================================
LangGraph-style stateful pipeline:
  Guardrail → Parser → Router → Memory → Solver → Verifier → [retry loop, max 3]
  → Explainer → Memory Store

Retry logic: if verifier confidence < 0.8 and retry_count < 3, route back to
Solver with the failure_reason so the LLM can fix its approach.
"""

from typing import TypedDict
from utils.logger import get_logger
from utils.trace_logger import TraceLogger

logger = get_logger("agents.orchestrator")

_CONFIDENCE_THRESHOLD = 0.8
_MAX_RETRIES = 3


# ────────────────────────────────────────────────
# Pipeline State
# ────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """Shared state passed between all agents in the pipeline."""
    # Input
    input_type: str            # 'text', 'image', 'audio'
    raw_input: str             # Raw user input / OCR / ASR text
    input_confidence: float    # OCR or ASR confidence

    # Guardrail
    guardrail_passed: bool
    guardrail_reason: str

    # Parser
    problem_text: str
    topic: str
    problem_type: str          # from router (e.g. 'equation_solving')
    variables: list
    constraints: list
    needs_clarification: bool
    clarification_reason: str

    # Router
    router_topic: str
    router_confidence: float
    solver_strategy: str
    tool_chain: list
    rag_query: str
    fallback_strategy: str

    # Retry loop
    retry_count: int           # current retry attempt (0 = first pass)
    failure_reason: str        # why the verifier rejected the last answer

    # Memory
    memory_results: list

    # Solver
    approach: str
    solution_steps: list
    final_answer: str
    tools_used: list
    tool_results: list
    rag_context: str
    rag_sources: list

    # Verifier
    confidence: float
    score_breakdown: dict
    verification_issues: list
    needs_human_review: bool
    needs_retry: bool
    is_verified: bool

    # Explainer
    explanation: dict

    # Metadata
    trace: list
    interaction_id: int
    error: str
    status: str                # 'SOLVED' | 'UNSOLVED' | 'PARTIAL' | 'BLOCKED'
    hitl_required: bool
    hitl_type: str             # 'parser_ambiguity' | 'low_confidence' | 'ocr_review'


# ────────────────────────────────────────────────
# Node Functions
# ────────────────────────────────────────────────

def guardrail_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Run guardrail checks on input."""
    from agents.guardrail_agent import guardrail_check

    result = guardrail_check(state["raw_input"])

    trace.log(
        "Guardrail Agent",
        "Input validation",
        state["raw_input"][:80],
        f"safe={result['is_safe']}, math={result['is_math']}",
        metadata={"blocked_reason": result.get("blocked_reason")},
    )

    state["guardrail_passed"] = result["is_safe"] and result["is_math"]
    state["guardrail_reason"] = result.get("blocked_reason") or ""
    return state


def parser_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Parse and structure the math problem."""
    from agents.parser_agent import parse_problem

    parsed = parse_problem(state["raw_input"], state["input_type"])

    trace.log(
        "Parser Agent",
        "Parsed problem",
        state["raw_input"][:80],
        f"topic={parsed['topic']}, vars={parsed['variables']}",
        metadata={"needs_clarification": parsed["needs_clarification"]},
    )

    state["problem_text"] = parsed["problem_text"]
    state["topic"] = parsed["topic"]
    state["problem_type"] = parsed.get("problem_type", "")
    state["variables"] = parsed["variables"]
    state["constraints"] = parsed["constraints"]
    state["needs_clarification"] = parsed["needs_clarification"]
    state["clarification_reason"] = parsed.get("clarification_reason", "")

    if parsed["needs_clarification"]:
        state["hitl_required"] = True
        state["hitl_type"] = "parser_ambiguity"

    return state


def router_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Classify topic and determine solver strategy."""
    from agents.router_agent import classify_and_route

    result = classify_and_route(state["problem_text"], state)

    trace.log(
        "Router Agent",
        "Classified problem",
        state["problem_text"][:80],
        f"type={result['problem_type']}, confidence={result['router_confidence']:.2f}",
        metadata={"all_scores": result.get("all_scores", {})},
    )

    state["router_topic"] = result.get("topic", state.get("topic", ""))
    state["router_confidence"] = result.get("router_confidence", 0.0)
    state["solver_strategy"] = result.get("solver_strategy", "")
    state["problem_type"] = result.get("problem_type", "")
    state["tool_chain"] = result.get("tool_chain", [])
    state["rag_query"] = result.get("rag_query", state["problem_text"])
    state["fallback_strategy"] = result.get("fallback_strategy", "")
    state["topic"] = result.get("topic", state.get("topic", ""))
    return state


def memory_retrieval_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Retrieve similar past problems from memory."""
    from memory.memory_store import MemoryStore

    try:
        store = MemoryStore()
        similar = store.get_similar_problems(state["problem_text"], top_k=3)
        state["memory_results"] = similar

        trace.log(
            "Memory Retrieval",
            "Searched past interactions",
            state["problem_text"][:80],
            f"Found {len(similar)} similar problems",
            metadata={
                "top_similarity": similar[0]["similarity_score"] if similar else 0
            },
        )
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)
        state["memory_results"] = []
        trace.log("Memory Retrieval", "Search failed", state["problem_text"][:80], str(e))

    return state


def solver_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Solve the math problem (called on first pass and each retry)."""
    from agents.solver_agent import solve_problem

    retry_attempt = state.get("retry_count", 0)
    failure_reason = state.get("failure_reason", "")

    result = solve_problem(
        problem_text=state["problem_text"],
        topic=state["topic"],
        variables=state.get("variables", []),
        constraints=state.get("constraints", []),
        strategy=state.get("solver_strategy", ""),
        memory_results=state.get("memory_results"),
        failure_reason=failure_reason,
        retry_attempt=retry_attempt,
    )

    trace.log(
        "Solver Agent",
        f"Solved (attempt {retry_attempt + 1})",
        state["problem_text"][:80],
        f"answer={str(result.get('final_answer', 'N/A'))[:100]}",
        metadata={
            "tools_used": result.get("tools_used", []),
            "rag_sources_count": len(result.get("rag_sources", [])),
            "memory_used": result.get("memory_used", False),
            "retry_attempt": retry_attempt,
        },
    )

    state["approach"] = result.get("approach", "")
    state["solution_steps"] = result.get("steps", [])

    raw_answer = result.get("final_answer", "")
    if isinstance(raw_answer, dict):
        state["final_answer"] = str(raw_answer.get("final_answer", raw_answer))
    else:
        state["final_answer"] = str(raw_answer) if raw_answer else ""

    state["tools_used"] = result.get("tools_used", [])
    state["tool_results"] = result.get("tool_results", [])
    state["rag_context"] = result.get("rag_context", "")
    state["rag_sources"] = result.get("rag_sources", [])
    return state


def verifier_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Verify the solution; set needs_retry=True when confidence < 0.8."""
    from agents.verifier_agent import run_verification

    result = run_verification(
        problem_text=state["problem_text"],
        final_answer=state.get("final_answer", ""),
        solution_steps=state.get("solution_steps", []),
        topic=state["topic"],
        constraints=state.get("constraints", []),
        rag_sources=state.get("rag_sources", []),
    )

    confidence = result.get("confidence_score", result.get("confidence", 0.0))
    needs_retry = result.get("needs_retry", confidence < _CONFIDENCE_THRESHOLD)

    trace.log(
        "Verifier Agent",
        f"Confidence = {confidence:.3f} (attempt {state.get('retry_count', 0) + 1})",
        f"Answer: {state.get('final_answer', '')[:80]}",
        f"verified={result.get('verified')}, needs_retry={needs_retry}",
        metadata=result.get("score_breakdown", {}),
    )

    state["confidence"] = confidence
    state["score_breakdown"] = result.get("score_breakdown", {})
    state["verification_issues"] = result.get("issues", [])
    state["is_verified"] = result.get("verified", False)
    state["needs_retry"] = needs_retry
    state["failure_reason"] = result.get("failure_reason", "")
    state["needs_human_review"] = result.get("needs_human_review", False)

    if result["needs_human_review"] and not needs_retry:
        state["hitl_required"] = True
        state["hitl_type"] = "low_confidence"

    return state


def explainer_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Generate the student-friendly JEE explanation."""
    from agents.explainer_agent import generate_explanation

    result = generate_explanation(
        problem_text=state["problem_text"],
        topic=state["topic"],
        solution_steps=state.get("solution_steps", []),
        final_answer=state.get("final_answer", ""),
        verification_result={
            "confidence_score": state.get("confidence", 0),
            "issues": state.get("verification_issues", []),
        },
        problem_type=state.get("problem_type", ""),
    )

    trace.log(
        "Explainer Agent",
        "Generated explanation",
        state.get("final_answer", "")[:80],
        f"concept={result.get('concept_identified', '?')}, difficulty={result.get('difficulty_level', '?')}",
    )

    state["explanation"] = result
    return state


def save_to_memory_node(state: AgentState, trace: TraceLogger) -> AgentState:
    """Save the full interaction to memory for self-learning."""
    from memory.memory_store import MemoryStore

    try:
        store = MemoryStore()
        interaction_id = store.save_interaction(
            input_type=state.get("input_type", "text"),
            raw_input=state.get("raw_input", ""),
            parsed_problem=state.get("problem_text", ""),
            topic=state.get("topic", ""),
            retrieved_context=state.get("rag_context", ""),
            solution_steps=str(state.get("solution_steps", [])),
            final_answer=state.get("final_answer", ""),
            explanation=str(
                state.get("explanation", {}).get("formatted_explanation")
                or state.get("explanation", {}).get("step_by_step", "")
            ),
            confidence=state.get("confidence", 0.0),
        )
        state["interaction_id"] = interaction_id

        trace.log(
            "Memory Store",
            "Saved interaction",
            f"ID: {interaction_id}",
            f"topic={state.get('topic', '?')}, confidence={state.get('confidence', 0)}",
        )
    except Exception as e:
        logger.error("Failed to save to memory: %s", e)
        trace.log("Memory Store", "Save failed", "", str(e))

    return state


# ────────────────────────────────────────────────
# Pipeline Runner
# ────────────────────────────────────────────────

def run_pipeline(
    input_type: str,
    raw_input: str,
    input_confidence: float = 1.0,
) -> dict:
    """
    Run the full agent pipeline with retry loop.

    Retry logic: if verifier confidence < 0.8 and retry_count < 3,
    the solver is called again with the failure_reason so it can
    correct its approach. If all retries are exhausted the status
    is set to 'UNSOLVED'.

    Args:
        input_type: 'text', 'image', or 'audio'.
        raw_input: The text to process.
        input_confidence: OCR/ASR confidence score.

    Returns:
        Full pipeline result dict with all agent outputs and trace.
    """
    trace = TraceLogger()

    state: AgentState = {
        "input_type": input_type,
        "raw_input": raw_input,
        "input_confidence": input_confidence,
        "guardrail_passed": False,
        "guardrail_reason": "",
        "problem_text": "",
        "topic": "",
        "problem_type": "",
        "variables": [],
        "constraints": [],
        "needs_clarification": False,
        "clarification_reason": "",
        "router_topic": "",
        "router_confidence": 0.0,
        "solver_strategy": "",
        "tool_chain": [],
        "rag_query": "",
        "fallback_strategy": "",
        "retry_count": 0,
        "failure_reason": "",
        "memory_results": [],
        "approach": "",
        "solution_steps": [],
        "final_answer": "",
        "tools_used": [],
        "tool_results": [],
        "rag_context": "",
        "rag_sources": [],
        "confidence": 0.0,
        "score_breakdown": {},
        "verification_issues": [],
        "needs_human_review": False,
        "needs_retry": False,
        "is_verified": False,
        "explanation": {},
        "trace": [],
        "interaction_id": 0,
        "error": "",
        "status": "PARTIAL",
        "hitl_required": False,
        "hitl_type": "",
    }

    try:
        # ── Step 1: Guardrail ──
        logger.info("Pipeline: Running guardrail check...")
        state = guardrail_node(state, trace)

        if not state["guardrail_passed"]:
            state["status"] = "BLOCKED"
            state["trace"] = trace.to_dict_list()
            state["error"] = state["guardrail_reason"]
            return state

        # ── Step 2: Parser ──
        logger.info("Pipeline: Parsing problem...")
        state = parser_node(state, trace)

        if state.get("needs_clarification") and not state.get("raw_input"):
            state["status"] = "PARTIAL"
            state["trace"] = trace.to_dict_list()
            return state

        # ── Step 3: Router ──
        logger.info("Pipeline: Routing to topic...")
        state = router_node(state, trace)

        # ── Step 4: Memory Retrieval ──
        logger.info("Pipeline: Retrieving from memory...")
        state = memory_retrieval_node(state, trace)

        # ── Steps 5-6: Solver → Verifier retry loop ──
        for attempt in range(_MAX_RETRIES):
            state["retry_count"] = attempt

            logger.info("Pipeline: Solving (attempt %d/%d)...", attempt + 1, _MAX_RETRIES)
            state = solver_node(state, trace)

            logger.info("Pipeline: Verifying solution...")
            state = verifier_node(state, trace)

            if not state.get("needs_retry", False):
                logger.info(
                    "Pipeline: Confidence %.3f >= %.1f — accepted after %d attempt(s).",
                    state["confidence"], _CONFIDENCE_THRESHOLD, attempt + 1,
                )
                break

            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "Pipeline: Confidence %.3f < %.1f — retrying (attempt %d/%d). Reason: %s",
                    state["confidence"], _CONFIDENCE_THRESHOLD,
                    attempt + 2, _MAX_RETRIES,
                    state.get("failure_reason", "unknown"),
                )
            else:
                logger.warning(
                    "Pipeline: All %d retries exhausted. Marking as UNSOLVED.", _MAX_RETRIES
                )
                state["status"] = "UNSOLVED"
                state["trace"] = trace.to_dict_list()
                state = save_to_memory_node(state, trace)
                state["trace"] = trace.to_dict_list()
                return state

        state["status"] = "SOLVED" if state.get("is_verified") else "PARTIAL"

        # ── Step 7: Explainer ──
        logger.info("Pipeline: Generating explanation...")
        state = explainer_node(state, trace)

        # ── Step 8: Save to Memory ──
        logger.info("Pipeline: Saving to memory...")
        state = save_to_memory_node(state, trace)

        state["trace"] = trace.to_dict_list()
        logger.info(
            "Pipeline completed (status=%s, confidence=%.3f, retries=%d)",
            state["status"], state["confidence"], state.get("retry_count", 0),
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        state["error"] = str(e)
        state["status"] = "PARTIAL"
        trace.log("Pipeline", "Error", "", str(e))
        state["trace"] = trace.to_dict_list()

    return state
