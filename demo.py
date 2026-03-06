"""
Demo Script for Multimodal Math Mentor.
Demonstrates the full pipeline without needing the UI.

Usage:
    python demo.py

Requires:
    - .env file with OPENAI_API_KEY
    - Run `python -m rag.ingest` first to build the knowledge base
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import run_pipeline
from rag.ingest import ingest_knowledge_base
from memory.memory_store import MemoryStore


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_trace(trace: list):
    print("\n🔍 Agent Trace:")
    print("─" * 40)
    for step in trace:
        print(f"  Step {step.get('step', '?')}: {step.get('agent_name', '?')}")
        print(f"    Action: {step.get('action', '')}")
        print(f"    Output: {step.get('output_summary', '')[:120]}")
        meta = step.get("metadata", {})
        if meta:
            for k, v in meta.items():
                print(f"    {k}: {v}")
        print()


def demo_text_question():
    """Demo 1: Solve a text question."""
    print_separator("DEMO 1: Text Question — Quadratic Equation")

    result = run_pipeline(
        input_type="text",
        raw_input="Solve the equation x² - 5x + 6 = 0. Find all roots.",
    )

    print(f"📐 Topic: {result.get('topic', 'N/A')}")
    print(f"✅ Answer: {result.get('final_answer', 'N/A')}")
    print(f"🎯 Confidence: {result.get('confidence', 0):.1%}")
    print(f"📊 Score Breakdown: {result.get('score_breakdown', {})}")

    explanation = result.get("explanation", {})
    if explanation.get("step_by_step"):
        print(f"\n📖 Explanation:\n{explanation['step_by_step'][:500]}")
    if explanation.get("key_insight"):
        print(f"\n💡 Key Insight: {explanation['key_insight']}")

    print_trace(result.get("trace", []))
    return result.get("interaction_id", 0)


def demo_memory_reuse(prev_id: int):
    """Demo 2: Ask a similar question — system should reuse past solution."""
    print_separator("DEMO 2: Memory Reuse — Similar Problem")

    result = run_pipeline(
        input_type="text",
        raw_input="Solve x² - 7x + 12 = 0",
    )

    print(f"📐 Topic: {result.get('topic', 'N/A')}")
    print(f"✅ Answer: {result.get('final_answer', 'N/A')}")
    print(f"🎯 Confidence: {result.get('confidence', 0):.1%}")

    # Check if memory was used
    trace = result.get("trace", [])
    for step in trace:
        if step.get("agent_name") == "Memory Retrieval":
            print(f"\n🧠 Memory Retrieved: {step.get('output_summary', '')}")
            break

    print_trace(trace)


def demo_feedback(interaction_id: int):
    """Demo 3: Submit feedback for self-learning."""
    print_separator("DEMO 3: User Feedback — Self-Learning")

    store = MemoryStore()

    # Simulate incorrect feedback
    success = store.apply_correction(
        interaction_id=interaction_id,
        feedback="correct",
        comment="Good solution!",
        corrected_answer="",
    )
    print(f"✅ Feedback saved: {success}")

    # Show history
    history = store.get_history(limit=5)
    print(f"\n📜 Recent History ({len(history)} entries):")
    for h in history:
        print(f"  #{h['id']} | {h.get('topic', '?')} | Answer: {h.get('final_answer', '?')[:50]} | Feedback: {h.get('user_feedback', 'none')}")


def demo_calculus():
    """Demo 4: Calculus question."""
    print_separator("DEMO 4: Calculus — Derivative")

    result = run_pipeline(
        input_type="text",
        raw_input="Find the derivative of f(x) = x³ · sin(x)",
    )

    print(f"📐 Topic: {result.get('topic', 'N/A')}")
    print(f"✅ Answer: {result.get('final_answer', 'N/A')}")
    print(f"🎯 Confidence: {result.get('confidence', 0):.1%}")
    print_trace(result.get("trace", []))


def demo_guardrail():
    """Demo 5: Guardrail blocking non-math input."""
    print_separator("DEMO 5: Guardrail — Non-Math Query")

    result = run_pipeline(
        input_type="text",
        raw_input="What is the capital of France?",
    )

    print(f"🛡️ Guardrail passed: {result.get('guardrail_passed', 'N/A')}")
    print(f"❌ Reason: {result.get('error', 'N/A')}")
    print_trace(result.get("trace", []))


if __name__ == "__main__":
    print("\n" + "🧮" * 20)
    print("  MULTIMODAL MATH MENTOR — DEMO")
    print("🧮" * 20)

    # Step 0: Ingest knowledge base
    print_separator("SETUP: Ingesting Knowledge Base")
    try:
        result = ingest_knowledge_base()
        print(f"✅ Ingested: {result}")
    except Exception as e:
        print(f"⚠️ Ingestion warning: {e}")
        print("  (Pipeline will still work, but RAG sources may be limited)")

    # Run demos
    interaction_id = demo_text_question()
    demo_memory_reuse(interaction_id)
    demo_feedback(interaction_id)
    demo_calculus()
    demo_guardrail()

    print_separator("DEMO COMPLETE")
    print("🎉 All demos finished! The system demonstrates:")
    print("  ✅ Multi-agent pipeline (Guardrail → Parser → Router → Solver → Verifier → Explainer)")
    print("  ✅ RAG retrieval from knowledge base")
    print("  ✅ Memory-based self-learning from corrections")
    print("  ✅ Agent trace transparency")
    print("  ✅ Hybrid confidence scoring")
    print("  ✅ Guardrail protection")
