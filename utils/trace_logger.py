"""
Agent Trace Logger for Multimodal Math Mentor.
Tracks every agent's actions, inputs, and outputs for full transparency.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TraceStep:
    """A single step in the agent trace."""
    agent_name: str
    action: str
    input_summary: str
    output_summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class TraceLogger:
    """
    Collects agent trace steps for a single pipeline run.

    Usage:
        trace = TraceLogger()
        trace.log("Parser Agent", "Parsed problem", "Raw OCR text", "Extracted x² - 4 = 0")
        trace.log("Router Agent", "Classified topic", "x² - 4 = 0", "algebra")
        print(trace.format())
    """

    def __init__(self):
        self.steps: list[TraceStep] = []
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(
        self,
        agent_name: str,
        action: str,
        input_summary: str,
        output_summary: str,
        metadata: dict | None = None,
    ) -> None:
        """Record a trace step."""
        step = TraceStep(
            agent_name=agent_name,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata or {},
        )
        self.steps.append(step)

    def format(self) -> str:
        """Format the trace as a human-readable string for the UI."""
        lines = [f"🔍 Agent Trace (Run: {self.run_id})", "─" * 50]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.agent_name}")
            lines.append(f"  Action : {step.action}")
            lines.append(f"  Input  : {step.input_summary[:100]}")
            lines.append(f"  Output : {step.output_summary[:200]}")
            if step.metadata:
                for k, v in step.metadata.items():
                    lines.append(f"  {k}: {v}")
            lines.append("")
        return "\n".join(lines)

    def to_dict_list(self) -> list[dict]:
        """Convert trace to a list of dicts for JSON serialization."""
        return [
            {
                "step": i + 1,
                "agent_name": s.agent_name,
                "action": s.action,
                "input_summary": s.input_summary,
                "output_summary": s.output_summary,
                "timestamp": s.timestamp,
                "metadata": s.metadata,
            }
            for i, s in enumerate(self.steps)
        ]

    def clear(self) -> None:
        """Reset the trace for a new run."""
        self.steps.clear()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
